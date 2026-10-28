"""Out-of-Distribution (OOD) Guardrail for Pinch Analysis ANN Surrogates.

This module provides multi-tier safety checks for surrogate predictions:
1. Operational domain range checks (stream counts, temperatures, approach, U).
2. Feature-space density / distance-based anomaly detection (Isolation Forest).
3. Thermodynamic physical consistency validation (energy conservation, positive area).
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class OODGuardrail:
    """Out-of-distribution guardrail and physical consistency validator.

    Attributes
    ----------
    contamination : float
        Expected fraction of outliers for the density detector (default 0.05).
    random_state : int
        Seed for detector reproducibility.
    tolerance : float
        Allowable fractional error in energy balance verification.
    """

    contamination: float = 0.05
    random_state: int = 42
    tolerance: float = 0.15

    # Feature bounding boxes learned during training
    dtmin_range: tuple[float, float] = (3.0, 50.0)
    u_range: tuple[float, float] = (0.02, 1.0)
    temp_range: tuple[float, float] = (10.0, 500.0)
    cp_max: float = 25.0
    total_cp_max: float = 60.0
    max_hot_capacity: int = 5
    max_cold_capacity: int = 5

    # Density estimator
    detector: IsolationForest | None = None
    is_fitted: bool = False

    def fit(self, X: np.ndarray, feature_names_list: list[str] | None = None) -> OODGuardrail:
        """Fit the domain bounds and anomaly detector on training features."""
        X_arr = np.asarray(X, dtype=float)
        if len(X_arr) == 0:
            raise ValueError("Cannot fit OODGuardrail on empty feature matrix.")

        # Fit IsolationForest density detector on training feature vectors
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.detector.fit(X_arr)
        self.is_fitted = True

        # Compute empirical bounds with safety margin
        if X_arr.shape[1] >= 2:
            dt_min_val, dt_max_val = float(np.min(X_arr[:, 0])), float(np.max(X_arr[:, 0]))
            self.dtmin_range = (max(1.0, dt_min_val * 0.8), dt_max_val * 1.25)

            u_min_val, u_max_val = float(np.min(X_arr[:, 1])), float(np.max(X_arr[:, 1]))
            self.u_range = (max(0.01, u_min_val * 0.8), u_max_val * 1.25)

        if X_arr.shape[1] >= 14:
            min_c_supply = float(np.min(X_arr[:, 11]))
            max_h_supply = float(np.max(X_arr[:, 9]))
            self.temp_range = (max(0.0, min_c_supply * 0.5), max_h_supply * 1.25)

            max_tot_cp = max(float(np.max(X_arr[:, 7])), float(np.max(X_arr[:, 8])))
            self.total_cp_max = max_tot_cp * 1.5

        if X_arr.shape[1] >= 53:
            cp_cols = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
            self.cp_max = float(np.max(X_arr[:, cp_cols])) * 1.5

        return self

    def check_input(self, x: np.ndarray) -> tuple[bool, float, str | None]:
        """Check whether an input feature vector is out-of-distribution.

        Returns
        -------
        is_ood : bool
            True if the input is flagged as outside the training distribution.
        score : float
            Anomaly decision score (higher is more normal, negative indicates outlier).
        reason : str or None
            Explanation if flagged, else None.
        """
        x_arr = np.asarray(x, dtype=float).reshape(1, -1)

        # Tier 1: Domain bounding checks
        delta_t_min = float(x_arr[0, 0])
        u_val = float(x_arr[0, 1])
        n_hot = int(round(x_arr[0, 2]))
        n_cold = int(round(x_arr[0, 3]))

        if delta_t_min <= 0:
            return True, -1.0, f"Non-physical ΔTmin: {delta_t_min:.1f} °C (must be > 0)."

        if delta_t_min < self.dtmin_range[0] or delta_t_min > self.dtmin_range[1]:
            return True, -0.8, (
                f"ΔTmin {delta_t_min:.1f} °C outside training range "
                f"[{self.dtmin_range[0]:.1f}, {self.dtmin_range[1]:.1f}] °C."
            )

        if u_val <= 0:
            return True, -1.0, f"Non-physical heat transfer coefficient U: {u_val} (must be > 0)."

        if u_val < self.u_range[0] or u_val > self.u_range[1]:
            return True, -0.8, (
                f"U coefficient {u_val:.3f} outside training range "
                f"[{self.u_range[0]:.3f}, {self.u_range[1]:.3f}]."
            )

        if n_hot <= 0 or n_hot > self.max_hot_capacity:
            return True, -0.9, (
                f"Number of hot streams ({n_hot}) outside supported range "
                f"[1, {self.max_hot_capacity}]."
            )

        if n_cold <= 0 or n_cold > self.max_cold_capacity:
            return True, -0.9, (
                f"Number of cold streams ({n_cold}) outside supported range "
                f"[1, {self.max_cold_capacity}]."
            )

        # Check stream temperatures in features
        if x_arr.shape[1] >= 12:
            max_hot_supply = float(x_arr[0, 9])
            min_cold_supply = float(x_arr[0, 11])
            if max_hot_supply > self.temp_range[1] or min_cold_supply < self.temp_range[0]:
                return True, -0.7, (
                    f"Stream temperatures span [{min_cold_supply:.1f}, {max_hot_supply:.1f}] °C, "
                    f"outside operational envelope [{self.temp_range[0]:.1f}, {self.temp_range[1]:.1f}] °C."
                )

        # Check stream CP in features
        if x_arr.shape[1] >= 9:
            tot_h_cp = float(x_arr[0, 7])
            tot_c_cp = float(x_arr[0, 8])
            if tot_h_cp > self.total_cp_max or tot_c_cp > self.total_cp_max:
                return True, -0.7, (
                    f"Total CP ({max(tot_h_cp, tot_c_cp):.2f} kW/°C) exceeds training bound "
                    f"({self.total_cp_max:.2f} kW/°C)."
                )

        if x_arr.shape[1] >= 53:
            cp_cols = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
            max_stream_cp = float(np.max(x_arr[0, cp_cols]))
            if max_stream_cp > self.cp_max:
                return True, -0.7, (
                    f"Stream CP ({max_stream_cp:.2f} kW/°C) exceeds individual CP envelope "
                    f"({self.cp_max:.2f} kW/°C)."
                )

        # Tier 2: Density / Distance detector
        score = 0.0
        if self.is_fitted and self.detector is not None:
            score = float(self.detector.decision_function(x_arr)[0])
            is_inlier = int(self.detector.predict(x_arr)[0]) == 1
            if not is_inlier:
                return True, score, f"Feature density score ({score:.3f}) falls below nominal inlier boundary."

        return False, score, None

    def check_input_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
        """Vectorized batch out-of-distribution detection for feature matrix X.

        Parameters
        ----------
        X : np.ndarray
            Matrix of shape (N, n_features).

        Returns
        -------
        is_ood_mask : np.ndarray of bool
            Boolean array of length N.
        scores : np.ndarray of float
            Decision function anomaly scores of length N.
        reasons : list of str or None
            Explanations for flagged rows.
        """
        X_arr = np.asarray(X, dtype=float)
        n_samples = len(X_arr)
        is_ood = np.zeros(n_samples, dtype=bool)
        reasons: list[str | None] = [None] * n_samples

        # Tier 1 Vectorized Bounds
        dtmin = X_arr[:, 0]
        u_val = X_arr[:, 1]
        n_hot = np.round(X_arr[:, 2])
        n_cold = np.round(X_arr[:, 3])

        t1_mask = (
            (dtmin <= 0) | (dtmin < self.dtmin_range[0]) | (dtmin > self.dtmin_range[1]) |
            (u_val <= 0) | (u_val < self.u_range[0]) | (u_val > self.u_range[1]) |
            (n_hot <= 0) | (n_hot > self.max_hot_capacity) |
            (n_cold <= 0) | (n_cold > self.max_cold_capacity)
        )
        if X_arr.shape[1] >= 12:
            t1_mask |= (X_arr[:, 9] > self.temp_range[1]) | (X_arr[:, 11] < self.temp_range[0])
        if X_arr.shape[1] >= 9:
            t1_mask |= (X_arr[:, 7] > self.total_cp_max) | (X_arr[:, 8] > self.total_cp_max)
        if X_arr.shape[1] >= 53:
            cp_cols = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]
            t1_mask |= (np.max(X_arr[:, cp_cols], axis=1) > self.cp_max)

        is_ood |= t1_mask
        for idx in np.where(t1_mask)[0]:
            reasons[idx] = "Operational bounds or capacity violated"

        # Tier 2 Vectorized Isolation Forest
        scores = np.zeros(n_samples, dtype=float)
        if self.is_fitted and self.detector is not None:
            scores = self.detector.decision_function(X_arr)
            density_outliers = (self.detector.predict(X_arr) == -1)
            t2_new = density_outliers & ~is_ood
            is_ood |= density_outliers
            for idx in np.where(t2_new)[0]:
                reasons[idx] = f"Feature density score ({scores[idx]:.3f}) below inlier threshold"

        return is_ood, scores, reasons

    def validate_physics(
        self,
        predictions: np.ndarray | list[float],
        total_hot_duty: float,
        total_cold_duty: float,
    ) -> tuple[bool, str | None]:
        """Verify thermodynamic consistency of surrogate predictions.

        Parameters
        ----------
        predictions : array-like
            Target predictions: [A_target, Q_H_min, Q_C_min].
        total_hot_duty : float
            Sum of hot stream sensible duties (kW).
        total_cold_duty : float
            Sum of cold stream sensible duties (kW).

        Returns
        -------
        is_valid : bool
            True if predictions satisfy physical thermodynamic constraints.
        reason : str or None
            Explanation if violated.
        """
        preds = [float(p) for p in predictions]
        if len(preds) < 3:
            return True, None

        area, q_h, q_c = preds[0], preds[1], preds[2]

        # 1. Non-negativity check
        if area < 0.0 or math.isnan(area):
            return False, f"Predicted area is negative or NaN ({area:.2f} m²)."

        scale = max(total_hot_duty, total_cold_duty, 10.0)
        neg_limit = -max(1e-2, 0.01 * scale)
        if q_h < neg_limit or math.isnan(q_h):
            return False, f"Predicted hot utility is negative ({q_h:.2f} kW, exceeds noise threshold {neg_limit:.2f} kW)."

        if q_c < neg_limit or math.isnan(q_c):
            return False, f"Predicted cold utility is negative ({q_c:.2f} kW, exceeds noise threshold {neg_limit:.2f} kW)."

        # 2. First Law Enthalpy Conservation Check
        # In pinch analysis: Q_C,min - Q_H,min = total_hot_duty - total_cold_duty
        expected_difference = total_hot_duty - total_cold_duty
        predicted_difference = q_c - q_h
        discrepancy = abs(predicted_difference - expected_difference)

        relative_discrepancy = discrepancy / scale

        if relative_discrepancy > self.tolerance:
            return False, (
                f"First Law energy conservation violated: "
                f"predicted (Q_C - Q_H) = {predicted_difference:.1f} kW vs "
                f"net sensible duty balance (Q_hot - Q_cold) = {expected_difference:.1f} kW "
                f"(relative discrepancy: {relative_discrepancy:.1%}, exceeds threshold {self.tolerance:.1%})."
            )


        return True, None

    def validate_physics_batch(
        self,
        predictions: np.ndarray,
        total_hot_duties: np.ndarray,
        total_cold_duties: np.ndarray,
    ) -> tuple[np.ndarray, list[str | None]]:
        """Vectorized batch validation of physical consistency.

        Returns
        -------
        is_valid_mask : np.ndarray of bool
            True for samples that satisfy all thermodynamic constraints.
        reasons : list of str or None
            Explanations for rejected samples.
        """
        preds = np.asarray(predictions, dtype=float)
        tot_h = np.asarray(total_hot_duties, dtype=float)
        tot_c = np.asarray(total_cold_duties, dtype=float)
        n_samples = len(preds)

        scale = np.maximum(np.maximum(tot_h, tot_c), 10.0)
        neg_limits = -np.maximum(1e-2, 0.01 * scale)

        area = preds[:, 0]
        q_h = preds[:, 1]
        q_c = preds[:, 2]

        area_fail = (area < 0.0) | np.isnan(area)
        qh_fail = (q_h < neg_limits) | np.isnan(q_h)
        qc_fail = (q_c < neg_limits) | np.isnan(q_c)

        expected_diff = tot_h - tot_c
        pred_diff = q_c - q_h
        discrepancy = np.abs(pred_diff - expected_diff)
        rel_disc = discrepancy / scale
        balance_fail = rel_disc > self.tolerance

        fail_mask = area_fail | qh_fail | qc_fail | balance_fail
        valid_mask = ~fail_mask

        reasons: list[str | None] = [None] * n_samples
        for idx in np.where(fail_mask)[0]:
            if area_fail[idx]:
                reasons[idx] = f"Predicted area is negative ({area[idx]:.2f} m²)"
            elif qh_fail[idx] or qc_fail[idx]:
                reasons[idx] = f"Negative utility predicted (Qh={q_h[idx]:.1f}, Qc={q_c[idx]:.1f} kW)"
            elif balance_fail[idx]:
                reasons[idx] = f"First Law violation: rel_error={rel_disc[idx]:.1%} > {self.tolerance:.1%}"

        return valid_mask, reasons

    def save(self, filepath: str | Path) -> None:
        """Serialize guardrail to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> OODGuardrail:
        """Load serialized guardrail from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

