"""High-level prediction and guarded surrogate inference for PinchPy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..analysis.area_target import area_target
from ..analysis.pinch_point import utility_targets
from ..models.stream import Stream
from .features import case_features
from .guardrail import OODGuardrail


@dataclass
class FastANNPredictor:
    """NumPy-accelerated inference engine for trained ANN pipelines.

    Extracts StandardScaler parameters, MLP weights and biases, and
    target transformer scaling to execute forward passes and inverse
    transformations directly in vectorized BLAS NumPy without scikit-learn
    validation overhead.
    """

    in_mean: np.ndarray | None
    in_scale: np.ndarray | None
    coefs: list[np.ndarray]
    intercepts: list[np.ndarray]
    out_mean: np.ndarray | None = None
    out_scale: np.ndarray | None = None
    has_log1p_area: bool = True

    @classmethod
    def from_model(cls, model: Any) -> FastANNPredictor | None:
        """Extract neural network parameters from a fitted pipeline or regressor."""
        if model is None:
            return None

        # 1. Inspect TransformedTargetRegressor wrapper
        target_transformer = getattr(model, "transformer_", getattr(model, "transformer", None))
        regressor = getattr(model, "regressor_", getattr(model, "regressor", model))

        # 2. Inspect Regressor Pipeline or MLP
        feature_scaler = None
        mlp = None

        if hasattr(regressor, "named_steps"):
            feature_scaler = regressor.named_steps.get("scaler")
            mlp = regressor.named_steps.get("ann", regressor.named_steps.get("mlp"))
        elif hasattr(regressor, "coefs_"):
            mlp = regressor

        if mlp is None or not hasattr(mlp, "coefs_") or not hasattr(mlp, "intercepts_"):
            return None

        # 3. Extract Feature Scaler Parameters
        in_mean = None
        in_scale = None
        if feature_scaler is not None and hasattr(feature_scaler, "mean_") and hasattr(feature_scaler, "scale_"):
            in_mean = np.asarray(feature_scaler.mean_, dtype=float)
            in_scale = np.asarray(feature_scaler.scale_, dtype=float)

        # 4. Extract MLP Weights and Biases
        coefs = [np.asarray(w, dtype=float) for w in mlp.coefs_]
        intercepts = [np.asarray(b, dtype=float) for b in mlp.intercepts_]

        # 5. Extract Target Transformer Scaler Parameters
        out_mean = None
        out_scale = None
        has_log1p = False

        if target_transformer is not None:
            t_scaler = getattr(target_transformer, "scaler", None)
            if t_scaler is not None and hasattr(t_scaler, "mean_") and hasattr(t_scaler, "scale_"):
                out_mean = np.asarray(t_scaler.mean_, dtype=float)
                out_scale = np.asarray(t_scaler.scale_, dtype=float)
            if hasattr(target_transformer, "inverse_transform"):
                cls_name = target_transformer.__class__.__name__.lower()
                if "log" in cls_name or "log1p" in cls_name:
                    has_log1p = True

        return cls(
            in_mean=in_mean,
            in_scale=in_scale,
            coefs=coefs,
            intercepts=intercepts,
            out_mean=out_mean,
            out_scale=out_scale,
            has_log1p_area=has_log1p,
        )

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """Execute single-case forward pass and inverse in pure NumPy."""
        x_vec = np.asarray(x, dtype=float).ravel()
        if self.in_mean is not None and self.in_scale is not None:
            h = (x_vec - self.in_mean) / self.in_scale
        else:
            h = x_vec.copy()

        n_layers = len(self.coefs)
        for i in range(n_layers - 1):
            h = np.maximum(0.0, h @ self.coefs[i] + self.intercepts[i])
        out = h @ self.coefs[-1] + self.intercepts[-1]

        if self.out_scale is not None and self.out_mean is not None:
            out = out * self.out_scale + self.out_mean

        if self.has_log1p_area:
            clipped_exp = np.clip(out[0], -50.0, 50.0)
            out[0] = np.expm1(clipped_exp)

        return np.maximum(0.0, out)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Execute batch forward pass and inverse in pure NumPy."""
        X_mat = np.asarray(X, dtype=float)
        if self.in_mean is not None and self.in_scale is not None:
            H = (X_mat - self.in_mean) / self.in_scale
        else:
            H = X_mat.copy()

        n_layers = len(self.coefs)
        for i in range(n_layers - 1):
            H = np.maximum(0.0, H @ self.coefs[i] + self.intercepts[i])
        Out = H @ self.coefs[-1] + self.intercepts[-1]

        if self.out_scale is not None and self.out_mean is not None:
            Out = Out * self.out_scale + self.out_mean

        if self.has_log1p_area:
            clipped_exp = np.clip(Out[:, 0], -50.0, 50.0)
            Out[:, 0] = np.expm1(clipped_exp)

        return np.maximum(0.0, Out)


def get_fast_ann_predictor(model: Any) -> FastANNPredictor | None:
    """Retrieve or construct a cached NumPy fast predictor for the model."""
    fast_predictor = getattr(model, "_fast_ann_predictor", None)
    if fast_predictor is not None:
        return fast_predictor
    predictor = FastANNPredictor.from_model(model)
    if predictor is not None:
        try:
            model._fast_ann_predictor = predictor
        except Exception:
            pass
    return predictor


def predict_ann_fast(model: Any, x: np.ndarray) -> np.ndarray:
    """Predict targets for single case with NumPy acceleration if supported."""
    predictor = get_fast_ann_predictor(model)
    if predictor is not None:
        return predictor.predict_single(x)
    return np.asarray(model.predict(np.asarray(x, dtype=float).reshape(1, -1))[0], dtype=float)


def predict_targets(model: Any, X: Any) -> np.ndarray:
    """Predict targets using fitted model pipeline (backward-compatible API).

    Returns
    -------
    np.ndarray
        Array of predicted targets [A_target, Q_H_min, Q_C_min].
    """
    X_arr = np.asarray(X, dtype=float)
    predictor = get_fast_ann_predictor(model)
    if predictor is not None:
        if X_arr.ndim == 1:
            return predictor.predict_single(X_arr)
        return predictor.predict_batch(X_arr)
    return np.asarray(model.predict(X_arr))


def predict_network_targets(
    streams: list[Stream],
    delta_t_min: float = 10.0,
    overall_heat_transfer_coefficient: float = 0.1,
    model: Any | None = None,
    guardrail: OODGuardrail | None = None,
    auto_fallback: bool = True,
) -> dict[str, Any]:
    """Predict network targets (Area, Hot Utility, Cold Utility) with OOD guardrails.

    If the input stream configuration is identified as Out-of-Distribution or if
    the surrogate output violates thermodynamic physical consistency, this function
    automatically and safely falls back to the exact deterministic engine.

    Parameters
    ----------
    streams : list of Stream
        Process streams requiring heating or cooling.
    delta_t_min : float, default 10.0
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, default 0.1
        Overall heat transfer coefficient U in kW/(m²·°C).
    model : Pipeline, optional
        Trained ANN surrogate model. If None, uses deterministic engine directly.
    guardrail : OODGuardrail, optional
        Fitted OOD guardrail instance for anomaly detection.
    auto_fallback : bool, default True
        If True, automatically compute exact values from deterministic engine
        when inputs are flagged OOD or non-physical.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - `total_area`: Area target (m²)
        - `minimum_hot_utility`: Hot utility (kW)
        - `minimum_cold_utility`: Cold utility (kW)
        - `source`: 'ann_surrogate' or 'deterministic_fallback'
        - `is_ood`: Boolean flag
        - `fallback_reason`: Explanation if fallback was triggered
        - `ood_score`: Decision score from density detector
    """
    # If no model provided, compute directly from deterministic engine
    if model is None:
        u_res = utility_targets(streams, delta_t_min)
        a_res = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
        return {
            "total_area": float(a_res["total_area"]),
            "minimum_hot_utility": float(u_res["minimum_hot_utility"]),
            "minimum_cold_utility": float(u_res["minimum_cold_utility"]),
            "source": "deterministic_engine",
            "is_ood": False,
            "fallback_reason": None,
            "ood_score": 1.0,
        }

    x = case_features(
        streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
    )

    is_ood = False
    ood_score = 1.0
    fallback_reason = None

    # Step 1: Input OOD detection
    if guardrail is not None:
        is_ood, ood_score, fallback_reason = guardrail.check_input(x)
        if is_ood and auto_fallback:
            u_res = utility_targets(streams, delta_t_min)
            a_res = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
            return {
                "total_area": float(a_res["total_area"]),
                "minimum_hot_utility": float(u_res["minimum_hot_utility"]),
                "minimum_cold_utility": float(u_res["minimum_cold_utility"]),
                "source": "deterministic_fallback",
                "is_ood": True,
                "fallback_reason": fallback_reason,
                "ood_score": ood_score,
            }
        elif is_ood and not auto_fallback:
            raise ValueError(f"Input is out-of-distribution: {fallback_reason}")

    # Step 2: ANN Surrogate Inference (NumPy accelerated fast path)
    fast_predictor = get_fast_ann_predictor(model)
    if fast_predictor is not None:
        raw_preds = fast_predictor.predict_single(x)
    else:
        raw_preds = model.predict(x.reshape(1, -1))[0]
    preds = [float(p) for p in raw_preds]

    total_hot = sum(float(s.duty()) for s in streams if s.stream_type == "hot")
    total_cold = sum(float(s.duty()) for s in streams if s.stream_type == "cold")

    # Step 3: Output Physical Consistency Validation
    if guardrail is not None:
        physics_valid, physics_reason = guardrail.validate_physics(preds, total_hot, total_cold)
        if not physics_valid:
            if auto_fallback:
                u_res = utility_targets(streams, delta_t_min)
                a_res = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
                return {
                    "total_area": float(a_res["total_area"]),
                    "minimum_hot_utility": float(u_res["minimum_hot_utility"]),
                    "minimum_cold_utility": float(u_res["minimum_cold_utility"]),
                    "source": "deterministic_fallback",
                    "is_ood": True,
                    "fallback_reason": f"Physical violation: {physics_reason}",
                    "ood_score": ood_score,
                }
            else:
                raise ValueError(f"Surrogate output physical violation: {physics_reason}")

    return {
        "total_area": max(0.0, preds[0]),
        "minimum_hot_utility": max(0.0, preds[1]),
        "minimum_cold_utility": max(0.0, preds[2]),
        "source": "ann_surrogate",
        "is_ood": False,
        "fallback_reason": None,
        "ood_score": ood_score,
    }


def predict_batch_network_targets(
    cases: list[tuple[list[Stream], float, float]],
    model: Any | None = None,
    guardrail: OODGuardrail | None = None,
    auto_fallback: bool = True,
) -> list[dict[str, Any]]:
    """Predict network targets for a batch of problem cases with vectorized acceleration.

    Parameters
    ----------
    cases : list of tuple(list of Stream, float, float)
        List of tuples: (streams, delta_t_min, overall_heat_transfer_coefficient).
    model : Pipeline, optional
        Trained ANN surrogate model. If None, evaluates deterministically.
    guardrail : OODGuardrail, optional
        Fitted OOD guardrail for batch anomaly and physics validation.
    auto_fallback : bool, default True
        If True, invokes deterministic engine for cases flagged OOD or non-physical.

    Returns
    -------
    list of dict
        List of prediction dictionaries with target values, source, and guardrail flags.
    """
    n_cases = len(cases)
    if n_cases == 0:
        return []

    # If no model provided, compute entirely via deterministic engine
    if model is None:
        out = []
        for streams, dtmin, u_val in cases:
            u_res = utility_targets(streams, dtmin)
            a_res = area_target(streams, dtmin, u_val)
            out.append({
                "total_area": float(a_res["total_area"]),
                "minimum_hot_utility": float(u_res["minimum_hot_utility"]),
                "minimum_cold_utility": float(u_res["minimum_cold_utility"]),
                "source": "deterministic_engine",
                "is_ood": False,
                "fallback_reason": None,
                "ood_score": 1.0,
            })
        return out

    # Step 1: Batch Feature Extraction
    X_batch = np.array([case_features(s, dt, u) for s, dt, u in cases], dtype=float)

    # Step 2: Vectorized OOD Input Check
    is_ood_mask = np.zeros(n_cases, dtype=bool)
    scores = np.ones(n_cases, dtype=float)
    reasons: list[str | None] = [None] * n_cases

    if guardrail is not None:
        is_ood_mask, scores, reasons = guardrail.check_input_batch(X_batch)
        if not auto_fallback and np.any(is_ood_mask):
            first_fail = int(np.where(is_ood_mask)[0][0])
            raise ValueError(f"Input case {first_fail} is out-of-distribution: {reasons[first_fail]}")

    # Step 3: Vectorized Single-Pass ANN Inference (NumPy accelerated fast path)
    fast_predictor = get_fast_ann_predictor(model)
    if fast_predictor is not None:
        raw_preds = fast_predictor.predict_batch(X_batch)
    else:
        raw_preds = np.asarray(model.predict(X_batch), dtype=float)

    # Step 4: Vectorized Physical Consistency Check
    tot_hot = X_batch[:, 4]
    tot_cold = X_batch[:, 5]
    fallback_mask = is_ood_mask.copy()

    if guardrail is not None:
        valid_mask, phys_reasons = guardrail.validate_physics_batch(raw_preds, tot_hot, tot_cold)
        for idx in np.where(~valid_mask)[0]:
            fallback_mask[idx] = True
            if reasons[idx] is None:
                reasons[idx] = f"Physical violation: {phys_reasons[idx]}"

        if not auto_fallback and np.any(~valid_mask):
            first_phys = int(np.where(~valid_mask)[0][0])
            raise ValueError(f"Surrogate output case {first_phys} physical violation: {phys_reasons[first_phys]}")

    # Step 5: Assemble Results with Selective Deterministic Fallback
    results: list[dict[str, Any]] = []
    for idx, (streams, dtmin, u_val) in enumerate(cases):
        if fallback_mask[idx]:
            if auto_fallback:
                u_res = utility_targets(streams, dtmin)
                a_res = area_target(streams, dtmin, u_val)
                results.append({
                    "total_area": float(a_res["total_area"]),
                    "minimum_hot_utility": float(u_res["minimum_hot_utility"]),
                    "minimum_cold_utility": float(u_res["minimum_cold_utility"]),
                    "source": "deterministic_fallback",
                    "is_ood": bool(fallback_mask[idx]),
                    "fallback_reason": reasons[idx],
                    "ood_score": float(scores[idx]),
                })
            else:
                # Should not reach here if auto_fallback is False due to earlier checks
                pass
        else:
            results.append({
                "total_area": max(0.0, float(raw_preds[idx, 0])),
                "minimum_hot_utility": max(0.0, float(raw_preds[idx, 1])),
                "minimum_cold_utility": max(0.0, float(raw_preds[idx, 2])),
                "source": "ann_surrogate",
                "is_ood": False,
                "fallback_reason": None,
                "ood_score": float(scores[idx]),
            })

    return results

