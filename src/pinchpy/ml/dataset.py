"""Reproducible synthetic dataset generation for Pinch Analysis ML surrogates.

Generates physically plausible process stream collections across realistic
industrial operating envelopes and labels them using PinchPy's deterministic
Phases 1-5 engine.
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.area_target import area_target
from ..analysis.pinch_point import utility_targets
from ..models.stream import Stream
from .features import case_features, feature_names

TARGET_COLUMNS = ["A_target", "Q_H_min", "Q_C_min"]

HEN_TARGET_COLUMNS = [
    "total_area",
    "exchanger_count",
    "total_heater_duty",
    "total_cooler_duty",
    "hot_utility_above_target",
    "tac",
]

HEN_PREDICTION_TARGETS = [
    "total_area",
    "exchanger_count",
    "total_heater_duty",
    "total_cooler_duty",
    "hot_utility_above_target",
]


def generate_case(
    n_hot: int = 2,
    n_cold: int = 2,
    delta_t_min: float = 10.0,
    overall_heat_transfer_coefficient: float = 0.1,
    seed: int | None = None,
    max_hot: int = 5,
    max_cold: int = 5,
) -> tuple[list[Stream], np.ndarray, list[float]]:
    """Generate a single verified process stream case with ground-truth targets.

    Returns
    -------
    streams : list of Stream
        The generated process streams.
    x : np.ndarray
        Feature vector for the case.
    y : list of float
        Deterministic ground-truth targets: [A_target, Q_H_min, Q_C_min].
    """
    rng = random.Random(seed)
    streams: list[Stream] = []

    # Generate hot streams (require cooling: supply > target)
    for i in range(n_hot):
        supply = rng.uniform(130.0, 420.0)
        target = rng.uniform(35.0, supply - 15.0)
        cpf = rng.uniform(0.5, 7.5)
        streams.append(
            Stream(
                id=f"H{i+1}",
                type="hot",
                supply_temperature=round(supply, 1),
                target_temperature=round(target, 1),
                cp=round(cpf, 2),
            )
        )

    # Generate cold streams (require heating: target > supply)
    for i in range(n_cold):
        supply = rng.uniform(30.0, 260.0)
        target = rng.uniform(supply + 15.0, 390.0)
        cpf = rng.uniform(0.5, 7.5)
        streams.append(
            Stream(
                id=f"C{i+1}",
                type="cold",
                supply_temperature=round(supply, 1),
                target_temperature=round(target, 1),
                cp=round(cpf, 2),
            )
        )

    # Compute deterministic ground-truth labels
    targets = utility_targets(streams, delta_t_min)
    area_res = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)

    x = case_features(
        streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        max_hot=max_hot,
        max_cold=max_cold,
    )

    y = [
        float(area_res["total_area"]),
        float(targets["minimum_hot_utility"]),
        float(targets["minimum_cold_utility"]),
    ]

    return streams, x, y


def generate_dataset(
    n_cases: int = 2000,
    seed: int = 42,
    max_hot: int = 5,
    max_cold: int = 5,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a reproducible, verified synthetic dataset for ANN training.

    Parameters
    ----------
    n_cases : int, default 2000
        Number of valid cases to generate.
    seed : int, default 42
        Base random seed for reproducibility.
    max_hot : int, default 5
        Maximum hot streams per problem.
    max_cold : int, default 5
        Maximum cold streams per problem.
    output_dir : str or Path, optional
        Directory to save `pinch_dataset.csv` and `dataset_metadata.json`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature columns and the 3 target columns:
        ['A_target', 'Q_H_min', 'Q_C_min'].
    """
    rng = random.Random(seed)
    rows: list[list[float]] = []

    attempts = 0
    rejected_count = 0
    rejection_reasons: dict[str, int] = {}

    delta_t_choices = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
    u_choices = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]

    while len(rows) < n_cases:
        attempts += 1
        case_seed = seed + attempts
        n_h = rng.randint(1, max_hot)
        n_c = rng.randint(1, max_cold)
        dtmin = rng.choice(delta_t_choices)
        u_val = rng.choice(u_choices)

        try:
            _, x, y = generate_case(
                n_hot=n_h,
                n_cold=n_c,
                delta_t_min=dtmin,
                overall_heat_transfer_coefficient=u_val,
                seed=case_seed,
                max_hot=max_hot,
                max_cold=max_cold,
            )
            # Physical validity check: area and utilities must be non-negative
            if y[0] < 0 or y[1] < 0 or y[2] < 0 or np.isnan(y).any():
                raise ValueError("Encountered non-physical negative target value.")
            rows.append(list(x) + list(y))
        except Exception as exc:
            rejected_count += 1
            reason = type(exc).__name__ + ": " + str(exc).split("\n")[0]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    col_names = feature_names(max_hot=max_hot, max_cold=max_cold) + TARGET_COLUMNS
    df = pd.DataFrame(rows, columns=col_names)

    metadata: dict[str, Any] = {
        "n_cases": len(df),
        "seed": seed,
        "max_hot": max_hot,
        "max_cold": max_cold,
        "total_attempts": attempts,
        "rejected_cases": rejected_count,
        "rejection_reasons": rejection_reasons,
        "target_columns": TARGET_COLUMNS,
        "primary_target": "A_target (m²)",
        "secondary_targets": ["Q_H_min (kW)", "Q_C_min (kW)"],
        "feature_count": len(feature_names(max_hot, max_cold)),
        "delta_t_min_range": [min(delta_t_choices), max(delta_t_choices)],
        "u_coefficient_range": [min(u_choices), max(u_choices)],
    }

    df.attrs["metadata"] = metadata

    if output_dir is not None:
        save_dataset(df, metadata, output_dir)

    return df


def save_dataset(df: pd.DataFrame, metadata: dict[str, Any], output_dir: str | Path) -> None:
    """Save dataset DataFrame and metadata to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "pinch_dataset.csv"
    meta_path = out_path / "dataset_metadata.json"

    df.to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    target_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into Train, Validation, and Test partitions without leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and target columns.
    train_ratio : float, default 0.70
        Proportion for training.
    val_ratio : float, default 0.15
        Proportion for validation/early stopping.
    test_ratio : float, default 0.15
        Proportion for final evaluation.
    random_state : int, default 42
        Seed for split reproducibility.
    target_columns : list of str, optional
        Target column names. Defaults to TARGET_COLUMNS (3 targets).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray
    """
    n_targets = len(target_columns) if target_columns is not None else len(TARGET_COLUMNS)
    X = df.iloc[:, :-n_targets].to_numpy(dtype=float)
    y = df.iloc[:, -n_targets:].to_numpy(dtype=float)

    n_samples = len(df)
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def generate_hen_case(
    n_hot: int = 2,
    n_cold: int = 2,
    delta_t_min: float = 10.0,
    overall_heat_transfer_coefficient: float = 0.1,
    seed: int | None = None,
    max_hot: int = 5,
    max_cold: int = 5,
    search_budget: int = 500,
) -> tuple[list[Stream], np.ndarray, list[float]]:
    """Generate a single verified process stream case labeled via Phase 6 network design and TAC.

    Returns
    -------
    streams : list of Stream
        The generated process streams.
    x : np.ndarray
        Feature vector for the case.
    y : list of float
        Targets: [total_area, exchanger_count, total_heater_duty,
                  total_cooler_duty, hot_utility_above_target, tac].
    """
    from ..synthesis.network_design import design_network
    from ..synthesis.network_evaluation import network_tac

    streams, x, _ = generate_case(
        n_hot=n_hot,
        n_cold=n_cold,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        seed=seed,
        max_hot=max_hot,
        max_cold=max_cold,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = design_network(
            streams,
            delta_t_min,
            overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
            search_budget=search_budget,
        )
        tac_res = network_tac(net)

    heaters = [e for e in net.exchangers if getattr(e, "is_utility", False) and (e.hot_stream in ("HU", "hot_utility") or e.name.upper().startswith("HU"))]
    coolers = [e for e in net.exchangers if getattr(e, "is_utility", False) and (e.cold_stream in ("CU", "cold_utility") or e.name.upper().startswith("CU"))]

    tot_area = sum(float(e.area) for e in net.exchangers if e.area is not None)
    n_units = float(len(net.exchangers))
    tot_qh = sum(float(e.duty) for e in heaters)
    tot_qc = sum(float(e.duty) for e in coolers)

    u_targets = utility_targets(streams, delta_t_min)
    qh_min = float(u_targets["minimum_hot_utility"])
    qh_above = max(0.0, tot_qh - qh_min)
    tac = float(tac_res["total"])

    y_hen = [tot_area, n_units, tot_qh, tot_qc, qh_above, tac]
    return streams, x, y_hen


def generate_hen_dataset(
    n_cases: int = 1000,
    seed: int = 42,
    max_hot: int = 5,
    max_cold: int = 5,
    output_dir: str | Path | None = None,
    search_budget: int = 500,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for Phase 6 HEN surrogate modeling.

    Parameters
    ----------
    n_cases : int, default 1000
        Number of valid synthesized network cases.
    seed : int, default 42
        Random seed for reproducibility.
    max_hot : int, default 5
        Maximum hot streams per problem.
    max_cold : int, default 5
        Maximum cold streams per problem.
    output_dir : str or Path, optional
        Directory to save `hen_dataset.csv` and `hen_dataset_metadata.json`.
    search_budget : int, default 500
        Backtracking search budget passed through to design_network.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature columns and the 6 HEN labels:
        ['total_area', 'exchanger_count', 'total_heater_duty',
         'total_cooler_duty', 'hot_utility_above_target', 'tac'].
    """
    rng = random.Random(seed)
    rows: list[list[float]] = []

    attempts = 0
    rejected_count = 0
    rejection_reasons: dict[str, int] = {}

    delta_t_choices = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
    u_choices = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]

    while len(rows) < n_cases:
        attempts += 1
        case_seed = seed + attempts
        n_h = rng.randint(1, max_hot)
        n_c = rng.randint(1, max_cold)
        dtmin = rng.choice(delta_t_choices)
        u_val = rng.choice(u_choices)

        try:
            _, x, y = generate_hen_case(
                n_hot=n_h,
                n_cold=n_c,
                delta_t_min=dtmin,
                overall_heat_transfer_coefficient=u_val,
                seed=case_seed,
                max_hot=max_hot,
                max_cold=max_cold,
                search_budget=search_budget,
            )
            # Physical sanity check: area, duties, count, and TAC must be non-negative
            if any(val < 0 for val in y) or np.isnan(y).any():
                raise ValueError("Encountered non-physical negative target value.")
            rows.append(list(x) + list(y))
        except Exception as exc:
            rejected_count += 1
            reason = type(exc).__name__ + ": " + str(exc).split("\n")[0]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    col_names = feature_names(max_hot=max_hot, max_cold=max_cold) + HEN_TARGET_COLUMNS
    df = pd.DataFrame(rows, columns=col_names)

    metadata: dict[str, Any] = {
        "n_cases": len(df),
        "seed": seed,
        "max_hot": max_hot,
        "max_cold": max_cold,
        "search_budget": search_budget,
        "total_attempts": attempts,
        "rejected_cases": rejected_count,
        "rejection_reasons": rejection_reasons,
        "target_columns": HEN_TARGET_COLUMNS,
        "primary_prediction_targets": HEN_PREDICTION_TARGETS,
        "feature_count": len(feature_names(max_hot, max_cold)),
        "delta_t_min_range": [min(delta_t_choices), max(delta_t_choices)],
        "u_coefficient_range": [min(u_choices), max(u_choices)],
    }
    df.attrs["metadata"] = metadata

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p / "hen_dataset.csv", index=False)
        with open(out_p / "hen_dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return df

