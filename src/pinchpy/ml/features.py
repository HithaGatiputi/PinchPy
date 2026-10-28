"""Feature construction for Pinch Analysis machine-learning models.

This module converts variable-size process stream sets and operating conditions
into fixed-size, physically meaningful feature vectors for surrogate modeling.
"""
from __future__ import annotations

import numpy as np

STREAM_FEATURES_COUNT = 4  # supply_temperature, target_temperature, cp, duty


def stream_features(stream) -> list[float]:
    """Extract individual stream features: [T_supply, T_target, CP, duty]."""
    return [
        float(stream.supply_temperature),
        float(stream.target_temperature),
        float(stream.heat_capacity_flow),
        float(stream.duty()),
    ]


def feature_names(max_hot: int = 5, max_cold: int = 5) -> list[str]:
    """Return the ordered list of descriptive feature names."""
    names = [
        "delta_t_min",
        "overall_u",
        "n_hot",
        "n_cold",
        "total_hot_duty",
        "total_cold_duty",
        "net_duty_deficit",
        "total_hot_cp",
        "total_cold_cp",
        "max_hot_supply",
        "min_hot_target",
        "min_cold_supply",
        "max_cold_target",
        "max_temp_overlap",
    ]
    for i in range(1, max_hot + 1):
        names.extend([
            f"hot_{i}_supply_t",
            f"hot_{i}_target_t",
            f"hot_{i}_cp",
            f"hot_{i}_duty",
        ])
    for i in range(1, max_cold + 1):
        names.extend([
            f"cold_{i}_supply_t",
            f"cold_{i}_target_t",
            f"cold_{i}_cp",
            f"cold_{i}_duty",
        ])
    return names


def case_features(
    streams,
    delta_t_min: float,
    overall_heat_transfer_coefficient: float = 0.1,
    max_hot: int = 5,
    max_cold: int = 5,
) -> np.ndarray:
    """Build a fixed-size feature vector for a stream set and operating conditions.

    The representation combines permutation-invariant thermodynamic summary
    descriptors with ordered, padded stream parameters.

    Parameters
    ----------
    streams : list of Stream
        The collection of hot and cold process streams.
    delta_t_min : float
        Minimum heat exchanger approach temperature (°C).
    overall_heat_transfer_coefficient : float, default 0.1
        Overall heat transfer coefficient U in kW/(m²·°C).
    max_hot : int, default 5
        Maximum hot streams capacity (padded with zeros if fewer).
    max_cold : int, default 5
        Maximum cold streams capacity (padded with zeros if fewer).

    Returns
    -------
    np.ndarray
        1D array of float features.
    """
    hot = [s for s in streams if s.stream_type == "hot"]
    cold = [s for s in streams if s.stream_type == "cold"]

    # Order streams deterministically: hot by descending supply T, cold by ascending supply T
    hot_sorted = sorted(hot, key=lambda s: float(s.supply_temperature), reverse=True)[:max_hot]
    cold_sorted = sorted(cold, key=lambda s: float(s.supply_temperature))[:max_cold]

    total_hot_duty = sum(float(s.duty()) for s in hot)
    total_cold_duty = sum(float(s.duty()) for s in cold)
    total_hot_cp = sum(float(s.heat_capacity_flow) for s in hot)
    total_cold_cp = sum(float(s.heat_capacity_flow) for s in cold)

    max_hot_supply = max((float(s.supply_temperature) for s in hot), default=0.0)
    min_hot_target = min((float(s.target_temperature) for s in hot), default=0.0)
    min_cold_supply = min((float(s.supply_temperature) for s in cold), default=0.0)
    max_cold_target = max((float(s.target_temperature) for s in cold), default=0.0)

    max_temp_overlap = max(0.0, max_hot_supply - min_cold_supply) if (hot and cold) else 0.0

    # Global summary statistics
    row = [
        float(delta_t_min),
        float(overall_heat_transfer_coefficient),
        float(len(hot)),
        float(len(cold)),
        total_hot_duty,
        total_cold_duty,
        total_cold_duty - total_hot_duty,
        total_hot_cp,
        total_cold_cp,
        max_hot_supply,
        min_hot_target,
        min_cold_supply,
        max_cold_target,
        max_temp_overlap,
    ]

    # Individual stream features (padded to max_hot and max_cold)
    for s in hot_sorted:
        row.extend(stream_features(s))
    row.extend([0.0] * STREAM_FEATURES_COUNT * (max_hot - len(hot_sorted)))

    for s in cold_sorted:
        row.extend(stream_features(s))
    row.extend([0.0] * STREAM_FEATURES_COUNT * (max_cold - len(cold_sorted)))

    return np.asarray(row, dtype=float)
