"""Feature construction for variable-size stream sets.

The baseline representation pads stream features to a fixed maximum number of
hot/cold streams. This keeps the first ANN implementation simple and reproducible.
"""
from __future__ import annotations

from typing import List
import numpy as np

FEATURES_PER_STREAM = 4  # supply T, target T, Cp flow, duty
TARGET_NAMES = ["minimum_hot_utility", "minimum_cold_utility", "pinch_temperature"]


def feature_names(max_hot: int = 5, max_cold: int = 5) -> List[str]:
    """Return descriptive column names corresponding to case_features output."""
    names = ["delta_t_min", "n_hot", "n_cold"]
    for i in range(max_hot):
        names.extend([f"H{i+1}_Ts", f"H{i+1}_Tt", f"H{i+1}_CP", f"H{i+1}_Q"])
    for i in range(max_cold):
        names.extend([f"C{i+1}_Ts", f"C{i+1}_Tt", f"C{i+1}_CP", f"C{i+1}_Q"])
    return names


def stream_features(stream) -> List[float]:
    """Extract numeric features from a single process stream."""
    return [
        float(stream.supply_temperature),
        float(stream.target_temperature),
        float(stream.heat_capacity_flow),
        float(stream.duty()),
    ]


def case_features(streams, delta_t_min, max_hot: int = 5, max_cold: int = 5) -> np.ndarray:
    """Encode a system of streams and delta_t_min into a fixed-length feature vector."""
    hot = [s for s in streams if s.stream_type == "hot"]
    cold = [s for s in streams if s.stream_type == "cold"]
    hot = sorted(hot, key=lambda s: s.name)[:max_hot]
    cold = sorted(cold, key=lambda s: s.name)[:max_cold]

    row = [float(delta_t_min), float(len(hot)), float(len(cold))]
    for group, limit in ((hot, max_hot), (cold, max_cold)):
        for s in group:
            row.extend(stream_features(s))
        row.extend([0.0] * FEATURES_PER_STREAM * (limit - len(group)))
    return np.asarray(row, dtype=float)
