"""Feature construction for variable-size stream sets.

The baseline representation pads stream features to a fixed maximum number of
hot/cold streams. This keeps the first ANN implementation simple and reproducible.
"""
from __future__ import annotations
import numpy as np

FEATURES_PER_STREAM = 4  # supply T, target T, Cp flow, duty


def stream_features(stream):
    return [
        float(stream.supply_temperature),
        float(stream.target_temperature),
        float(stream.heat_capacity_flow),
        float(stream.duty()),
    ]


def case_features(streams, delta_t_min, max_hot=5, max_cold=5):
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
