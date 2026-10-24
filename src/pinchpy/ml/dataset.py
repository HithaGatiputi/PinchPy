"""Synthetic training-case generation from the exact pinch-analysis engine."""
from __future__ import annotations
import random
import pandas as pd
from ..models.stream import Stream
from ..analysis.pinch_point import utility_targets
from .features import case_features


def generate_case(n_hot=2, n_cold=2, delta_t_min=10.0, seed=None):
    rng = random.Random(seed)
    streams = []
    for i in range(n_hot):
        supply = rng.uniform(120, 450)
        target = rng.uniform(40, supply - 10)
        cpf = rng.uniform(0.5, 8.0)
        streams.append(Stream(f"H{i+1}", "hot", supply, target, cpf))
    for i in range(n_cold):
        target = rng.uniform(100, 400)
        supply = rng.uniform(30, target - 10)
        cpf = rng.uniform(0.5, 8.0)
        streams.append(Stream(f"C{i+1}", "cold", supply, target, cpf))

    targets = utility_targets(streams, delta_t_min)
    x = case_features(streams, delta_t_min)
    y = [
        targets["minimum_hot_utility"],
        targets["minimum_cold_utility"],
        targets["pinch_shifted_temperature"],
    ]
    return streams, x, y


def generate_dataset(n_cases=1000, seed=42, max_hot=5, max_cold=5):
    rows = []
    for i in range(n_cases):
        rng = random.Random(seed + i)
        n_hot = rng.randint(1, max_hot)
        n_cold = rng.randint(1, max_cold)
        dtmin = rng.choice([5.0, 10.0, 15.0, 20.0])
        streams, x, y = generate_case(n_hot, n_cold, dtmin, seed + i)
        rows.append(list(x) + list(y))
    return pd.DataFrame(rows)
