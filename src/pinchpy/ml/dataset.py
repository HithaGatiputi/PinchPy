"""Synthetic training-case generation from the exact pinch-analysis engine."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Iterable, List, Optional, Tuple
import pandas as pd

from ..models.stream import Stream
from ..analysis.pinch_point import utility_targets
from .features import case_features, feature_names, TARGET_NAMES


def generate_case(
    n_hot: int = 2,
    n_cold: int = 2,
    delta_t_min: float = 10.0,
    seed: Optional[int] = None,
    max_hot: int = 5,
    max_cold: int = 5,
) -> Tuple[List[Stream], Any, List[float]]:
    """Generate a single random feasible stream system and its exact pinch targets.

    Returns
    -------
    streams : list[Stream]
        The generated process streams.
    x : np.ndarray
        Fixed-length padded feature vector.
    y : list[float]
        [minimum_hot_utility, minimum_cold_utility, pinch_shifted_temperature].
    """
    rng = random.Random(seed)
    streams: List[Stream] = []
    for i in range(n_hot):
        supply = round(rng.uniform(120.0, 450.0), 2)
        target = round(rng.uniform(40.0, supply - 10.0), 2)
        cpf = round(rng.uniform(0.5, 8.0), 2)
        streams.append(Stream(f"H{i+1}", "hot", supply, target, cpf))
    for i in range(n_cold):
        target = round(rng.uniform(100.0, 400.0), 2)
        supply = round(rng.uniform(30.0, target - 10.0), 2)
        cpf = round(rng.uniform(0.5, 8.0), 2)
        streams.append(Stream(f"C{i+1}", "cold", supply, target, cpf))

    targets = utility_targets(streams, delta_t_min)
    x = case_features(streams, delta_t_min, max_hot=max_hot, max_cold=max_cold)
    y = [
        float(targets["minimum_hot_utility"]),
        float(targets["minimum_cold_utility"]),
        float(targets["pinch_shifted_temperature"]),
    ]
    return streams, x, y


def generate_dataset(
    n_cases: int = 1000,
    seed: int = 42,
    max_hot: int = 5,
    max_cold: int = 5,
    dt_min_choices: Iterable[float] = (5.0, 10.0, 15.0, 20.0),
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Generate a DataFrame of synthetic pinch-analysis training cases.

    Parameters
    ----------
    n_cases : int
        Number of distinct stream network cases to generate.
    seed : int
        Base random seed for reproducibility.
    max_hot : int
        Maximum number of hot streams allowed per case.
    max_cold : int
        Maximum number of cold streams allowed per case.
    dt_min_choices : sequence of float
        Discretized set of delta_t_min values to sample from.
    output_path : str or Path, optional
        If specified, save the resulting DataFrame to CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns and target columns.
    """
    choices = list(dt_min_choices)
    columns = feature_names(max_hot=max_hot, max_cold=max_cold) + TARGET_NAMES
    rows = []

    for i in range(n_cases):
        rng = random.Random(seed + i)
        n_hot = rng.randint(1, max_hot)
        n_cold = rng.randint(1, max_cold)
        dtmin = rng.choice(choices)
        _, x, y = generate_case(
            n_hot=n_hot,
            n_cold=n_cold,
            delta_t_min=dtmin,
            seed=seed + i,
            max_hot=max_hot,
            max_cold=max_cold,
        )
        rows.append(list(x) + list(y))

    df = pd.DataFrame(rows, columns=columns)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)

    return df
