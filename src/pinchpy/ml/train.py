"""ANN training utilities for pinch-analysis targets."""
from __future__ import annotations

from typing import Iterable
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ann(hidden_layer_sizes=(64, 32), random_state=42) -> Pipeline:
    """Build the project's primary feed-forward ANN (MLP) model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=random_state,
        )),
    ])


def train_ann(X, y, hidden_layer_sizes=(64, 32), random_state=42):
    """Fit the ANN and return the trained pipeline."""
    model = build_ann(hidden_layer_sizes, random_state)
    model.fit(np.asarray(X), np.asarray(y))
    return model


def train_mlp(*args, **kwargs):
    """Backward-compatible alias for the project's ANN/MLP model."""
    return train_ann(*args, **kwargs)
