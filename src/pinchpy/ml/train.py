"""ANN training utilities for pinch-analysis targets."""

from __future__ import annotations

from typing import Any, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ann(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    random_state: int = 42,
    max_iter: int = 2000,
    learning_rate_init: float = 1e-3,
    early_stopping: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Build the project's primary feed-forward ANN (MLP) model."""
    mlp_kwargs = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "early_stopping": early_stopping,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": random_state,
    }
    mlp_kwargs.update(kwargs)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(**mlp_kwargs)),
    ])


def train_ann(
    X: Any,
    y: Any,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Fit the ANN and return the trained pipeline."""
    model = build_ann(
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X, y)
    return model


def train_mlp(*args: Any, **kwargs: Any) -> Pipeline:
    """Backward-compatible alias for the project's ANN/MLP model."""
    return train_ann(*args, **kwargs)
