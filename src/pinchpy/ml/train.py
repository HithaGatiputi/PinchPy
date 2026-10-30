"""ANN and baseline model training utilities for Pinch Analysis surrogates."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .guardrail import OODGuardrail


class LogAreaTargetTransformer(BaseEstimator, TransformerMixin):
    """Target transformer applying log1p to A_target (column 0) and standardizing targets."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit(self, y: np.ndarray, **fit_params: Any) -> LogAreaTargetTransformer:
        y_arr = np.asarray(y, dtype=float).copy()
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_arr[:, 0] = np.log1p(np.maximum(0.0, y_arr[:, 0]))
        self.scaler.fit(y_arr)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float).copy()
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_arr[:, 0] = np.log1p(np.maximum(0.0, y_arr[:, 0]))
        return self.scaler.transform(y_arr)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y_unscaled = self.scaler.inverse_transform(y)
        y_out = y_unscaled.copy()
        # Clip exponent to avoid numerical overflow when called on extreme OOD samples
        clipped_exp = np.clip(y_unscaled[:, 0], -50.0, 50.0)
        y_out[:, 0] = np.expm1(clipped_exp)
        y_out = np.maximum(0.0, y_out)
        return y_out


def build_ann(
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    learning_rate_init: float = 1e-3,
    max_iter: int = 1000,
    random_state: int = 42,
    use_log1p_area: bool = True,
) -> TransformedTargetRegressor | Pipeline:
    """Build the project's primary feed-forward ANN (MLP) pipeline."""
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=random_state,
        )),
    ])
    if use_log1p_area:
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=LogAreaTargetTransformer(),
        )
    return base_pipeline


def train_ann(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
    learning_rate_init: float = 1e-3,
    max_iter: int = 1000,
    random_state: int = 42,
    use_log1p_area: bool = True,
) -> TransformedTargetRegressor | Pipeline:
    """Fit the ANN surrogate on training data and return the trained model."""
    model = build_ann(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        use_log1p_area=use_log1p_area,
    )
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model


def train_mlp(*args, **kwargs) -> Any:
    """Backward-compatible alias for train_ann."""
    return train_ann(*args, **kwargs)


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train comparative baseline models on the same training partition."""
    X_arr = np.asarray(X_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=float)

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_arr, y_arr)

    linear = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    linear.fit(X_arr, y_arr)

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_arr, y_arr)

    return {
        "dummy": dummy,
        "linear": linear,
        "random_forest": rf,
    }


def save_artifacts(
    model: Any,
    guardrail: OODGuardrail | None = None,
    metadata: dict[str, Any] | None = None,
    output_dir: str | Path = "models",
) -> None:
    """Save trained ANN model, guardrail, and training metadata to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ann_model.pkl", "wb") as f:
        pickle.dump(model, f)

    if guardrail is not None:
        guardrail.save(out_dir / "ood_guardrail.pkl")

    if metadata is not None:
        with open(out_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def load_artifacts(
    model_dir: str | Path = "models",
) -> tuple[Any, OODGuardrail | None, dict[str, Any] | None]:
    """Load serialized ANN model, guardrail, and metadata from disk."""
    m_dir = Path(model_dir)

    model_path = m_dir / "ann_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    guardrail = None
    guardrail_path = m_dir / "ood_guardrail.pkl"
    if guardrail_path.exists():
        guardrail = OODGuardrail.load(guardrail_path)

    metadata = None
    meta_path = m_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, guardrail, metadata
