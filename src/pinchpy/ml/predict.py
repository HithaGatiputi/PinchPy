"""Prediction helpers for the ANN."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from ..models.stream import Stream


def predict_targets(model: Any, X: Any) -> np.ndarray:
    """Predict [minimum hot utility, minimum cold utility, pinch temperature].

    Parameters
    ----------
    model : Pipeline or Regressor
        Fitted ANN model pipeline.
    X : array-like of shape (n_samples, n_features)
        Padded feature matrix.

    Returns
    -------
    np.ndarray of shape (n_samples, 3)
        Predicted targets for each sample.
    """
    return np.asarray(model.predict(X))


def predict_case(
    model: Any,
    streams: List[Stream],
    delta_t_min: float = 10.0,
    max_hot: int = 5,
    max_cold: int = 5,
) -> Dict[str, float]:
    """Predict pinch targets directly from a list of process Stream objects.

    Parameters
    ----------
    model : Pipeline or Regressor
        Trained ANN model.
    streams : list[Stream]
        List of Stream instances.
    delta_t_min : float
        Minimum approach temperature.
    max_hot : int
        Maximum number of hot streams supported by model architecture.
    max_cold : int
        Maximum number of cold streams supported by model architecture.

    Returns
    -------
    dict
        Dictionary with predicted targets:
        - 'minimum_hot_utility'
        - 'minimum_cold_utility'
        - 'pinch_temperature'
    """
    from .features import case_features, feature_names

    x_arr = case_features(streams, delta_t_min, max_hot=max_hot, max_cold=max_cold).reshape(1, -1)

    # Supply DataFrame with matching column names if model was fitted on named columns
    if hasattr(model, "feature_names_in_"):
        X_in = pd.DataFrame(x_arr, columns=model.feature_names_in_)
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("scaler"), "feature_names_in_"):
        X_in = pd.DataFrame(x_arr, columns=model.named_steps["scaler"].feature_names_in_)
    else:
        cols = feature_names(max_hot=max_hot, max_cold=max_cold)
        X_in = pd.DataFrame(x_arr, columns=cols)

    pred = predict_targets(model, X_in)[0]
    return {
        "minimum_hot_utility": float(pred[0]),
        "minimum_cold_utility": float(pred[1]),
        "pinch_temperature": float(pred[2]),
    }
