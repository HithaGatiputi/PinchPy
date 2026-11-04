"""Evaluation metrics for ANN predictions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import TARGET_NAMES


def evaluate_regression(
    y_true: Any,
    y_pred: Any,
    target_names: Optional[Sequence[str]] = TARGET_NAMES,
) -> Dict[str, Any]:
    """Calculate MAE, RMSE, and R² scores overall and per target.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_targets)
        Ground truth exact values.
    y_pred : array-like of shape (n_samples, n_targets)
        ANN predicted target values.
    target_names : sequence of str, optional
        Descriptive names for each target column.

    Returns
    -------
    dict
        Dictionary with overall 'mae', 'rmse', 'r2', and optional 'per_target' metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    metrics: Dict[str, Any] = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }

    if y_true_arr.ndim > 1 and y_true_arr.shape[1] > 1:
        per_target: Dict[str, Dict[str, float]] = {}
        names = list(target_names) if target_names else [f"target_{i}" for i in range(y_true_arr.shape[1])]
        for i, name in enumerate(names[: y_true_arr.shape[1]]):
            per_target[name] = {
                "mae": float(mean_absolute_error(y_true_arr[:, i], y_pred_arr[:, i])),
                "rmse": float(np.sqrt(mean_squared_error(y_true_arr[:, i], y_pred_arr[:, i]))),
                "r2": float(r2_score(y_true_arr[:, i], y_pred_arr[:, i])),
            }
        metrics["per_target"] = per_target

    return metrics
