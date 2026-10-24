"""Prediction helpers for the ANN."""
from __future__ import annotations
import numpy as np


def predict_targets(model, X):
    """Predict [minimum hot utility, minimum cold utility, pinch temperature]."""
    return np.asarray(model.predict(X))
