"""Unit tests for Phase 8: Train and evaluate an ANN to predict pinch-analysis targets."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from pinchpy.models.stream import Stream
from pinchpy.ml.train import build_ann, train_ann, train_mlp
from pinchpy.ml.predict import predict_targets, predict_case
from pinchpy.ml.evaluate import evaluate_regression
from pinchpy.ml.dataset import generate_dataset


@pytest.fixture(scope="module")
def small_dataset():
    """Generate a quick reproducible dataset for ML testing."""
    df = generate_dataset(n_cases=100, seed=42, max_hot=3, max_cold=3)
    X = df.iloc[:, :-3]
    y = df.iloc[:, -3:]
    return X, y


def test_build_ann_pipeline_structure():
    """Verify build_ann constructs a valid StandardScaler + MLPRegressor Pipeline."""
    model = build_ann(hidden_layer_sizes=(32, 16), random_state=42)

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert "ann" in model.named_steps
    assert isinstance(model.named_steps["ann"], MLPRegressor)
    assert model.named_steps["ann"].hidden_layer_sizes == (32, 16)


def test_train_ann_returns_fitted_pipeline(small_dataset):
    """Verify train_ann fits the model without errors."""
    X, y = small_dataset
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=50, random_state=42)

    assert hasattr(model.named_steps["ann"], "n_layers_")
    assert model.named_steps["ann"].n_outputs_ == 3


def test_train_mlp_backward_compatibility(small_dataset):
    """Verify train_mlp alias works identically to train_ann."""
    X, y = small_dataset
    model = train_mlp(X, y, hidden_layer_sizes=(16,), max_iter=50, random_state=42)
    assert isinstance(model, Pipeline)
    assert hasattr(model.named_steps["ann"], "n_layers_")


def test_predict_targets_shape_and_finite(small_dataset):
    """Verify predict_targets produces finite numeric outputs of correct shape."""
    X, y = small_dataset
    model = train_ann(X[:70], y[:70], hidden_layer_sizes=(16,), max_iter=50, random_state=42)

    preds = predict_targets(model, X[70:])
    assert preds.shape == (30, 3)
    assert np.all(np.isfinite(preds))


def test_predict_case_from_stream_objects(small_dataset):
    """Verify predict_case accepts Stream objects directly and returns dictionary targets."""
    X, y = small_dataset
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=50, random_state=42)

    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    result = predict_case(model, streams, delta_t_min=10.0, max_hot=3, max_cold=3)

    assert isinstance(result, dict)
    assert "minimum_hot_utility" in result
    assert "minimum_cold_utility" in result
    assert "pinch_temperature" in result
    assert isinstance(result["minimum_hot_utility"], float)
    assert isinstance(result["minimum_cold_utility"], float)
    assert isinstance(result["pinch_temperature"], float)


def test_evaluate_regression_perfect_prediction():
    """Verify metrics calculation for a perfect predictor."""
    y_true = np.array([[10.0, 20.0, 85.0], [30.0, 40.0, 95.0]])
    y_pred = np.array([[10.0, 20.0, 85.0], [30.0, 40.0, 95.0]])

    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_regression_per_target_breakdown():
    """Verify evaluate_regression provides per-target breakdown."""
    y_true = np.array([[100.0, 50.0], [200.0, 150.0]])
    y_pred = np.array([[110.0, 50.0], [190.0, 140.0]])

    metrics = evaluate_regression(
        y_true,
        y_pred,
        target_names=["target_A", "target_B"],
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "per_target" in metrics
    assert "target_A" in metrics["per_target"]
    assert "target_B" in metrics["per_target"]
    assert metrics["per_target"]["target_A"]["mae"] == pytest.approx(10.0)
    assert metrics["per_target"]["target_B"]["mae"] == pytest.approx(5.0)


def test_ann_training_on_generated_dataset():
    """Verify that the ANN learns valid target relationships on the generated dataset."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "pinch_dataset_1000.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_dataset(n_cases=300, seed=42, max_hot=5, max_cold=5)

    X_train = df.iloc[:400, :-3]
    y_train = df.iloc[:400, -3:]
    X_test = df.iloc[400:500, :-3]
    y_test = df.iloc[400:500, -3:]

    model = train_ann(
        X_train,
        y_train,
        hidden_layer_sizes=(64, 32),
        max_iter=400,
        random_state=42,
    )

    preds = predict_targets(model, X_test)
    metrics = evaluate_regression(y_test, preds)

    # Generalization check: positive R² showing meaningful statistical pattern learning
    assert metrics["r2"] > 0.40
    assert np.all(np.isfinite(preds))
    assert metrics["mae"] < 250.0
