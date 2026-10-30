"""Unit and integration tests for Phase 7-8 machine learning surrogate and OOD guardrails."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pinchpy.analysis.area_target import area_target
from pinchpy.analysis.pinch_point import utility_targets
from pinchpy.ml.dataset import generate_case, generate_dataset, split_dataset
from pinchpy.ml.evaluate import (
    benchmark_latency,
    compare_models,
    evaluate_ood_detection,
    evaluate_regression,
)
from pinchpy.ml.features import case_features, feature_names
from pinchpy.ml.guardrail import OODGuardrail
from pinchpy.ml.predict import predict_network_targets, predict_targets
from pinchpy.ml.train import (
    build_ann,
    load_artifacts,
    save_artifacts,
    train_ann,
    train_baselines,
    train_mlp,
)
from pinchpy.models.stream import Stream

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


@pytest.fixture
def sample_streams():
    return [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]


@pytest.fixture
def four_streams():
    return [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]


# ============================================================================
# 1. Feature Extraction Tests
# ============================================================================


def test_feature_names_length_and_content():
    names = feature_names(max_hot=5, max_cold=5)
    assert len(names) == 54
    assert names[0] == "delta_t_min"
    assert names[1] == "overall_u"
    assert "total_hot_duty" in names
    assert "hot_1_supply_t" in names
    assert "cold_5_duty" in names


def test_case_features_vector_shape_and_values(sample_streams):
    feats = case_features(sample_streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (54,)
    assert feats[0] == 10.0  # delta_t_min
    assert feats[1] == 0.1  # U
    assert feats[2] == 1.0  # n_hot
    assert feats[3] == 1.0  # n_cold
    assert feats[4] == 200.0  # total_hot_duty (2.0 * 100)
    assert feats[5] == 195.0  # total_cold_duty (1.5 * 130)


# ============================================================================
# 2. Dataset Generation & Reproducibility Tests
# ============================================================================


def test_generate_case_validity():
    streams, x, y = generate_case(n_hot=2, n_cold=2, delta_t_min=10.0, seed=123)
    assert len(streams) == 4
    assert x.shape == (54,)
    assert len(y) == 3
    # Target values: A_target >= 0, Q_H >= 0, Q_C >= 0
    assert y[0] >= 0.0
    assert y[1] >= 0.0
    assert y[2] >= 0.0


def test_dataset_generation_reproducibility():
    df1 = generate_dataset(n_cases=20, seed=100)
    df2 = generate_dataset(n_cases=20, seed=100)
    df3 = generate_dataset(n_cases=20, seed=200)

    pd_assert = pytest.importorskip("pandas.testing").assert_frame_equal
    pd_assert(df1, df2)
    assert not df1.equals(df3)


def test_split_dataset_partitions():
    df = generate_dataset(n_cases=100, seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    assert X_train.shape[1] == 54
    assert y_train.shape[1] == 3


# ============================================================================
# 3. Model Training & Baseline Tests
# ============================================================================


def test_train_ann_and_baselines():
    df = generate_dataset(n_cases=40, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()

    ann = train_ann(X, y, hidden_layer_sizes=(32, 16), max_iter=50, random_state=42)
    preds = ann.predict(X[:5])
    assert preds.shape == (5, 3)

    baselines = train_baselines(X, y, random_state=42)
    assert "dummy" in baselines
    assert "linear" in baselines
    assert "random_forest" in baselines


def test_train_mlp_backward_compatibility():
    df = generate_dataset(n_cases=20, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_mlp(X, y, hidden_layer_sizes=(16,), max_iter=20)
    preds = predict_targets(model, X[:3])
    assert preds.shape == (3, 3)


# ============================================================================
# 4. Out-of-Distribution (OOD) Guardrail Tests
# ============================================================================


def test_ood_guardrail_training_and_detection():
    df = generate_dataset(n_cases=100, seed=42)
    X = df.iloc[:, :-3].to_numpy()

    guardrail = OODGuardrail(contamination=0.05, random_state=42)
    guardrail.fit(X)

    # In-distribution check (normal case from test partition)
    x_in = X[0]
    is_ood, score, reason = guardrail.check_input(x_in)
    assert not is_ood
    assert reason is None

    # OOD check 1: Extreme high delta_t_min
    x_ood_dt = x_in.copy()
    x_ood_dt[0] = 120.0
    is_ood, score, reason = guardrail.check_input(x_ood_dt)
    assert is_ood
    assert "ΔTmin" in reason

    # OOD check 2: Non-physical negative delta_t_min
    x_ood_neg = x_in.copy()
    x_ood_neg[0] = -5.0
    is_ood, score, reason = guardrail.check_input(x_ood_neg)
    assert is_ood
    assert "Non-physical" in reason

    # OOD check 3: Extreme hot temperature (upper bound)
    x_ood_temp = x_in.copy()
    x_ood_temp[9] = 750.0  # max hot supply
    is_ood, score, reason = guardrail.check_input(x_ood_temp)
    assert is_ood
    assert "outside operational envelope" in reason

    # OOD check 4: Extreme low temperature (lower bound)
    x_ood_low_temp = x_in.copy()
    x_ood_low_temp[11] = -20.0  # min cold supply
    is_ood, score, reason = guardrail.check_input(x_ood_low_temp)
    assert is_ood
    assert "outside operational envelope" in reason

    # OOD check 5: Extremely low individual stream CP (lower bound)
    x_ood_low_cp = x_in.copy()
    x_ood_low_cp[16] = 0.001  # hot 1 CP below minimum active envelope
    is_ood, score, reason = guardrail.check_input(x_ood_low_cp)
    assert is_ood
    assert "falls below individual CP envelope" in reason

    # OOD check 6: Extremely low total CP (lower bound)
    x_ood_low_tot_cp = x_in.copy()
    x_ood_low_tot_cp[7] = 0.01  # total hot CP below bound
    is_ood, score, reason = guardrail.check_input(x_ood_low_tot_cp)
    assert is_ood
    assert "falls below training bound" in reason


def test_guardrail_physics_validation():
    guardrail = OODGuardrail(tolerance=0.15)
    # Valid physical prediction
    # total_hot = 200, total_cold = 195 -> expected (Q_C - Q_H) = 5.0
    valid_preds = [44.0, 0.0, 5.0]
    is_valid, reason = guardrail.validate_physics(valid_preds, total_hot_duty=200.0, total_cold_duty=195.0)
    assert is_valid
    assert reason is None

    # Invalid: Negative area
    invalid_area = [-10.0, 0.0, 5.0]
    is_valid, reason = guardrail.validate_physics(invalid_area, 200.0, 195.0)
    assert not is_valid
    assert "negative" in reason.lower()


    # Invalid: Gross First Law violation (predicts Q_C - Q_H = 100 kW when expected is 5 kW)
    invalid_energy = [44.0, 0.0, 105.0]
    is_valid, reason = guardrail.validate_physics(invalid_energy, 200.0, 195.0)
    assert not is_valid
    assert "First Law" in reason


# ============================================================================
# 5. Guarded Inference & Deterministic Fallback Tests
# ============================================================================


def test_predict_network_targets_in_distribution(four_streams):
    df = generate_dataset(n_cases=50, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(32,), max_iter=50, random_state=42)
    guardrail = OODGuardrail().fit(X)

    # In-distribution call
    result = predict_network_targets(
        four_streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.1,
        model=model,
        guardrail=guardrail,
    )
    assert result["total_area"] > 0
    assert result["minimum_hot_utility"] >= 0
    assert result["minimum_cold_utility"] >= 0
    assert result["source"] in ("ann_surrogate", "deterministic_fallback")


def test_predict_network_targets_ood_fallback(four_streams):
    df = generate_dataset(n_cases=50, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(32,), max_iter=50, random_state=42)
    guardrail = OODGuardrail().fit(X)

    # OOD call with extreme delta_t_min = 120 °C
    result = predict_network_targets(
        four_streams,
        delta_t_min=120.0,
        overall_heat_transfer_coefficient=0.1,
        model=model,
        guardrail=guardrail,
        auto_fallback=True,
    )
    assert result["is_ood"] is True
    assert result["source"] == "deterministic_fallback"
    assert "outside training range" in result["fallback_reason"]

    # Verify fallback matches exact deterministic engine
    exact_util = utility_targets(four_streams, 120.0)
    assert result["minimum_hot_utility"] == pytest.approx(exact_util["minimum_hot_utility"])


def test_predict_network_targets_raises_without_fallback(four_streams):
    df = generate_dataset(n_cases=50, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(32,), max_iter=50, random_state=42)
    guardrail = OODGuardrail().fit(X)

    with pytest.raises(ValueError, match="out-of-distribution"):
        predict_network_targets(
            four_streams,
            delta_t_min=120.0,
            overall_heat_transfer_coefficient=0.1,
            model=model,
            guardrail=guardrail,
            auto_fallback=False,
        )


# ============================================================================
# 6. Serialization & Latency Tests
# ============================================================================


def test_save_and_load_artifacts():
    df = generate_dataset(n_cases=20, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=20)
    guardrail = OODGuardrail().fit(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_artifacts(model, guardrail, {"test": 123}, output_dir=tmp_dir)
        loaded_model, loaded_guard, loaded_meta = load_artifacts(tmp_dir)

        assert loaded_model is not None
        assert loaded_guard is not None
        assert loaded_meta["test"] == 123
        assert loaded_guard.is_fitted


def test_benchmark_latency(sample_streams):
    df = generate_dataset(n_cases=20, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=20)

    bench = benchmark_latency(
        model,
        sample_streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.1,
        warmup_iterations=5,
        n_rounds=2,
        n_iterations=10,
    )
    assert "deterministic_latency_us" in bench
    assert "ann_pure_latency_us" in bench
    assert "feature_extraction_us" in bench
    assert "guarded_latency_us" in bench
    assert bench["deterministic_latency_us"] > 0
    assert bench["ann_pure_latency_us"] > 0


# ============================================================================
# 7. Batch Screening, Energy Balance & Regression Tests
# ============================================================================


def test_deterministic_fallback_identity(four_streams):
    """Verify that deterministic fallback returns exact bitwise/float values from Phase 1-5."""
    df = generate_dataset(n_cases=30, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=20, random_state=42)
    guardrail = OODGuardrail().fit(X)

    # Trigger Tier 1 fallback with extreme approach temperature
    dtmin_ood = 150.0
    u_val = 0.15
    res = predict_network_targets(
        four_streams,
        delta_t_min=dtmin_ood,
        overall_heat_transfer_coefficient=u_val,
        model=model,
        guardrail=guardrail,
        auto_fallback=True,
    )
    assert res["source"] == "deterministic_fallback"
    assert res["is_ood"] is True

    # Exact deterministic recalculation
    exact_u = utility_targets(four_streams, dtmin_ood)
    exact_a = area_target(four_streams, dtmin_ood, u_val)

    assert res["total_area"] == exact_a["total_area"]
    assert res["minimum_hot_utility"] == exact_u["minimum_hot_utility"]
    assert res["minimum_cold_utility"] == exact_u["minimum_cold_utility"]


def test_energy_balance_mathematical_consistency():
    """Verify First Law energy conservation across multiple stream network regimes."""
    guardrail = OODGuardrail(tolerance=0.05)

    # 1. Balanced duty case: Qh = 50, Qc = 50 -> Qc - Qh = 0 = total_hot - total_cold
    valid, reason = guardrail.validate_physics([100.0, 50.0, 50.0], total_hot_duty=500.0, total_cold_duty=500.0)
    assert valid
    assert reason is None

    # 2. Hot-dominated case: hot streams release more heat than cold streams absorb
    # total_hot = 600, total_cold = 400 -> expected deficit = 600 - 400 = 200 kW
    # If Qh = 0 (threshold), Qc must be 200 kW -> Qc - Qh = 200 kW
    valid, reason = guardrail.validate_physics([80.0, 0.0, 200.0], total_hot_duty=600.0, total_cold_duty=400.0)
    assert valid

    # 3. Cold-dominated case: cold streams require more heat than hot streams release
    # total_hot = 300, total_cold = 500 -> expected deficit = 300 - 500 = -200 kW
    # If Qc = 0 (threshold), Qh must be 200 kW -> Qc - Qh = -200 kW
    valid, reason = guardrail.validate_physics([60.0, 200.0, 0.0], total_hot_duty=300.0, total_cold_duty=500.0)
    assert valid


def test_predict_batch_network_targets_vectorization(four_streams, sample_streams):
    """Verify that vectorized batch prediction matches single-case prediction."""
    from pinchpy.ml.predict import predict_batch_network_targets

    df = generate_dataset(n_cases=40, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(32,), max_iter=30, random_state=42)
    guardrail = OODGuardrail().fit(X)

    cases = [
        (four_streams, 10.0, 0.1),       # nominal in-distribution
        (sample_streams, 10.0, 0.1),     # nominal in-distribution
        (four_streams, 140.0, 0.1),      # OOD: high approach
        (sample_streams, 10.0, 2.5),     # OOD: high U
    ]

    # Batch prediction
    batch_res = predict_batch_network_targets(cases, model=model, guardrail=guardrail, auto_fallback=True)
    assert len(batch_res) == 4

    # Sequential predictions
    seq_res = [
        predict_network_targets(s, dt, u, model=model, guardrail=guardrail, auto_fallback=True)
        for s, dt, u in cases
    ]

    for b, s in zip(batch_res, seq_res):
        assert b["is_ood"] == s["is_ood"]
        assert b["source"] == s["source"]
        assert b["total_area"] == pytest.approx(s["total_area"], rel=1e-4)


def test_no_data_leakage_between_splits():
    """Verify that train, validation, and test partitions are completely disjoint."""
    df = generate_dataset(n_cases=100, seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )

    # Convert rows to bytes or tuples to test set intersection
    train_set = {tuple(row) for row in X_train}
    val_set = {tuple(row) for row in X_val}
    test_set = {tuple(row) for row in X_test}

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_untrusted_prediction_never_leaked(four_streams):
    """Verify that untrusted/corrupted model outputs are strictly discarded on failure."""
    class CorruptedModel:
        def __init__(self, corrupted_output):
            self.corrupted_output = corrupted_output

        def predict(self, X):
            return np.tile(self.corrupted_output, (len(X), 1))

    guardrail = OODGuardrail()
    # Fit guardrail on nominal data
    df = generate_dataset(n_cases=20, seed=42)
    guardrail.fit(df.iloc[:, :-3].to_numpy())

    # Case 1: Corrupted model predicts negative area (-500 m²)
    bad_area_model = CorruptedModel([-500.0, 50.0, 100.0])
    res1 = predict_network_targets(four_streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=bad_area_model, guardrail=guardrail)
    assert res1["source"] == "deterministic_fallback"
    assert res1["is_ood"] is True
    assert "Physical violation" in res1["fallback_reason"]
    # Returned area must be exact positive deterministic ground truth
    assert res1["total_area"] > 0
    assert res1["total_area"] != -500.0

    # Case 2: Corrupted model predicts gross First Law violation (Qc - Qh off by 10,000 kW)
    bad_physics_model = CorruptedModel([200.0, 10.0, 10000.0])
    res2 = predict_network_targets(four_streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=bad_physics_model, guardrail=guardrail)
    assert res2["source"] == "deterministic_fallback"
    assert res2["is_ood"] is True
    assert "First Law" in res2["fallback_reason"]
    assert res2["minimum_cold_utility"] < 1000.0


def test_guarded_inference_latency_budget(four_streams):
    """Verify that single-case guarded surrogate inference executes under 5 ms (robust threshold)."""
    import time
    df = generate_dataset(n_cases=30, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=20, random_state=42)
    guardrail = OODGuardrail().fit(X)

    # Warm-up
    for _ in range(10):
        _ = predict_network_targets(
            four_streams,
            delta_t_min=10.0,
            overall_heat_transfer_coefficient=0.1,
            model=model,
            guardrail=guardrail,
        )

    n_calls = 50
    t0 = time.perf_counter_ns()
    for _ in range(n_calls):
        _ = predict_network_targets(
            four_streams,
            delta_t_min=10.0,
            overall_heat_transfer_coefficient=0.1,
            model=model,
            guardrail=guardrail,
        )
    t1 = time.perf_counter_ns()
    mean_latency_ms = (t1 - t0) / (n_calls * 1e6)

    # Target: relaxed to strictly under 5.0 ms to avoid CI flakiness
    assert mean_latency_ms < 5.0, f"Latency {mean_latency_ms:.3f} ms exceeded 5 ms budget"


def test_oversized_benchmark_case_flagged_ood():
    """Verify that oversized literature benchmark problems (>5 hot or cold streams) are flagged OOD, not crashing."""
    from pinchpy.utils.io import load_furman_sahinidis_dat

    # 12sp1 has 9 hot streams and 3 cold streams
    bench_file = Path("data/benchmarks/furman_sahinidis/dat_files/12sp1.dat")
    streams, dtmin = load_furman_sahinidis_dat(bench_file)
    n_hot = sum(1 for s in streams if s.stream_type == "hot")
    assert n_hot == 9, "12sp1 should have 9 hot streams"

    df = generate_dataset(n_cases=30, seed=42)
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -3:].to_numpy()
    model = train_ann(X, y, hidden_layer_sizes=(16,), max_iter=20, random_state=42)
    guardrail = OODGuardrail().fit(X)

    # Auto fallback mode: flags OOD and safely computes exact deterministic result without crashing
    res = predict_network_targets(
        streams,
        delta_t_min=dtmin,
        overall_heat_transfer_coefficient=0.1,
        model=model,
        guardrail=guardrail,
        auto_fallback=True,
    )
    assert res["is_ood"] is True
    assert res["source"] == "deterministic_fallback"
    assert "outside supported range" in res["fallback_reason"]
    assert res["total_area"] > 0

    # Strict mode: raises ValueError explaining the capacity boundary
    with pytest.raises(ValueError, match="out-of-distribution"):
        predict_network_targets(
            streams,
            delta_t_min=dtmin,
            overall_heat_transfer_coefficient=0.1,
            model=model,
            guardrail=guardrail,
            auto_fallback=False,
        )


def test_pinch_targets_linear_scaling_in_cp(four_streams):
    """Verify that exact utility_targets and area_target scale strictly linearly with CP."""
    delta_t_min = 10.0
    u_val = 0.1

    base_util = utility_targets(four_streams, delta_t_min)
    base_area = area_target(four_streams, delta_t_min, u_val)

    # Test scaling by various positive factors k
    for k in [0.1, 0.5, 2.0, 5.0, 10.0]:
        scaled_streams = [
            Stream(
                id=s.id,
                stream_type=s.stream_type,
                supply_temperature=s.supply_temperature,
                target_temperature=s.target_temperature,
                heat_capacity_flow=s.heat_capacity_flow * k,
            )
            for s in four_streams
        ]

        scaled_util = utility_targets(scaled_streams, delta_t_min)
        scaled_area = area_target(scaled_streams, delta_t_min, u_val)

        # Q_H,min, Q_C,min, and total area must scale strictly by k
        assert scaled_util["minimum_hot_utility"] == pytest.approx(base_util["minimum_hot_utility"] * k, rel=1e-5)
        assert scaled_util["minimum_cold_utility"] == pytest.approx(base_util["minimum_cold_utility"] * k, rel=1e-5)
        assert scaled_area["total_area"] == pytest.approx(base_area["total_area"] * k, rel=1e-5)




