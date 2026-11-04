"""Unit tests for Phase 7: Dataset generation using the validated pinch engine."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from pinchpy.models.stream import Stream
from pinchpy.ml.dataset import generate_case, generate_dataset
from pinchpy.ml.features import (
    case_features,
    stream_features,
    feature_names,
    TARGET_NAMES,
    FEATURES_PER_STREAM,
)


def test_generate_case_returns_valid_streams_and_shapes():
    """Verify that generate_case produces physical Stream objects and valid vectors."""
    streams, x, y = generate_case(n_hot=2, n_cold=3, delta_t_min=10.0, seed=123)

    assert len(streams) == 5
    assert sum(1 for s in streams if s.type == "hot") == 2
    assert sum(1 for s in streams if s.type == "cold") == 3

    # All streams must be valid and positive
    for s in streams:
        assert isinstance(s, Stream)
        assert s.cp > 0
        assert s.delta_t > 0
        assert s.duty() > 0

    # Feature vector shape: 3 + 4 * 5 + 4 * 5 = 43
    expected_len = 3 + 2 * 5 * FEATURES_PER_STREAM
    assert len(x) == expected_len
    assert len(y) == 3  # QH,min, QC,min, pinch_T


def test_generate_case_reproducible_with_seed():
    """Identical seeds must produce identical streams and target values."""
    s1, x1, y1 = generate_case(n_hot=3, n_cold=2, delta_t_min=15.0, seed=999)
    s2, x2, y2 = generate_case(n_hot=3, n_cold=2, delta_t_min=15.0, seed=999)

    assert [s.id for s in s1] == [s.id for s in s2]
    assert [s.supply_temperature for s in s1] == [s.supply_temperature for s in s2]
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_generate_case_first_law_enthalpy_balance():
    """Every generated case must satisfy First Law enthalpy conservation:
    sum(Q_hot) + QH,min == sum(Q_cold) + QC,min.
    """
    for seed in [10, 42, 100, 777]:
        streams, _, y = generate_case(n_hot=3, n_cold=3, delta_t_min=10.0, seed=seed)
        total_hot = sum(s.duty() for s in streams if s.type == "hot")
        total_cold = sum(s.duty() for s in streams if s.type == "cold")
        qh_min, qc_min, _ = y

        assert total_hot + qh_min == pytest.approx(total_cold + qc_min, rel=1e-5)


def test_generate_case_pinch_within_shifted_temperature_bounds():
    """Pinch temperature must lie within the stream system's shifted temperature range."""
    for seed in [1, 2, 3]:
        streams, _, y = generate_case(n_hot=2, n_cold=2, delta_t_min=10.0, seed=seed)
        _, _, pinch_shifted = y

        shifted_hot_supplies = [s.supply_temperature - 5.0 for s in streams if s.type == "hot"]
        shifted_cold_targets = [s.target_temperature + 5.0 for s in streams if s.type == "cold"]
        shifted_hot_targets = [s.target_temperature - 5.0 for s in streams if s.type == "hot"]
        shifted_cold_supplies = [s.supply_temperature + 5.0 for s in streams if s.type == "cold"]

        max_shifted = max(shifted_hot_supplies + shifted_cold_targets)
        min_shifted = min(shifted_hot_targets + shifted_cold_supplies)

        assert min_shifted <= pinch_shifted <= max_shifted


def test_case_features_zero_padding():
    """Systems with fewer streams than max_hot/max_cold must be zero-padded correctly."""
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    x = case_features(streams, delta_t_min=10.0, max_hot=3, max_cold=3)

    # Header: delta_t_min, n_hot, n_cold
    assert x[0] == 10.0
    assert x[1] == 1.0
    assert x[2] == 1.0

    # H1 features (4 values): Ts, Tt, CP, Q
    assert x[3] == 180.0
    assert x[4] == 80.0
    assert x[5] == 2.0
    assert x[6] == 200.0

    # H2 and H3 are padded with zeros (8 values: indices 7 to 14)
    assert np.all(x[7:15] == 0.0)


def test_feature_names_matches_case_features_length():
    """Feature names list length must strictly match case_features length."""
    for max_h in [2, 5]:
        for max_c in [2, 5]:
            names = feature_names(max_hot=max_h, max_cold=max_c)
            streams = [Stream("H1", "hot", 100, 50, 1.0)]
            x = case_features(streams, 10.0, max_hot=max_h, max_cold=max_c)
            assert len(names) == len(x)


def test_generate_dataset_shape_and_no_nans():
    """Generate a dataset and verify columns, row count, and lack of nulls."""
    df = generate_dataset(n_cases=30, seed=42, max_hot=3, max_cold=3)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30

    expected_cols = feature_names(max_hot=3, max_cold=3) + TARGET_NAMES
    assert list(df.columns) == expected_cols
    assert df.isnull().sum().sum() == 0
    assert np.all(np.isfinite(df.values))


def test_generate_dataset_reproducibility():
    """Datasets generated with the same parameters must be identical."""
    df1 = generate_dataset(n_cases=20, seed=123, max_hot=3, max_cold=3)
    df2 = generate_dataset(n_cases=20, seed=123, max_hot=3, max_cold=3)

    pd.testing.assert_frame_equal(df1, df2)


def test_generate_dataset_all_energy_balances_hold():
    """Verify that every single row generated in the dataset respects first law energy balance."""
    df = generate_dataset(n_cases=50, seed=77, max_hot=3, max_cold=3)

    for _, row in df.iterrows():
        # Sum hot duties: H1_Q, H2_Q, H3_Q
        hot_duties = sum(row[f"H{i+1}_Q"] for i in range(3))
        cold_duties = sum(row[f"C{i+1}_Q"] for i in range(3))
        qh = row["minimum_hot_utility"]
        qc = row["minimum_cold_utility"]

        assert hot_duties + qh == pytest.approx(cold_duties + qc, rel=1e-5)


def test_generate_dataset_csv_export():
    """Verify output_path writes a valid, loadable CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_dataset.csv")
        df = generate_dataset(n_cases=15, seed=42, output_path=csv_path)

        assert os.path.exists(csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 15
        assert list(loaded.columns) == list(df.columns)

