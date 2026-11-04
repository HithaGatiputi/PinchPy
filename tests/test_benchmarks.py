"""Validation tests for PinchPy against published peer-reviewed benchmarks.

Benchmarks verified:
1. Ian C. Kemp (2007), 'Pinch Analysis and Process Integration', Elsevier:
   4-stream benchmark problem at ΔTmin = 10 °C and ΔTmin = 20 °C.
2. Gavin Towler & Ray Sinnott (2013) / Bodo Linnhoff et al. (1982),
   'Chemical Engineering Design', Butterworth-Heinemann:
   4-stream benchmark problem at ΔTmin = 20 °C.
3. Robin Smith (2005), 'Chemical Process Design and Integration', Wiley:
   2-stream benchmark problem (Table 16.1) at ΔTmin = 10 °C.
4. K. F. Lee, A. H. Masso & D. F. Rudd (1970) / S. A. Papoulias & I. E. Grossmann (1983):
   4SP1 benchmark at ΔTmin = 20 °C and ΔTmin = 10 °C (threshold case).
"""

import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets
from pinchpy.analysis.benchmarks import BENCHMARK_CASES, get_benchmark


# ---------------------------------------------------------------------------
# Kemp (2007) Benchmark Validation
# ---------------------------------------------------------------------------

def test_benchmark_kemp_2007_delta_t_min_10():
    """Verify Kemp (2007) 4-stream benchmark at ΔTmin = 10 °C.

    Published targets:
      QH,min = 20 kW, QC,min = 60 kW,
      Tpinch,shifted = 85 °C, Tpinch,hot = 90 °C, Tpinch,cold = 80 °C.
    """
    case = get_benchmark("kemp_2007_4stream")
    streams = case["streams"]
    expected = case["evaluations"][10.0]

    result = utility_targets(streams, delta_t_min=10.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is False


def test_benchmark_kemp_2007_delta_t_min_20():
    """Verify Kemp (2007) 4-stream benchmark at ΔTmin = 20 °C.

    Published targets:
      QH,min = 65 kW, QC,min = 105 kW,
      Tpinch,shifted = 90 °C, Tpinch,hot = 100 °C, Tpinch,cold = 80 °C.
    """
    case = get_benchmark("kemp_2007_4stream")
    streams = case["streams"]
    expected = case["evaluations"][20.0]

    result = utility_targets(streams, delta_t_min=20.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is False


# ---------------------------------------------------------------------------
# Towler & Sinnott / Linnhoff et al. Benchmark Validation
# ---------------------------------------------------------------------------

def test_benchmark_towler_sinnott_linnhoff():
    """Verify Towler & Sinnott / Linnhoff et al. (1982) benchmark at ΔTmin = 20 °C.

    Published targets:
      QH,min = 2900 kW, QC,min = 600 kW,
      Tpinch,shifted = 90 °C, Tpinch,hot = 100 °C, Tpinch,cold = 80 °C,
      Heat recovery = 7700 kW.
    """
    case = get_benchmark("towler_sinnott_linnhoff_4stream")
    streams = case["streams"]
    expected = case["evaluations"][20.0]

    result = utility_targets(streams, delta_t_min=20.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is False


# ---------------------------------------------------------------------------
# Robin Smith (2005) Benchmark Validation
# ---------------------------------------------------------------------------

def test_benchmark_robin_smith_table_16_1():
    """Verify Robin Smith (2005) Table 16.1 benchmark at ΔTmin = 10 °C.

    Published targets:
      QH,min = 3000 kW (3 MW), QC,min = 1000 kW (1 MW),
      Tpinch,shifted = 45 °C, Tpinch,hot = 50 °C, Tpinch,cold = 40 °C,
      Heat recovery = 11000 kW (11 MW).
    """
    case = get_benchmark("robin_smith_2stream")
    streams = case["streams"]
    expected = case["evaluations"][10.0]

    result = utility_targets(streams, delta_t_min=10.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is False


# ---------------------------------------------------------------------------
# Lee, Masso & Rudd / Papoulias & Grossmann 4SP1 Benchmark Validation
# ---------------------------------------------------------------------------

def test_benchmark_4sp1_interior_pinch():
    """Verify 4SP1 benchmark at ΔTmin = 20 °C (interior pinch).

    Published targets:
      QH,min = 188.48 kW, QC,min = 42.94 kW,
      Tpinch,shifted = 239 °C, Tpinch,hot = 249 °C, Tpinch,cold = 229 °C,
      Heat recovery = 1449.04 kW.
    """
    case = get_benchmark("lee_masso_rudd_4sp1")
    streams = case["streams"]
    expected = case["evaluations"][20.0]

    result = utility_targets(streams, delta_t_min=20.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is False


def test_benchmark_4sp1_threshold_pinch():
    """Verify 4SP1 benchmark at ΔTmin = 10 °C (threshold problem, zero cold utility).

    Published targets:
      QH,min = 145.54 kW, QC,min = 0 kW,
      Tpinch,shifted = 65 °C, Tpinch,hot = 70 °C, Tpinch,cold = 60 °C,
      Heat recovery = 1491.98 kW.
    """
    case = get_benchmark("lee_masso_rudd_4sp1")
    streams = case["streams"]
    expected = case["evaluations"][10.0]

    result = utility_targets(streams, delta_t_min=10.0)

    assert result["minimum_hot_utility"] == pytest.approx(expected["minimum_hot_utility"], abs=1e-2)
    assert result["minimum_cold_utility"] == pytest.approx(expected["minimum_cold_utility"], abs=1e-2)
    assert result["pinch_shifted_temperature"] == pytest.approx(expected["pinch_shifted_temperature"], abs=1e-2)
    assert result["pinch_hot_temperature"] == pytest.approx(expected["pinch_hot_temperature"], abs=1e-2)
    assert result["pinch_cold_temperature"] == pytest.approx(expected["pinch_cold_temperature"], abs=1e-2)
    assert result["heat_recovery"] == pytest.approx(expected["heat_recovery"], abs=1e-2)
    assert result["is_threshold_problem"] is True


# ---------------------------------------------------------------------------
# Benchmark Registry Error Handling
# ---------------------------------------------------------------------------

def test_get_benchmark_unknown_key_raises():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_benchmark("non_existent_key")

