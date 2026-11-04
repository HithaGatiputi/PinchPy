"""Unit tests for Phase 5: Pinch points and minimum utility targets calculation.

Tests verify:
- Exact minimum hot and cold utility targets
- Shifted and real pinch temperatures (hot and cold pinch)
- Heat cascade zero-point identification
- Single-pinch and multiple-pinch detection
- Threshold problems (pure hot utility or pure cold utility)
- First Law of Thermodynamics energy conservation
- Second Law pinch principle (zero net heat flow across the pinch)
- Monotonicity and sensitivity with respect to delta_t_min
- Edge cases: empty stream list, single stream type, non-positive delta_t_min
- Module exports and API stability
"""

import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import (
    utility_targets,
    pinch_temperature,
    pinch_temperatures,
)
from pinchpy.analysis.heat_cascade import heat_cascade


# ---------------------------------------------------------------------------
# Fixtures and test data
# ---------------------------------------------------------------------------

def _two_stream_system():
    """Two-stream threshold case from the README example."""
    return [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]


def _four_stream_system():
    """Classic 4-stream interior pinch problem with known analytical solution.

    H1: 170 -> 60 °C, CP = 3.0 kW/°C (Q = 330 kW)
    H2: 150 -> 30 °C, CP = 1.5 kW/°C (Q = 180 kW)
    C1:  20 -> 135 °C, CP = 2.0 kW/°C (Q = 230 kW)
    C2:  80 -> 140 °C, CP = 4.0 kW/°C (Q = 240 kW)

    At ΔTmin = 10 °C:
      Total hot duty  = 510 kW
      Total cold duty = 470 kW
      Shifted pinch T = 85.0 °C
      Hot pinch T     = 90.0 °C
      Cold pinch T    = 80.0 °C
      QH,min          = 20.0 kW
      QC,min          = 60.0 kW
      Heat recovery   = 450.0 kW
    """
    return [
        Stream("H1", "hot", 170, 60, 3.0),
        Stream("H2", "hot", 150, 30, 1.5),
        Stream("C1", "cold", 20, 135, 2.0),
        Stream("C2", "cold", 80, 140, 4.0),
    ]


# ---------------------------------------------------------------------------
# Core Phase 5 Targeting Tests
# ---------------------------------------------------------------------------

def test_pinch_target_runs():
    """Preserve existing test contract."""
    streams = _two_stream_system()
    result = utility_targets(streams, 10)
    assert "minimum_hot_utility" in result
    assert "minimum_cold_utility" in result


def test_pinch_targets_two_stream_exact_values():
    """Verify two-stream threshold problem exact targeting results."""
    streams = _two_stream_system()
    result = utility_targets(streams, 10)

    assert result["minimum_hot_utility"] == pytest.approx(0.0)
    assert result["minimum_cold_utility"] == pytest.approx(5.0)
    assert result["pinch_shifted_temperature"] == pytest.approx(175.0)
    assert result["pinch_hot_temperature"] == pytest.approx(180.0)
    assert result["pinch_cold_temperature"] == pytest.approx(170.0)
    assert result["pinch_shifted_temperatures"] == [175.0]
    assert result["is_threshold_problem"] is True
    assert result["heat_recovery"] == pytest.approx(195.0)


def test_pinch_targets_four_stream_analytical_interior_pinch():
    """Verify 4-stream problem with an interior pinch against manual derivation."""
    streams = _four_stream_system()
    result = utility_targets(streams, delta_t_min=10)

    assert result["minimum_hot_utility"] == pytest.approx(20.0)
    assert result["minimum_cold_utility"] == pytest.approx(60.0)
    assert result["pinch_shifted_temperature"] == pytest.approx(85.0)
    assert result["pinch_hot_temperature"] == pytest.approx(90.0)
    assert result["pinch_cold_temperature"] == pytest.approx(80.0)
    assert result["pinch_shifted_temperatures"] == [85.0]
    assert result["is_threshold_problem"] is False
    assert result["heat_recovery"] == pytest.approx(450.0)

    # Temperature approach at pinch must strictly equal delta_t_min
    pinch_gap = result["pinch_hot_temperature"] - result["pinch_cold_temperature"]
    assert pinch_gap == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# First and Second Laws of Thermodynamics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtmin", [5.0, 10.0, 15.0, 20.0])
def test_first_law_enthalpy_conservation(dtmin):
    """Overall enthalpy balance: sum(Q_hot) + QH,min == sum(Q_cold) + QC,min."""
    streams = _four_stream_system()
    targets = utility_targets(streams, delta_t_min=dtmin)

    total_hot = sum(s.duty() for s in streams if s.type == "hot")
    total_cold = sum(s.duty() for s in streams if s.type == "cold")

    lhs = total_hot + targets["minimum_hot_utility"]
    rhs = total_cold + targets["minimum_cold_utility"]
    assert lhs == pytest.approx(rhs, rel=1e-9)

    # Heat recovery consistency: Q_rec = sum(Q_hot) - QC,min = sum(Q_cold) - QH,min
    q_rec = targets["heat_recovery"]
    assert q_rec == pytest.approx(total_hot - targets["minimum_cold_utility"], rel=1e-9)
    assert q_rec == pytest.approx(total_cold - targets["minimum_hot_utility"], rel=1e-9)


def test_second_law_zero_heat_flow_across_pinch():
    """Verify the Pinch Principle: net heat flow across the pinch is zero.

    Above pinch: sum(Q_hot_above) + QH,min == sum(Q_cold_above)
    Below pinch: sum(Q_hot_below) == sum(Q_cold_below) + QC,min
    """
    streams = _four_stream_system()
    targets = utility_targets(streams, delta_t_min=10)

    t_pinch_hot = targets["pinch_hot_temperature"]
    t_pinch_cold = targets["pinch_cold_temperature"]

    # Calculate duties strictly above the pinch
    q_hot_above = 0.0
    q_cold_above = 0.0
    for s in streams:
        if s.type == "hot" and s.supply_temperature > t_pinch_hot:
            t_low = max(s.target_temperature, t_pinch_hot)
            q_hot_above += s.cp * (s.supply_temperature - t_low)
        elif s.type == "cold" and s.target_temperature > t_pinch_cold:
            t_low = max(s.supply_temperature, t_pinch_cold)
            q_cold_above += s.cp * (s.target_temperature - t_low)

    assert q_hot_above + targets["minimum_hot_utility"] == pytest.approx(q_cold_above, rel=1e-9)

    # Calculate duties strictly below the pinch
    q_hot_below = 0.0
    q_cold_below = 0.0
    for s in streams:
        if s.type == "hot" and s.target_temperature < t_pinch_hot:
            t_high = min(s.supply_temperature, t_pinch_hot)
            q_hot_below += s.cp * (t_high - s.target_temperature)
        elif s.type == "cold" and s.supply_temperature < t_pinch_cold:
            t_high = min(s.target_temperature, t_pinch_cold)
            q_cold_below += s.cp * (t_high - s.supply_temperature)

    assert q_hot_below == pytest.approx(q_cold_below + targets["minimum_cold_utility"], rel=1e-9)


# ---------------------------------------------------------------------------
# Threshold and Balanced Problem Tests
# ---------------------------------------------------------------------------

def test_threshold_cold_utility_only():
    """Hot duty exceeds cold duty; no hot utility required."""
    streams = [
        Stream("H1", "hot", 200, 50, 2.0),   # Q = 300 kW
        Stream("C1", "cold", 30, 120, 1.0),  # Q = 90 kW
    ]
    targets = utility_targets(streams, delta_t_min=10)
    assert targets["is_threshold_problem"] is True
    assert targets["minimum_hot_utility"] == pytest.approx(0.0)
    assert targets["minimum_cold_utility"] == pytest.approx(210.0)
    assert targets["heat_recovery"] == pytest.approx(90.0)
    # Pinch for pure cold utility sits at the hot end of the shifted cascade
    assert targets["pinch_shifted_temperature"] == pytest.approx(195.0)


def test_threshold_hot_utility_only():
    """Cold duty exceeds hot duty and cold temps are high; no cold utility required."""
    streams = [
        Stream("H1", "hot", 100, 40, 1.0),   # Q = 60 kW
        Stream("C1", "cold", 30, 180, 2.0),  # Q = 300 kW
    ]
    targets = utility_targets(streams, delta_t_min=10)
    assert targets["is_threshold_problem"] is True
    assert targets["minimum_hot_utility"] == pytest.approx(240.0)
    assert targets["minimum_cold_utility"] == pytest.approx(0.0)
    assert targets["heat_recovery"] == pytest.approx(60.0)
    # Pinch for pure hot utility sits at the cold end of the shifted cascade
    assert targets["pinch_shifted_temperature"] == pytest.approx(35.0)


def test_zero_utility_balanced_network():
    """Hot and cold stream duties match and thermal feasibility requires 0 utilities."""
    streams = [
        Stream("H1", "hot", 150, 50, 2.0),   # Q = 200 kW
        Stream("C1", "cold", 30, 130, 2.0),  # Q = 200 kW
    ]
    targets = utility_targets(streams, delta_t_min=10)
    assert targets["is_threshold_problem"] is True
    assert targets["minimum_hot_utility"] == pytest.approx(0.0)
    assert targets["minimum_cold_utility"] == pytest.approx(0.0)
    assert targets["heat_recovery"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Multiple Pinch Points
# ---------------------------------------------------------------------------

def test_multiple_pinch_points_identified():
    """A cascade reaching minimum (zero) at multiple distinct temperatures."""
    streams = [
        # Two distinct sub-networks that both touch the minimum approach
        Stream("H1", "hot", 200, 120, 2.0),  # 160 kW, shifted 195 -> 115
        Stream("C1", "cold", 110, 190, 2.0), # 160 kW, shifted 115 -> 195
        Stream("H2", "hot", 100, 40, 2.0),   # 120 kW, shifted 95 -> 35
        Stream("C2", "cold", 30, 90, 2.0),   # 120 kW, shifted 35 -> 95
    ]
    targets = utility_targets(streams, delta_t_min=10)

    # In this configuration, both intervals are exactly balanced with zero net utilities
    assert targets["minimum_hot_utility"] == pytest.approx(0.0)
    assert targets["minimum_cold_utility"] == pytest.approx(0.0)
    # Check that multiple pinches are captured
    pinches = targets["pinch_shifted_temperatures"]
    assert len(pinches) >= 2
    assert len(targets["pinch_hot_temperatures"]) == len(pinches)
    assert len(targets["pinch_cold_temperatures"]) == len(pinches)


# ---------------------------------------------------------------------------
# Sensitivity and Monotonicity
# ---------------------------------------------------------------------------

def test_utility_targets_monotonicity_with_delta_t_min():
    """As delta_t_min increases, utility requirements must not decrease."""
    streams = _four_stream_system()
    dt_values = [5.0, 10.0, 15.0, 20.0, 25.0]

    hot_utilities = []
    cold_utilities = []
    heat_recoveries = []

    for dt in dt_values:
        t = utility_targets(streams, delta_t_min=dt)
        hot_utilities.append(t["minimum_hot_utility"])
        cold_utilities.append(t["minimum_cold_utility"])
        heat_recoveries.append(t["heat_recovery"])

    # Monotonically non-decreasing utilities
    for i in range(len(dt_values) - 1):
        assert hot_utilities[i + 1] >= hot_utilities[i] - 1e-9
        assert cold_utilities[i + 1] >= cold_utilities[i] - 1e-9
        assert heat_recoveries[i + 1] <= heat_recoveries[i] + 1e-9


# ---------------------------------------------------------------------------
# Direct Functions and Edge Cases
# ---------------------------------------------------------------------------

def test_pinch_temperature_and_temperatures_direct():
    """Test pinch_temperature and pinch_temperatures with synthetic inputs."""
    intervals = [(150, 100), (100, 50)]
    cascade = [0.0, 10.0, 20.0]  # min at index 0 -> shifted 150
    assert pinch_temperature(intervals, cascade) == 150
    assert pinch_temperatures(intervals, cascade) == [150]

    cascade_mid = [10.0, 0.0, 5.0]  # min at index 1 -> shifted 100
    assert pinch_temperature(intervals, cascade_mid) == 100
    assert pinch_temperatures(intervals, cascade_mid) == [100]

    cascade_end = [20.0, 10.0, 0.0]  # min at index 2 -> shifted 50
    assert pinch_temperature(intervals, cascade_end) == 50
    assert pinch_temperatures(intervals, cascade_end) == [50]


def test_pinch_temperatures_empty_cascade_or_intervals():
    """Safe handling of empty inputs."""
    assert pinch_temperatures([], []) == []
    assert pinch_temperatures([(100, 50)], []) == []
    assert pinch_temperatures([], [0.0]) == []
    assert pinch_temperature([], []) is None


def test_utility_targets_empty_stream_list():
    """Empty stream list returns graceful defaults without raising."""
    targets = utility_targets([], delta_t_min=10)
    assert targets["minimum_hot_utility"] == 0.0
    assert targets["minimum_cold_utility"] == 0.0
    assert targets["pinch_shifted_temperature"] is None
    assert targets["pinch_hot_temperature"] is None
    assert targets["pinch_cold_temperature"] is None
    assert targets["pinch_shifted_temperatures"] == []
    assert targets["pinch_hot_temperatures"] == []
    assert targets["pinch_cold_temperatures"] == []
    assert targets["heat_recovery"] == 0.0
    assert targets["is_threshold_problem"] is False


def test_utility_targets_only_hot_streams():
    """System with only hot streams requires cold utility equal to total hot duty."""
    streams = [Stream("H1", "hot", 160, 60, 2.0)]  # Q = 200 kW
    targets = utility_targets(streams, delta_t_min=10)
    assert targets["minimum_hot_utility"] == 0.0
    assert targets["minimum_cold_utility"] == pytest.approx(200.0)
    assert targets["heat_recovery"] == 0.0
    assert targets["is_threshold_problem"] is True
    assert targets["pinch_shifted_temperature"] == pytest.approx(155.0)


def test_utility_targets_only_cold_streams():
    """System with only cold streams requires hot utility equal to total cold duty."""
    streams = [Stream("C1", "cold", 40, 140, 2.0)]  # Q = 200 kW
    targets = utility_targets(streams, delta_t_min=10)
    assert targets["minimum_hot_utility"] == pytest.approx(200.0)
    assert targets["minimum_cold_utility"] == 0.0
    assert targets["heat_recovery"] == 0.0
    assert targets["is_threshold_problem"] is True
    assert targets["pinch_shifted_temperature"] == pytest.approx(45.0)


@pytest.mark.parametrize("bad_dt", [0.0, -5.0])
def test_utility_targets_rejects_non_positive_delta_t_min(bad_dt):
    """delta_t_min <= 0 must raise ValueError."""
    streams = _two_stream_system()
    with pytest.raises(ValueError, match="delta_t_min must be greater than zero"):
        utility_targets(streams, delta_t_min=bad_dt)


# ---------------------------------------------------------------------------
# API Exports Stability
# ---------------------------------------------------------------------------

def test_api_exports():
    """Verify clean top-level and analysis-level exports."""
    import pinchpy
    assert hasattr(pinchpy, "utility_targets")
    assert hasattr(pinchpy, "pinch_temperature")
    assert hasattr(pinchpy, "pinch_temperatures")
    assert hasattr(pinchpy, "heat_cascade")

    import pinchpy.analysis
    assert hasattr(pinchpy.analysis, "utility_targets")
    assert hasattr(pinchpy.analysis, "pinch_temperature")
    assert hasattr(pinchpy.analysis, "pinch_temperatures")
    assert hasattr(pinchpy.analysis, "heat_cascade")
