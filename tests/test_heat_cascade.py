import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis.heat_cascade import heat_cascade
from pinchpy.analysis.pinch_point import utility_targets, pinch_temperatures


def test_heat_cascade_is_always_non_negative_after_shift():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    _, cascade, _, _ = heat_cascade(streams, 10)
    assert all(value >= -1e-9 for value in cascade)


def test_heat_cascade_satisfies_enthalpy_balance():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    _, _, hot_utility, cold_utility = heat_cascade(streams, 10)

    total_hot_duty = sum(s.duty() for s in streams if s.type == "hot")
    total_cold_duty = sum(s.duty() for s in streams if s.type == "cold")

    assert cold_utility - hot_utility == pytest.approx(total_hot_duty - total_cold_duty)


def test_heat_cascade_threshold_problem_needs_no_hot_utility():
    # Cold demand (195) is less than hot supply (200): pure threshold case.
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    _, _, hot_utility, cold_utility = heat_cascade(streams, 10)
    assert hot_utility == 0.0
    assert cold_utility == pytest.approx(5.0)


def test_utility_targets_flags_threshold_problem():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    result = utility_targets(streams, 10)
    assert result["is_threshold_problem"] is True
    assert result["minimum_hot_utility"] == 0.0


def test_utility_targets_interior_pinch_not_flagged_as_threshold():
    # Four-stream problem with an interior pinch requiring both utilities.
    streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]
    result = utility_targets(streams, 10)
    assert result["minimum_hot_utility"] > 0
    assert result["minimum_cold_utility"] > 0
    assert result["is_threshold_problem"] is False


def test_pinch_temperatures_returns_all_zero_points():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    intervals, cascade, _, _ = heat_cascade(streams, 10)
    points = pinch_temperatures(intervals, cascade)
    assert len(points) >= 1
    assert all(isinstance(p, float) for p in points)


def test_utility_targets_empty_streams_returns_none_pinch():
    result = utility_targets([], 10)
    assert result["minimum_hot_utility"] == 0.0
    assert result["minimum_cold_utility"] == 0.0
    assert result["pinch_shifted_temperature"] is None
    assert result["pinch_shifted_temperatures"] == []
