import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis import plot_data
from pinchpy.analysis.composite_curves import composite_curve_data
from pinchpy.analysis.heat_cascade import heat_cascade
from pinchpy.analysis.plot_data import (
    composite_curve_plot_data,
    grand_composite_curve_data,
    minimum_approach,
)


def _simple_streams():
    return [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]


def _four_streams():
    return [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]


STREAM_SETS = pytest.mark.parametrize(
    "make_streams", [_simple_streams, _four_streams], ids=["two-stream", "four-stream"]
)


def _duty(streams, stream_type):
    return sum(s.duty() for s in streams if s.type == stream_type)


# ---------------------------------------------------------------------------
# Composite curves
# ---------------------------------------------------------------------------


@STREAM_SETS
def test_hot_curve_is_the_phase3_hot_curve(make_streams):
    streams = make_streams()
    data = composite_curve_plot_data(streams, 10)
    assert data["hot_curve"] == composite_curve_data(streams, 10)["hot_curve"]


@STREAM_SETS
def test_cold_curve_is_phase3_cold_curve_shifted_by_cold_utility(make_streams):
    streams = make_streams()
    data = composite_curve_plot_data(streams, 10)
    raw = composite_curve_data(streams, 10)["cold_curve"]
    shift = data["minimum_cold_utility"]
    assert len(data["cold_curve"]) == len(raw)
    for (h, t), (h_raw, t_raw) in zip(data["cold_curve"], raw):
        assert h == pytest.approx(h_raw + shift)
        assert t == t_raw


@STREAM_SETS
def test_curve_endpoints_follow_duties_and_utilities(make_streams):
    streams = make_streams()
    data = composite_curve_plot_data(streams, 10)
    hot_end = data["hot_curve"][-1][0]
    cold_end = data["cold_curve"][-1][0]
    assert hot_end == pytest.approx(_duty(streams, "hot"))
    assert cold_end == pytest.approx(data["minimum_cold_utility"] + _duty(streams, "cold"))
    # The cold curve overhangs the hot curve by exactly the hot utility.
    assert cold_end - hot_end == pytest.approx(data["minimum_hot_utility"])


@STREAM_SETS
def test_heat_recovery_is_consistent_from_both_sides(make_streams):
    streams = make_streams()
    data = composite_curve_plot_data(streams, 10)
    assert data["heat_recovery"] == pytest.approx(
        _duty(streams, "hot") - data["minimum_cold_utility"]
    )
    assert data["heat_recovery"] == pytest.approx(
        _duty(streams, "cold") - data["minimum_hot_utility"]
    )


@STREAM_SETS
def test_curves_are_monotone(make_streams):
    data = composite_curve_plot_data(make_streams(), 10)
    for curve in (data["hot_curve"], data["cold_curve"]):
        enthalpies = [h for h, _ in curve]
        temperatures = [t for _, t in curve]
        assert enthalpies == sorted(enthalpies)
        assert all(b > a for a, b in zip(temperatures, temperatures[1:]))


@STREAM_SETS
def test_pinch_markers_lie_inside_their_curves(make_streams):
    data = composite_curve_plot_data(make_streams(), 10)
    assert len(data["pinch_points"]) == len(data["pinch_shifted_temperatures"]) >= 1
    for pinch in data["pinch_points"]:
        for key, curve in (("hot", data["hot_curve"]), ("cold", data["cold_curve"])):
            marker = pinch[key]
            if marker is None:
                continue
            temperatures = [t for _, t in curve]
            assert min(temperatures) - 1e-9 <= marker[1] <= max(temperatures) + 1e-9


def test_threshold_problem_is_reported():
    data = composite_curve_plot_data(_simple_streams(), 10)
    assert data["is_threshold_problem"] is True
    assert data["minimum_hot_utility"] == 0.0
    assert data["minimum_cold_utility"] == pytest.approx(5.0)


def test_interior_pinch_problem_is_not_threshold():
    data = composite_curve_plot_data(_four_streams(), 10)
    assert data["is_threshold_problem"] is False
    assert data["minimum_hot_utility"] > 0
    assert data["minimum_cold_utility"] > 0


def test_only_hot_streams_gives_empty_cold_curve():
    streams = [Stream("H1", "hot", 180, 80, 2.0)]
    data = composite_curve_plot_data(streams, 10)
    assert data["cold_curve"] == []
    assert len(data["hot_curve"]) >= 2
    assert data["heat_recovery"] == pytest.approx(0.0)


def test_only_cold_streams_gives_empty_hot_curve():
    streams = [Stream("C1", "cold", 20, 150, 1.5)]
    data = composite_curve_plot_data(streams, 10)
    assert data["hot_curve"] == []
    assert len(data["cold_curve"]) >= 2
    assert data["heat_recovery"] == pytest.approx(0.0)


def test_composite_empty_input_gives_empty_curves():
    data = composite_curve_plot_data([], 10)
    assert data["hot_curve"] == []
    assert data["cold_curve"] == []
    assert data["pinch_points"] == []
    assert data["pinch_shifted_temperatures"] == []
    assert data["minimum_hot_utility"] == 0.0
    assert data["minimum_cold_utility"] == 0.0
    assert data["minimum_approach"] is None


def test_multiple_pinches_each_get_a_marker(monkeypatch):
    fake_targets = {
        "minimum_hot_utility": 0.0,
        "minimum_cold_utility": 5.0,
        "pinch_shifted_temperature": 175.0,
        "pinch_hot_temperature": 180.0,
        "pinch_cold_temperature": 170.0,
        "pinch_shifted_temperatures": [175.0, 100.0],
        "is_threshold_problem": False,
    }
    monkeypatch.setattr(plot_data, "utility_targets", lambda s, d: fake_targets)
    data = composite_curve_plot_data(_simple_streams(), 10)

    assert [p["shifted_temperature"] for p in data["pinch_points"]] == [175.0, 100.0]
    top, lower = data["pinch_points"]
    # Pinch 1: hot end of the hot curve (180 °C, all 200 kW); cold side is
    # above the cold curve's 150 °C target, so it has no marker.
    assert top["hot"] == pytest.approx((200.0, 180.0))
    assert top["cold"] is None
    # Pinch 2: both real temperatures (105 / 95 °C) fall on their curves.
    assert lower["hot"][1] == pytest.approx(105.0)
    assert lower["cold"][1] == pytest.approx(95.0)


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
@pytest.mark.parametrize("make_streams", [_simple_streams, list], ids=["streams", "empty"])
def test_invalid_delta_t_min_raises_value_error(make_streams, bad_delta_t_min):
    with pytest.raises(ValueError):
        composite_curve_plot_data(make_streams(), bad_delta_t_min)
    with pytest.raises(ValueError):
        grand_composite_curve_data(make_streams(), bad_delta_t_min)


@pytest.mark.xfail(
    reason=(
        "Phase 2/3 shift hot streams up and cold streams down by ΔTmin/2 (standard "
        "pinch analysis does the opposite), so the cold-utility target used to place "
        "the cold curve is too small and the curves come closer than ΔTmin. This "
        "should start passing once the shift direction is corrected upstream."
    ),
    strict=False,
)
def test_composite_curves_keep_delta_t_min_at_closest_approach():
    data = composite_curve_plot_data(_four_streams(), 10)
    assert data["minimum_approach"] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# minimum_approach
# ---------------------------------------------------------------------------


def test_minimum_approach_of_parallel_curves():
    hot = [(0.0, 100.0), (100.0, 200.0)]
    cold = [(10.0, 90.0), (110.0, 190.0)]
    assert minimum_approach(hot, cold) == pytest.approx(20.0)


def test_minimum_approach_is_found_at_a_breakpoint():
    hot = [(0.0, 100.0), (100.0, 200.0)]
    cold = [(0.0, 50.0), (100.0, 190.0)]
    assert minimum_approach(hot, cold) == pytest.approx(10.0)


def test_minimum_approach_negative_when_curves_cross():
    hot = [(0.0, 100.0), (100.0, 150.0)]
    cold = [(0.0, 80.0), (100.0, 200.0)]
    assert minimum_approach(hot, cold) == pytest.approx(-50.0)


def test_minimum_approach_none_without_overlap_or_curve():
    hot = [(0.0, 100.0), (10.0, 110.0)]
    cold = [(10.0, 50.0), (20.0, 60.0)]
    assert minimum_approach(hot, cold) is None
    assert minimum_approach([], cold) is None
    assert minimum_approach(hot, []) is None


# ---------------------------------------------------------------------------
# Grand Composite Curve
# ---------------------------------------------------------------------------


@STREAM_SETS
def test_gcc_matches_the_phase3_cascade(make_streams):
    streams = make_streams()
    intervals, cascade, hot_utility, cold_utility = heat_cascade(streams, 10)
    data = grand_composite_curve_data(streams, 10)

    assert data["heat_flow"] == cascade
    assert len(data["temperatures"]) == len(intervals) + 1
    assert data["temperatures"][0] == intervals[0][0]
    assert data["temperatures"][1:] == [low for _, low in intervals]
    assert all(b < a for a, b in zip(data["temperatures"], data["temperatures"][1:]))
    assert data["points"] == list(zip(cascade, data["temperatures"]))
    assert data["minimum_hot_utility"] == hot_utility
    assert data["minimum_cold_utility"] == cold_utility


@STREAM_SETS
def test_gcc_end_values_are_the_utilities_and_never_negative(make_streams):
    data = grand_composite_curve_data(make_streams(), 10)
    flow = data["heat_flow"]
    assert flow[0] == pytest.approx(data["minimum_hot_utility"])
    assert flow[-1] == pytest.approx(data["minimum_cold_utility"])
    assert min(flow) == pytest.approx(0.0, abs=1e-9)
    assert all(value >= -1e-9 for value in flow)


@STREAM_SETS
def test_gcc_is_zero_at_every_reported_pinch(make_streams):
    data = grand_composite_curve_data(make_streams(), 10)
    assert len(data["pinch_shifted_temperatures"]) >= 1
    for temperature in data["pinch_shifted_temperatures"]:
        index = data["temperatures"].index(temperature)
        assert data["heat_flow"][index] == pytest.approx(0.0, abs=1e-9)


def test_gcc_threshold_problem_touches_zero_at_the_top():
    data = grand_composite_curve_data(_simple_streams(), 10)
    assert data["is_threshold_problem"] is True
    assert data["heat_flow"][0] == pytest.approx(0.0)
    assert data["heat_flow"][-1] == pytest.approx(5.0)
    assert data["pinch_shifted_temperatures"] == [data["temperatures"][0]]


def test_gcc_single_hot_stream_is_a_threshold_problem():
    data = grand_composite_curve_data([Stream("H1", "hot", 180, 80, 2.0)], 10)
    assert data["is_threshold_problem"] is True
    assert data["minimum_hot_utility"] == 0.0
    assert data["heat_flow"][0] == 0.0
    assert data["heat_flow"][-1] == pytest.approx(200.0)


def test_gcc_empty_input_gives_empty_curve():
    data = grand_composite_curve_data([], 10)
    assert data["temperatures"] == []
    assert data["heat_flow"] == []
    assert data["points"] == []
    assert data["pinch_shifted_temperatures"] == []
    assert data["minimum_hot_utility"] == 0.0
    assert data["minimum_cold_utility"] == 0.0


def test_gcc_multiple_pinches_are_all_reported(monkeypatch):
    intervals = [(100.0, 80.0), (80.0, 60.0), (60.0, 40.0)]
    cascade = [0.0, 20.0, 0.0, 15.0]
    monkeypatch.setattr(
        plot_data, "heat_cascade", lambda s, d: (intervals, cascade, 0.0, 15.0)
    )
    monkeypatch.setattr(
        plot_data,
        "utility_targets",
        lambda s, d: {
            "pinch_shifted_temperatures": [100.0, 60.0],
            "is_threshold_problem": True,
        },
    )
    data = grand_composite_curve_data(_simple_streams(), 10)
    assert data["temperatures"] == [100.0, 80.0, 60.0, 40.0]
    assert data["heat_flow"] == [0.0, 20.0, 0.0, 15.0]
    assert data["pinch_shifted_temperatures"] == [100.0, 60.0]