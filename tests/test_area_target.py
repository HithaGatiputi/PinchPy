import math
import warnings

import numpy as np
import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis import area_target as area_module
from pinchpy.analysis.area_target import area_target, enthalpy_strips

U = 0.1  # overall heat-transfer coefficient, kW/(m²·°C)


def _simple_streams():
    return [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]


def _multi_streams():
    """Two hot and two cold streams with no pinch (a threshold problem)."""
    return [
        Stream("H1", "hot", 200, 100, 2.0),
        Stream("H2", "hot", 150, 60, 1.0),
        Stream("C1", "cold", 30, 120, 1.5),
        Stream("C2", "cold", 50, 90, 1.0),
    ]


def _four_streams():
    return [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]


def _lmtd(gap_1, gap_2):
    """Independent LMTD formula, so these tests do not rely on the library's."""
    if math.isclose(gap_1, gap_2):
        return gap_1
    return (gap_1 - gap_2) / math.log(gap_1 / gap_2)


def _fake_data(hot_curve, cold_curve, threshold=False):
    return {
        "hot_curve": hot_curve,
        "cold_curve": cold_curve,
        "is_threshold_problem": threshold,
    }


def _patch_curves(monkeypatch, hot_curve, cold_curve):
    monkeypatch.setattr(
        area_module,
        "composite_curve_plot_data",
        lambda streams, delta_t_min: _fake_data(hot_curve, cold_curve),
    )


def _independent_area(streams, coefficient, points=20001):
    """Area by numerically integrating dH / (U * gap) over the heat recovered.

    Rebuilds both composite curves straight from the streams (enthalpy as a
    function of temperature) instead of using the library's curves. Only valid
    for problems with no hot utility, where the cold curve starts at
    ``hot duty - cold duty`` and every stream range is contiguous.
    """
    hot = [s for s in streams if s.type == "hot"]
    cold = [s for s in streams if s.type == "cold"]

    def enthalpy_at(group, temperatures):
        total = np.zeros_like(temperatures)
        for s in group:
            low = min(s.supply_temperature, s.target_temperature)
            high = max(s.supply_temperature, s.target_temperature)
            total += s.cp * (np.clip(temperatures, low, high) - low)
        return total

    hot_grid = np.linspace(min(min(s.supply_temperature, s.target_temperature) for s in hot),
                           max(max(s.supply_temperature, s.target_temperature) for s in hot), 14001)
    cold_grid = np.linspace(min(min(s.supply_temperature, s.target_temperature) for s in cold),
                            max(max(s.supply_temperature, s.target_temperature) for s in cold), 9001)
    hot_h = enthalpy_at(hot, hot_grid)
    cold_h = enthalpy_at(cold, cold_grid)

    offset = hot_h[-1] - cold_h[-1]
    heat = np.linspace(offset, hot_h[-1], points)
    hot_t = np.interp(heat, hot_h, hot_grid)
    cold_t = np.interp(heat - offset, cold_h, cold_grid)
    integrand = 1.0 / (coefficient * (hot_t - cold_t))
    return float(np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(heat)))


# ---------------------------------------------------------------------------
# area_target on real streams
# ---------------------------------------------------------------------------


def test_single_strip_matches_hand_calculation():
    result = area_target(_simple_streams(), 10, U)
    assert len(result["strips"]) == 1
    strip = result["strips"][0]
    assert strip["h_start"] == pytest.approx(5.0)
    assert strip["h_end"] == pytest.approx(200.0)
    assert strip["hot_out"] == pytest.approx(82.5)
    assert strip["hot_in"] == pytest.approx(180.0)
    assert strip["cold_in"] == pytest.approx(20.0)
    assert strip["cold_out"] == pytest.approx(150.0)
    assert strip["delta_t_cold_end"] == pytest.approx(62.5)
    assert strip["delta_t_hot_end"] == pytest.approx(30.0)

    expected = 195.0 / (U * _lmtd(62.5, 30.0))
    assert result["total_area"] == pytest.approx(expected)
    assert strip["lmtd"] == pytest.approx(_lmtd(62.5, 30.0))


def test_result_summary_for_threshold_problem():
    result = area_target(_simple_streams(), 10, U)
    assert result["is_threshold_problem"] is True
    assert result["heat_recovery"] == pytest.approx(195.0)
    assert result["minimum_approach"] == pytest.approx(30.0)
    assert result["delta_t_min"] == 10
    assert result["overall_heat_transfer_coefficient"] == U


def test_multi_stream_area_matches_independent_integration():
    streams = _multi_streams()
    result = area_target(streams, 10, U)
    assert result["total_area"] == pytest.approx(_independent_area(streams, U), rel=1e-5)


def test_strips_cover_the_recovered_heat_and_sum_to_the_total():
    result = area_target(_multi_streams(), 10, U)
    strips = result["strips"]

    assert len(strips) == 4
    assert strips[0]["h_start"] == pytest.approx(115.0)   # cold utility
    assert strips[-1]["h_end"] == pytest.approx(290.0)    # all hot-stream heat
    for previous, following in zip(strips, strips[1:]):
        assert previous["h_end"] == pytest.approx(following["h_start"])
    assert sum(s["duty"] for s in strips) == pytest.approx(result["heat_recovery"])
    assert result["heat_recovery"] == pytest.approx(175.0)  # = total cold duty

    for strip in strips:
        gaps = (strip["delta_t_cold_end"], strip["delta_t_hot_end"])
        assert all(gap > 0 for gap in gaps)
        assert strip["area"] == pytest.approx(strip["duty"] / (U * _lmtd(*gaps)))
    assert result["total_area"] == pytest.approx(sum(s["area"] for s in strips))
    assert result["minimum_approach"] == pytest.approx(
        min(min(s["delta_t_cold_end"], s["delta_t_hot_end"]) for s in strips)
    )


def test_area_is_inversely_proportional_to_the_coefficient():
    base = area_target(_multi_streams(), 10, 0.1)["total_area"]
    assert area_target(_multi_streams(), 10, 0.2)["total_area"] == pytest.approx(base / 2)
    assert area_target(_multi_streams(), 10, 0.05)["total_area"] == pytest.approx(base * 2)


def test_integer_and_float_coefficients_are_accepted():
    assert area_target(_simple_streams(), 10, 1)["overall_heat_transfer_coefficient"] == 1.0


@pytest.mark.xfail(
    reason=(
        "Phase 2/3 shift hot streams up and cold streams down by ΔTmin/2 (standard "
        "pinch analysis does the opposite), so on this problem the composite curves "
        "cross and the area is undefined. This should start passing once the shift "
        "direction is corrected upstream."
    ),
    strict=False,
)
def test_four_stream_area_target_with_pinch():
    result = area_target(_four_streams(), 10, U)
    assert result["total_area"] > 0
    assert result["is_threshold_problem"] is False
    assert result["minimum_approach"] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Empty and one-sided input
# ---------------------------------------------------------------------------


def test_empty_input_gives_zero_area():
    result = area_target([], 10, U)
    assert result["total_area"] == 0.0
    assert result["strips"] == []
    assert result["heat_recovery"] == 0.0
    assert result["minimum_approach"] is None


@pytest.mark.parametrize(
    "streams",
    [
        [Stream("H1", "hot", 180, 80, 2.0)],
        [Stream("C1", "cold", 20, 150, 1.5)],
    ],
    ids=["only-hot", "only-cold"],
)
def test_one_sided_input_gives_zero_area(streams):
    result = area_target(streams, 10, U)
    assert result["total_area"] == 0.0
    assert result["strips"] == []


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
@pytest.mark.parametrize("make_streams", [_simple_streams, list], ids=["streams", "empty"])
def test_invalid_delta_t_min_raises_value_error(make_streams, bad_delta_t_min):
    with pytest.raises(ValueError):
        area_target(make_streams(), bad_delta_t_min, U)


@pytest.mark.parametrize("bad_coefficient", [0, -1, 0.0, float("nan"), float("inf")])
def test_invalid_coefficient_raises_value_error(bad_coefficient):
    with pytest.raises(ValueError):
        area_target(_simple_streams(), 10, bad_coefficient)
    with pytest.raises(ValueError):
        area_target([], 10, bad_coefficient)


@pytest.mark.parametrize("bad_coefficient", ["0.1", None, True, [0.1]])
def test_non_numeric_coefficient_raises_type_error(bad_coefficient):
    with pytest.raises(TypeError):
        area_target(_simple_streams(), 10, bad_coefficient)


# ---------------------------------------------------------------------------
# enthalpy_strips
# ---------------------------------------------------------------------------


def test_strips_split_at_every_bend_of_either_curve():
    hot = [(0.0, 100.0), (50.0, 150.0), (100.0, 200.0)]
    cold = [(0.0, 20.0), (100.0, 120.0)]
    strips = enthalpy_strips(hot, cold)
    assert [(s["h_start"], s["h_end"]) for s in strips] == [(0.0, 50.0), (50.0, 100.0)]
    first, second = strips
    assert (first["hot_out"], first["hot_in"]) == pytest.approx((100.0, 150.0))
    assert (first["cold_in"], first["cold_out"]) == pytest.approx((20.0, 70.0))
    assert (second["hot_out"], second["hot_in"]) == pytest.approx((150.0, 200.0))
    assert (second["cold_in"], second["cold_out"]) == pytest.approx((70.0, 120.0))
    assert all(s["delta_t_cold_end"] == pytest.approx(80.0) for s in strips)
    assert all(s["delta_t_hot_end"] == pytest.approx(80.0) for s in strips)


def test_strips_use_only_the_horizontal_overlap():
    hot = [(0.0, 100.0), (100.0, 200.0)]
    cold = [(30.0, 20.0), (130.0, 120.0)]
    (strip,) = enthalpy_strips(hot, cold)
    assert (strip["h_start"], strip["h_end"], strip["duty"]) == pytest.approx((30.0, 100.0, 70.0))
    assert (strip["hot_out"], strip["hot_in"]) == pytest.approx((130.0, 200.0))
    assert (strip["cold_in"], strip["cold_out"]) == pytest.approx((20.0, 90.0))


def test_vertical_piece_of_a_curve_is_not_used_for_temperatures():
    # The hot curve jumps from 140 to 160 °C at H = 40 (no CP in that range).
    hot = [(0.0, 100.0), (40.0, 140.0), (40.0, 160.0), (100.0, 220.0)]
    cold = [(0.0, 50.0), (100.0, 150.0)]
    first, second = enthalpy_strips(hot, cold)
    assert (first["hot_out"], first["hot_in"]) == pytest.approx((100.0, 140.0))
    assert (second["hot_out"], second["hot_in"]) == pytest.approx((160.0, 220.0))
    assert (first["delta_t_cold_end"], first["delta_t_hot_end"]) == pytest.approx((50.0, 50.0))
    assert (second["delta_t_cold_end"], second["delta_t_hot_end"]) == pytest.approx((70.0, 70.0))


def test_strips_empty_without_overlap_or_curve():
    hot = [(0.0, 100.0), (10.0, 110.0)]
    cold = [(10.0, 50.0), (20.0, 60.0)]
    assert enthalpy_strips(hot, cold) == []
    assert enthalpy_strips([], cold) == []
    assert enthalpy_strips(hot, []) == []


# ---------------------------------------------------------------------------
# Crossing curves, warnings and multiple pinches (curves supplied directly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cold_start", [100.0, 120.0], ids=["touching", "crossing"])
def test_curves_that_touch_or_cross_raise_a_clear_error(monkeypatch, cold_start):
    hot = [(0.0, 100.0), (100.0, 150.0)]
    cold = [(0.0, cold_start), (100.0, 130.0)]
    _patch_curves(monkeypatch, hot, cold)
    with pytest.raises(ValueError, match="cross or touch"):
        area_target(_simple_streams(), 10, U)


def test_gap_smaller_than_delta_t_min_gives_a_warning(monkeypatch):
    _patch_curves(monkeypatch, [(0.0, 100.0), (100.0, 150.0)], [(0.0, 96.0), (100.0, 146.0)])
    with pytest.warns(RuntimeWarning, match="ΔTmin"):
        result = area_target(_simple_streams(), 10, U)
    assert result["total_area"] == pytest.approx(100.0 / (U * 4.0))


def test_gap_equal_to_delta_t_min_is_silent(monkeypatch):
    _patch_curves(monkeypatch, [(0.0, 100.0), (100.0, 150.0)], [(0.0, 90.0), (100.0, 140.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = area_target(_simple_streams(), 10, U)
    assert result["minimum_approach"] == pytest.approx(10.0)


def test_curves_touching_delta_t_min_at_two_places_are_handled(monkeypatch):
    # Gap pattern 30, 10, 30, 10: two pinch-like points of closest approach.
    hot = [(0.0, 100.0), (50.0, 140.0), (100.0, 180.0), (150.0, 200.0)]
    cold = [(0.0, 70.0), (50.0, 130.0), (100.0, 150.0), (150.0, 190.0)]
    _patch_curves(monkeypatch, hot, cold)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = area_target(_simple_streams(), 10, U)
    assert len(result["strips"]) == 3
    assert result["minimum_approach"] == pytest.approx(10.0)
    expected = 3 * 50.0 / (U * _lmtd(30.0, 10.0))
    assert result["total_area"] == pytest.approx(expected)