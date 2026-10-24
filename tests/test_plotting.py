import warnings

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from pinchpy.models.stream import Stream
from pinchpy.analysis.plot_data import (
    composite_curve_plot_data,
    grand_composite_curve_data,
)
from pinchpy.visualization import cascade_plot, composite_curve
from pinchpy.visualization.cascade_plot import plot_cascade, plot_grand_composite_curve
from pinchpy.visualization.composite_curve import plot_composite_curves

# The four-stream data below can trigger the "closer than ΔTmin" warning while
# Phase 2/3 apply the shift in the opposite direction; that is covered
# explicitly in test_composite_plot_warns_when_curves_violate_delta_t_min.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Composite curves approach:RuntimeWarning"
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


def _line(ax, label):
    matches = [line for line in ax.lines if line.get_label() == label]
    assert len(matches) == 1, f"expected one line labelled {label!r}"
    return matches[0]


def _legend_labels(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _texts(ax):
    return [text.get_text() for text in ax.texts]


def _diamond_markers(ax):
    return [line for line in ax.lines if line.get_marker() == "D"]


def _fake_composite_data(minimum_approach=10.0, pinch_points=()):
    return {
        "delta_t_min": 10,
        "hot_curve": [(0.0, 80.0), (200.0, 180.0)],
        "cold_curve": [(5.0, 20.0), (205.0, 150.0)],
        "minimum_hot_utility": 5.0,
        "minimum_cold_utility": 5.0,
        "heat_recovery": 195.0,
        "pinch_shifted_temperatures": [p["shifted_temperature"] for p in pinch_points],
        "pinch_points": list(pinch_points),
        "minimum_approach": minimum_approach,
        "is_threshold_problem": False,
    }


def _fake_gcc_data(pinches=(100.0, 60.0)):
    return {
        "delta_t_min": 10,
        "temperatures": [100.0, 80.0, 60.0, 40.0],
        "heat_flow": [0.0, 20.0, 0.0, 15.0],
        "points": [(0.0, 100.0), (20.0, 80.0), (0.0, 60.0), (15.0, 40.0)],
        "pinch_shifted_temperatures": list(pinches),
        "minimum_hot_utility": 0.0,
        "minimum_cold_utility": 15.0,
        "is_threshold_problem": False,
    }


# ---------------------------------------------------------------------------
# Composite curves
# ---------------------------------------------------------------------------


def test_composite_plot_returns_figure_and_axes():
    fig, ax = plot_composite_curves(_simple_streams(), 10)
    assert isinstance(fig, Figure)
    assert ax in fig.axes


def test_composite_plot_axes_labels_and_title():
    _, ax = plot_composite_curves(_simple_streams(), 10)
    assert "H" in ax.get_xlabel() and "kW" in ax.get_xlabel()
    assert "T" in ax.get_ylabel() and "°C" in ax.get_ylabel()
    assert "Composite curves" in ax.get_title()
    assert "10" in ax.get_title()


def test_composite_plot_legend_names_both_curves():
    _, ax = plot_composite_curves(_four_streams(), 10)
    labels = _legend_labels(ax)
    assert "Hot composite curve" in labels
    assert "Cold composite curve" in labels


@pytest.mark.parametrize("make_streams", [_simple_streams, _four_streams])
def test_composite_plot_draws_the_plot_data(make_streams):
    streams = make_streams()
    data = composite_curve_plot_data(streams, 10)
    _, ax = plot_composite_curves(streams, 10)
    for label, key in (("Hot composite curve", "hot_curve"),
                       ("Cold composite curve", "cold_curve")):
        line = _line(ax, label)
        assert list(zip(line.get_xdata(), line.get_ydata())) == pytest.approx(data[key])


def test_composite_plot_summarises_utilities():
    _, ax = plot_composite_curves(_four_streams(), 10)
    text = "\n".join(_texts(ax))
    assert "Q_H,min" in text and "Q_C,min" in text and "kW" in text
    assert "Threshold" not in text


def test_composite_plot_flags_threshold_problem():
    _, ax = plot_composite_curves(_simple_streams(), 10)
    assert any("Threshold problem: no hot utility" in t for t in _texts(ax))


def test_composite_plot_marks_every_pinch_with_one_legend_entry(monkeypatch):
    pinches = [
        {"shifted_temperature": 175.0, "hot": (200.0, 180.0), "cold": (205.0, 150.0)},
        {"shifted_temperature": 100.0, "hot": (100.0, 130.0), "cold": None},
    ]
    monkeypatch.setattr(
        composite_curve, "composite_curve_plot_data",
        lambda s, d: _fake_composite_data(pinch_points=pinches),
    )
    _, ax = plot_composite_curves(_simple_streams(), 10)
    assert len(_diamond_markers(ax)) == 3  # the missing cold marker is skipped
    assert _legend_labels(ax).count("Pinch") == 1


def test_composite_plot_can_hide_pinch_markers(monkeypatch):
    pinches = [{"shifted_temperature": 100.0, "hot": (100.0, 130.0), "cold": (100.0, 120.0)}]
    monkeypatch.setattr(
        composite_curve, "composite_curve_plot_data",
        lambda s, d: _fake_composite_data(pinch_points=pinches),
    )
    _, ax = plot_composite_curves(_simple_streams(), 10, show_pinch=False)
    assert _diamond_markers(ax) == []
    assert "Pinch" not in _legend_labels(ax)


def test_composite_plot_draws_into_a_given_axes():
    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_composite_curves(_simple_streams(), 10, ax=ax)
    assert returned_ax is ax
    assert returned_fig is fig
    assert plt.get_fignums() == [fig.number]


def test_composite_plot_empty_input_shows_labelled_empty_axes():
    _, ax = plot_composite_curves([], 10)
    assert len(ax.lines) == 0
    assert any("No streams" in t for t in _texts(ax))
    assert "kW" in ax.get_xlabel() and "°C" in ax.get_ylabel()


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
def test_composite_plot_rejects_invalid_delta_t_min_without_creating_a_figure(bad_delta_t_min):
    with pytest.raises(ValueError):
        plot_composite_curves(_simple_streams(), bad_delta_t_min)
    with pytest.raises(ValueError):
        plot_composite_curves([], bad_delta_t_min)
    assert plt.get_fignums() == []


def test_composite_plot_warns_when_curves_violate_delta_t_min(monkeypatch):
    monkeypatch.setattr(
        composite_curve, "composite_curve_plot_data",
        lambda s, d: _fake_composite_data(minimum_approach=2.0),
    )
    with pytest.warns(RuntimeWarning, match="ΔTmin"):
        plot_composite_curves(_simple_streams(), 10)


def test_composite_plot_is_silent_when_approach_equals_delta_t_min(monkeypatch):
    monkeypatch.setattr(
        composite_curve, "composite_curve_plot_data",
        lambda s, d: _fake_composite_data(minimum_approach=10.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_composite_curves(_simple_streams(), 10)


def test_composite_plot_can_be_saved(tmp_path):
    fig, _ = plot_composite_curves(_four_streams(), 10)
    path = tmp_path / "composite.png"
    fig.savefig(path)
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Grand Composite Curve
# ---------------------------------------------------------------------------


def test_gcc_plot_returns_figure_and_axes():
    fig, ax = plot_grand_composite_curve(_simple_streams(), 10)
    assert isinstance(fig, Figure)
    assert ax in fig.axes


def test_gcc_plot_axes_labels_and_title():
    _, ax = plot_grand_composite_curve(_simple_streams(), 10)
    assert "Residual heat flow" in ax.get_xlabel() and "kW" in ax.get_xlabel()
    assert "Shifted temperature" in ax.get_ylabel() and "°C" in ax.get_ylabel()
    assert "Grand composite curve" in ax.get_title()
    assert "10" in ax.get_title()


@pytest.mark.parametrize("make_streams", [_simple_streams, _four_streams])
def test_gcc_plot_draws_the_plot_data(make_streams):
    streams = make_streams()
    data = grand_composite_curve_data(streams, 10)
    _, ax = plot_grand_composite_curve(streams, 10)
    line = _line(ax, "Grand composite curve")
    assert list(zip(line.get_xdata(), line.get_ydata())) == pytest.approx(data["points"])
    assert "Grand composite curve" in _legend_labels(ax)


def test_gcc_plot_has_a_zero_heat_flow_reference_line():
    _, ax = plot_grand_composite_curve(_four_streams(), 10)
    vertical = [l for l in ax.lines if list(l.get_xdata()) == [0.0, 0.0]]
    assert len(vertical) == 1
    low, high = ax.get_xlim()
    assert low < 0 < high


@pytest.mark.parametrize("make_streams", [_simple_streams, _four_streams])
def test_gcc_plot_marks_pinches_on_the_zero_line(make_streams):
    streams = make_streams()
    data = grand_composite_curve_data(streams, 10)
    _, ax = plot_grand_composite_curve(streams, 10)
    pinch = _line(ax, "Pinch")
    assert set(pinch.get_xdata()) == {0.0}
    assert list(pinch.get_ydata()) == data["pinch_shifted_temperatures"]


def test_gcc_plot_marks_multiple_pinches(monkeypatch):
    monkeypatch.setattr(cascade_plot, "grand_composite_curve_data", lambda s, d: _fake_gcc_data())
    _, ax = plot_grand_composite_curve(_simple_streams(), 10)
    pinch = _line(ax, "Pinch")
    assert list(pinch.get_ydata()) == [100.0, 60.0]


def test_gcc_plot_can_hide_pinch_markers():
    _, ax = plot_grand_composite_curve(_four_streams(), 10, show_pinch=False)
    assert _diamond_markers(ax) == []


def test_gcc_plot_labels_both_utilities():
    _, ax = plot_grand_composite_curve(_four_streams(), 10)
    text = "\n".join(_texts(ax))
    assert "Q_H,min" in text and "Q_C,min" in text


def test_gcc_plot_flags_threshold_problem():
    _, ax = plot_grand_composite_curve(_simple_streams(), 10)
    assert any("Threshold problem: no hot utility" in t for t in _texts(ax))


def test_gcc_plot_draws_into_a_given_axes():
    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_grand_composite_curve(_simple_streams(), 10, ax=ax)
    assert returned_ax is ax
    assert returned_fig is fig
    assert plt.get_fignums() == [fig.number]


def test_gcc_plot_empty_input_shows_labelled_empty_axes():
    _, ax = plot_grand_composite_curve([], 10)
    assert len(ax.lines) == 0
    assert any("No streams" in t for t in _texts(ax))
    assert "kW" in ax.get_xlabel() and "°C" in ax.get_ylabel()


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
def test_gcc_plot_rejects_invalid_delta_t_min_without_creating_a_figure(bad_delta_t_min):
    with pytest.raises(ValueError):
        plot_grand_composite_curve(_simple_streams(), bad_delta_t_min)
    with pytest.raises(ValueError):
        plot_grand_composite_curve([], bad_delta_t_min)
    assert plt.get_fignums() == []


def test_plot_cascade_is_an_alias_for_the_gcc_plot():
    assert plot_cascade is plot_grand_composite_curve


def test_gcc_plot_can_be_saved(tmp_path):
    fig, _ = plot_grand_composite_curve(_four_streams(), 10)
    path = tmp_path / "gcc.png"
    fig.savefig(path)
    assert path.stat().st_size > 0