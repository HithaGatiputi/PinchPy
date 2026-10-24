"""Hot and cold composite curves on temperature vs cumulative-enthalpy axes.

The numbers come from ``pinchpy.analysis.plot_data``; this module only draws.
Reading the plot: the horizontal overlap of the two curves is the heat
recovered between process streams, the hot curve's overhang at low
temperature is the cold utility, the cold curve's overhang at high
temperature is the hot utility, and the closest approach marks the pinch.
"""

import warnings

from ..analysis.plot_data import composite_curve_plot_data
from ._common import (
    COLD_COLOR,
    HOT_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    note_empty,
    prepare_axes,
    style_axes,
    threshold_note,
)

_APPROACH_TOLERANCE = 1e-6


def plot_composite_curves(streams, delta_t_min, ax=None, show_pinch=True):
    """Plot the hot and cold composite curves; return ``(fig, ax)``.

    Pass an existing ``ax`` to draw into it. Empty input gives labelled axes
    with a "No streams to plot" note. A non-positive ``delta_t_min`` raises
    ``ValueError`` before any figure is created. A ``RuntimeWarning`` is
    issued when the plotted curves approach closer than ``delta_t_min``,
    which means the utility targets are inconsistent with ΔTmin.
    """
    data = composite_curve_plot_data(streams, delta_t_min)
    fig, ax = prepare_axes(ax)
    style_axes(ax)

    ax.set_xlabel("Heat flow, H (kW)")
    ax.set_ylabel("Temperature, T (°C)")
    ax.set_title(f"Composite curves (ΔTmin = {delta_t_min:g} °C)", color=TEXT_PRIMARY)

    hot_curve, cold_curve = data["hot_curve"], data["cold_curve"]
    if not hot_curve and not cold_curve:
        note_empty(ax)
        return fig, ax

    if hot_curve:
        enthalpy, temperature = zip(*hot_curve)
        ax.plot(enthalpy, temperature, color=HOT_COLOR, linewidth=2, marker="o",
                markersize=6, label="Hot composite curve")
    if cold_curve:
        enthalpy, temperature = zip(*cold_curve)
        ax.plot(enthalpy, temperature, color=COLD_COLOR, linewidth=2, marker="s",
                markersize=6, label="Cold composite curve")

    if show_pinch:
        labelled = False
        for pinch in data["pinch_points"]:
            for key in ("hot", "cold"):
                point = pinch[key]
                if point is None:
                    continue
                ax.plot([point[0]], [point[1]], linestyle="none", marker="D",
                        markersize=9, markerfacecolor="white",
                        markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.6,
                        zorder=5, label=None if labelled else "Pinch")
                labelled = True

    lines = [
        f"Hot utility, Q_H,min = {data['minimum_hot_utility']:.1f} kW",
        f"Cold utility, Q_C,min = {data['minimum_cold_utility']:.1f} kW",
        f"Heat recovery = {data['heat_recovery']:.1f} kW",
    ]
    if data["is_threshold_problem"]:
        lines.append(threshold_note(data["minimum_hot_utility"], data["minimum_cold_utility"]))
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=TEXT_SECONDARY,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "#d8d7d2"})

    ax.margins(x=0.04, y=0.06)
    ax.legend(loc="lower right", frameon=False)

    approach = data["minimum_approach"]
    if approach is not None and approach < delta_t_min - _APPROACH_TOLERANCE:
        warnings.warn(
            f"Composite curves approach each other by {approach:.2f} °C, less than "
            f"ΔTmin = {delta_t_min:g} °C; the curves are not a feasible ΔTmin design. "
            "Check the shifted-temperature convention behind the utility targets.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fig, ax