"""Grand Composite Curve: residual heat flow vs shifted temperature.

The numbers come from ``pinchpy.analysis.plot_data``; this module only draws.
The curve touches R = 0 at each pinch, its top value is the hot utility and
its bottom value is the cold utility. The y-axis is the *shifted* temperature
exactly as produced by the Phase 2/3 heat cascade.
"""

from ..analysis.plot_data import grand_composite_curve_data
from ._common import (
    GCC_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    note_empty,
    prepare_axes,
    style_axes,
    threshold_note,
)


def _label_end(ax, text, heat_flow, temperature, span, above):
    """Label an end of the curve on whichever side leaves room inside the axes."""
    to_right = heat_flow <= span / 2
    ax.annotate(text, xy=(heat_flow, temperature),
                xytext=(10 if to_right else -10, 6 if above else -8),
                textcoords="offset points",
                ha="left" if to_right else "right",
                va="bottom" if above else "top",
                fontsize=9, color=TEXT_SECONDARY)


def plot_grand_composite_curve(streams, delta_t_min, ax=None, show_pinch=True):
    """Plot the Grand Composite Curve; return ``(fig, ax)``.

    Pass an existing ``ax`` to draw into it. Empty input gives labelled axes
    with a "No streams to plot" note. A non-positive ``delta_t_min`` raises
    ``ValueError`` before any figure is created.
    """
    data = grand_composite_curve_data(streams, delta_t_min)
    fig, ax = prepare_axes(ax)
    style_axes(ax)

    ax.set_xlabel("Residual heat flow, R (kW)")
    ax.set_ylabel("Shifted temperature, T* (°C)")
    ax.set_title(f"Grand composite curve (ΔTmin = {delta_t_min:g} °C)", color=TEXT_PRIMARY)

    if not data["points"]:
        note_empty(ax)
        return fig, ax

    heat_flow, temperature = zip(*data["points"])
    ax.plot(heat_flow, temperature, color=GCC_COLOR, linewidth=2, marker="o",
            markersize=6, label="Grand composite curve")
    ax.axvline(0.0, color=TEXT_SECONDARY, linewidth=1, zorder=1)

    if show_pinch and data["pinch_shifted_temperatures"]:
        ax.plot([0.0] * len(data["pinch_shifted_temperatures"]),
                data["pinch_shifted_temperatures"], linestyle="none", marker="D",
                markersize=9, markerfacecolor="white", markeredgecolor=TEXT_PRIMARY,
                markeredgewidth=1.6, zorder=5, label="Pinch")

    span = max(heat_flow)
    _label_end(ax, f"Hot utility, Q_H,min = {data['minimum_hot_utility']:.1f} kW",
               heat_flow[0], temperature[0], span, above=True)
    _label_end(ax, f"Cold utility, Q_C,min = {data['minimum_cold_utility']:.1f} kW",
               heat_flow[-1], temperature[-1], span, above=False)
    if data["is_threshold_problem"]:
        ax.text(0.98, 0.84, threshold_note(data["minimum_hot_utility"],
                                           data["minimum_cold_utility"]),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=TEXT_SECONDARY)

    pad = 0.05 * span if span > 0 else 1.0
    ax.set_xlim(-pad, span + 0.35 * span if span > 0 else 1.0 + pad)
    ax.margins(y=0.10)
    ax.legend(loc="upper right", frameon=False)
    return fig, ax


# Name of the earlier stub, kept so existing imports keep working.
plot_cascade = plot_grand_composite_curve