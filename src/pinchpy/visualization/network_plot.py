"""Grid diagram for a synthesized heat exchanger network (Phase 6).

Each stream is drawn as a horizontal line on a shared (real) temperature
axis -- hot streams above, cold streams below -- running from its target to
its supply temperature. Each process match is a pair of circles, one on the
hot line and one on the cold line at that match's actual terminal
temperatures, joined by a line and labelled with its duty. Utility
exchangers (heaters/coolers) are drawn as a single circle on the process
stream's line, labelled with the utility duty.
"""

from ._common import (
    COLD_COLOR,
    HOT_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    note_empty,
    prepare_axes,
    style_axes,
)

_UTILITY_COLOR = "#8a8a86"


def plot_network(streams, network, ax=None):
    """Plot a grid diagram of ``network`` over ``streams``; return ``(fig, ax)``."""
    fig, ax = prepare_axes(ax, figsize=(9, 5.5))
    style_axes(ax)
    ax.set_xlabel("Temperature, T (°C)")
    ax.set_yticks([])
    ax.set_title("Heat exchanger network grid diagram", color=TEXT_PRIMARY)

    if not streams:
        note_empty(ax)
        return fig, ax

    hot_streams = [s for s in streams if s.type == "hot"]
    cold_streams = [s for s in streams if s.type == "cold"]

    y_positions = {}
    for i, s in enumerate(hot_streams):
        y_positions[s.id] = len(hot_streams) - i
    for i, s in enumerate(cold_streams):
        y_positions[s.id] = -(i + 1)

    for s in hot_streams + cold_streams:
        y = y_positions[s.id]
        color = HOT_COLOR if s.type == "hot" else COLD_COLOR
        lo, hi = min(s.supply_temperature, s.target_temperature), max(s.supply_temperature, s.target_temperature)
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=2)
        ax.text(hi + 1, y, s.id, color=TEXT_PRIMARY, va="center", fontsize=9, fontweight="bold")
        arrow_x = s.supply_temperature + (0.5 if s.type == "cold" else -0.5)
        ax.annotate("", xy=(arrow_x, y), xytext=(s.supply_temperature, y),
                    arrowprops={"arrowstyle": "-|>", "color": color})

    base_id_to_original = {}
    for s in streams:
        base_id_to_original[s.id] = s.id
        base_id_to_original[f"{s.id}_above"] = s.id
        base_id_to_original[f"{s.id}_below"] = s.id

    for exchanger in network.exchangers:
        if exchanger.is_utility:
            is_heater = exchanger.hot_stream == "HU"
            stream_id = exchanger.cold_stream if is_heater else exchanger.hot_stream
            original_id = base_id_to_original.get(stream_id, stream_id)
            if original_id not in y_positions:
                continue
            y = y_positions[original_id]
            x = (exchanger.cold_outlet if is_heater else exchanger.hot_outlet) or 0.0
            ax.plot([x], [y], marker="o", markersize=12, markerfacecolor=_UTILITY_COLOR,
                    markeredgecolor=TEXT_PRIMARY, zorder=4)
            label = "HU" if is_heater else "CU"
            ax.text(x, y + (0.25 if is_heater else -0.35), f"{label} {exchanger.duty:.0f}",
                    ha="center", color=TEXT_SECONDARY, fontsize=8)
            continue

        hot_original = base_id_to_original.get(exchanger.hot_stream, exchanger.hot_stream)
        cold_original = base_id_to_original.get(exchanger.cold_stream, exchanger.cold_stream)
        if hot_original not in y_positions or cold_original not in y_positions:
            continue
        hot_y = y_positions[hot_original]
        cold_y = y_positions[cold_original]
        hot_x = (exchanger.hot_inlet + exchanger.hot_outlet) / 2
        cold_x = (exchanger.cold_inlet + exchanger.cold_outlet) / 2

        ax.plot([hot_x, cold_x], [hot_y, cold_y], color=TEXT_SECONDARY, linewidth=1.4, zorder=3)
        for x, y in ((hot_x, hot_y), (cold_x, cold_y)):
            ax.plot([x], [y], marker="o", markersize=11, markerfacecolor="white",
                     markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.5, zorder=4)
        mid_x, mid_y = (hot_x + cold_x) / 2, (hot_y + cold_y) / 2
        ax.text(mid_x, mid_y, exchanger.name, ha="center", va="center", fontsize=7,
                color=TEXT_PRIMARY)
        ax.text(mid_x, mid_y - 0.3, f"{exchanger.duty:.0f} kW", ha="center", va="top",
                fontsize=7, color=TEXT_SECONDARY)

    ax.margins(x=0.12, y=0.25)
    return fig, ax
