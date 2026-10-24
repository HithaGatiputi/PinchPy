"""Shared Matplotlib helpers for the PinchPy plots (styling only, no physics)."""

import matplotlib.pyplot as plt

# Validated categorical colours (colour-blind safe pair); hot = red, cold = blue.
HOT_COLOR = "#e34948"
COLD_COLOR = "#2a78d6"
GCC_COLOR = "#6250d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"

_UTILITY_ZERO = 1e-9


def prepare_axes(ax=None, figsize=(7.5, 5.5)):
    """Return ``(fig, ax)``, creating a new figure only when ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    return fig, ax


def style_axes(ax):
    """Light grid, no top/right frame, readable tick labels."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)


def note_empty(ax, message="No streams to plot"):
    ax.text(
        0.5, 0.5, message, transform=ax.transAxes,
        ha="center", va="center", color=TEXT_SECONDARY, fontsize=11,
    )


def threshold_note(minimum_hot_utility, minimum_cold_utility):
    """Text for a threshold problem (one utility target is zero)."""
    if minimum_hot_utility <= _UTILITY_ZERO:
        return "Threshold problem: no hot utility required"
    if minimum_cold_utility <= _UTILITY_ZERO:
        return "Threshold problem: no cold utility required"
    return "Threshold problem"