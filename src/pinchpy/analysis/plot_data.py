"""Plot-ready data for the composite curves and the Grand Composite Curve.

This module only *re-arranges* results that Phase 3 already computes; it adds
no thermodynamic calculation of its own and it does not import Matplotlib.

Composite curves
    ``composite_curve_data`` returns the hot and cold curves as
    ``(cumulative enthalpy, temperature)`` points, both starting at H = 0.
    On a real T-H diagram the cold curve starts at the cold-utility load
    (the heat the hot streams must still reject to cooling water), so the
    cold curve is translated to the right by ``minimum_cold_utility``. After
    the translation the horizontal overlap of the two curves is the heat
    recovered between process streams, the hot curve's left overhang is the
    cold utility, and the cold curve's right overhang is the hot utility.

Grand Composite Curve (GCC)
    The feasible heat cascade from ``heat_cascade`` gives, for each shifted
    interval boundary, the residual heat flow R(T*) -- the net heat that must
    cross that shifted temperature. Plotting R against the shifted
    temperature T* gives the GCC. It touches R = 0 at every pinch, its top
    value is the hot utility and its bottom value is the cold utility. The
    temperature axis is the *shifted* scale used by Phase 2/3, unchanged.

Nothing here hard-codes how ΔTmin is applied to the streams: temperatures and
cascades are taken as Phase 2/3 produce them.
"""

import numpy as np

from .composite_curves import composite_curve_data
from .heat_cascade import heat_cascade
from .pinch_point import utility_targets

_TOLERANCE = 1e-9


def _translate_enthalpy(curve, offset):
    return [(enthalpy + offset, temperature) for enthalpy, temperature in curve]


def _enthalpy_at(curve, temperature):
    """Interpolate enthalpy at ``temperature``; None if outside the curve."""
    if len(curve) < 2:
        return None
    enthalpies = [h for h, _ in curve]
    temperatures = [t for _, t in curve]
    if temperature < temperatures[0] - _TOLERANCE or temperature > temperatures[-1] + _TOLERANCE:
        return None
    return float(np.interp(temperature, temperatures, enthalpies))


def minimum_approach(hot_curve, cold_curve):
    """Smallest vertical gap between the curves where they overlap in H.

    Returns None when either curve is empty or the curves do not overlap
    horizontally. Both curves are piecewise linear, so the minimum gap occurs
    at one of their breakpoints.
    """
    if len(hot_curve) < 2 or len(cold_curve) < 2:
        return None
    hot_h, hot_t = zip(*hot_curve)
    cold_h, cold_t = zip(*cold_curve)
    low = max(hot_h[0], cold_h[0])
    high = min(hot_h[-1], cold_h[-1])
    if high - low <= _TOLERANCE:
        return None
    grid = {low, high}
    grid.update(h for h in (*hot_h, *cold_h) if low <= h <= high)
    gaps = [
        float(np.interp(h, hot_h, hot_t) - np.interp(h, cold_h, cold_t))
        for h in sorted(grid)
    ]
    return min(gaps)


def composite_curve_plot_data(streams, delta_t_min):
    """Return both composite curves positioned for a T-H plot.

    Keys: ``hot_curve`` and ``cold_curve`` (lists of ``(H, T)``; empty when
    there are no streams of that type), ``minimum_hot_utility``,
    ``minimum_cold_utility``, ``heat_recovery``, ``pinch_shifted_temperatures``,
    ``pinch_points`` (one dict per pinch with the ``(H, T)`` marker on the hot
    and cold curve, or None when it lies outside that curve),
    ``minimum_approach``, ``is_threshold_problem`` and ``delta_t_min``.

    Raises ``ValueError`` for a non-positive ``delta_t_min`` (raised by the
    Phase 2/3 code, also for empty input).
    """
    curves = composite_curve_data(streams, delta_t_min)
    targets = utility_targets(streams, delta_t_min)

    has_hot = any(s.stream_type == "hot" for s in streams)
    has_cold = any(s.stream_type == "cold" for s in streams)
    hot_curve = list(curves["hot_curve"]) if has_hot else []
    cold_curve = list(curves["cold_curve"]) if has_cold else []

    cold_utility = targets["minimum_cold_utility"]
    cold_curve = _translate_enthalpy(cold_curve, cold_utility)

    hot_total = hot_curve[-1][0] if hot_curve else 0.0
    shifted_pinches = list(targets["pinch_shifted_temperatures"])

    pinch_points = []
    if shifted_pinches:
        # Offsets between shifted and real pinch temperatures are read from
        # the Phase 3 result rather than re-derived here.
        reference = targets["pinch_shifted_temperature"]
        hot_offset = targets["pinch_hot_temperature"] - reference
        cold_offset = targets["pinch_cold_temperature"] - reference
        for shifted in shifted_pinches:
            hot_t = shifted + hot_offset
            cold_t = shifted + cold_offset
            hot_h = _enthalpy_at(hot_curve, hot_t)
            cold_h = _enthalpy_at(cold_curve, cold_t)
            pinch_points.append({
                "shifted_temperature": shifted,
                "hot": None if hot_h is None else (hot_h, hot_t),
                "cold": None if cold_h is None else (cold_h, cold_t),
            })

    return {
        "delta_t_min": delta_t_min,
        "hot_curve": hot_curve,
        "cold_curve": cold_curve,
        "minimum_hot_utility": targets["minimum_hot_utility"],
        "minimum_cold_utility": cold_utility,
        "heat_recovery": hot_total - cold_utility,
        "pinch_shifted_temperatures": shifted_pinches,
        "pinch_points": pinch_points,
        "minimum_approach": minimum_approach(hot_curve, cold_curve),
        "is_threshold_problem": targets["is_threshold_problem"],
    }


def grand_composite_curve_data(streams, delta_t_min):
    """Return the Grand Composite Curve as residual heat vs shifted temperature.

    Keys: ``temperatures`` (shifted, highest first), ``heat_flow`` (residual
    heat flow at each temperature, same length), ``points`` (``(R, T*)``
    pairs ready to plot), ``pinch_shifted_temperatures``,
    ``minimum_hot_utility``, ``minimum_cold_utility``,
    ``is_threshold_problem`` and ``delta_t_min``. Empty input gives empty
    lists and zero utilities.

    Raises ``ValueError`` for a non-positive ``delta_t_min``.
    """
    intervals, cascade, hot_utility, cold_utility = heat_cascade(streams, delta_t_min)
    targets = utility_targets(streams, delta_t_min)

    if intervals:
        temperatures = [intervals[0][0]] + [low for _, low in intervals]
        heat_flow = list(cascade)
    else:
        temperatures = []
        heat_flow = []

    return {
        "delta_t_min": delta_t_min,
        "temperatures": temperatures,
        "heat_flow": heat_flow,
        "points": list(zip(heat_flow, temperatures)),
        "pinch_shifted_temperatures": list(targets["pinch_shifted_temperatures"]),
        "minimum_hot_utility": hot_utility,
        "minimum_cold_utility": cold_utility,
        "is_threshold_problem": targets["is_threshold_problem"],
    }