"""Minimum heat-transfer area target from the composite curves (Phase 5).

A heat exchanger is sized with ``Q = U * A * LMTD``. The composite curves give
the hot and cold temperature at every heat-flow value, so the T-H diagram can
be cut into vertical *enthalpy strips* inside which both curves are straight
lines. Inside one strip the hot and cold sides behave like a single
counter-current exchanger whose four terminal temperatures are known, so its
area is ``Q_strip / (U * LMTD_strip)``. Adding the strips gives the minimum
area target for the chosen ΔTmin (vertical heat transfer, one uniform overall
coefficient ``U`` in kW/(m²·°C)).

Nothing here recomputes utilities, pinches or curves: the curves come from
``composite_curve_plot_data`` (Phase 3/4), the log-mean temperature difference
from ``calculations.lmtd`` and the area from ``calculations.area``.

Strip temperatures use the standard counter-current reading of the diagram:
at the low-enthalpy end of a strip the hot stream leaves (``hot_out``) and the
cold stream enters (``cold_in``); at the high-enthalpy end the hot stream
enters (``hot_in``) and the cold stream leaves (``cold_out``).
"""

import math
import numbers
import warnings
from bisect import bisect_right

from ..calculations.area import required_area
from ..calculations.lmtd import lmtd
from .plot_data import composite_curve_plot_data

_TOLERANCE = 1e-9
_APPROACH_TOLERANCE = 1e-6


def _validate_coefficient(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            "overall_heat_transfer_coefficient must be a real number (kW/m²/°C)."
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "overall_heat_transfer_coefficient must be finite and greater than zero."
        )
    return float(value)


def _strip_breakpoints(hot_curve, cold_curve, low, high):
    """Enthalpies where either curve bends, inside the overlap [low, high]."""
    points = [low]
    candidates = sorted({h for h, _ in hot_curve} | {h for h, _ in cold_curve})
    for enthalpy in candidates:
        if low + _TOLERANCE < enthalpy < high - _TOLERANCE and enthalpy - points[-1] > _TOLERANCE:
            points.append(enthalpy)
    points.append(high)
    return points


def _temperatures_in_strip(curve, h_start, h_end):
    """Temperatures at both ends of a strip, read from the straight piece of
    ``curve`` that contains it (so vertical pieces of a curve are never used)."""
    enthalpies = [h for h, _ in curve]
    middle = (h_start + h_end) / 2
    index = min(max(bisect_right(enthalpies, middle) - 1, 0), len(curve) - 2)
    (h0, t0), (h1, t1) = curve[index], curve[index + 1]
    slope = (t1 - t0) / (h1 - h0)
    return t0 + slope * (h_start - h0), t0 + slope * (h_end - h0)


def enthalpy_strips(hot_curve, cold_curve):
    """Cut the overlap of two composite curves into straight-line strips.

    Both curves are lists of ``(H, T)`` points as returned by
    ``composite_curve_plot_data``. Returns one dict per strip, lowest enthalpy
    first, with ``h_start``, ``h_end``, ``duty``, ``hot_in``, ``hot_out``,
    ``cold_in``, ``cold_out``, ``delta_t_cold_end`` (``hot_out - cold_in``) and
    ``delta_t_hot_end`` (``hot_in - cold_out``). Returns ``[]`` when either
    curve is empty or the curves do not overlap in enthalpy.
    """
    if len(hot_curve) < 2 or len(cold_curve) < 2:
        return []
    low = max(hot_curve[0][0], cold_curve[0][0])
    high = min(hot_curve[-1][0], cold_curve[-1][0])
    if high - low <= _TOLERANCE:
        return []

    points = _strip_breakpoints(hot_curve, cold_curve, low, high)
    strips = []
    for start, end in zip(points, points[1:]):
        hot_out, hot_in = _temperatures_in_strip(hot_curve, start, end)
        cold_in, cold_out = _temperatures_in_strip(cold_curve, start, end)
        strips.append({
            "h_start": start,
            "h_end": end,
            "duty": end - start,
            "hot_in": hot_in,
            "hot_out": hot_out,
            "cold_in": cold_in,
            "cold_out": cold_out,
            "delta_t_cold_end": hot_out - cold_in,
            "delta_t_hot_end": hot_in - cold_out,
        })
    return strips


def area_target(streams, delta_t_min, overall_heat_transfer_coefficient):
    """Minimum heat-transfer area target for the given streams and ΔTmin.

    ``overall_heat_transfer_coefficient`` is U in kW/(m²·°C), the same heat
    units as the stream CP values. Returns a dict with ``total_area`` (m²),
    ``strips`` (each strip from ``enthalpy_strips`` plus its ``lmtd`` and
    ``area``), ``heat_recovery`` (kW, the duty of all strips),
    ``minimum_approach`` (smallest end gap, ``None`` when there are no strips),
    ``is_threshold_problem``, ``delta_t_min`` and
    ``overall_heat_transfer_coefficient``.

    Empty input, or only hot or only cold streams, gives zero area and no
    strips. Raises ``ValueError`` for a non-positive ``delta_t_min`` or ``U``
    (``TypeError`` if ``U`` is not a number), and ``ValueError`` if the
    composite curves cross or touch, because the area is then undefined. A
    ``RuntimeWarning`` is issued if the curves come closer than ``delta_t_min``.
    """
    coefficient = _validate_coefficient(overall_heat_transfer_coefficient)
    data = composite_curve_plot_data(streams, delta_t_min)
    strips = enthalpy_strips(data["hot_curve"], data["cold_curve"])

    sized = []
    for strip in strips:
        smallest = min(strip["delta_t_cold_end"], strip["delta_t_hot_end"])
        if smallest <= _TOLERANCE:
            raise ValueError(
                "The composite curves cross or touch between "
                f"H = {strip['h_start']:.2f} and {strip['h_end']:.2f} kW "
                f"(temperature gap {smallest:.2f} °C), so the area target is undefined. "
                "The utility targets are inconsistent with ΔTmin; check the "
                "shifted-temperature convention in the Phase 2/3 code."
            )
        mean_difference = lmtd(strip["delta_t_cold_end"], strip["delta_t_hot_end"])
        sized.append({
            **strip,
            "lmtd": mean_difference,
            "area": required_area(strip["duty"], coefficient, mean_difference),
        })

    approach = min(
        (min(s["delta_t_cold_end"], s["delta_t_hot_end"]) for s in sized),
        default=None,
    )
    if approach is not None and approach < delta_t_min - _APPROACH_TOLERANCE:
        warnings.warn(
            f"Composite curves approach each other by {approach:.2f} °C, less than "
            f"ΔTmin = {delta_t_min:g} °C; the area target is not for a feasible ΔTmin "
            "design. Check the shifted-temperature convention behind the utility targets.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "delta_t_min": delta_t_min,
        "overall_heat_transfer_coefficient": coefficient,
        "strips": sized,
        "total_area": sum(s["area"] for s in sized),
        "heat_recovery": sum(s["duty"] for s in sized),
        "minimum_approach": approach,
        "is_threshold_problem": data["is_threshold_problem"],
    }