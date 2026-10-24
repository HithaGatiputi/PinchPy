"""Pinch-point identification and utility targeting.

The pinch is the shifted temperature at which the feasible heat cascade
touches zero -- the point of closest approach between the hot and cold
composite curves, where heat transfer is most constrained. A problem can
have more than one shifted temperature at which the cascade is zero
(multiple pinches); a "threshold problem" is one where the cascade is
already at its minimum at the very top (no hot utility is required) or at
the very bottom (no cold utility is required), so the "pinch" sits at the
extreme end of the temperature range rather than at an interior point.
"""

_TOLERANCE = 1e-9


def pinch_temperature(intervals, cascade):
    """Return the shifted temperature of the first (coldest-indexed) pinch."""
    points = pinch_temperatures(intervals, cascade)
    return points[0]


def pinch_temperatures(intervals, cascade):
    """Return every shifted temperature at which the cascade reaches zero."""
    minimum = min(cascade)
    points = []
    for index, value in enumerate(cascade):
        if abs(value - minimum) <= _TOLERANCE:
            if index >= len(intervals):
                points.append(intervals[-1][1])
            else:
                points.append(intervals[index][0])
    return points


def utility_targets(streams, delta_t_min):
    from .heat_cascade import heat_cascade

    intervals, cascade, hot_utility, cold_utility = heat_cascade(streams, delta_t_min)

    if not intervals:
        return {
            "minimum_hot_utility": hot_utility,
            "minimum_cold_utility": cold_utility,
            "pinch_shifted_temperature": None,
            "pinch_hot_temperature": None,
            "pinch_cold_temperature": None,
            "pinch_shifted_temperatures": [],
            "is_threshold_problem": False,
        }

    pinch_points = pinch_temperatures(intervals, cascade)
    pinch_shifted = pinch_points[0]
    is_threshold_problem = hot_utility <= _TOLERANCE or cold_utility <= _TOLERANCE

    return {
        "minimum_hot_utility": hot_utility,
        "minimum_cold_utility": cold_utility,
        "pinch_shifted_temperature": pinch_shifted,
        "pinch_hot_temperature": pinch_shifted + delta_t_min / 2,
        "pinch_cold_temperature": pinch_shifted - delta_t_min / 2,
        "pinch_shifted_temperatures": pinch_points,
        "is_threshold_problem": is_threshold_problem,
    }
