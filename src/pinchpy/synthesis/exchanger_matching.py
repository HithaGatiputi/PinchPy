"""Match feasibility rules for pinch-based HEN synthesis.

A match is thermally feasible when the hot stream's supply is hot enough to
reach at least ``delta_t_min`` above the cold stream's target. Within a
pinch sub-problem there is a second, stricter rule at the pinch itself: to
avoid creating a smaller pinch inside the match, the hot stream's CP must be
less than or equal to the cold stream's CP above the pinch, and greater than
or equal to it below the pinch. ``region`` selects which rule applies;
``None`` skips the CP check (useful away from the pinch, or for callers that
enforce it themselves).
"""


def feasible_pair(hot_stream, cold_stream, delta_t_min):
    return hot_stream.supply_temperature > cold_stream.target_temperature + delta_t_min


def cp_rule_satisfied(hot_stream, cold_stream, region):
    if region == "above":
        return hot_stream.cp <= cold_stream.cp
    if region == "below":
        return hot_stream.cp >= cold_stream.cp
    return True


def feasible_match(hot_stream, cold_stream, delta_t_min, region=None):
    return feasible_pair(hot_stream, cold_stream, delta_t_min) and cp_rule_satisfied(
        hot_stream, cold_stream, region
    )
