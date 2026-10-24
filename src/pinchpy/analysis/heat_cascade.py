"""Heat cascade construction for Pinch Analysis.

Within each shifted-temperature interval, every stream that spans the full
interval contributes its heat-capacity flow rate (positive for hot streams,
negative for cold streams). Multiplying the net heat-capacity flow rate by
the interval width gives the heat surplus/deficit of that interval. Running
a cumulative sum of these values from the highest temperature downward
gives the "problem table cascade". Shifting that cascade up by the amount
needed to make its lowest point exactly zero gives the feasible cascade,
whose required shift is the minimum hot utility and whose final value is
the minimum cold utility.
"""


def interval_heat_balance(streams, high_t, low_t, delta_t_min):
    d_t = high_t - low_t
    net_cph = 0.0
    for s in streams:
        shift = delta_t_min / 2 if s.type == "hot" else -delta_t_min / 2
        ts = s.supply_temperature + shift
        tt = s.target_temperature + shift
        upper, lower = max(ts, tt), min(ts, tt)
        if upper >= high_t and lower <= low_t:
            net_cph += s.cp if s.type == "hot" else -s.cp
    return net_cph * d_t


def heat_cascade(streams, delta_t_min):
    from .temperature_intervals import build_intervals

    intervals = build_intervals(streams, delta_t_min)
    cascade = [0.0]
    for high_t, low_t in intervals:
        cascade.append(cascade[-1] + interval_heat_balance(streams, high_t, low_t, delta_t_min))
    minimum = min(cascade)
    hot_utility = max(0.0, -minimum)
    shifted_cascade = [x + hot_utility for x in cascade]
    cold_utility = shifted_cascade[-1]
    return intervals, shifted_cascade, hot_utility, cold_utility
