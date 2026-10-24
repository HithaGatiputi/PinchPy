def pinch_temperature(intervals, cascade):
    min_value = min(cascade)
    index = cascade.index(min_value)
    if index >= len(intervals):
        return intervals[-1][1]
    return intervals[index][0]

def utility_targets(streams, delta_t_min):
    from .heat_cascade import heat_cascade
    intervals, cascade, hot_utility, cold_utility = heat_cascade(streams, delta_t_min)
    pinch_shifted = pinch_temperature(intervals, cascade)
    pinch_hot = pinch_shifted + delta_t_min / 2
    pinch_cold = pinch_shifted - delta_t_min / 2
    return {
        "minimum_hot_utility": hot_utility,
        "minimum_cold_utility": cold_utility,
        "pinch_shifted_temperature": pinch_shifted,
        "pinch_hot_temperature": pinch_hot,
        "pinch_cold_temperature": pinch_cold,
    }
