def _segments(streams, delta_t_min):
    from .temperature_intervals import shifted_temperatures

    temps = shifted_temperatures(streams, delta_t_min)
    segments = []
    for high_t, low_t in zip(temps[:-1], temps[1:]):
        d_t = high_t - low_t
        net_cph = 0.0
        for s in streams:
            shift = delta_t_min / 2 if s.stream_type == "hot" else -delta_t_min / 2
            ts = s.supply_temperature + shift
            tt = s.target_temperature + shift
            upper, lower = max(ts, tt), min(ts, tt)
            if upper >= high_t and lower <= low_t:
                net_cph += s.heat_capacity_flow
        segments.append((high_t, low_t, net_cph))
    return segments


def _curve(streams, stream_type, delta_t_min):
    segments = _segments([s for s in streams if s.stream_type == stream_type], delta_t_min)
    shift = delta_t_min / 2 if stream_type == "hot" else -delta_t_min / 2

    enthalpy = 0.0
    points = [(0.0, segments[-1][1] - shift if segments else 0.0)]
    for high_t, low_t, net_cph in reversed(segments):
        enthalpy += net_cph * (high_t - low_t)
        points.append((enthalpy, high_t - shift))
    return points


def composite_curve_data(streams, delta_t_min):
    hot_streams = [s for s in streams if s.stream_type == "hot"]
    cold_streams = [s for s in streams if s.stream_type == "cold"]
    return {
        "hot_curve": _curve(hot_streams, "hot", delta_t_min),
        "cold_curve": _curve(cold_streams, "cold", delta_t_min),
    }
