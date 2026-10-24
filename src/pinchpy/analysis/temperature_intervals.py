def shifted_temperatures(streams, delta_t_min):
    hot_shift = delta_t_min / 2
    cold_shift = -delta_t_min / 2
    temperatures = []
    for s in streams:
        shift = hot_shift if s.stream_type == "hot" else cold_shift
        temperatures.extend([s.supply_temperature + shift, s.target_temperature + shift])
    return sorted(set(temperatures), reverse=True)

def build_intervals(streams, delta_t_min):
    temps = shifted_temperatures(streams, delta_t_min)
    return list(zip(temps[:-1], temps[1:]))
