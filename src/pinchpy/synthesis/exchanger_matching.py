def feasible_pair(hot_stream, cold_stream, delta_t_min):
    return hot_stream.supply_temperature > cold_stream.target_temperature + delta_t_min
