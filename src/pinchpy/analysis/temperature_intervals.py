"""Temperature shifting and interval construction for Pinch Analysis.

Hot streams are shifted down and cold streams are shifted up by half of
``delta_t_min``, placing every stream on a common shifted-temperature scale
on which a ``delta_t_min`` feasible heat transfer corresponds to equal
shifted temperatures. The shifted stream endpoints are merged into a sorted,
deduplicated list of breakpoints that define the temperature intervals used
by the heat cascade.
"""


def _validate_delta_t_min(delta_t_min):
    if delta_t_min <= 0:
        raise ValueError("delta_t_min must be greater than zero.")


def shifted_temperatures(streams, delta_t_min):
    _validate_delta_t_min(delta_t_min)
    hot_shift = delta_t_min / 2
    cold_shift = -delta_t_min / 2
    temperatures = []
    for s in streams:
        shift = hot_shift if s.type == "hot" else cold_shift
        temperatures.extend([s.supply_temperature + shift, s.target_temperature + shift])
    return sorted(set(temperatures), reverse=True)


def build_intervals(streams, delta_t_min):
    temps = shifted_temperatures(streams, delta_t_min)
    return list(zip(temps[:-1], temps[1:]))
