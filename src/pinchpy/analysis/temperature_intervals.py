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


def stream_shift(stream_type, delta_t_min):
    """Return the shifted-temperature-scale offset for a given stream type.

    Hot streams are shifted down and cold streams are shifted up by half of
    ``delta_t_min``, which is the single source of truth for this sign used
    everywhere in the analysis package. Changing the convention only
    requires editing this function.
    """
    return -delta_t_min / 2 if stream_type == "hot" else delta_t_min / 2


def shifted_endpoints(stream, delta_t_min):
    """Return a stream's (supply, target) temperatures on the shifted scale."""
    shift = stream_shift(stream.type, delta_t_min)
    return stream.supply_temperature + shift, stream.target_temperature + shift


def shifted_temperatures(streams, delta_t_min):
    _validate_delta_t_min(delta_t_min)
    temperatures = []
    for s in streams:
        ts, tt = shifted_endpoints(s, delta_t_min)
        temperatures.extend([ts, tt])
    return sorted(set(temperatures), reverse=True)


def build_intervals(streams, delta_t_min):
    temps = shifted_temperatures(streams, delta_t_min)
    return list(zip(temps[:-1], temps[1:]))
