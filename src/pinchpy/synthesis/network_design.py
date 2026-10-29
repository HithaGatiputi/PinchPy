"""Heat exchanger network synthesis by the pinch design method (Phase 6).

The problem is split at the pinch (``pinch_split.split_streams_at_pinch``)
into an above-pinch and a below-pinch sub-problem, each solved independently
so that no match is ever asked to carry heat across the pinch. Within each
sub-problem, feasible process-to-process matches (``exchanger_matching``) are
made greedily, always transferring as much duty as possible ("tick-off") to
keep the number of units low, until no further process match is feasible.
Any duty left on a stream after that is met by a utility exchanger -- a
heater above the pinch, a cooler below it.

Each match is sized with the same LMTD/area formulas used in Phase 5
(``calculations.lmtd``, ``calculations.area``), computed from the actual
terminal temperatures of that match rather than re-using the composite
curves, since here the duty handed to any one match is not necessarily the
stream's full duty.
"""

import warnings

from ..calculations.area import required_area
from ..calculations.lmtd import lmtd
from ..models.exchanger import HeatExchanger
from ..models.network import HeatExchangerNetwork
from ..analysis.pinch_point import utility_targets
from .exchanger_matching import feasible_match
from .pinch_split import split_streams_at_pinch

_TOLERANCE = 1e-9


class _StreamState:
    """Mutable running state of one stream while it is matched up."""

    def __init__(self, stream):
        self.stream = stream
        self.remaining = stream.duty()
        self.current_temperature = stream.supply_temperature

    @property
    def done(self):
        return self.remaining <= _TOLERANCE


def _size_match(hot_state, cold_state, duty, U, delta_t_min, name, region):
    hot_outlet = hot_state.current_temperature - duty / hot_state.stream.cp
    cold_outlet = cold_state.current_temperature + duty / cold_state.stream.cp

    delta_t_hot_end = hot_state.current_temperature - cold_outlet
    delta_t_cold_end = hot_outlet - cold_state.current_temperature
    if min(delta_t_hot_end, delta_t_cold_end) < delta_t_min - 1e-6:
        return None

    mean_difference = lmtd(delta_t_hot_end, delta_t_cold_end)
    area = required_area(duty, U, mean_difference) if U else None

    return HeatExchanger(
        name=name,
        hot_stream=hot_state.stream.id,
        cold_stream=cold_state.stream.id,
        duty=duty,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=U,
        lmtd=mean_difference,
        area=area,
        hot_inlet=hot_state.current_temperature,
        hot_outlet=hot_outlet,
        cold_inlet=cold_state.current_temperature,
        cold_outlet=cold_outlet,
        is_utility=False,
    ), hot_outlet, cold_outlet


def _match_region(streams, delta_t_min, U, region, counter):
    hot_states = [_StreamState(s) for s in streams if s.type == "hot"]
    cold_states = [_StreamState(s) for s in streams if s.type == "cold"]
    exchangers = []

    while True:
        best = None
        for hot_state in hot_states:
            if hot_state.done:
                continue
            for cold_state in cold_states:
                if cold_state.done:
                    continue
                if not feasible_match(hot_state.stream, cold_state.stream, delta_t_min, region):
                    continue
                duty = min(hot_state.remaining, cold_state.remaining)
                result = _size_match(
                    hot_state, cold_state, duty, U, delta_t_min,
                    name=f"E{counter[0] + len(exchangers) + 1}", region=region,
                )
                if result is None:
                    continue
                exchanger, hot_outlet, cold_outlet = result
                if best is None or duty > best[0]:
                    best = (duty, hot_state, cold_state, exchanger, hot_outlet, cold_outlet)

        if best is None:
            break
        duty, hot_state, cold_state, exchanger, hot_outlet, cold_outlet = best
        exchangers.append(exchanger)
        hot_state.remaining -= duty
        cold_state.remaining -= duty
        hot_state.current_temperature = hot_outlet
        cold_state.current_temperature = cold_outlet

    counter[0] += len(exchangers)
    return exchangers, hot_states, cold_states


def _utility_exchangers(states, delta_t_min, U, utility_name, utility_type, counter, warn_label=None):
    exchangers = []
    for state in states:
        if state.done:
            continue
        duty = state.remaining
        if warn_label is not None:
            warnings.warn(
                f"'{state.stream.id}' could not be fully matched by process streams "
                f"{warn_label}; closing the remaining {duty:.2f} kW with a "
                f"{'cooler' if utility_type == 'cooler' else 'heater'} there instead. "
                "This means the design exceeds the Phase 3 minimum utility targets; "
                "a better match order or stream splitting could reduce it.",
                RuntimeWarning,
                stacklevel=2,
            )
        # No explicit utility-stream temperature is modelled here, so the
        # driving force is approximated by the process stream's own
        # remaining temperature change, floored at delta_t_min.
        if utility_type == "cooler":
            mean_difference = max(state.current_temperature - state.stream.target_temperature, delta_t_min)
        else:
            mean_difference = max(state.stream.target_temperature - state.current_temperature, delta_t_min)

        area = required_area(duty, U, mean_difference) if U else None
        counter[0] += 1
        exchangers.append(HeatExchanger(
            name=f"{utility_name}{counter[0]}",
            hot_stream=state.stream.id if utility_type == "cooler" else utility_name,
            cold_stream=utility_name if utility_type == "cooler" else state.stream.id,
            duty=duty,
            delta_t_min=delta_t_min,
            overall_heat_transfer_coefficient=U,
            lmtd=mean_difference,
            area=area,
            is_utility=True,
        ))
        state.remaining = 0.0
    return exchangers


def design_network(streams, delta_t_min, overall_heat_transfer_coefficient=None):
    """Synthesize a heat exchanger network meeting the Phase 3 utility targets.

    Returns a ``HeatExchangerNetwork`` whose exchangers are the process
    matches (``is_utility=False``) found above and below the pinch, plus the
    heaters/coolers (``is_utility=True``) needed to close any remaining duty.
    ``overall_heat_transfer_coefficient`` (U, kW/(m^2*C)) is optional; when
    given, each exchanger's ``area`` is also computed.
    """
    targets = utility_targets(streams, delta_t_min)
    if targets["pinch_hot_temperature"] is None:
        return HeatExchangerNetwork(exchangers=[])

    above, below = split_streams_at_pinch(
        streams, targets["pinch_hot_temperature"], targets["pinch_cold_temperature"]
    )

    counter = [0]
    above_exchangers, above_hot, above_cold = _match_region(
        above, delta_t_min, overall_heat_transfer_coefficient, "above", counter
    )
    below_exchangers, below_hot, below_cold = _match_region(
        below, delta_t_min, overall_heat_transfer_coefficient, "below", counter
    )

    heaters = _utility_exchangers(
        above_cold, delta_t_min, overall_heat_transfer_coefficient, "HU", "heater", counter
    )
    coolers = _utility_exchangers(
        below_hot, delta_t_min, overall_heat_transfer_coefficient, "CU", "cooler", counter
    )
    # Streams left over on the side of the pinch that should not need a
    # utility (hot above the pinch, cold below it) only happen if the greedy
    # matcher could not find a feasible partner for them; still close them
    # off so no stream is left unsatisfied, but say so loudly.
    stray_coolers = _utility_exchangers(
        above_hot, delta_t_min, overall_heat_transfer_coefficient, "CU", "cooler", counter,
        warn_label="above the pinch",
    )
    stray_heaters = _utility_exchangers(
        below_cold, delta_t_min, overall_heat_transfer_coefficient, "HU", "heater", counter,
        warn_label="below the pinch",
    )

    return HeatExchangerNetwork(
        exchangers=(
            above_exchangers + below_exchangers
            + heaters + coolers + stray_coolers + stray_heaters
        )
    )
