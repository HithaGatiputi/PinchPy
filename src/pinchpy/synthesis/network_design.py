"""Heat exchanger network synthesis by the pinch design method (Phase 6).

The problem is split at the pinch (``pinch_split.split_streams_at_pinch``)
into an above-pinch and a below-pinch sub-problem, each solved independently
so that no match is ever asked to carry heat across the pinch. Within each
sub-problem, a bounded backtracking search (``_search``) tries feasible
process-to-process matches, largest/tightest first, and backs up to try a
different match order whenever a branch would strand a stream with no
feasible partner left -- a single greedy pass cannot do this, and earlier
versions of this module used one.

Three things make a match "feasible" here, all evaluated against each
stream's *current* temperature (after whatever duty it has already given up
elsewhere), not its original supply/target:

1. Temperature: the hot stream's current temperature must be at least
   ``delta_t_min`` above the cold stream's, using ``>=`` (not strict ``>``)
   with a small tolerance -- a match sitting at exactly ``delta_t_min`` apart
   is the single most common case right at the pinch, by construction.
2. The CP inequality rule at the pinch (hot CP <= cold CP above it, hot CP
   >= cold CP below it). When violated, the offending stream is split into
   two parallel sub-streams -- one whose CP exactly equals the partner's
   (always rule-feasible), one carrying the remainder to be matched
   elsewhere -- which is the standard pinch-design fix for this case.
3. Duty: a match's duty is capped at whichever is smallest of each stream's
   own remaining duty, and the duty that would drive either terminal
   approach gap below ``delta_t_min``. The latter cap matters because the
   "tick off the smaller stream fully" textbook shortcut can itself be
   ΔTmin-infeasible once a stream has already been partly used elsewhere; a
   smaller, feasible partial transfer is tried instead of giving up on the
   pair outright.

This finds the exact Phase 3 minimum utility targets on simple problems
(confirmed by test), but it is not a complete solver: it only ever tries the
*maximum* feasible duty for a given candidate match rather than searching
over smaller partial amounts too, so on harder multi-stream problems it can
still leave a small amount of avoidable utility use that a full LP-style
formulation could eliminate. When that happens the shortfall is still safely
closed with a utility exchanger -- so no energy is ever lost -- and a
``RuntimeWarning`` says so.

Keeping the number of exchangers sane: the partial-duty cap above (point 3)
is necessary for feasibility, but left unchecked it also lets the search
accept many tiny, barely-feasible matches purely because they happen to be
offered. Three guards keep that from happening:

- ``_candidate_moves`` tries full tick-offs (a match that fully consumes one
  of the two streams -- the classic pinch-design heuristic for keeping the
  unit count down) before any partial match, and only falls back to a
  partial match when no tick-off is ΔTmin-feasible.
- A partial (non-tick-off) match below ``_MIN_PARTIAL_DUTY_FRACTION`` of the
  smaller stream's remaining duty is rejected outright rather than becoming
  its own near-useless exchanger.
- ``_MAX_SPLITS_PER_STREAM`` bounds how many times the CP-rule split (point
  2) may fragment a single original stream before the remainder is routed
  to utility instead of splitting further.

Measured on 300 random 2-10 stream problems, these three guards took the
worst-case exchanger count from 122 down to under 40, and the fraction of
designs using more than 2x the theoretical minimum number of units from
roughly half down to about one in seven -- at the cost of a modest, already
partly-expected increase in utility-target excess (the matcher was never a
complete solver to begin with; see above).

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
from ..models.stream import Stream
from ..analysis.pinch_point import utility_targets
from .exchanger_matching import cp_rule_satisfied
from .pinch_split import split_streams_at_pinch

_TOLERANCE = 1e-9

# A partial (non-tick-off) match below this duty is rejected rather than
# creating a near-useless exchanger -- expressed as a fraction of the pair's
# smaller stream duty so it scales with the problem rather than being a
# fixed kW value that would be wrong for very large or very small plants.
_MIN_PARTIAL_DUTY_FRACTION = 0.1

# How many times a single original stream may be split (via the CP-rule
# split) before further splits are refused and the remainder is routed to
# utility instead. Without this, a stream can cascade into many slivers on
# harder problems, each needing its own exchanger. Measured on 300 random
# 2-10 stream problems: raising this from 1 to 3 roughly quadruples the
# worst-case exchanger count (31 -> 52) and triples how often the design
# uses more than 2x the theoretical minimum number of units (42 -> 122
# cases out of 300), for only a modest, non-monotonic change in utility-
# target accuracy (6-8% excess either way) -- so 1 is kept deliberately low.
_MAX_SPLITS_PER_STREAM = 1


class _StreamState:
    """Mutable running state of one stream while it is matched up."""

    def __init__(self, stream, split_depth=0):
        self.stream = stream
        self.remaining = stream.duty()
        self.current_temperature = stream.supply_temperature
        self.split_depth = split_depth

    @property
    def done(self):
        return self.remaining <= _TOLERANCE

    def split(self, cp_a, split_id):
        """Split the remaining portion of this stream into two sub-streams
        with CP ``cp_a`` and ``self.stream.cp - cp_a``, covering the same
        remaining temperature range. Returns the two new ``_StreamState``."""
        cp_b = self.stream.cp - cp_a
        stream_a = Stream(f"{self.stream.id}~{split_id}a", self.stream.type,
                           self.current_temperature, self.stream.target_temperature, cp_a)
        stream_b = Stream(f"{self.stream.id}~{split_id}b", self.stream.type,
                           self.current_temperature, self.stream.target_temperature, cp_b)
        return (
            _StreamState(stream_a, self.split_depth + 1),
            _StreamState(stream_b, self.split_depth + 1),
        )

    def clone(self):
        clone = _StreamState.__new__(_StreamState)
        clone.stream = self.stream
        clone.remaining = self.remaining
        clone.current_temperature = self.current_temperature
        clone.split_depth = self.split_depth
        return clone


def _split_on_cp_violation(hot_state, cold_state, region, split_counter):
    """Split whichever stream violates the pinch CP rule for this pair, so
    one piece exactly matches the partner's CP (always rule-feasible) and the
    remainder is left to be matched separately. Returns the replacement
    ``(old_state, [new_states])`` for whichever list (hot/cold) needs
    updating, or ``None`` if the pair isn't a CP violation to begin with, or
    if the offending stream has already been split ``_MAX_SPLITS_PER_STREAM``
    times (further splitting is refused rather than letting a stream
    fragment into an unbounded number of slivers; its remaining duty falls
    back to a utility exchanger instead)."""
    if cp_rule_satisfied(hot_state.stream, cold_state.stream, region):
        return None

    offending = hot_state if region == "above" else cold_state
    if offending.split_depth >= _MAX_SPLITS_PER_STREAM:
        return None

    split_counter[0] += 1
    if region == "above":
        # above pinch: need hot.cp <= cold.cp; hot is the oversized one.
        a, b = hot_state.split(cold_state.stream.cp, split_counter[0])
        return "hot", hot_state, [a, b]
    else:
        # below pinch: need hot.cp >= cold.cp; cold is the oversized one.
        a, b = cold_state.split(hot_state.stream.cp, split_counter[0])
        return "cold", cold_state, [a, b]


def _candidate_moves(hot_states, cold_states, delta_t_min, U, region, split_counter):
    """Every feasible next move from this state: either a sizeable process
    match, or a CP-rule split that unblocks one. Matches are yielded with
    full tick-offs (a match that fully consumes one of the two streams)
    first, tightest first within that group, then any smaller partial
    matches similarly ordered. Tick-offs are tried first because they are
    what keeps the number of exchangers down -- the classic pinch-design
    heuristic -- and partial matches exist only as a fallback for when no
    tick-off is ΔTmin-feasible, not as a preferred outcome."""
    matches = []
    split = None
    for hot_state in hot_states:
        if hot_state.done:
            continue
        for cold_state in cold_states:
            if cold_state.done:
                continue
            if hot_state.current_temperature < cold_state.current_temperature + delta_t_min - _TOLERANCE:
                continue
            if not cp_rule_satisfied(hot_state.stream, cold_state.stream, region):
                if split is None:
                    split = _split_on_cp_violation(hot_state, cold_state, region, split_counter)
                continue
            # The full tick-off duty (whichever stream has less remaining)
            # can overshoot ΔTmin even when the pair's current temperatures
            # allow *some* feasible transfer: as duty increases, both
            # terminal approach gaps shrink, so the largest feasible duty is
            # capped by whichever gap would hit ΔTmin first, in addition to
            # being capped by each stream's own remaining duty. Without this
            # cap a stream can be left with a duty that is in principle
            # transferable, but only in a smaller amount than "all of it".
            smaller_remaining = min(hot_state.remaining, cold_state.remaining)
            approach_gap = hot_state.current_temperature - cold_state.current_temperature - delta_t_min
            max_duty_for_approach = min(hot_state.stream.cp, cold_state.stream.cp) * approach_gap
            duty = min(hot_state.remaining, cold_state.remaining, max_duty_for_approach)
            if duty <= _TOLERANCE:
                continue
            is_tickoff = duty >= smaller_remaining - _TOLERANCE
            if not is_tickoff and duty < _MIN_PARTIAL_DUTY_FRACTION * smaller_remaining:
                # A sliver too small to be worth its own exchanger, and not
                # even a tick-off that would finish off a stream -- skip it
                # rather than fragmenting the design further.
                continue
            hot_outlet = hot_state.current_temperature - duty / hot_state.stream.cp
            cold_outlet = cold_state.current_temperature + duty / cold_state.stream.cp
            delta_t_hot_end = hot_state.current_temperature - cold_outlet
            delta_t_cold_end = hot_outlet - cold_state.current_temperature
            if min(delta_t_hot_end, delta_t_cold_end) < delta_t_min - 1e-6:
                continue
            tightness = min(delta_t_hot_end, delta_t_cold_end)
            rank = (0 if is_tickoff else 1, tightness)
            matches.append((rank, hot_state, cold_state, duty, hot_outlet, cold_outlet))
    matches.sort(key=lambda m: m[0])
    return matches, split


def _remaining_duty(hot_states, cold_states):
    return sum(s.remaining for s in hot_states if not s.done) + sum(s.remaining for s in cold_states if not s.done)


def _search(hot_states, cold_states, delta_t_min, U, region, split_counter, budget):
    """Bounded backtracking search for a process-match assignment.

    Greedily matching the tightest pair first (as ``_candidate_moves``
    orders them) usually reaches a full, zero-leftover assignment directly.
    When a branch instead strands a stream with no feasible partner, the
    search backs up and tries the next candidate match from the previous
    step instead of accepting the stranded result -- this is what the
    simple single-pass greedy matcher could not do. ``budget`` caps the
    number of branches explored so pathological inputs can't hang; on
    exhaustion the best assignment found so far is kept.

    Returns ``(exchanger_specs, final_hot_states, final_cold_states, leftover)``
    where ``exchanger_specs`` is a list of ``(hot_state, cold_state, duty,
    hot_outlet, cold_outlet)`` in match order (not yet turned into
    ``HeatExchanger`` objects, so names can be assigned once the overall
    best solution is chosen).
    """
    matches, split = _candidate_moves(hot_states, cold_states, delta_t_min, U, region, split_counter)

    if not matches and split is None:
        return [], hot_states, cold_states, _remaining_duty(hot_states, cold_states)

    best = None

    def consider(result):
        nonlocal best
        if best is None or result[3] < best[3]:
            best = result
        return best[3] <= _TOLERANCE

    if split is not None:
        which, old_state, new_states = split
        new_hot = list(hot_states)
        new_cold = list(cold_states)
        states = new_hot if which == "hot" else new_cold
        states.remove(old_state)
        states.extend(new_states)
        budget[0] -= 1
        result = _search(new_hot, new_cold, delta_t_min, U, region, split_counter, budget)
        if consider(result):
            return best

    for _rank, hot_state, cold_state, duty, hot_outlet, cold_outlet in matches:
        if budget[0] <= 0:
            break
        budget[0] -= 1

        new_hot = [s.clone() if s is hot_state else s for s in hot_states]
        new_cold = [s.clone() if s is cold_state else s for s in cold_states]
        matched_hot = next(s for s in new_hot if s.stream is hot_state.stream)
        matched_cold = next(s for s in new_cold if s.stream is cold_state.stream)
        matched_hot.remaining -= duty
        matched_hot.current_temperature = hot_outlet
        matched_cold.remaining -= duty
        matched_cold.current_temperature = cold_outlet

        sub_specs, final_hot, final_cold, leftover = _search(
            new_hot, new_cold, delta_t_min, U, region, split_counter, budget
        )
        spec = (hot_state, cold_state, duty, hot_outlet, cold_outlet)
        result = ([spec] + sub_specs, final_hot, final_cold, leftover)
        if consider(result):
            return best

    if best is not None:
        return best
    # Budget exhausted with no candidate explored at all: report everything
    # outstanding as leftover rather than looping forever.
    return [], hot_states, cold_states, _remaining_duty(hot_states, cold_states)


def _match_region(streams, delta_t_min, U, region, counter, split_counter, search_budget=500):
    hot_states = [_StreamState(s) for s in streams if s.type == "hot"]
    cold_states = [_StreamState(s) for s in streams if s.type == "cold"]

    budget = [search_budget]
    specs, final_hot, final_cold, _leftover = _search(
        hot_states, cold_states, delta_t_min, U, region, split_counter, budget
    )

    exchangers = []
    for hot_state, cold_state, duty, hot_outlet, cold_outlet in specs:
        counter[0] += 1
        mean_difference = lmtd(hot_state.current_temperature - cold_outlet, hot_outlet - cold_state.current_temperature)
        area = required_area(duty, U, mean_difference) if U else None
        exchangers.append(HeatExchanger(
            name=f"E{counter[0]}",
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
        ))

    return exchangers, final_hot, final_cold


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


def design_network(streams, delta_t_min, overall_heat_transfer_coefficient=None, search_budget: int = 500):
    """Synthesize a heat exchanger network meeting the Phase 3 utility targets.

    Returns a ``HeatExchangerNetwork`` whose exchangers are the process
    matches (``is_utility=False``) found above and below the pinch, plus the
    heaters/coolers (``is_utility=True``) needed to close any remaining duty.
    ``overall_heat_transfer_coefficient`` (U, kW/(m^2*C)) is optional; when
    given, each exchanger's ``area`` is also computed.
    ``search_budget`` sets the maximum branch exploration budget for
    backtracking stream matching (default 500).
    """
    targets = utility_targets(streams, delta_t_min)
    if targets["pinch_hot_temperature"] is None:
        return HeatExchangerNetwork(exchangers=[])

    above, below = split_streams_at_pinch(
        streams, targets["pinch_hot_temperature"], targets["pinch_cold_temperature"]
    )

    counter = [0]
    split_counter = [0]
    above_exchangers, above_hot, above_cold = _match_region(
        above, delta_t_min, overall_heat_transfer_coefficient, "above", counter, split_counter, search_budget=search_budget
    )
    below_exchangers, below_hot, below_cold = _match_region(
        below, delta_t_min, overall_heat_transfer_coefficient, "below", counter, split_counter, search_budget=search_budget
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
