"""Quality metrics and pinch-rule validation for a synthesized heat exchanger network."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ..analysis.pinch_point import utility_targets

_TOLERANCE = 1e-5


def evaluate_network(network, streams=None, delta_t_min: Optional[float] = None) -> Dict[str, Any]:
    """Summarize a ``HeatExchangerNetwork``.

    Parameters
    ----------
    network : HeatExchangerNetwork
        The network to evaluate.
    streams : list[Stream], optional
        Original process streams, used to compare against theoretical MER targets.
    delta_t_min : float, optional
        Minimum approach temperature, used to calculate targets.

    Returns
    -------
    dict
        Summary dictionary containing:
        - 'number_of_exchangers'
        - 'number_of_process_matches'
        - 'number_of_utility_exchangers'
        - 'total_recovered_heat'
        - 'total_utility_heat'
        - 'total_heater_heat'
        - 'total_cooler_heat'
        - 'total_area' (if exchangers sized)
        - 'minimum_units' (if streams provided)
        - 'target_hot_utility' (if streams and delta_t_min provided)
        - 'target_cold_utility' (if streams and delta_t_min provided)
        - 'energy_recovery_efficiency' (if streams and delta_t_min provided)
    """
    exchangers = network.exchangers
    process_exchangers = [e for e in exchangers if not getattr(e, "is_utility", False)]
    utility_exchangers = [e for e in exchangers if getattr(e, "is_utility", False)]

    heaters = [e for e in utility_exchangers if e.hot_stream in ("HU", "hot_utility", "steam")]
    coolers = [e for e in utility_exchangers if e.cold_stream in ("CU", "cold_utility", "water")]

    result: Dict[str, Any] = {
        "number_of_exchangers": len(exchangers),
        "number_of_process_matches": len(process_exchangers),
        "number_of_utility_exchangers": len(utility_exchangers),
        "total_recovered_heat": sum(e.duty for e in process_exchangers),
        "total_utility_heat": sum(e.duty for e in utility_exchangers),
        "total_heater_heat": sum(e.duty for e in heaters),
        "total_cooler_heat": sum(e.duty for e in coolers),
    }

    areas = [e.area for e in exchangers if e.area is not None]
    if areas:
        result["total_area"] = sum(areas)

    if streams is not None:
        n_streams = len(streams)
        n_utility_types = len(
            {
                e.hot_stream if e.hot_stream in ("HU", "CU") else e.cold_stream
                for e in utility_exchangers
            }
        )
        result["minimum_units"] = max(n_streams + n_utility_types - 1, 0)

        dtmin = delta_t_min
        if dtmin is None and exchangers:
            dtmin = exchangers[0].delta_t_min

        if dtmin is not None and len(streams) > 0:
            targets = utility_targets(streams, dtmin)
            qh_target = targets["minimum_hot_utility"]
            qc_target = targets["minimum_cold_utility"]
            qrec_target = targets.get("heat_recovery", 0.0)

            result["target_hot_utility"] = qh_target
            result["target_cold_utility"] = qc_target
            result["target_heat_recovery"] = qrec_target

            if qrec_target > _TOLERANCE:
                efficiency = min(1.0, result["total_recovered_heat"] / qrec_target)
            else:
                efficiency = 1.0
            result["energy_recovery_efficiency"] = efficiency

    return result


def validate_network_pinch_rules(network, streams, delta_t_min: float) -> Dict[str, Any]:
    """Validate that a synthesized network respects thermodynamic and pinch principles.

    Checks:
    1. Stream energy balances (every process stream's enthalpy change is satisfied).
    2. Overall First Law energy conservation.
    3. Minimum approach temperature feasibility for all process matches.
    4. Pinch rule adherence: no heat transferred across the pinch boundary.

    Returns
    -------
    dict
        Validation report with boolean flags and numerical checks.
    """
    targets = utility_targets(streams, delta_t_min)
    t_pinch_hot = targets["pinch_hot_temperature"]
    t_pinch_cold = targets["pinch_cold_temperature"]

    process_exchangers = [e for e in network.exchangers if not getattr(e, "is_utility", False)]
    utility_exchangers = [e for e in network.exchangers if getattr(e, "is_utility", False)]

    # 1. Approach gap checks
    delta_t_violations = []
    for e in process_exchangers:
        if e.hot_inlet is not None and e.cold_outlet is not None:
            gap_hot = e.hot_inlet - e.cold_outlet
            if gap_hot < delta_t_min - 1e-5:
                delta_t_violations.append((e.name, "hot_end", gap_hot))
        if e.hot_outlet is not None and e.cold_inlet is not None:
            gap_cold = e.hot_outlet - e.cold_inlet
            if gap_cold < delta_t_min - 1e-5:
                delta_t_violations.append((e.name, "cold_end", gap_cold))

    # 2. Check for heat transferred across the pinch
    cross_pinch_matches = []
    if t_pinch_hot is not None and t_pinch_cold is not None:
        for e in process_exchangers:
            if (
                e.hot_inlet is not None
                and e.cold_inlet is not None
                and e.hot_inlet > t_pinch_hot + 1e-5
                and e.cold_inlet < t_pinch_cold - 1e-5
            ):
                cross_pinch_matches.append(e.name)

    # 3. Stream duty balance check
    stream_duties: Dict[str, float] = {}
    for e in network.exchangers:
        if e.hot_stream not in ("HU", "CU", "hot_utility", "cold_utility"):
            sid = e.hot_stream.split("~")[0].replace("_above", "").replace("_below", "")
            stream_duties[sid] = stream_duties.get(sid, 0.0) + e.duty
        if e.cold_stream not in ("HU", "CU", "hot_utility", "cold_utility"):
            sid = e.cold_stream.split("~")[0].replace("_above", "").replace("_below", "")
            stream_duties[sid] = stream_duties.get(sid, 0.0) + e.duty

    stream_balances_satisfied = True
    stream_errors = {}
    for s in streams:
        actual_q = stream_duties.get(s.id, 0.0)
        expected_q = s.duty()
        err = abs(actual_q - expected_q)
        if err > 1e-2:
            stream_balances_satisfied = False
            stream_errors[s.id] = {"expected": expected_q, "actual": actual_q, "error": err}

    # 4. Total energy conservation
    total_hot = sum(s.duty() for s in streams if s.type == "hot")
    total_cold = sum(s.duty() for s in streams if s.type == "cold")
    heater_heat = sum(e.duty for e in utility_exchangers if e.hot_stream in ("HU", "hot_utility", "steam"))
    cooler_heat = sum(e.duty for e in utility_exchangers if e.cold_stream in ("CU", "cold_utility", "water"))

    energy_balance_conserved = abs((total_hot + heater_heat) - (total_cold + cooler_heat)) < 1e-3

    is_valid = (
        len(delta_t_violations) == 0
        and len(cross_pinch_matches) == 0
        and stream_balances_satisfied
        and energy_balance_conserved
    )

    return {
        "is_valid": is_valid,
        "energy_balance_conserved": energy_balance_conserved,
        "stream_balances_satisfied": stream_balances_satisfied,
        "delta_t_min_respected": len(delta_t_violations) == 0,
        "no_cross_pinch_transfer": len(cross_pinch_matches) == 0,
        "delta_t_violations": delta_t_violations,
        "cross_pinch_matches": cross_pinch_matches,
        "stream_errors": stream_errors,
        "total_heat_recovered": sum(e.duty for e in process_exchangers),
        "total_heater_duty": heater_heat,
        "total_cooler_duty": cooler_heat,
    }
