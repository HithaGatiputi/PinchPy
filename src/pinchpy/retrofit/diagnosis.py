"""Diagnosis of existing Heat Exchanger Networks against Pinch Analysis principles (Phase 11).

Detects thermodynamic inefficiencies in existing networks:
1. Cross-pinch process exchangers (transferring heat across the pinch boundary).
2. Heaters placed below the pinch (consuming hot utility where process heat suffices).
3. Coolers placed above the pinch (rejecting high-temperature heat to cooling water).
4. Approach temperature violations (DT < delta_t_min).
5. Energy penalty assessment: quantified excess hot and cold utility usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analysis.pinch_point import utility_targets
from ..analysis.area_target import area_target


@dataclass
class PinchViolation:
    """Represents a specific pinch-rule violation in an existing heat exchanger network.

    Attributes
    ----------
    exchanger_name : str
        Identifier of the violating heat exchanger.
    violation_type : str
        Type of violation:
        - "cross_pinch_match": Process exchanger transferring heat across the pinch.
        - "heater_below_pinch": Utility heater supplying heat below the pinch.
        - "cooler_above_pinch": Utility cooler removing heat above the pinch.
        - "approach_temperature_violation": Exchanger operating with approach < delta_t_min.
    severity : str
        "critical" for thermodynamic pinch violations; "warning" for small approach violations.
    penalty_kw : float
        Excess utility penalty directly attributable to this violation in kW.
    details : dict
        Contextual details (temperatures, stream IDs, pinch temperatures).
    recommendation : str
        Human-readable recommendation for corrective modification.
    """

    exchanger_name: str
    violation_type: str
    severity: str
    penalty_kw: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


def _extract_stream_id(stream_name: str) -> str:
    """Normalize stream identifier by stripping split and region suffixes."""
    return stream_name.split("~")[0].replace("_above", "").replace("_below", "")


def diagnose_network(
    network,
    streams: List[Any],
    delta_t_min: float,
    overall_heat_transfer_coefficient: Optional[float] = None,
) -> Dict[str, Any]:
    """Diagnose an existing Heat Exchanger Network against pinch principles.

    Parameters
    ----------
    network : HeatExchangerNetwork
        The existing industrial heat exchanger network to evaluate.
    streams : list[Stream]
        The process stream definitions (supply, target, CP).
    delta_t_min : float
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U in kW/(m^2*°C) for area target comparison.

    Returns
    -------
    dict
        Comprehensive diagnostic report detailing violations, utility penalties,
        energy recovery potential, and area comparisons.
    """
    if delta_t_min <= 0:
        raise ValueError(f"delta_t_min must be positive, got {delta_t_min}")

    targets = utility_targets(streams, delta_t_min)
    t_pinch_hot = targets["pinch_hot_temperature"]
    t_pinch_cold = targets["pinch_cold_temperature"]
    qh_target = targets["minimum_hot_utility"]
    qc_target = targets["minimum_cold_utility"]
    qrec_target = targets.get("heat_recovery", 0.0)

    exchangers = network.exchangers
    process_exchangers = [e for e in exchangers if not getattr(e, "is_utility", False)]
    utility_exchangers = [e for e in exchangers if getattr(e, "is_utility", False)]

    heaters = [e for e in utility_exchangers if e.hot_stream in ("HU", "hot_utility", "steam")]
    coolers = [e for e in utility_exchangers if e.cold_stream in ("CU", "cold_utility", "water")]

    qh_exist = sum(e.duty for e in heaters)
    qc_exist = sum(e.duty for e in coolers)
    qrec_exist = sum(e.duty for e in process_exchangers)

    violations: List[PinchViolation] = []

    # Stream CP lookup
    stream_map = {s.id: s for s in streams}

    # 1. Inspect Process Exchangers for Cross-Pinch Transfer
    if t_pinch_hot is not None and t_pinch_cold is not None:
        for e in process_exchangers:
            # Check if exchanger operates with known inlet/outlet temperatures
            h_in = e.hot_inlet
            h_out = e.hot_outlet
            c_in = e.cold_inlet
            c_out = e.cold_outlet

            # If temperatures are missing, infer from stream definitions if full stream is used
            if h_in is None and e.hot_stream in stream_map:
                h_in = stream_map[e.hot_stream].supply_temperature
                h_out = stream_map[e.hot_stream].target_temperature
            if c_in is None and e.cold_stream in stream_map:
                c_in = stream_map[e.cold_stream].supply_temperature
                c_out = stream_map[e.cold_stream].target_temperature

            if h_in is not None and h_out is not None and c_in is not None and c_out is not None:
                # Hot stream temperatures (hot stream cools: h_in > h_out)
                t_h_high = max(h_in, h_out)
                t_h_low = min(h_in, h_out)

                # Cold stream temperatures (cold stream heats: c_out > c_in)
                t_c_high = max(c_in, c_out)
                t_c_low = min(c_in, c_out)

                # Check if hot stream enters above the pinch and cold stream enters below the pinch
                is_cross_pinch = (t_h_high > t_pinch_hot + 1e-4) and (t_c_low < t_pinch_cold - 1e-4)

                if is_cross_pinch:
                    # Calculate duty transferred above pinch on hot side vs cold side
                    h_span = max(t_h_high - t_h_low, 1e-6)
                    c_span = max(t_c_high - t_c_low, 1e-6)

                    # Fraction of hot duty above pinch
                    frac_h_above = max(0.0, min(1.0, (t_h_high - t_pinch_hot) / h_span))
                    q_h_above = e.duty * frac_h_above

                    # Fraction of cold duty above pinch
                    frac_c_above = max(0.0, min(1.0, (t_c_high - t_pinch_cold) / c_span))
                    q_c_above = e.duty * frac_c_above

                    # Net cross-pinch duty transferred from above to below
                    q_cross = max(0.0, q_h_above - q_c_above)
                    if q_cross < 1e-4 and (t_h_low < t_pinch_hot and t_c_high > t_pinch_cold):
                        # Both streams cross the pinch; estimate duty spanning across boundary
                        q_cross = min(q_h_above, e.duty - q_c_above)

                    if q_cross > 1e-4:
                        violations.append(
                            PinchViolation(
                                exchanger_name=e.name,
                                violation_type="cross_pinch_match",
                                severity="critical",
                                penalty_kw=q_cross,
                                details={
                                    "hot_stream": e.hot_stream,
                                    "cold_stream": e.cold_stream,
                                    "hot_inlet": h_in,
                                    "hot_outlet": h_out,
                                    "cold_inlet": c_in,
                                    "cold_outlet": c_out,
                                    "pinch_hot": t_pinch_hot,
                                    "pinch_cold": t_pinch_cold,
                                    "cross_pinch_duty_kw": q_cross,
                                },
                                recommendation=(
                                    f"Relocate or split exchanger '{e.name}'. It transfers {q_cross:.2f} kW "
                                    f"across the pinch, causing a dual penalty of +{q_cross:.2f} kW hot utility "
                                    f"and +{q_cross:.2f} kW cold utility."
                                ),
                            )
                        )

            # Approach temperature check
            if h_in is not None and c_out is not None:
                gap_hot = h_in - c_out
                if gap_hot < delta_t_min - 1e-4:
                    violations.append(
                        PinchViolation(
                            exchanger_name=e.name,
                            violation_type="approach_temperature_violation",
                            severity="warning",
                            penalty_kw=0.0,
                            details={"location": "hot_end", "actual_dt": gap_hot, "delta_t_min": delta_t_min},
                            recommendation=(
                                f"Hot end approach in '{e.name}' is {gap_hot:.2f} °C, below "
                                f"specified delta_t_min = {delta_t_min:.2f} °C."
                            ),
                        )
                    )
            if h_out is not None and c_in is not None:
                gap_cold = h_out - c_in
                if gap_cold < delta_t_min - 1e-4:
                    violations.append(
                        PinchViolation(
                            exchanger_name=e.name,
                            violation_type="approach_temperature_violation",
                            severity="warning",
                            penalty_kw=0.0,
                            details={"location": "cold_end", "actual_dt": gap_cold, "delta_t_min": delta_t_min},
                            recommendation=(
                                f"Cold end approach in '{e.name}' is {gap_cold:.2f} °C, below "
                                f"specified delta_t_min = {delta_t_min:.2f} °C."
                            ),
                        )
                    )

    # 2. Inspect Heaters Placed Below the Pinch
    if t_pinch_cold is not None:
        for e in heaters:
            c_in = e.cold_inlet
            c_out = e.cold_outlet
            sid = _extract_stream_id(e.cold_stream)
            if c_in is None and sid in stream_map:
                c_in = stream_map[sid].supply_temperature
                c_out = stream_map[sid].target_temperature

            if c_in is not None:
                t_low = min(c_in, c_out if c_out is not None else c_in)
                t_high = max(c_in, c_out if c_out is not None else c_in)
                if t_low < t_pinch_cold - 1e-4:
                    span = max(t_high - t_low, 1e-6)
                    # Duty delivered below pinch temperature
                    frac_below = max(0.0, min(1.0, (t_pinch_cold - t_low) / span))
                    duty_below = e.duty * frac_below if c_out is not None else e.duty
                    if duty_below > 1e-4:
                        violations.append(
                            PinchViolation(
                                exchanger_name=e.name,
                                violation_type="heater_below_pinch",
                                severity="critical",
                                penalty_kw=duty_below,
                                details={
                                    "process_stream": e.cold_stream,
                                    "cold_inlet": c_in,
                                    "cold_outlet": c_out,
                                    "pinch_cold": t_pinch_cold,
                                    "duty_below_pinch_kw": duty_below,
                                },
                                recommendation=(
                                    f"Eliminate or reposition heater '{e.name}'. It supplies {duty_below:.2f} kW "
                                    f"below the pinch ({t_pinch_cold:.1f} °C), which should be satisfied by process heat recovery."
                                ),
                            )
                        )

    # 3. Inspect Coolers Placed Above the Pinch
    if t_pinch_hot is not None:
        for e in coolers:
            h_in = e.hot_inlet
            h_out = e.hot_outlet
            sid = _extract_stream_id(e.hot_stream)
            if h_in is None and sid in stream_map:
                h_in = stream_map[sid].supply_temperature
                h_out = stream_map[sid].target_temperature

            if h_in is not None:
                t_high = max(h_in, h_out if h_out is not None else h_in)
                t_low = min(h_in, h_out if h_out is not None else h_in)
                if t_high > t_pinch_hot + 1e-4:
                    span = max(t_high - t_low, 1e-6)
                    # Duty removed above pinch temperature
                    frac_above = max(0.0, min(1.0, (t_high - t_pinch_hot) / span))
                    duty_above = e.duty * frac_above if h_out is not None else e.duty
                    if duty_above > 1e-4:
                        violations.append(
                            PinchViolation(
                                exchanger_name=e.name,
                                violation_type="cooler_above_pinch",
                                severity="critical",
                                penalty_kw=duty_above,
                                details={
                                    "process_stream": e.hot_stream,
                                    "hot_inlet": h_in,
                                    "hot_outlet": h_out,
                                    "pinch_hot": t_pinch_hot,
                                    "duty_above_pinch_kw": duty_above,
                                },
                                recommendation=(
                                    f"Eliminate or reposition cooler '{e.name}'. It rejects {duty_above:.2f} kW "
                                    f"above the pinch ({t_pinch_hot:.1f} °C), wasting heat that could be recovered."
                                ),
                            )
                        )

    # Penalty and Savings Metrics
    excess_hu = max(0.0, qh_exist - qh_target)
    excess_cu = max(0.0, qc_exist - qc_target)
    potential_energy_savings_kw = excess_hu
    energy_savings_percent = (
        (potential_energy_savings_kw / qh_exist) * 100.0 if qh_exist > 1e-5 else 0.0
    )

    # Area Analysis
    existing_area = network.total_area()
    target_area_val = None
    area_difference = None
    if overall_heat_transfer_coefficient is not None:
        try:
            a_tgt = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
            target_area_val = a_tgt["total_area"]
            if existing_area is not None:
                area_difference = target_area_val - existing_area
        except Exception:
            pass

    # Heat recovery efficiency
    recovery_efficiency = (
        min(1.0, qrec_exist / qrec_target) if qrec_target > 1e-5 else 1.0
    )

    is_pinch_compliant = (
        len([v for v in violations if v.severity == "critical"]) == 0
        and excess_hu < 1e-3
    )

    return {
        "is_pinch_compliant": is_pinch_compliant,
        "number_of_violations": len(violations),
        "critical_violations_count": len([v for v in violations if v.severity == "critical"]),
        "violations": violations,
        "existing_performance": {
            "hot_utility_duty_kw": qh_exist,
            "cold_utility_duty_kw": qc_exist,
            "recovered_heat_kw": qrec_exist,
            "total_area_m2": existing_area,
            "number_of_units": len(exchangers),
        },
        "theoretical_targets": {
            "minimum_hot_utility_kw": qh_target,
            "minimum_cold_utility_kw": qc_target,
            "maximum_heat_recovery_kw": qrec_target,
            "pinch_hot_temperature": t_pinch_hot,
            "pinch_cold_temperature": t_pinch_cold,
            "target_area_m2": target_area_val,
        },
        "penalties_and_savings": {
            "excess_hot_utility_kw": excess_hu,
            "excess_cold_utility_kw": excess_cu,
            "potential_energy_savings_kw": potential_energy_savings_kw,
            "energy_savings_percent": energy_savings_percent,
            "recovery_efficiency": recovery_efficiency,
            "area_difference_m2": area_difference,
        },
    }

