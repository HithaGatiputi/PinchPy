"""Retrofit planning, modification recommendations, and techno-economic justification (Phase 11).

Provides:
1. Actionable modification plans (relocations, additions, eliminations) to resolve pinch violations.
2. Target MER network benchmark comparison.
3. Retrofit capital expenditure estimation (new area + repiping allowance).
4. Financial feasibility: Simple payback period, NPV, and ROI for the retrofit investment.
5. Environmental decarbonization: Annual CO2 reduction and carbon tax savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .diagnosis import PinchViolation, diagnose_network
from ..economics.cost import ExchangerCostModel, UtilityCostModel
from ..economics.emissions import EmissionsModel
from ..synthesis.network_design import design_network


@dataclass
class RetrofitModification:
    """Specific unit-level modification recommended in the retrofit plan."""

    action: str  # "relocate", "eliminate", "add", "resequence"
    unit_name: str
    description: str
    target_energy_impact_kw: float
    estimated_cost_usd: float


@dataclass
class RetrofitProject:
    """Comprehensive retrofit project proposal for an existing heat exchanger network."""

    is_already_optimal: bool
    modifications: List[RetrofitModification]
    energy_savings_kw: float
    energy_savings_percent: float
    annual_cost_savings_usd: float
    estimated_retrofit_capex_usd: float
    simple_payback_years: Optional[float]
    net_present_value_usd: float
    return_on_investment_percent: Optional[float]
    annual_co2_reduction_tonnes: float
    carbon_tax_savings_usd: float
    target_network: Any
    assumptions: Dict[str, Any]


def plan_retrofit(
    existing_network,
    streams: List[Any],
    delta_t_min: float,
    overall_heat_transfer_coefficient: Optional[float] = None,
    exchanger_cost_model: Optional[ExchangerCostModel] = None,
    utility_cost_model: Optional[UtilityCostModel] = None,
    emissions_model: Optional[EmissionsModel] = None,
    repiping_factor: float = 1.30,
    unit_modification_allowance: float = 5000.0,
) -> RetrofitProject:
    """Generate an actionable retrofit plan with engineering, economic, and CO2 evaluations.

    Parameters
    ----------
    existing_network : HeatExchangerNetwork
        Existing heat exchanger network to be improved.
    streams : list[Stream]
        Process streams.
    delta_t_min : float
        Target minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U in kW/(m^2*°C).
    exchanger_cost_model : ExchangerCostModel, optional
        Cost parameters for equipment sizing.
    utility_cost_model : UtilityCostModel, optional
        Utility pricing and economic evaluation parameters.
    emissions_model : EmissionsModel, optional
        Greenhouse gas emission factors and carbon pricing.
    repiping_factor : float
        Multiplier on equipment cost for piping, civil, and instrumentation modifications.
        Default is 1.30 (30% installation/repiping allowance).
    unit_modification_allowance : float
        Fixed engineering and relocation allowance per modified unit (USD).
        Default is 5,000 USD.

    Returns
    -------
    RetrofitProject
        Complete retrofit proposal.
    """
    c_model = exchanger_cost_model or ExchangerCostModel()
    u_model = utility_cost_model or UtilityCostModel()
    e_model = emissions_model or EmissionsModel()

    # 1. Run Diagnosis
    diag = diagnose_network(
        existing_network,
        streams,
        delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
    )

    violations = diag["violations"]
    penalties = diag["penalties_and_savings"]
    perf = diag["existing_performance"]
    targets = diag["theoretical_targets"]

    energy_savings_kw = penalties["potential_energy_savings_kw"]
    energy_savings_pct = penalties["energy_savings_percent"]

    # 2. Synthesize Benchmark Target Network
    target_net = design_network(
        streams,
        delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
    )

    # 3. Formulate Actionable Modifications
    modifications: List[RetrofitModification] = []

    for v in violations:
        if v.violation_type == "cross_pinch_match":
            mod_cost = unit_modification_allowance
            modifications.append(
                RetrofitModification(
                    action="relocate_or_split",
                    unit_name=v.exchanger_name,
                    description=(
                        f"Relocate or split '{v.exchanger_name}' to remain completely on one side "
                        f"of the pinch. Eliminates {v.penalty_kw:.2f} kW cross-pinch heat transfer."
                    ),
                    target_energy_impact_kw=v.penalty_kw,
                    estimated_cost_usd=mod_cost,
                )
            )
        elif v.violation_type == "heater_below_pinch":
            mod_cost = unit_modification_allowance * 0.75
            modifications.append(
                RetrofitModification(
                    action="eliminate_or_shift_heater",
                    unit_name=v.exchanger_name,
                    description=(
                        f"Remove heater '{v.exchanger_name}' from below the pinch. Shift heating load "
                        f"of {v.penalty_kw:.2f} kW to process heat recovery."
                    ),
                    target_energy_impact_kw=v.penalty_kw,
                    estimated_cost_usd=mod_cost,
                )
            )
        elif v.violation_type == "cooler_above_pinch":
            mod_cost = unit_modification_allowance * 0.75
            modifications.append(
                RetrofitModification(
                    action="eliminate_or_shift_cooler",
                    unit_name=v.exchanger_name,
                    description=(
                        f"Remove cooler '{v.exchanger_name}' from above the pinch. Divert hot stream heat "
                        f"of {v.penalty_kw:.2f} kW to preheat cold process streams."
                    ),
                    target_energy_impact_kw=v.penalty_kw,
                    estimated_cost_usd=mod_cost,
                )
            )

    # If new exchangers or additional area are needed
    existing_area = perf["total_area_m2"] or 0.0
    target_area = targets["target_area_m2"] or target_net.total_area() or 0.0
    delta_area = max(0.0, target_area - existing_area)

    new_area_cost = 0.0
    if delta_area > 1e-4:
        new_area_cost = c_model.exchanger_cost(delta_area) * repiping_factor
        modifications.append(
            RetrofitModification(
                action="add_area",
                unit_name="New/Expanded Surface",
                description=(
                    f"Add ~{delta_area:.2f} m² of heat exchange surface to achieve the optimal "
                    f"energy recovery target of {targets['maximum_heat_recovery_kw']:.2f} kW."
                ),
                target_energy_impact_kw=energy_savings_kw,
                estimated_cost_usd=new_area_cost,
            )
        )

    # Total estimated retrofit CAPEX
    modifications_cost = sum(m.estimated_cost_usd for m in modifications)
    total_retrofit_capex = modifications_cost

    # 4. Economic Calculations
    # Utility savings from reduced hot and cold utility
    # Each kW of saved hot utility also saves 1 kW of cold utility (dual pinch penalty elimination)
    excess_cu = penalties["excess_cold_utility_kw"]
    annual_opex_savings = (
        (energy_savings_kw * u_model.hot_utility_unit_cost + excess_cu * u_model.cold_utility_unit_cost)
        * u_model.annual_operating_hours
    )

    payback_years = u_model.simple_payback(total_retrofit_capex, annual_opex_savings)
    npv = u_model.net_present_value(total_retrofit_capex, annual_opex_savings)

    acc = u_model.annualized_capital_cost(total_retrofit_capex) if total_retrofit_capex > 0 else 0.0
    roi = (
        ((annual_opex_savings - acc) / total_retrofit_capex) * 100.0
        if total_retrofit_capex > 0
        else None
    )

    # 5. CO2 Reductions
    # Emissions avoided from saved hot utility and saved cooling utility
    saved_fuel_kw = energy_savings_kw / e_model.hot_utility_efficiency
    assert e_model.hot_utility_emission_factor is not None
    hourly_saved_co2_kg = (
        saved_fuel_kw * e_model.hot_utility_emission_factor
        + excess_cu * e_model.cold_utility_emission_factor
    )
    annual_co2_reduction_tonnes = (
        hourly_saved_co2_kg * e_model.annual_operating_hours
    ) / 1000.0
    carbon_tax_savings_usd = annual_co2_reduction_tonnes * e_model.carbon_price

    is_already_optimal = len(violations) == 0 and energy_savings_kw < 1e-4

    assumptions = {
        "delta_t_min": delta_t_min,
        "repiping_factor": repiping_factor,
        "unit_modification_allowance_usd": unit_modification_allowance,
        "hot_utility_unit_cost_usd_per_kwh": u_model.hot_utility_unit_cost,
        "cold_utility_unit_cost_usd_per_kwh": u_model.cold_utility_unit_cost,
        "annual_operating_hours": u_model.annual_operating_hours,
        "discount_rate": u_model.discount_rate,
        "plant_lifetime_years": u_model.plant_lifetime,
        "fuel_type": e_model.fuel_type,
        "boiler_efficiency": e_model.hot_utility_efficiency,
        "carbon_price_usd_per_tonne": e_model.carbon_price,
    }

    return RetrofitProject(
        is_already_optimal=is_already_optimal,
        modifications=modifications,
        energy_savings_kw=energy_savings_kw,
        energy_savings_percent=energy_savings_pct,
        annual_cost_savings_usd=annual_opex_savings,
        estimated_retrofit_capex_usd=total_retrofit_capex,
        simple_payback_years=payback_years,
        net_present_value_usd=npv,
        return_on_investment_percent=roi,
        annual_co2_reduction_tonnes=annual_co2_reduction_tonnes,
        carbon_tax_savings_usd=carbon_tax_savings_usd,
        target_network=target_net,
        assumptions=assumptions,
    )

