"""Retrofit analysis and optimization of existing Heat Exchanger Networks (Phase 11)."""

from typing import Any, Dict, List, Optional

from .diagnosis import PinchViolation, diagnose_network
from .recommendations import RetrofitModification, RetrofitProject, plan_retrofit
from ..economics.cost import ExchangerCostModel, UtilityCostModel
from ..economics.emissions import EmissionsModel


def retrofit_analysis(
    existing_network,
    streams: List[Any],
    delta_t_min: float,
    overall_heat_transfer_coefficient: Optional[float] = None,
    exchanger_cost_model: Optional[ExchangerCostModel] = None,
    utility_cost_model: Optional[UtilityCostModel] = None,
    emissions_model: Optional[EmissionsModel] = None,
    repiping_factor: float = 1.30,
    unit_modification_allowance: float = 5000.0,
) -> Dict[str, Any]:
    """Perform complete retrofit diagnosis and improvement analysis for an existing network.

    Parameters
    ----------
    existing_network : HeatExchangerNetwork
        The existing network.
    streams : list[Stream]
        Original process streams.
    delta_t_min : float
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U in kW/(m^2*°C).
    exchanger_cost_model : ExchangerCostModel, optional
        Equipment capital cost parameters.
    utility_cost_model : UtilityCostModel, optional
        Economic and operating cost parameters.
    emissions_model : EmissionsModel, optional
        Emissions and decarbonization parameters.
    repiping_factor : float, optional
        Piping and civil works cost multiplier (default 1.30).
    unit_modification_allowance : float, optional
        Fixed relocation/modification fee per unit (USD, default 5000.0).

    Returns
    -------
    dict
        Consolidated retrofit report containing diagnosis, modification plan,
        economics, decarbonization, and benchmark target network.
    """
    project = plan_retrofit(
        existing_network=existing_network,
        streams=streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        exchanger_cost_model=exchanger_cost_model,
        utility_cost_model=utility_cost_model,
        emissions_model=emissions_model,
        repiping_factor=repiping_factor,
        unit_modification_allowance=unit_modification_allowance,
    )

    diagnosis = diagnose_network(
        network=existing_network,
        streams=streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
    )

    return {
        "is_already_optimal": project.is_already_optimal,
        "diagnosis": diagnosis,
        "modifications": [
            {
                "action": m.action,
                "unit_name": m.unit_name,
                "description": m.description,
                "energy_impact_kw": m.target_energy_impact_kw,
                "estimated_cost_usd": m.estimated_cost_usd,
            }
            for m in project.modifications
        ],
        "economics": {
            "energy_savings_kw": project.energy_savings_kw,
            "energy_savings_percent": project.energy_savings_percent,
            "annual_cost_savings_usd": project.annual_cost_savings_usd,
            "estimated_retrofit_capex_usd": project.estimated_retrofit_capex_usd,
            "simple_payback_years": project.simple_payback_years,
            "net_present_value_usd": project.net_present_value_usd,
            "return_on_investment_percent": project.return_on_investment_percent,
        },
        "emissions": {
            "annual_co2_reduction_tonnes": project.annual_co2_reduction_tonnes,
            "carbon_tax_savings_usd": project.carbon_tax_savings_usd,
        },
        "target_network": project.target_network,
        "assumptions": project.assumptions,
    }


__all__ = [
    "PinchViolation",
    "diagnose_network",
    "RetrofitModification",
    "RetrofitProject",
    "plan_retrofit",
    "retrofit_analysis",
]

