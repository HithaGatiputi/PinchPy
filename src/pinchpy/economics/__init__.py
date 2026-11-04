"""Economics and environmental impact assessment for Heat Exchanger Networks (Phase 10)."""

from typing import Any, Dict, List, Optional

from .cost import ExchangerCostModel, UtilityCostModel, cost_analysis
from .emissions import EmissionsModel, emissions_analysis, FUEL_EMISSION_FACTORS


def economic_and_environmental_analysis(
    network=None,
    streams: Optional[List[Any]] = None,
    delta_t_min: Optional[float] = None,
    overall_heat_transfer_coefficient: Optional[float] = None,
    exchanger_cost_model: Optional[ExchangerCostModel] = None,
    utility_cost_model: Optional[UtilityCostModel] = None,
    emissions_model: Optional[EmissionsModel] = None,
) -> Dict[str, Any]:
    """Perform combined economic and environmental (CO2) evaluation.

    Parameters
    ----------
    network : HeatExchangerNetwork, optional
        Synthesized heat exchanger network.
    streams : list[Stream], optional
        Process streams for baseline and savings estimation.
    delta_t_min : float, optional
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U (kW/(m^2*°C)).
    exchanger_cost_model : ExchangerCostModel, optional
        Capital cost parameters.
    utility_cost_model : UtilityCostModel, optional
        Operational cost and financial parameters.
    emissions_model : EmissionsModel, optional
        Emissions and decarbonization parameters.

    Returns
    -------
    dict
        Combined report with cost, payback, emissions, and assumptions sections.
    """
    cost_report = cost_analysis(
        network=network,
        streams=streams,
        delta_t_min=delta_t_min,
        overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
        exchanger_cost_model=exchanger_cost_model,
        utility_cost_model=utility_cost_model,
    )

    emissions_report = emissions_analysis(
        network=network,
        streams=streams,
        delta_t_min=delta_t_min,
        emissions_model=emissions_model,
    )

    return {
        "economics": cost_report,
        "emissions": emissions_report,
        "summary": {
            "installed_capital_cost_usd": cost_report["installed_capital_cost_usd"],
            "annual_operating_cost_usd_per_yr": cost_report["annual_operating_cost_usd_per_yr"],
            "total_annualized_cost_usd_per_yr": cost_report["total_annualized_cost_usd_per_yr"],
            "annual_operating_savings_usd_per_yr": cost_report["annual_operating_savings_usd_per_yr"],
            "simple_payback_years": cost_report["simple_payback_years"],
            "net_present_value_usd": cost_report["net_present_value_usd"],
            "annual_co2_reduction_tonnes": emissions_report["annual_co2_reduction_tonnes"],
            "co2_reduction_percentage": emissions_report["co2_reduction_percentage"],
            "annual_carbon_cost_savings_usd": emissions_report["annual_carbon_cost_savings_usd"],
        },
    }


__all__ = [
    "ExchangerCostModel",
    "UtilityCostModel",
    "cost_analysis",
    "EmissionsModel",
    "emissions_analysis",
    "FUEL_EMISSION_FACTORS",
    "economic_and_environmental_analysis",
]

