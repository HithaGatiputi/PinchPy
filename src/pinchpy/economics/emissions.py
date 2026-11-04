"""CO2 emissions and decarbonization analysis for Heat Exchanger Networks (Phase 10).

Provides emission estimation based on:
1. Fuel-specific emission factors (IPCC / EPA standard factors).
2. Thermal efficiency of hot utility generation (boilers, furnaces, heat pumps).
3. Auxiliary electric emissions from cooling utilities (pumps, cooling tower fans).
4. Un-integrated process baseline emissions and net carbon reduction.
5. Carbon tax / carbon credit financial savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..analysis.pinch_point import utility_targets

# Standard emission factors in kg CO2 per kWh of thermal energy delivered by fuel
# Sources: IPCC Guidelines for National Greenhouse Gas Inventories; UK DEFRA / BEIS; US EPA.
FUEL_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas": 0.202,
    "fuel_oil": 0.279,
    "coal": 0.354,
    "lpg": 0.227,
    "biomass": 0.030,
    "grid_electricity": 0.450,
    "solar_electric": 0.0,
}


@dataclass
class EmissionsModel:
    """Parameters for greenhouse gas (CO2) emissions estimation.

    Parameters
    ----------
    fuel_type : str
        Primary fuel or energy source for hot utility generation.
        Default is "natural_gas".
    hot_utility_emission_factor : float, optional
        Carbon emission factor for hot utility fuel in kg CO2/kWh.
        If omitted, looked up automatically from ``FUEL_EMISSION_FACTORS``.
    hot_utility_efficiency : float
        Combustion / generation thermal efficiency (eta) of the heater or boiler.
        Default is 0.85 (85% thermal efficiency).
    cold_utility_emission_factor : float
        Indirect electricity emission factor for cooling water circulation pumps
        and cooling tower fans in kg CO2/kWh_cooling. Default is 0.010 kg CO2/kWh.
    carbon_price : float
        Carbon tax or carbon offset value in USD per metric tonne of CO2.
        Default is 50.0 USD/tonne.
    annual_operating_hours : float
        Operating hours per year. Default is 8000.0 hours/year.
    """

    fuel_type: str = "natural_gas"
    hot_utility_emission_factor: Optional[float] = None
    hot_utility_efficiency: float = 0.85
    cold_utility_emission_factor: float = 0.010
    carbon_price: float = 50.0
    annual_operating_hours: float = 8000.0

    def __post_init__(self) -> None:
        if self.hot_utility_emission_factor is None:
            if self.fuel_type.lower() in FUEL_EMISSION_FACTORS:
                self.hot_utility_emission_factor = FUEL_EMISSION_FACTORS[self.fuel_type.lower()]
            else:
                available = list(FUEL_EMISSION_FACTORS.keys())
                raise ValueError(
                    f"Unknown fuel type '{self.fuel_type}'. Specify a recognized type from "
                    f"{available} or provide an explicit 'hot_utility_emission_factor'."
                )
        if self.hot_utility_emission_factor < 0:
            raise ValueError("Hot utility emission factor cannot be negative.")
        if self.cold_utility_emission_factor < 0:
            raise ValueError("Cold utility emission factor cannot be negative.")
        if self.hot_utility_efficiency <= 0 or self.hot_utility_efficiency > 1.0:
            raise ValueError(
                f"Thermal efficiency must be between 0 and 1.0, got {self.hot_utility_efficiency}."
            )
        if self.carbon_price < 0:
            raise ValueError("Carbon price cannot be negative.")
        if self.annual_operating_hours <= 0 or self.annual_operating_hours > 8760:
            raise ValueError("Annual operating hours must be between 0 and 8760.")

    def calculate_emissions(self, q_hot: float, q_cold: float) -> Dict[str, float]:
        """Calculate hourly and annual CO2 emissions for given utility duties.

        Parameters
        ----------
        q_hot : float
            Hot utility duty in kW.
        q_cold : float
            Cold utility duty in kW.

        Returns
        -------
        dict
            Emissions metrics (rates, annual totals, and carbon tax costs).
        """
        if q_hot < 0 or q_cold < 0:
            raise ValueError("Utility duties cannot be negative.")

        assert self.hot_utility_emission_factor is not None
        # Fuel consumption rate considering thermal efficiency
        fuel_consumption_kw = q_hot / self.hot_utility_efficiency
        hourly_hot_co2_kg = fuel_consumption_kw * self.hot_utility_emission_factor
        hourly_cold_co2_kg = q_cold * self.cold_utility_emission_factor
        hourly_total_co2_kg = hourly_hot_co2_kg + hourly_cold_co2_kg

        annual_co2_tonnes = (hourly_total_co2_kg * self.annual_operating_hours) / 1000.0
        annual_carbon_cost = annual_co2_tonnes * self.carbon_price

        return {
            "hourly_co2_kg": float(hourly_total_co2_kg),
            "hourly_hot_utility_co2_kg": float(hourly_hot_co2_kg),
            "hourly_cold_utility_co2_kg": float(hourly_cold_co2_kg),
            "annual_co2_tonnes": float(annual_co2_tonnes),
            "annual_carbon_cost_usd": float(annual_carbon_cost),
        }


def emissions_analysis(
    network=None,
    streams: Optional[List[Any]] = None,
    delta_t_min: Optional[float] = None,
    emissions_model: Optional[EmissionsModel] = None,
) -> Dict[str, Any]:
    """Perform carbon dioxide (CO2) emissions and savings analysis.

    Parameters
    ----------
    network : HeatExchangerNetwork, optional
        Synthesized network.
    streams : list[Stream], optional
        Process streams. Required to estimate baseline un-integrated emissions
        and environmental savings.
    delta_t_min : float, optional
        Minimum approach temperature (°C) for target-level emissions analysis.
    emissions_model : EmissionsModel, optional
        Emission parameters and fuel characteristics.

    Returns
    -------
    dict
        CO2 emissions report with actual, baseline, reduction metrics, and assumptions.
    """
    model = emissions_model or EmissionsModel()

    if network is not None:
        utility_exchangers = [e for e in network.exchangers if getattr(e, "is_utility", False)]
        heaters = [e for e in utility_exchangers if e.hot_stream in ("HU", "hot_utility", "steam")]
        coolers = [e for e in utility_exchangers if e.cold_stream in ("CU", "cold_utility", "water")]
        q_hot = sum(e.duty for e in heaters)
        q_cold = sum(e.duty for e in coolers)
    elif streams is not None and delta_t_min is not None:
        targets = utility_targets(streams, delta_t_min)
        q_hot = targets["minimum_hot_utility"]
        q_cold = targets["minimum_cold_utility"]
    else:
        raise ValueError(
            "Either 'network' or ('streams' and 'delta_t_min') must be provided for emissions analysis."
        )

    actual_emissions = model.calculate_emissions(q_hot, q_cold)

    unintegrated_emissions = None
    annual_reduction_tonnes = None
    reduction_percentage = None
    carbon_savings_usd = None

    if streams is not None:
        # In an un-integrated process, cold streams require hot utility and hot streams require cold utility
        total_hot_duty = sum(s.duty() for s in streams if s.type == "hot")
        total_cold_duty = sum(s.duty() for s in streams if s.type == "cold")

        baseline = model.calculate_emissions(total_cold_duty, total_hot_duty)
        unintegrated_emissions = baseline

        annual_reduction_tonnes = max(0.0, baseline["annual_co2_tonnes"] - actual_emissions["annual_co2_tonnes"])
        if baseline["annual_co2_tonnes"] > 1e-6:
            reduction_percentage = (annual_reduction_tonnes / baseline["annual_co2_tonnes"]) * 100.0
        else:
            reduction_percentage = 0.0

        carbon_savings_usd = annual_reduction_tonnes * model.carbon_price

    assumptions = {
        "fuel_type": model.fuel_type,
        "hot_utility_emission_factor_kg_per_kwh": model.hot_utility_emission_factor,
        "hot_utility_efficiency": model.hot_utility_efficiency,
        "cold_utility_emission_factor_kg_per_kwh": model.cold_utility_emission_factor,
        "carbon_price_usd_per_tonne": model.carbon_price,
        "annual_operating_hours": model.annual_operating_hours,
        "methodology": (
            "Hot utility emissions calculated as (Q_HU / efficiency) * EF_fuel. "
            "Cold utility emissions calculated from pumping/cooling tower electricity EF. "
            "Un-integrated baseline represents total stream heating/cooling demand without heat recovery."
        ),
    }

    return {
        "actual_annual_co2_tonnes": actual_emissions["annual_co2_tonnes"],
        "actual_hourly_co2_kg": actual_emissions["hourly_co2_kg"],
        "actual_annual_carbon_cost_usd": actual_emissions["annual_carbon_cost_usd"],
        "unintegrated_annual_co2_tonnes": (
            unintegrated_emissions["annual_co2_tonnes"] if unintegrated_emissions else None
        ),
        "annual_co2_reduction_tonnes": annual_reduction_tonnes,
        "co2_reduction_percentage": reduction_percentage,
        "annual_carbon_cost_savings_usd": carbon_savings_usd,
        "assumptions": assumptions,
    }

