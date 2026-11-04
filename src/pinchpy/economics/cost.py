"""Capital and operating cost estimation for Heat Exchanger Networks (Phase 10).

Provides engineering correlations for:
1. Capital expenditure (CAPEX): Guthrie/Hall standard installed cost correlation
   Cost = (a + b * Area^c) * Fm * Fp * Fi
2. Annualized Capital Cost (ACC): Capital recovery factor based on plant life and discount rate.
3. Operating expenditure (OPEX): Annual cost of hot and cold utilities.
4. Total Annualized Cost (TAC): ACC + OPEX.
5. Economic feasibility: Simple payback period, Net Present Value (NPV), and ROI
   relative to an un-integrated process baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..analysis.pinch_point import utility_targets
from ..analysis.area_target import area_target


@dataclass
class ExchangerCostModel:
    """Capital cost correlation parameters for heat exchangers.

    Formula for each exchanger:
        Installed Cost = (a + b * Area^c) * material_factor * pressure_factor * installation_factor

    Parameters
    ----------
    a : float
        Base fixed cost per exchanger shell (USD). Default is 10,000 USD
        (standard Hall et al., 1988 / Guthrie correlation baseline).
    b : float
        Area cost coefficient (USD/(m^2)^c). Default is 800.0 USD/m^(2*c).
    c : float
        Area exponent reflecting economy of scale (typical range 0.6 - 0.85).
        Default is 0.8.
    material_factor : float
        Material correction factor Fm (e.g. 1.0 for carbon steel, 1.7 for SS304,
        2.2 for SS316, 3.0 for titanium). Default is 1.0.
    pressure_factor : float
        Pressure rating correction factor Fp (1.0 for < 10 bar, 1.15 for 10-20 bar,
        1.3 for 20-50 bar). Default is 1.0.
    installation_factor : float
        Installation / Lang factor multiplier Fi (default 1.0 if incorporated in
        coefficients a and b).
    """

    a: float = 10000.0
    b: float = 800.0
    c: float = 0.8
    material_factor: float = 1.0
    pressure_factor: float = 1.0
    installation_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.a < 0 or self.b < 0:
            raise ValueError("Cost coefficients 'a' and 'b' must be non-negative.")
        if self.c <= 0:
            raise ValueError("Area exponent 'c' must be positive.")
        if self.material_factor <= 0 or self.pressure_factor <= 0 or self.installation_factor <= 0:
            raise ValueError("Correction factors (material, pressure, installation) must be positive.")

    def exchanger_cost(self, area: float) -> float:
        """Calculate installed capital cost for a single heat exchanger.

        Parameters
        ----------
        area : float
            Heat-transfer surface area in square meters (m^2).

        Returns
        -------
        float
            Installed capital cost in USD.
        """
        if area < 0:
            raise ValueError(f"Exchanger area cannot be negative, got {area} m^2.")
        if area == 0.0:
            return 0.0

        base_cost = self.a + self.b * (area ** self.c)
        return float(base_cost * self.material_factor * self.pressure_factor * self.installation_factor)


@dataclass
class UtilityCostModel:
    """Operating cost and financial evaluation parameters.

    Parameters
    ----------
    hot_utility_unit_cost : float
        Cost of hot utility (e.g. steam or natural gas) in USD/kWh.
        Default is 0.03 USD/kWh (~8.33 USD/GJ).
    cold_utility_unit_cost : float
        Cost of cold utility (e.g. cooling water) in USD/kWh.
        Default is 0.005 USD/kWh (~1.39 USD/GJ).
    annual_operating_hours : float
        Annual operating hours (hours/year). Default is 8000.0 hours/year (~91% uptime).
    discount_rate : float
        Annual discount rate / interest rate (fraction, e.g. 0.10 for 10%).
        Default is 0.10.
    plant_lifetime : int
        Expected plant / equipment lifetime in years. Default is 10 years.
    """

    hot_utility_unit_cost: float = 0.03
    cold_utility_unit_cost: float = 0.005
    annual_operating_hours: float = 8000.0
    discount_rate: float = 0.10
    plant_lifetime: int = 10

    def __post_init__(self) -> None:
        if self.hot_utility_unit_cost < 0 or self.cold_utility_unit_cost < 0:
            raise ValueError("Utility unit costs must be non-negative.")
        if self.annual_operating_hours <= 0 or self.annual_operating_hours > 8760:
            raise ValueError("Annual operating hours must be between 0 and 8760.")
        if self.discount_rate < 0:
            raise ValueError("Discount rate must be non-negative.")
        if self.plant_lifetime <= 0:
            raise ValueError("Plant lifetime must be a positive integer.")

    def annualization_factor(self) -> float:
        """Capital recovery factor (CRF) used to annualize capital expenditure.

        CRF = i * (1 + i)^n / ((1 + i)^n - 1)
        If discount_rate == 0, CRF = 1 / n.
        """
        i = self.discount_rate
        n = self.plant_lifetime
        if i == 0.0:
            return 1.0 / n
        factor = (i * ((1.0 + i) ** n)) / (((1.0 + i) ** n) - 1.0)
        return float(factor)

    def operating_cost(self, q_hot: float, q_cold: float) -> float:
        """Calculate annual operating cost for hot and cold utility usage.

        Parameters
        ----------
        q_hot : float
            Hot utility duty in kW.
        q_cold : float
            Cold utility duty in kW.

        Returns
        -------
        float
            Annual operating cost in USD/year.
        """
        if q_hot < 0 or q_cold < 0:
            raise ValueError("Utility duties cannot be negative.")
        hourly_cost = (
            q_hot * self.hot_utility_unit_cost + q_cold * self.cold_utility_unit_cost
        )
        return float(hourly_cost * self.annual_operating_hours)

    def annualized_capital_cost(self, capex: float) -> float:
        """Calculate annualized capital cost (ACC) in USD/year."""
        return float(capex * self.annualization_factor())

    def total_annualized_cost(self, capex: float, annual_opex: float) -> float:
        """Calculate Total Annualized Cost (TAC = ACC + OPEX) in USD/year."""
        return float(self.annualized_capital_cost(capex) + annual_opex)

    def simple_payback(self, capex: float, annual_savings: float) -> Optional[float]:
        """Calculate simple payback period in years.

        Returns None if annual_savings <= 0.
        """
        if annual_savings <= 0 or capex <= 0:
            return None
        return float(capex / annual_savings)

    def net_present_value(self, capex: float, annual_savings: float) -> float:
        """Calculate Net Present Value (NPV) of energy savings over plant lifetime.

        NPV = -CAPEX + sum_{t=1}^n [ savings / (1 + i)^t ]
        """
        i = self.discount_rate
        n = self.plant_lifetime
        if i == 0.0:
            pv = annual_savings * n
        else:
            pv = annual_savings * ((1.0 - (1.0 + i) ** (-n)) / i)
        return float(pv - capex)


def cost_analysis(
    network=None,
    streams: Optional[List[Any]] = None,
    delta_t_min: Optional[float] = None,
    overall_heat_transfer_coefficient: Optional[float] = None,
    exchanger_cost_model: Optional[ExchangerCostModel] = None,
    utility_cost_model: Optional[UtilityCostModel] = None,
) -> Dict[str, Any]:
    """Perform capital, operational, and total annualized cost analysis.

    Can be invoked with either:
    1. A synthesized ``HeatExchangerNetwork`` (calculates cost based on actual exchangers).
    2. Process ``streams``, ``delta_t_min``, and ``overall_heat_transfer_coefficient``
       (calculates target-based theoretical minimum cost).

    Parameters
    ----------
    network : HeatExchangerNetwork, optional
        Synthesized network.
    streams : list[Stream], optional
        Process streams. Required to estimate un-integrated baseline, savings, and payback.
    delta_t_min : float, optional
        Minimum approach temperature (°C).
    overall_heat_transfer_coefficient : float, optional
        Overall heat transfer coefficient U in kW/(m^2*°C).
    exchanger_cost_model : ExchangerCostModel, optional
        Parameters for capital cost correlation.
    utility_cost_model : UtilityCostModel, optional
        Parameters for utility pricing and financial evaluation.

    Returns
    -------
    dict
        Comprehensive cost and financial report.
    """
    c_model = exchanger_cost_model or ExchangerCostModel()
    u_model = utility_cost_model or UtilityCostModel()

    # Determine calculation mode
    if network is not None:
        # Network-level analysis
        exchangers = network.exchangers
        process_exchangers = [e for e in exchangers if not getattr(e, "is_utility", False)]
        utility_exchangers = [e for e in exchangers if getattr(e, "is_utility", False)]

        heaters = [e for e in utility_exchangers if e.hot_stream in ("HU", "hot_utility", "steam")]
        coolers = [e for e in utility_exchangers if e.cold_stream in ("CU", "cold_utility", "water")]

        q_hot = sum(e.duty for e in heaters)
        q_cold = sum(e.duty for e in coolers)
        q_recovered = sum(e.duty for e in process_exchangers)

        num_units = len(exchangers)
        num_process_units = len(process_exchangers)
        num_utility_units = len(utility_exchangers)

        # Sizing and CAPEX
        unit_details = []
        total_capex = 0.0
        total_area = 0.0
        all_sized = True

        for e in exchangers:
            area = getattr(e, "area", None)
            if area is not None:
                cost = c_model.exchanger_cost(area)
                total_area += area
                total_capex += cost
                unit_details.append({
                    "name": e.name,
                    "duty": e.duty,
                    "area": area,
                    "cost": cost,
                    "is_utility": getattr(e, "is_utility", False),
                })
            else:
                all_sized = False
                unit_details.append({
                    "name": e.name,
                    "duty": e.duty,
                    "area": None,
                    "cost": None,
                    "is_utility": getattr(e, "is_utility", False),
                })

        if not all_sized:
            total_capex_reported = None
            total_area_reported = None
        else:
            total_capex_reported = total_capex
            total_area_reported = total_area

    elif streams is not None and delta_t_min is not None:
        # Target-level analysis from pinch analysis
        targets = utility_targets(streams, delta_t_min)
        q_hot = targets["minimum_hot_utility"]
        q_cold = targets["minimum_cold_utility"]
        q_recovered = targets.get("heat_recovery", 0.0)

        num_process_units = max(len(streams) - 1, 0)
        num_utility_units = (1 if q_hot > 1e-5 else 0) + (1 if q_cold > 1e-5 else 0)
        num_units = num_process_units + num_utility_units

        unit_details = []
        if overall_heat_transfer_coefficient is not None:
            a_target = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
            total_area_reported = a_target["total_area"]
            # Estimate target capital cost assuming area distributed across minimum process units
            if num_process_units > 0 and total_area_reported > 0:
                area_per_unit = total_area_reported / num_process_units
                cost_per_unit = c_model.exchanger_cost(area_per_unit)
                total_capex_reported = cost_per_unit * num_process_units
            else:
                total_capex_reported = 0.0
        else:
            total_area_reported = None
            total_capex_reported = None
    else:
        raise ValueError(
            "Either 'network' or ('streams' and 'delta_t_min') must be provided for cost analysis."
        )

    # Annual Operating Cost (OPEX)
    annual_opex = u_model.operating_cost(q_hot, q_cold)

    # Annualized Capital Cost (ACC) & Total Annualized Cost (TAC)
    if total_capex_reported is not None:
        acc = u_model.annualized_capital_cost(total_capex_reported)
        tac = u_model.total_annualized_cost(total_capex_reported, annual_opex)
    else:
        acc = None
        tac = None

    # Baseline comparison (un-integrated process)
    unintegrated_opex = None
    annual_savings = None
    payback_years = None
    npv = None
    roi = None

    if streams is not None:
        total_hot_duty = sum(s.duty() for s in streams if s.type == "hot")
        total_cold_duty = sum(s.duty() for s in streams if s.type == "cold")
        unintegrated_opex = u_model.operating_cost(total_cold_duty, total_hot_duty)
        annual_savings = max(0.0, unintegrated_opex - annual_opex)

        if total_capex_reported is not None and total_capex_reported > 0:
            payback_years = u_model.simple_payback(total_capex_reported, annual_savings)
            npv = u_model.net_present_value(total_capex_reported, annual_savings)
            if acc is not None:
                roi = ((annual_savings - acc) / total_capex_reported) * 100.0

    assumptions = {
        "cost_correlation": f"Cost = ({c_model.a:g} + {c_model.b:g} * Area^{c_model.c:g}) * Fm * Fp * Fi",
        "material_factor_Fm": c_model.material_factor,
        "pressure_factor_Fp": c_model.pressure_factor,
        "installation_factor_Fi": c_model.installation_factor,
        "hot_utility_unit_cost_usd_per_kwh": u_model.hot_utility_unit_cost,
        "cold_utility_unit_cost_usd_per_kwh": u_model.cold_utility_unit_cost,
        "annual_operating_hours": u_model.annual_operating_hours,
        "discount_rate": u_model.discount_rate,
        "plant_lifetime_years": u_model.plant_lifetime,
        "capital_recovery_factor": u_model.annualization_factor(),
    }

    return {
        "number_of_units": num_units,
        "number_of_process_units": num_process_units,
        "number_of_utility_units": num_utility_units,
        "total_recovered_heat_kw": q_recovered,
        "hot_utility_duty_kw": q_hot,
        "cold_utility_duty_kw": q_cold,
        "total_area_m2": total_area_reported,
        "installed_capital_cost_usd": total_capex_reported,
        "annualized_capital_cost_usd_per_yr": acc,
        "annual_operating_cost_usd_per_yr": annual_opex,
        "total_annualized_cost_usd_per_yr": tac,
        "unintegrated_operating_cost_usd_per_yr": unintegrated_opex,
        "annual_operating_savings_usd_per_yr": annual_savings,
        "simple_payback_years": payback_years,
        "net_present_value_usd": npv,
        "return_on_investment_percent": roi,
        "unit_details": unit_details,
        "assumptions": assumptions,
    }

