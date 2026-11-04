import pytest

from pinchpy.models.stream import Stream
from pinchpy.synthesis.network_design import design_network
from pinchpy.analysis.benchmarks import get_benchmark
from pinchpy.economics.cost import (
    ExchangerCostModel,
    UtilityCostModel,
    cost_analysis,
)
from pinchpy.economics.emissions import (
    EmissionsModel,
    emissions_analysis,
    FUEL_EMISSION_FACTORS,
)
from pinchpy.economics import economic_and_environmental_analysis


# =============================================================================
# ExchangerCostModel Tests
# =============================================================================

def test_exchanger_cost_model_default_calculation():
    model = ExchangerCostModel(a=10000.0, b=800.0, c=0.8)
    area = 25.0
    expected = 10000.0 + 800.0 * (25.0 ** 0.8)
    assert model.exchanger_cost(area) == pytest.approx(expected)


def test_exchanger_cost_zero_area_returns_zero():
    model = ExchangerCostModel()
    assert model.exchanger_cost(0.0) == 0.0


def test_exchanger_cost_negative_area_raises():
    model = ExchangerCostModel()
    with pytest.raises(ValueError, match="negative"):
        model.exchanger_cost(-5.0)


def test_exchanger_cost_factors_scaling():
    base = ExchangerCostModel(a=10000, b=800, c=0.8, material_factor=1.0)
    ss316 = ExchangerCostModel(a=10000, b=800, c=0.8, material_factor=2.2)
    assert ss316.exchanger_cost(20.0) == pytest.approx(base.exchanger_cost(20.0) * 2.2)


def test_exchanger_cost_model_invalid_parameters():
    with pytest.raises(ValueError, match="non-negative"):
        ExchangerCostModel(a=-100)
    with pytest.raises(ValueError, match="positive"):
        ExchangerCostModel(c=-0.5)
    with pytest.raises(ValueError, match="positive"):
        ExchangerCostModel(material_factor=0)


# =============================================================================
# UtilityCostModel Tests
# =============================================================================

def test_utility_cost_model_operating_cost():
    model = UtilityCostModel(
        hot_utility_unit_cost=0.03,
        cold_utility_unit_cost=0.005,
        annual_operating_hours=8000.0,
    )
    # Q_hot = 100 kW, Q_cold = 50 kW
    hourly = 100.0 * 0.03 + 50.0 * 0.005
    expected_annual = hourly * 8000.0
    assert model.operating_cost(100.0, 50.0) == pytest.approx(expected_annual)


def test_utility_cost_model_annualization_factor():
    model = UtilityCostModel(discount_rate=0.10, plant_lifetime=10)
    # CRF = 0.10 * 1.1^10 / (1.1^10 - 1) ~ 0.162745
    crf = model.annualization_factor()
    assert crf == pytest.approx(0.162745, rel=1e-4)

    # Zero discount rate edge case: 1 / n
    zero_i_model = UtilityCostModel(discount_rate=0.0, plant_lifetime=10)
    assert zero_i_model.annualization_factor() == pytest.approx(0.10)


def test_utility_cost_model_payback_and_npv():
    model = UtilityCostModel(discount_rate=0.10, plant_lifetime=10)
    capex = 50000.0
    annual_savings = 25000.0
    assert model.simple_payback(capex, annual_savings) == pytest.approx(2.0)
    assert model.simple_payback(capex, 0.0) is None
    assert model.simple_payback(capex, -1000.0) is None

    # NPV = -50000 + 25000 * ((1 - 1.1^-10) / 0.1) ~ -50000 + 25000 * 6.14457 = 103614.3
    npv = model.net_present_value(capex, annual_savings)
    assert npv == pytest.approx(103614.3, rel=1e-3)


def test_utility_cost_model_invalid_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        UtilityCostModel(hot_utility_unit_cost=-0.01)
    with pytest.raises(ValueError, match="operating hours"):
        UtilityCostModel(annual_operating_hours=10000)
    with pytest.raises(ValueError, match="lifetime"):
        UtilityCostModel(plant_lifetime=0)


# =============================================================================
# cost_analysis Tests
# =============================================================================

def test_cost_analysis_with_synthesized_network():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    report = cost_analysis(network=network, streams=streams)

    assert report["number_of_units"] == len(network.exchangers)
    assert report["total_area_m2"] > 0
    assert report["installed_capital_cost_usd"] > 0
    assert report["annualized_capital_cost_usd_per_yr"] > 0
    assert report["annual_operating_cost_usd_per_yr"] >= 0
    assert report["total_annualized_cost_usd_per_yr"] > 0
    assert report["unintegrated_operating_cost_usd_per_yr"] > report["annual_operating_cost_usd_per_yr"]
    assert report["annual_operating_savings_usd_per_yr"] > 0
    assert report["simple_payback_years"] is not None and report["simple_payback_years"] > 0
    assert report["net_present_value_usd"] is not None
    assert "assumptions" in report
    assert len(report["unit_details"]) == len(network.exchangers)


def test_cost_analysis_unsized_network():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10)  # U is None, so area is None
    report = cost_analysis(network=network, streams=streams)

    assert report["total_area_m2"] is None
    assert report["installed_capital_cost_usd"] is None
    assert report["annual_operating_cost_usd_per_yr"] >= 0


def test_cost_analysis_target_based():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    report = cost_analysis(
        streams=streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.5,
    )

    assert report["total_area_m2"] > 0
    assert report["installed_capital_cost_usd"] > 0
    assert report["annual_operating_cost_usd_per_yr"] >= 0
    assert report["annual_operating_savings_usd_per_yr"] > 0


def test_cost_analysis_missing_inputs_raises():
    with pytest.raises(ValueError, match="Either 'network' or"):
        cost_analysis()


# =============================================================================
# EmissionsModel Tests
# =============================================================================

def test_emissions_model_standard_fuels():
    for fuel in FUEL_EMISSION_FACTORS:
        model = EmissionsModel(fuel_type=fuel)
        assert model.hot_utility_emission_factor == FUEL_EMISSION_FACTORS[fuel]


def test_emissions_model_unknown_fuel_raises():
    with pytest.raises(ValueError, match="Unknown fuel type"):
        EmissionsModel(fuel_type="antimatter")


def test_emissions_model_custom_factor():
    model = EmissionsModel(fuel_type="custom_fuel", hot_utility_emission_factor=0.185)
    assert model.hot_utility_emission_factor == 0.185


def test_emissions_model_efficiency_calculation():
    model = EmissionsModel(
        fuel_type="natural_gas",
        hot_utility_efficiency=0.80,
        cold_utility_emission_factor=0.010,
        annual_operating_hours=1000.0,
        carbon_price=50.0,
    )
    # Q_hot = 100 kW, Q_cold = 50 kW
    # Fuel kw = 100 / 0.80 = 125 kW
    # Hot CO2 rate = 125 * 0.202 = 25.25 kg/h
    # Cold CO2 rate = 50 * 0.010 = 0.5 kg/h
    # Total rate = 25.75 kg/h
    # Annual tonnes = 25.75 * 1000 / 1000 = 25.75 tonnes
    res = model.calculate_emissions(100.0, 50.0)
    assert res["hourly_hot_utility_co2_kg"] == pytest.approx(25.25)
    assert res["hourly_cold_utility_co2_kg"] == pytest.approx(0.5)
    assert res["annual_co2_tonnes"] == pytest.approx(25.75)
    assert res["annual_carbon_cost_usd"] == pytest.approx(25.75 * 50.0)


# =============================================================================
# emissions_analysis Tests
# =============================================================================

def test_emissions_analysis_with_network():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    report = emissions_analysis(network=network, streams=streams)

    assert report["actual_annual_co2_tonnes"] >= 0
    assert report["unintegrated_annual_co2_tonnes"] > report["actual_annual_co2_tonnes"]
    assert report["annual_co2_reduction_tonnes"] > 0
    assert 0 < report["co2_reduction_percentage"] <= 100.0
    assert report["annual_carbon_cost_savings_usd"] > 0
    assert "assumptions" in report


def test_emissions_analysis_target_based():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    report = emissions_analysis(streams=streams, delta_t_min=10.0)

    assert report["actual_annual_co2_tonnes"] >= 0
    assert report["annual_co2_reduction_tonnes"] > 0
    assert report["co2_reduction_percentage"] > 0


# =============================================================================
# Combined Analysis & Kemp Benchmark Tests
# =============================================================================

def test_economic_and_environmental_analysis_kemp_benchmark():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    network = design_network(streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5)

    full_report = economic_and_environmental_analysis(
        network=network,
        streams=streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.5,
    )

    summary = full_report["summary"]
    assert summary["installed_capital_cost_usd"] > 0
    assert summary["annual_operating_cost_usd_per_yr"] > 0
    assert summary["total_annualized_cost_usd_per_yr"] > 0
    assert summary["annual_operating_savings_usd_per_yr"] > 0
    assert summary["simple_payback_years"] is not None and summary["simple_payback_years"] < 5.0
    assert summary["annual_co2_reduction_tonnes"] > 0
    assert summary["co2_reduction_percentage"] > 70.0  # High heat recovery in Kemp 4-stream
    assert summary["annual_carbon_cost_savings_usd"] > 0

