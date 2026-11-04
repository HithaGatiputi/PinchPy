import pytest

from pinchpy.models.stream import Stream
from pinchpy.models.exchanger import HeatExchanger
from pinchpy.models.network import HeatExchangerNetwork
from pinchpy.analysis.benchmarks import get_benchmark
from pinchpy.synthesis.network_design import design_network
from pinchpy.retrofit.diagnosis import (
    diagnose_network,
    PinchViolation,
)
from pinchpy.retrofit.recommendations import (
    plan_retrofit,
    RetrofitProject,
    RetrofitModification,
)
from pinchpy.retrofit import retrofit_analysis


# =============================================================================
# Diagnostic Tests
# =============================================================================

def test_diagnose_network_on_optimal_network():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    optimal_net = design_network(streams, 10.0, overall_heat_transfer_coefficient=0.5)
    report = diagnose_network(optimal_net, streams, delta_t_min=10.0)

    assert report["is_pinch_compliant"] is True
    assert report["critical_violations_count"] == 0
    assert report["penalties_and_savings"]["potential_energy_savings_kw"] == pytest.approx(0.0)
    assert report["penalties_and_savings"]["recovery_efficiency"] == pytest.approx(1.0)


def test_diagnose_network_invalid_delta_t_min_raises():
    streams = [Stream("H1", "hot", 180, 80, 2.0)]
    net = HeatExchangerNetwork([])
    with pytest.raises(ValueError, match="positive"):
        diagnose_network(net, streams, delta_t_min=0.0)


def test_diagnose_network_detects_cross_pinch_match():
    # Kemp 4-stream at dTmin=10 has pinch at Th=90, Tc=80.
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]

    # Process exchanger transferring heat from H1 above pinch (170 -> 100)
    # to C1 below pinch (20 -> 80), duty = 120 kW
    cross_pinch_hex = HeatExchanger(
        name="E_CROSS",
        hot_stream="H1",
        cold_stream="C1",
        duty=120.0,
        delta_t_min=10.0,
        hot_inlet=170.0,
        hot_outlet=100.0,
        cold_inlet=20.0,
        cold_outlet=80.0,
        is_utility=False,
    )
    # Plus utility exchangers to satisfy remaining duties
    heater = HeatExchanger(
        name="HTR",
        hot_stream="HU",
        cold_stream="C2",
        duty=140.0,
        delta_t_min=10.0,
        cold_inlet=80.0,
        cold_outlet=140.0,
        is_utility=True,
    )
    cooler = HeatExchanger(
        name="CLR",
        hot_stream="H2",
        cold_stream="CU",
        duty=180.0,
        delta_t_min=10.0,
        hot_inlet=150.0,
        hot_outlet=30.0,
        is_utility=True,
    )

    bad_net = HeatExchangerNetwork(exchangers=[cross_pinch_hex, heater, cooler])
    report = diagnose_network(bad_net, streams, delta_t_min=10.0)

    assert report["is_pinch_compliant"] is False
    assert report["number_of_violations"] >= 1
    cross_violations = [v for v in report["violations"] if v.violation_type == "cross_pinch_match"]
    assert len(cross_violations) == 1
    assert cross_violations[0].exchanger_name == "E_CROSS"
    assert cross_violations[0].penalty_kw > 0
    assert "relocate or split" in cross_violations[0].recommendation.lower()


def test_diagnose_network_detects_heater_below_pinch():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    # Pinch cold is 80 °C. Place a heater heating C1 from 20 °C to 60 °C (completely below pinch)
    bad_heater = HeatExchanger(
        name="BAD_HEATER",
        hot_stream="HU",
        cold_stream="C1",
        duty=80.0,
        delta_t_min=10.0,
        cold_inlet=20.0,
        cold_outlet=60.0,
        is_utility=True,
    )
    net = HeatExchangerNetwork(exchangers=[bad_heater])
    report = diagnose_network(net, streams, delta_t_min=10.0)

    heater_violations = [v for v in report["violations"] if v.violation_type == "heater_below_pinch"]
    assert len(heater_violations) == 1
    assert heater_violations[0].exchanger_name == "BAD_HEATER"
    assert heater_violations[0].penalty_kw == pytest.approx(80.0)


def test_diagnose_network_detects_cooler_above_pinch():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    # Pinch hot is 90 °C. Place a cooler cooling H1 from 170 °C to 110 °C (completely above pinch)
    bad_cooler = HeatExchanger(
        name="BAD_COOLER",
        hot_stream="H1",
        cold_stream="CU",
        duty=180.0,
        delta_t_min=10.0,
        hot_inlet=170.0,
        hot_outlet=110.0,
        is_utility=True,
    )
    net = HeatExchangerNetwork(exchangers=[bad_cooler])
    report = diagnose_network(net, streams, delta_t_min=10.0)

    cooler_violations = [v for v in report["violations"] if v.violation_type == "cooler_above_pinch"]
    assert len(cooler_violations) == 1
    assert cooler_violations[0].exchanger_name == "BAD_COOLER"
    assert cooler_violations[0].penalty_kw == pytest.approx(180.0)


def test_diagnose_network_approach_temperature_violation():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    # Exchanger above pinch but with approach gap = 4 °C < delta_t_min = 10 °C
    pinched_hex = HeatExchanger(
        name="E_TIGHT",
        hot_stream="H1",
        cold_stream="C2",
        duty=50.0,
        delta_t_min=10.0,
        hot_inlet=140.0,
        hot_outlet=110.0,
        cold_inlet=106.0,  # gap at cold end = 110 - 106 = 4 < 10
        cold_outlet=125.0,
        is_utility=False,
    )
    net = HeatExchangerNetwork(exchangers=[pinched_hex])
    report = diagnose_network(net, streams, delta_t_min=10.0)

    dt_violations = [v for v in report["violations"] if v.violation_type == "approach_temperature_violation"]
    assert len(dt_violations) >= 1
    assert dt_violations[0].severity == "warning"


# =============================================================================
# Retrofit Planning & Economic Feasibility Tests
# =============================================================================

def test_plan_retrofit_on_non_optimal_network():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]

    # Construct an inefficient network with a heater below the pinch and cooler above the pinch
    bad_heater = HeatExchanger(
        name="HTR1",
        hot_stream="HU",
        cold_stream="C1",
        duty=100.0,
        delta_t_min=10.0,
        cold_inlet=20.0,
        cold_outlet=70.0,
        is_utility=True,
    )
    bad_cooler = HeatExchanger(
        name="CLR1",
        hot_stream="H1",
        cold_stream="CU",
        duty=120.0,
        delta_t_min=10.0,
        hot_inlet=170.0,
        hot_outlet=130.0,
        is_utility=True,
    )
    existing_net = HeatExchangerNetwork(exchangers=[bad_heater, bad_cooler])

    project = plan_retrofit(
        existing_network=existing_net,
        streams=streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.5,
    )

    assert project.is_already_optimal is False
    assert len(project.modifications) > 0
    assert project.energy_savings_kw > 0
    assert project.energy_savings_percent > 0
    assert project.annual_cost_savings_usd > 0
    assert project.estimated_retrofit_capex_usd > 0
    assert project.simple_payback_years is not None and project.simple_payback_years > 0
    assert project.net_present_value_usd is not None
    assert project.annual_co2_reduction_tonnes > 0
    assert project.carbon_tax_savings_usd > 0
    assert project.target_network is not None
    assert "assumptions" in project.__dict__


def test_plan_retrofit_on_already_optimal_network():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    optimal_net = design_network(streams, 10.0, overall_heat_transfer_coefficient=0.5)
    project = plan_retrofit(optimal_net, streams, 10.0)

    assert project.is_already_optimal is True
    assert project.energy_savings_kw == pytest.approx(0.0)
    assert project.annual_cost_savings_usd == pytest.approx(0.0)


def test_retrofit_analysis_consolidated_report():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]

    bad_heater = HeatExchanger(
        name="HTR1",
        hot_stream="HU",
        cold_stream="C1",
        duty=60.0,
        delta_t_min=10.0,
        cold_inlet=20.0,
        cold_outlet=50.0,
        is_utility=True,
    )
    existing_net = HeatExchangerNetwork(exchangers=[bad_heater])

    report = retrofit_analysis(
        existing_network=existing_net,
        streams=streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.5,
    )

    assert "is_already_optimal" in report
    assert "diagnosis" in report
    assert "modifications" in report
    assert "economics" in report
    assert "emissions" in report
    assert "target_network" in report
    assert "assumptions" in report

    assert report["is_already_optimal"] is False
    assert len(report["modifications"]) >= 1
    assert report["economics"]["annual_cost_savings_usd"] > 0
    assert report["emissions"]["annual_co2_reduction_tonnes"] > 0

