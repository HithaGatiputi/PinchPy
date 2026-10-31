import pytest

from pinchpy.models.exchanger import HeatExchanger
from pinchpy.models.network import HeatExchangerNetwork
from pinchpy.models.stream import Stream
from pinchpy.synthesis.network_design import design_network
from pinchpy.synthesis.network_evaluation import network_tac


def test_network_heat():
    net = HeatExchangerNetwork([
        HeatExchanger("E1", "H1", "C1", 100, 10)
    ])
    assert net.total_recovered_heat() == 100


def test_network_tac_hand_calculated_example():
    """Verify TAC calculation against an analytical hand-calculated two-stream example."""
    # 3 exchangers: 1 process match, 1 utility heater, 1 utility cooler
    # E1: duty = 100 kW, area = 25.0 m², is_utility = False
    # HU1: duty = 50 kW, area = 10.0 m², is_utility = True, hot_stream = "HU", cold_stream = "C1"
    # CU1: duty = 20 kW, area = 5.0 m², is_utility = True, hot_stream = "H1", cold_stream = "CU"
    net = HeatExchangerNetwork([
        HeatExchanger("E1", "H1", "C1", duty=100.0, delta_t_min=10.0, area=25.0, is_utility=False),
        HeatExchanger("HU1", "HU", "C1", duty=50.0, delta_t_min=10.0, area=10.0, is_utility=True),
        HeatExchanger("CU1", "H1", "CU", duty=20.0, delta_t_min=10.0, area=5.0, is_utility=True),
    ])

    fixed_cost = 1000.0
    area_cost = 500.0
    area_exponent = 1.0  # linear for exact hand-checking
    hot_utility_cost = 70.0
    cold_utility_cost = 20.0

    res = network_tac(
        net,
        fixed_cost=fixed_cost,
        area_cost=area_cost,
        area_exponent=area_exponent,
        hot_utility_cost=hot_utility_cost,
        cold_utility_cost=cold_utility_cost,
    )

    # Capital = (1000 + 500*25) + (1000 + 500*10) + (1000 + 500*5)
    #         = 13500 + 6000 + 3500 = 23000
    # Hot utility = 70 * 50 = 3500
    # Cold utility = 20 * 20 = 400
    # Total = 23000 + 3500 + 400 = 26900
    assert res["capital"] == pytest.approx(23000.0)
    assert res["hot_utility"] == pytest.approx(3500.0)
    assert res["cold_utility"] == pytest.approx(400.0)
    assert res["total"] == pytest.approx(26900.0)


def test_network_tac_two_stream_synthesized_network():
    """Verify TAC on a synthesized 2-stream network from design_network."""
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    net = design_network(streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1)
    tac = network_tac(net)

    assert "capital" in tac
    assert "hot_utility" in tac
    assert "cold_utility" in tac
    assert "total" in tac
    assert tac["capital"] > 0
    assert tac["total"] == pytest.approx(tac["capital"] + tac["hot_utility"] + tac["cold_utility"])


def test_network_tac_zero_area_edge_case():
    """Verify TAC handles zero-area exchangers and empty networks gracefully."""
    # Exchanger with zero area
    net_zero = HeatExchangerNetwork([
        HeatExchanger("E0", "H1", "C1", duty=0.0, delta_t_min=10.0, area=0.0, is_utility=False)
    ])
    tac_zero = network_tac(net_zero, fixed_cost=1500.0, area_cost=800.0, area_exponent=0.8)
    assert tac_zero["capital"] == pytest.approx(1500.0)
    assert tac_zero["total"] == pytest.approx(1500.0)

    # Fixed cost 0 with zero area -> 0.0
    tac_zero_free = network_tac(net_zero, fixed_cost=0.0, area_cost=800.0, area_exponent=0.8)
    assert tac_zero_free["capital"] == pytest.approx(0.0)
    assert tac_zero_free["total"] == pytest.approx(0.0)

    # Empty network
    net_empty = HeatExchangerNetwork([])
    tac_empty = network_tac(net_empty)
    assert tac_empty["capital"] == 0.0
    assert tac_empty["hot_utility"] == 0.0
    assert tac_empty["cold_utility"] == 0.0
    assert tac_empty["total"] == 0.0


def test_network_tac_raises_when_area_is_none():
    """Verify TAC raises ValueError with clear message when exchanger area is None."""
    net = HeatExchangerNetwork([
        HeatExchanger("E1", "H1", "C1", duty=100.0, delta_t_min=10.0, area=None)
    ])
    with pytest.raises(ValueError, match="area None"):
        network_tac(net)

