from pinchpy.models.network import HeatExchangerNetwork
from pinchpy.models.exchanger import HeatExchanger

def test_network_heat():
    net = HeatExchangerNetwork([
        HeatExchanger("E1", "H1", "C1", 100, 10)
    ])
    assert net.total_recovered_heat() == 100
