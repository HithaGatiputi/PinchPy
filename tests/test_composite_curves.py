from pinchpy.models.stream import Stream
from pinchpy.analysis.composite_curves import composite_curve_data


def test_composite_curve_endpoints_match_duty():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    data = composite_curve_data(streams, 10)

    hot_enthalpy = data["hot_curve"][-1][0]
    cold_enthalpy = data["cold_curve"][-1][0]

    assert hot_enthalpy == streams[0].duty()
    assert cold_enthalpy == streams[1].duty()
