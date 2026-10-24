from pinchpy.calculations.lmtd import lmtd

def test_equal_temperature_difference():
    assert lmtd(20, 20) == 20
