from pinchpy.calculations.heat_duty import heat_duty

def test_heat_duty():
    assert heat_duty(2.0, 50) == 100
