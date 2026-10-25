import pytest

from pinchpy.calculations.area import required_area
from pinchpy.calculations.heat_transfer import heat_transfer_duty


def test_required_area():
    assert required_area(100, 0.5, 20) == pytest.approx(10.0)


def test_heat_transfer_duty():
    assert heat_transfer_duty(0.5, 10, 20) == pytest.approx(100.0)


def test_area_and_duty_are_inverse_of_each_other():
    area = required_area(250.0, 0.8, 17.5)
    assert heat_transfer_duty(0.8, area, 17.5) == pytest.approx(250.0)


@pytest.mark.parametrize("coefficient", [0, -0.5])
def test_non_positive_coefficient_is_rejected(coefficient):
    with pytest.raises(ValueError):
        required_area(100, coefficient, 20)


@pytest.mark.parametrize("mean_difference", [0, -5])
def test_non_positive_lmtd_is_rejected(mean_difference):
    with pytest.raises(ValueError):
        required_area(100, 0.5, mean_difference)