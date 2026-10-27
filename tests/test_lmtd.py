import math

import pytest

from pinchpy.calculations.lmtd import lmtd


def test_equal_temperature_difference():
    assert lmtd(20, 20) == 20


def test_unequal_temperature_differences():
    assert lmtd(30, 10) == pytest.approx(20 / math.log(3))  # about 18.2


def test_lmtd_is_symmetric():
    assert lmtd(10, 30) == pytest.approx(lmtd(30, 10))


def test_lmtd_lies_between_geometric_and_arithmetic_mean():
    value = lmtd(40, 10)
    assert math.sqrt(40 * 10) < value < (40 + 10) / 2


def test_nearly_equal_differences_are_continuous():
    assert lmtd(20, 20.000001) == pytest.approx(20, rel=1e-6)


@pytest.mark.parametrize("bad_pair", [(0, 10), (10, 0), (-5, 10), (10, -5)])
def test_non_positive_difference_is_rejected(bad_pair):
    with pytest.raises(ValueError):
        lmtd(*bad_pair)