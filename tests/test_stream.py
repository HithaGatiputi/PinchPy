"""Tests for PinchPy Stream model and physical validation (Phase 1)."""

import math
import pytest

from pinchpy.models.stream import (
    Stream,
    StreamTypeError,
    StreamValidationError,
)


# =====================================================================
# 1. Valid Streams
# =====================================================================


def test_valid_hot_stream():
    """Verify initialization and attributes of a valid hot stream."""
    stream = Stream(
        id="H1",
        type="hot",
        supply_temperature=180.0,
        target_temperature=80.0,
        cp=2.5,
    )
    assert stream.id == "H1"
    assert stream.type == "hot"
    assert stream.stream_type == "hot"
    assert stream.supply_temperature == 180.0
    assert stream.target_temperature == 80.0
    assert stream.cp == 2.5
    assert stream.heat_capacity_flow == 2.5
    assert stream.name == "H1"  # Defaults to id


def test_valid_cold_stream():
    """Verify initialization and attributes of a valid cold stream."""
    stream = Stream(
        id="C1",
        type="cold",
        supply_temperature=30.0,
        target_temperature=150.0,
        cp=3.0,
    )
    assert stream.id == "C1"
    assert stream.type == "cold"
    assert stream.stream_type == "cold"
    assert stream.supply_temperature == 30.0
    assert stream.target_temperature == 150.0
    assert stream.cp == 3.0
    assert stream.heat_capacity_flow == 3.0
    assert stream.name == "C1"  # Defaults to id


def test_stream_explicit_name():
    """Verify that an explicit name is preserved."""
    stream = Stream(
        id="H2",
        name="Crude Feed Stream",
        type="hot",
        supply_temperature=200,
        target_temperature=100,
        cp=1.8,
    )
    assert stream.id == "H2"
    assert stream.name == "Crude Feed Stream"


@pytest.mark.parametrize("stream_type_input", ["hot", "HOT", "Hot", "  hot  "])
def test_stream_type_case_normalization_hot(stream_type_input):
    """Verify case-insensitivity and whitespace trimming for hot stream type."""
    stream = Stream(
        id="H1",
        type=stream_type_input,
        supply_temperature=180,
        target_temperature=80,
        cp=2.5,
    )
    assert stream.type == "hot"


@pytest.mark.parametrize("stream_type_input", ["cold", "COLD", "Cold", "  cold  "])
def test_stream_type_case_normalization_cold(stream_type_input):
    """Verify case-insensitivity and whitespace trimming for cold stream type."""
    stream = Stream(
        id="C1",
        type=stream_type_input,
        supply_temperature=30,
        target_temperature=150,
        cp=3.0,
    )
    assert stream.type == "cold"


# =====================================================================
# 2. Invalid Streams — Physical Temperature Direction
# =====================================================================


def test_hot_stream_supply_less_than_target():
    """A hot stream requires cooling (supply > target); supply < target must fail."""
    with pytest.raises(StreamValidationError, match="supply temperature .* must be greater than target temperature"):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=80.0,
            target_temperature=180.0,
            cp=2.5,
        )


def test_hot_stream_supply_equal_to_target():
    """A hot stream with supply == target must fail."""
    with pytest.raises(StreamValidationError, match="supply temperature .* must be greater than target temperature"):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=100.0,
            target_temperature=100.0,
            cp=2.5,
        )


def test_cold_stream_target_less_than_supply():
    """A cold stream requires heating (target > supply); target < supply must fail."""
    with pytest.raises(StreamValidationError, match="target temperature .* must be greater than supply temperature"):
        Stream(
            id="C_err",
            type="cold",
            supply_temperature=150.0,
            target_temperature=30.0,
            cp=3.0,
        )


def test_cold_stream_target_equal_to_supply():
    """A cold stream with target == supply must fail."""
    with pytest.raises(StreamValidationError, match="target temperature .* must be greater than supply temperature"):
        Stream(
            id="C_err",
            type="cold",
            supply_temperature=50.0,
            target_temperature=50.0,
            cp=3.0,
        )


# =====================================================================
# 3. Invalid Streams — CP Validation
# =====================================================================


def test_zero_cp():
    """Zero CP must be rejected."""
    with pytest.raises(StreamValidationError, match="CP must be greater than zero"):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=180,
            target_temperature=80,
            cp=0.0,
        )


def test_negative_cp():
    """Negative CP must be rejected."""
    with pytest.raises(StreamValidationError, match="CP must be greater than zero"):
        Stream(
            id="C_err",
            type="cold",
            supply_temperature=30,
            target_temperature=150,
            cp=-2.5,
        )


@pytest.mark.parametrize("invalid_cp", ["3.0", "abc", None, [1.0], True, False])
def test_non_numeric_cp(invalid_cp):
    """Non-numeric or boolean CP must be rejected with TypeError / StreamTypeError."""
    with pytest.raises((TypeError, StreamTypeError)):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=180,
            target_temperature=80,
            cp=invalid_cp,
        )


@pytest.mark.parametrize("non_finite_cp", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_cp(non_finite_cp):
    """NaN and infinite CP values must be rejected."""
    with pytest.raises(StreamValidationError, match="CP must be a finite real number"):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=180,
            target_temperature=80,
            cp=non_finite_cp,
        )


# =====================================================================
# 4. Invalid Streams — Stream Type Validation
# =====================================================================


@pytest.mark.parametrize("invalid_type", ["warm", "ambient", "cool", "", "   "])
def test_invalid_stream_type_string(invalid_type):
    """Unknown stream type strings must raise StreamValidationError."""
    with pytest.raises(StreamValidationError, match="Stream type must be 'hot' or 'cold'"):
        Stream(
            id="S_err",
            type=invalid_type,
            supply_temperature=180,
            target_temperature=80,
            cp=2.5,
        )


@pytest.mark.parametrize("invalid_type", [123, None, True, ["hot"]])
def test_non_string_stream_type(invalid_type):
    """Non-string stream types must raise an error."""
    with pytest.raises((TypeError, StreamValidationError)):
        Stream(
            id="S_err",
            type=invalid_type,
            supply_temperature=180,
            target_temperature=80,
            cp=2.5,
        )


# =====================================================================
# 5. Invalid Streams — Temperatures Validation
# =====================================================================


@pytest.mark.parametrize("invalid_temp", ["180", "abc", None, [100], True, False])
def test_non_numeric_supply_temperature(invalid_temp):
    """Non-numeric supply temperature must be rejected."""
    with pytest.raises((TypeError, StreamTypeError)):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=invalid_temp,
            target_temperature=80,
            cp=2.5,
        )


@pytest.mark.parametrize("invalid_temp", ["80", "xyz", None, [50], True, False])
def test_non_numeric_target_temperature(invalid_temp):
    """Non-numeric target temperature must be rejected."""
    with pytest.raises((TypeError, StreamTypeError)):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=180,
            target_temperature=invalid_temp,
            cp=2.5,
        )


@pytest.mark.parametrize("non_finite_temp", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_temperature(non_finite_temp):
    """NaN or infinite temperature values must be rejected."""
    with pytest.raises(StreamValidationError, match="must be a finite real number"):
        Stream(
            id="H_err",
            type="hot",
            supply_temperature=non_finite_temp,
            target_temperature=80,
            cp=2.5,
        )


# =====================================================================
# 6. Stream ID Validation
# =====================================================================


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_empty_stream_id(empty_id):
    """Empty string ID must be rejected."""
    with pytest.raises(StreamValidationError, match="cannot be empty"):
        Stream(
            id=empty_id,
            type="hot",
            supply_temperature=180,
            target_temperature=80,
            cp=2.5,
        )


def test_none_stream_id():
    """None ID must be rejected."""
    with pytest.raises((TypeError, StreamTypeError)):
        Stream(
            id=None,
            type="hot",
            supply_temperature=180,
            target_temperature=80,
            cp=2.5,
        )


# =====================================================================
# 7. Calculations: ΔT and Heat Duty Q
# =====================================================================


def test_calculations_hot_stream():
    """Test temperature difference and heat duty calculation for hot stream."""
    stream = Stream(
        id="H1",
        type="hot",
        supply_temperature=180.0,
        target_temperature=80.0,
        cp=2.5,
    )
    # ΔT = supply - target = 180 - 80 = 100 °C
    assert stream.delta_t == 100.0
    assert stream.delta_T == 100.0
    assert stream.dt == 100.0

    # Q = CP * ΔT = 2.5 * 100 = 250.0 kW
    assert stream.heat_duty == 250.0
    assert stream.Q == 250.0
    assert stream.duty() == 250.0


def test_calculations_cold_stream():
    """Test temperature difference and heat duty calculation for cold stream."""
    stream = Stream(
        id="C1",
        type="cold",
        supply_temperature=30.0,
        target_temperature=150.0,
        cp=3.0,
    )
    # ΔT = target - supply = 150 - 30 = 120 °C
    assert stream.delta_t == 120.0
    assert stream.delta_T == 120.0
    assert stream.dt == 120.0

    # Q = CP * ΔT = 3.0 * 120 = 360.0 kW
    assert stream.heat_duty == 360.0
    assert stream.Q == 360.0
    assert stream.duty() == 360.0


# =====================================================================
# 8. Section 8 Example Test
# =====================================================================


def test_section_8_example():
    """Demonstrate the exact example from Section 8 of specification.

    hot:
      Supply = 180 °C, Target = 80 °C, CP = 2.5 kW/°C
      Expected: ΔT = 100 °C, Q = 250 kW

    cold:
      Supply = 30 °C, Target = 150 °C, CP = 3.0 kW/°C
      Expected: ΔT = 120 °C, Q = 360 kW
    """
    hot = Stream(
        id="H1",
        type="hot",
        supply_temperature=180,
        target_temperature=80,
        cp=2.5,
    )

    cold = Stream(
        id="C1",
        type="cold",
        supply_temperature=30,
        target_temperature=150,
        cp=3.0,
    )

    assert hot.delta_T == 100
    assert hot.Q == 250

    assert cold.delta_T == 120
    assert cold.Q == 360


# =====================================================================
# 9. Exception Inheritance Verification
# =====================================================================


def test_exception_inheritance():
    """Verify that StreamValidationError is a ValueError and StreamTypeError is a TypeError."""
    assert issubclass(StreamValidationError, ValueError)
    assert issubclass(StreamTypeError, TypeError)
    assert issubclass(StreamTypeError, StreamValidationError)
