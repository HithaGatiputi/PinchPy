"""Process stream data model and physical validation for Pinch Analysis.

This module defines the core `Stream` class, representing thermal process
streams requiring heating or cooling without phase change (sensible heat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

StreamType = Literal["hot", "cold"]


class StreamValidationError(ValueError):
    """Exception raised when stream parameters violate physical or domain rules."""
    pass


class StreamTypeError(StreamValidationError, TypeError):
    """Exception raised when a stream parameter is of an invalid or non-numeric type."""
    pass


@dataclass
class Stream:
    """Represents a thermal process stream for Pinch Analysis.

    A process stream is any process fluid that must be heated or cooled
    between a defined supply (inlet) temperature and target (outlet) temperature.

    - **Hot stream**: Requires cooling (supply_temperature > target_temperature).
      Releases heat to the system.
    - **Cold stream**: Requires heating (target_temperature > supply_temperature).
      Absorbs heat from the system.

    Parameters
    ----------
    id : str
        Unique identifier for the process stream (e.g., 'H1', 'C1').
    type : Literal["hot", "cold"] | str
        Stream type: 'hot' or 'cold'. Case-insensitive ('HOT', 'Hot' normalize to 'hot').
    supply_temperature : float
        Inlet/supply temperature of the stream in °C.
    target_temperature : float
        Outlet/target temperature of the stream in °C.
    cp : float
        Heat-capacity flow rate (mass flow rate × specific heat capacity, m * Cp)
        in kW/°C (or kW/K). Must be strictly positive (> 0).
    name : str | None, optional
        Descriptive name or label for the stream. If omitted or empty,
        defaults to the string representation of `id`.

    Raises
    ------
    StreamTypeError
        If temperature or CP values are non-numeric or boolean, or type is not a string.
    StreamValidationError
        If stream type is neither 'hot' nor 'cold', temperatures/CP are non-finite,
        CP is non-positive (<= 0), or temperature direction is thermodynamically invalid.
    """

    id: str
    type: StreamType | str
    supply_temperature: float
    target_temperature: float
    cp: float
    name: str = ""

    def __init__(
        self,
        id: str,
        type: StreamType | str | None = None,
        supply_temperature: float = 0.0,
        target_temperature: float = 0.0,
        cp: float | None = None,
        name: str | None = None,
        *,
        stream_type: StreamType | str | None = None,
        heat_capacity_flow: float | None = None,
    ) -> None:
        # Pre-assign default values so dataclass __repr__ is safe if validation fails early
        self.id = ""
        self.name = ""
        self.type = ""
        self.supply_temperature = 0.0
        self.target_temperature = 0.0
        self.cp = 0.0

        # Determine actual type and cp allowing backwards-compatible keyword arguments
        resolved_type = type if type is not None else stream_type
        resolved_cp = cp if cp is not None else heat_capacity_flow

        # Validate ID
        if id is None:
            raise StreamTypeError("Stream 'id' is required and cannot be None.")
        id_str = str(id).strip()
        if not id_str:
            raise StreamValidationError("Stream 'id' cannot be empty.")
        self.id = id_str

        # Descriptive name defaults to id
        if name is None or str(name).strip() == "":
            self.name = self.id
        else:
            self.name = str(name).strip()

        # Validate Stream Type
        if resolved_type is None:
            raise StreamTypeError(
                "Stream 'type' is required and cannot be None (must be 'hot' or 'cold')."
            )
        if not isinstance(resolved_type, str):
            raise StreamTypeError(
                f"Stream 'type' must be a string ('hot' or 'cold'), got {type(resolved_type).__name__}."
            )
        normalized_type = resolved_type.strip().lower()
        if normalized_type not in ("hot", "cold"):
            raise StreamValidationError(
                f"Invalid stream type '{resolved_type}'. Stream type must be 'hot' or 'cold'."
            )
        self.type: StreamType = normalized_type  # type: ignore[assignment]

        # Validate CP
        if resolved_cp is None:
            raise StreamTypeError("Heat-capacity flow rate 'cp' is required and cannot be None.")
        if isinstance(resolved_cp, bool) or not isinstance(resolved_cp, (int, float)):
            raise StreamTypeError(
                f"CP must be a numeric value, got {type(resolved_cp).__name__} ({resolved_cp!r})."
            )
        if math.isnan(resolved_cp) or math.isinf(resolved_cp):
            raise StreamValidationError(
                f"CP must be a finite real number, got {resolved_cp}."
            )
        if resolved_cp <= 0:
            raise StreamValidationError(
                f"CP must be greater than zero, got {resolved_cp}."
            )
        self.cp = float(resolved_cp)

        # Validate Temperatures
        for param_name, temp_val in (
            ("Supply temperature", supply_temperature),
            ("Target temperature", target_temperature),
        ):
            if temp_val is None:
                raise StreamTypeError(f"{param_name} is required and cannot be None.")
            if isinstance(temp_val, bool) or not isinstance(temp_val, (int, float)):
                raise StreamTypeError(
                    f"{param_name} must be a numeric value, got {type(temp_val).__name__} ({temp_val!r})."
                )
            if math.isnan(temp_val) or math.isinf(temp_val):
                raise StreamValidationError(
                    f"{param_name} must be a finite real number, got {temp_val}."
                )

        self.supply_temperature = float(supply_temperature)
        self.target_temperature = float(target_temperature)

        # Physical direction validation
        if self.type == "hot":
            if self.supply_temperature <= self.target_temperature:
                raise StreamValidationError(
                    f"Invalid hot stream '{self.id}': supply temperature "
                    f"({self.supply_temperature} °C) must be greater than target temperature "
                    f"({self.target_temperature} °C)."
                )
        elif self.type == "cold":
            if self.target_temperature <= self.supply_temperature:
                raise StreamValidationError(
                    f"Invalid cold stream '{self.id}': target temperature "
                    f"({self.target_temperature} °C) must be greater than supply temperature "
                    f"({self.supply_temperature} °C)."
                )

    @property
    def delta_t(self) -> float:
        """Temperature change ΔT in °C.

        Always strictly positive for every valid stream:
        - Hot stream: supply_temperature - target_temperature
        - Cold stream: target_temperature - supply_temperature
        """
        if self.type == "hot":
            return self.supply_temperature - self.target_temperature
        return self.target_temperature - self.supply_temperature

    @property
    def delta_T(self) -> float:
        """Alias for delta_t (temperature change ΔT in °C)."""
        return self.delta_t

    @property
    def dt(self) -> float:
        """Shorthand alias for delta_t (temperature change ΔT in °C)."""
        return self.delta_t

    @property
    def heat_duty(self) -> float:
        """Sensible heat duty Q = CP × ΔT in kW.

        Assuming units:
        - CP in kW/°C (or kW/K)
        - ΔT in °C (or K)
        - Q in kW
        """
        return self.cp * self.delta_t

    @property
    def Q(self) -> float:
        """Sensible heat duty Q in kW (alias for heat_duty)."""
        return self.heat_duty

    def duty(self) -> float:
        """Return the sensible heat duty Q in kW."""
        return self.heat_duty

    @property
    def stream_type(self) -> StreamType:
        """Alias for stream type ('hot' or 'cold') for compatibility."""
        return self.type

    @property
    def heat_capacity_flow(self) -> float:
        """Alias for heat-capacity flow rate cp (kW/°C) for compatibility."""
        return self.cp
