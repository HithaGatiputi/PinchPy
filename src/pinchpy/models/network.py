from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from .exchanger import HeatExchanger


@dataclass
class HeatExchangerNetwork:
    """Represents a synthesized Heat Exchanger Network (HEN)."""

    exchangers: List[HeatExchanger] = field(default_factory=list)

    def total_recovered_heat(self) -> float:
        """Total heat exchanged by process-to-process exchangers."""
        process = [e for e in self.exchangers if not getattr(e, "is_utility", False)]
        return sum(e.duty for e in process) if process else sum(e.duty for e in self.exchangers)

    def total_utility_heat(self) -> float:
        """Total utility duty consumed (heaters + coolers)."""
        return sum(e.duty for e in self.exchangers if getattr(e, "is_utility", False))

    def total_heater_heat(self) -> float:
        """Total hot utility duty consumed by heaters."""
        return sum(e.duty for e in self.exchangers if getattr(e, "is_utility", False) and e.hot_stream in ("HU", "hot_utility", "steam"))

    def total_cooler_heat(self) -> float:
        """Total cold utility duty rejected to coolers."""
        return sum(e.duty for e in self.exchangers if getattr(e, "is_utility", False) and e.cold_stream in ("CU", "cold_utility", "water"))

    def total_area(self) -> Optional[float]:
        """Total heat-transfer area across all sized exchangers."""
        areas = [e.area for e in self.exchangers if e.area is not None]
        return sum(areas) if areas else None

    def __len__(self) -> int:
        return len(self.exchangers)
