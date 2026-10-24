from dataclasses import dataclass
from typing import Literal

StreamType = Literal["hot", "cold"]

@dataclass
class Stream:
    name: str
    stream_type: StreamType
    supply_temperature: float
    target_temperature: float
    heat_capacity_flow: float

    def duty(self) -> float:
        return abs(self.heat_capacity_flow * (self.target_temperature - self.supply_temperature))
