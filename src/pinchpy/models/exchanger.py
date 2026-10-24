from dataclasses import dataclass

@dataclass
class HeatExchanger:
    name: str
    hot_stream: str
    cold_stream: str
    duty: float
    delta_t_min: float
    overall_heat_transfer_coefficient: float | None = None
