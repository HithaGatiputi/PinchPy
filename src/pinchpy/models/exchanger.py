from dataclasses import dataclass

@dataclass
class HeatExchanger:
    name: str
    hot_stream: str
    cold_stream: str
    duty: float
    delta_t_min: float
    overall_heat_transfer_coefficient: float | None = None
    lmtd: float | None = None
    area: float | None = None
    hot_inlet: float | None = None
    hot_outlet: float | None = None
    cold_inlet: float | None = None
    cold_outlet: float | None = None
    is_utility: bool = False
