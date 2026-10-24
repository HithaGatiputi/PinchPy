from dataclasses import dataclass, field
from .exchanger import HeatExchanger

@dataclass
class HeatExchangerNetwork:
    exchangers: list[HeatExchanger] = field(default_factory=list)

    def total_recovered_heat(self) -> float:
        return sum(e.duty for e in self.exchangers)
