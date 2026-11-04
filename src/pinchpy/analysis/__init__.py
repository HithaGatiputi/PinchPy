"""Analysis module for PinchPy."""

from .heat_cascade import heat_cascade, interval_heat_balance
from .pinch_point import utility_targets, pinch_temperature, pinch_temperatures
from .temperature_intervals import build_intervals, shifted_temperatures, shifted_endpoints
from .benchmarks import BENCHMARK_CASES, get_benchmark

__all__ = [
    "heat_cascade",
    "interval_heat_balance",
    "utility_targets",
    "pinch_temperature",
    "pinch_temperatures",
    "build_intervals",
    "shifted_temperatures",
    "shifted_endpoints",
    "BENCHMARK_CASES",
    "get_benchmark",
]
