"""Published literature benchmark cases for Pinch Analysis validation.

This module provides standard, peer-reviewed benchmark problems from:
1. Ian C. Kemp (2007), 'Pinch Analysis and Process Integration', Elsevier.
2. Gavin Towler & Ray Sinnott (2013) / Bodo Linnhoff et al. (1982),
   'Chemical Engineering Design', Butterworth-Heinemann.
3. Robin Smith (2005), 'Chemical Process Design and Integration', Wiley.
4. K. F. Lee, A. H. Masso & D. F. Rudd (1970) / S. A. Papoulias & I. E. Grossmann (1983),
   4SP1 Benchmark, Computers & Chemical Engineering.
"""

from __future__ import annotations

from typing import Any, Dict
from ..models.stream import Stream

BENCHMARK_CASES: Dict[str, Dict[str, Any]] = {
    "kemp_2007_4stream": {
        "title": "Kemp (2007) 4-Stream Process",
        "citation": "Ian C. Kemp, Pinch Analysis and Process Integration (2007), Tables 1.1, 2.2, 2.3",
        "streams": [
            Stream("H1", "hot", 170.0, 60.0, 3.0),
            Stream("H2", "hot", 150.0, 30.0, 1.5),
            Stream("C1", "cold", 20.0, 135.0, 2.0),
            Stream("C2", "cold", 80.0, 140.0, 4.0),
        ],
        "evaluations": {
            10.0: {
                "minimum_hot_utility": 20.0,
                "minimum_cold_utility": 60.0,
                "pinch_shifted_temperature": 85.0,
                "pinch_hot_temperature": 90.0,
                "pinch_cold_temperature": 80.0,
                "heat_recovery": 450.0,
                "is_threshold_problem": False,
            },
            20.0: {
                "minimum_hot_utility": 65.0,
                "minimum_cold_utility": 105.0,
                "pinch_shifted_temperature": 90.0,
                "pinch_hot_temperature": 100.0,
                "pinch_cold_temperature": 80.0,
                "heat_recovery": 405.0,
                "is_threshold_problem": False,
            },
        },
    },
    "towler_sinnott_linnhoff_4stream": {
        "title": "Towler & Sinnott / Linnhoff et al. 4-Stream Process",
        "citation": "Towler & Sinnott, Chemical Engineering Design; Linnhoff et al. (1982)",
        "streams": [
            Stream("H1", "hot", 180.0, 40.0, 40.0),
            Stream("H2", "hot", 150.0, 60.0, 30.0),
            Stream("C1", "cold", 30.0, 180.0, 60.0),
            Stream("C2", "cold", 80.0, 160.0, 20.0),
        ],
        "evaluations": {
            20.0: {
                "minimum_hot_utility": 2900.0,
                "minimum_cold_utility": 600.0,
                "pinch_shifted_temperature": 90.0,
                "pinch_hot_temperature": 100.0,
                "pinch_cold_temperature": 80.0,
                "heat_recovery": 7700.0,
                "is_threshold_problem": False,
            },
        },
    },
    "robin_smith_2stream": {
        "title": "Robin Smith (2005) 2-Stream Benchmark",
        "citation": "Robin Smith, Chemical Process Design and Integration (2005), Table 16.1",
        "streams": [
            Stream("H1", "hot", 160.0, 40.0, 100.0),
            Stream("C1", "cold", 40.0, 110.0, 200.0),
        ],
        "evaluations": {
            10.0: {
                "minimum_hot_utility": 3000.0,
                "minimum_cold_utility": 1000.0,
                "pinch_shifted_temperature": 45.0,
                "pinch_hot_temperature": 50.0,
                "pinch_cold_temperature": 40.0,
                "heat_recovery": 11000.0,
                "is_threshold_problem": False,
            },
        },
    },
    "lee_masso_rudd_4sp1": {
        "title": "Lee, Masso & Rudd (1970) / Papoulias & Grossmann (1983) 4SP1",
        "citation": "Lee et al. (1970) Ind. Eng. Chem. Fundam.; Papoulias & Grossmann (1983) Comp. & Chem. Eng.",
        "streams": [
            Stream("H1", "hot", 160.0, 93.0, 4.79),
            Stream("H2", "hot", 249.0, 138.0, 10.55),
            Stream("C1", "cold", 60.0, 160.0, 7.62),
            Stream("C2", "cold", 116.0, 260.0, 6.08),
        ],
        "evaluations": {
            20.0: {
                "minimum_hot_utility": 188.48,
                "minimum_cold_utility": 42.94,
                "pinch_shifted_temperature": 239.0,
                "pinch_hot_temperature": 249.0,
                "pinch_cold_temperature": 229.0,
                "heat_recovery": 1449.04,
                "is_threshold_problem": False,
            },
            10.0: {
                "minimum_hot_utility": 145.54,
                "minimum_cold_utility": 0.0,
                "pinch_shifted_temperature": 65.0,
                "pinch_hot_temperature": 70.0,
                "pinch_cold_temperature": 60.0,
                "heat_recovery": 1491.98,
                "is_threshold_problem": True,
            },
        },
    },
}


def get_benchmark(name: str) -> Dict[str, Any]:
    """Retrieve benchmark definition by key."""
    if name not in BENCHMARK_CASES:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available benchmarks: {list(BENCHMARK_CASES.keys())}"
        )
    return BENCHMARK_CASES[name]

