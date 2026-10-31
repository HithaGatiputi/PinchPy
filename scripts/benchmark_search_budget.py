"""Reproducible benchmark of design_network's search_budget trade-off.

Measures average stray utility duty and average run time per case across a
fixed pool of random multi-stream problems, for a few different
search_budget values, so the README's search_budget table can be regenerated
from a real measurement instead of hand-pasted numbers.

Usage: python scripts/benchmark_search_budget.py
"""
from __future__ import annotations

import random
import time
import warnings

from pinchpy.models.stream import Stream
from pinchpy.synthesis.network_design import design_network

N_CASES = 200
BUDGETS = [2000, 500, 200]


def _random_case(rng: random.Random, n_hot: int, n_cold: int) -> list[Stream]:
    streams = []
    for i in range(n_hot):
        supply = rng.uniform(130.0, 420.0)
        target = rng.uniform(35.0, supply - 15.0)
        cp = rng.uniform(0.5, 7.5)
        streams.append(Stream(f"H{i + 1}", "hot", round(supply, 1), round(target, 1), round(cp, 2)))
    for j in range(n_cold):
        supply = rng.uniform(20.0, 300.0)
        target = rng.uniform(supply + 15.0, supply + 300.0)
        cp = rng.uniform(0.5, 7.5)
        streams.append(Stream(f"C{j + 1}", "cold", round(supply, 1), round(target, 1), round(cp, 2)))
    return streams


def main() -> None:
    rng = random.Random(42)
    cases = [_random_case(rng, rng.randint(2, 5), rng.randint(2, 5)) for _ in range(N_CASES)]

    print(f"Benchmarking search_budget on {N_CASES} random 2-10 stream cases (seed=42)...")
    print("-" * 80)
    print(f"{'Budget':<10} | {'Avg stray utility (kW)':<24} | {'Avg time/case (ms)':<20} | {'Cases > target+1kW':<18}")
    print("-" * 80)

    # A fixed "effectively unbounded" reference to compute each case's true
    # Phase 3 utility target against, so "stray utility" is well-defined
    # without recomputing utility_targets separately here.
    from pinchpy.analysis.pinch_point import utility_targets

    targets_per_case = [utility_targets(case, 10) for case in cases]

    for budget in BUDGETS:
        total_stray = 0.0
        above_target = 0
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for case, targets in zip(cases, targets_per_case):
                net = design_network(case, 10, overall_heat_transfer_coefficient=0.5, search_budget=budget)
                heater = sum(e.duty for e in net.exchangers if e.is_utility and e.hot_stream == "HU")
                cooler = sum(e.duty for e in net.exchangers if e.is_utility and e.cold_stream == "CU")
                stray = (heater - targets["minimum_hot_utility"]) + (cooler - targets["minimum_cold_utility"])
                total_stray += max(stray, 0.0)
                if stray > 1.0:
                    above_target += 1
        elapsed = time.perf_counter() - t0

        avg_stray = total_stray / N_CASES
        avg_time_ms = (elapsed / N_CASES) * 1000.0
        print(f"{budget:<10} | {avg_stray:<24.2f} | {avg_time_ms:<20.3f} | {above_target}/{N_CASES}")

    print("-" * 80)


if __name__ == "__main__":
    main()
