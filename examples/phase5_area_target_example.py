"""Phase 5 demo: minimum heat-transfer area target from the composite curves."""

from pinchpy.analysis.area_target import area_target
from pinchpy.models.stream import Stream

U = 0.1  # overall heat-transfer coefficient, kW/(m²·°C)

CASES = {
    "two_stream": [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ],
    "four_stream_no_pinch": [
        Stream("H1", "hot", 200, 100, 2.0),
        Stream("H2", "hot", 150, 60, 1.0),
        Stream("C1", "cold", 30, 120, 1.5),
        Stream("C2", "cold", 50, 90, 1.0),
    ],
}


def print_report(name: str, streams, delta_t_min: float) -> None:
    result = area_target(streams, delta_t_min, U)
    print(f"\n=== {name} (ΔTmin = {delta_t_min:g} °C, U = {U:g} kW/m²/°C) ===")
    print(f"{'H from':>8} {'H to':>8} {'duty':>8} {'ΔT cold end':>12} "
          f"{'ΔT hot end':>11} {'LMTD':>8} {'area m²':>9}")
    for strip in result["strips"]:
        print(f"{strip['h_start']:8.1f} {strip['h_end']:8.1f} {strip['duty']:8.1f} "
              f"{strip['delta_t_cold_end']:12.2f} {strip['delta_t_hot_end']:11.2f} "
              f"{strip['lmtd']:8.2f} {strip['area']:9.2f}")
    print(f"Heat recovered : {result['heat_recovery']:.1f} kW")
    print(f"Closest approach: {result['minimum_approach']:.2f} °C")
    print(f"Minimum area target: {result['total_area']:.2f} m²")


def main() -> None:
    for name, streams in CASES.items():
        print_report(name, streams, delta_t_min=10)


if __name__ == "__main__":
    main()