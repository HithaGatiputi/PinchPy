"""Minimal example demonstrating PinchPy Phase 1 Process Stream model and calculations."""

from pinchpy.models.stream import Stream


def main() -> None:
    # Define a hot stream (requires cooling)
    hot = Stream(
        id="H1",
        type="hot",
        supply_temperature=180,
        target_temperature=80,
        cp=2.5,
    )

    # Define a cold stream (requires heating)
    cold = Stream(
        id="C1",
        type="cold",
        supply_temperature=30,
        target_temperature=150,
        cp=3.0,
    )

    # Output calculations
    print(f"H1 ΔT = {hot.delta_T:.0f} °C")
    print(f"H1 Q = {hot.Q:.0f} kW")
    print()
    print(f"C1 ΔT = {cold.delta_T:.0f} °C")
    print(f"C1 Q = {cold.Q:.0f} kW")


if __name__ == "__main__":
    main()
