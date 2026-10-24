"""Phase 4 demo: draw the composite curves and Grand Composite Curve."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from pinchpy.models.stream import Stream
from pinchpy.visualization.cascade_plot import plot_grand_composite_curve
from pinchpy.visualization.composite_curve import plot_composite_curves

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CASES = {
    "two_stream": [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ],
    "four_stream": [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, streams in CASES.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, _ = plot_composite_curves(streams, delta_t_min=10)
        for warning in caught:
            print(f"[{name}] WARNING: {warning.message}")
        fig.savefig(OUTPUT_DIR / f"{name}_composite_curves.png", dpi=150)

        fig, _ = plot_grand_composite_curve(streams, delta_t_min=10)
        fig.savefig(OUTPUT_DIR / f"{name}_grand_composite_curve.png", dpi=150)
        print(f"[{name}] saved plots to {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()