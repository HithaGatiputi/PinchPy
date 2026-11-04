"""Command-Line Interface (CLI) for PinchPy (Phase 12)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models.stream import Stream
from .analysis.pinch_point import utility_targets
from .analysis.benchmarks import get_benchmark
from .synthesis.network_design import design_network
from .synthesis.network_evaluation import validate_network_pinch_rules
from .reporting.report import (
    generate_report,
    export_report_html,
    export_report_markdown,
    export_report_json,
)
from .utils.io import load_streams_from_csv


def _load_streams(args) -> list[Stream]:
    if getattr(args, "benchmark", None):
        bench = get_benchmark(args.benchmark)
        return bench["streams"]
    if getattr(args, "file", None):
        return load_streams_from_csv(args.file)
    # Default fallback
    bench = get_benchmark("kemp_2007_4stream")
    return bench["streams"]


def cmd_analyze(args):
    streams = _load_streams(args)
    dtmin = args.dtmin
    targets = utility_targets(streams, dtmin)

    print("==========================================================")
    print("                PINCH ANALYSIS TARGETS                    ")
    print("==========================================================")
    print(f"Streams Loaded:                 {len(streams)}")
    print(f"Minimum Approach Temp (dTmin):  {dtmin:.1f} °C")
    print(f"Pinch Hot Temperature:          {targets['pinch_hot_temperature']:.1f} °C")
    print(f"Pinch Cold Temperature:         {targets['pinch_cold_temperature']:.1f} °C")
    print(f"Minimum Hot Utility (QH,min):   {targets['minimum_hot_utility']:.2f} kW")
    print(f"Minimum Cold Utility (QC,min):  {targets['minimum_cold_utility']:.2f} kW")
    print(f"Maximum Heat Recovery:          {targets.get('heat_recovery', 0.0):.2f} kW")
    print("==========================================================")


def cmd_design(args):
    streams = _load_streams(args)
    dtmin = args.dtmin
    u = args.u
    net = design_network(streams, dtmin, overall_heat_transfer_coefficient=u)
    validation = validate_network_pinch_rules(net, streams, dtmin)

    print("==========================================================")
    print("            SYNTHESIZED HEAT EXCHANGER NETWORK            ")
    print("==========================================================")
    print(f"Total Exchangers: {len(net.exchangers)}")
    print(f"Energy Conserved: {'YES' if validation['energy_balance_conserved'] else 'NO'}")
    print(f"No Cross-Pinch:   {'YES' if validation['no_cross_pinch_transfer'] else 'NO'}")
    print(f"Approach Gaps OK: {'YES' if validation['delta_t_min_respected'] else 'NO'}")
    print("----------------------------------------------------------")
    print(f"{'Unit':<8} {'Type':<8} {'Hot':<8} {'Cold':<8} {'Duty (kW)':<10} {'Area (m2)':<10}")
    print("----------------------------------------------------------")
    for e in net.exchangers:
        etype = "Utility" if getattr(e, "is_utility", False) else "Process"
        area_str = f"{e.area:.2f}" if getattr(e, "area", None) is not None else "-"
        print(f"{e.name:<8} {etype:<8} {e.hot_stream:<8} {e.cold_stream:<8} {e.duty:<10.1f} {area_str:<10}")
    print("==========================================================")


def cmd_report(args):
    streams = _load_streams(args)
    dtmin = args.dtmin
    u = args.u
    out_path = Path(args.out)

    report_data = generate_report(streams, dtmin, overall_heat_transfer_coefficient=u)

    if out_path.suffix.lower() == ".html":
        content = export_report_html(report_data)
    elif out_path.suffix.lower() == ".json":
        content = export_report_json(report_data)
    else:  # default markdown
        content = export_report_markdown(report_data)

    out_path.write_text(content, encoding="utf-8")
    print(f"Report exported successfully to: {out_path.resolve()}")


def cmd_app(args):
    import subprocess
    app_file = Path(__file__).parent / "ui" / "app.py"
    print(f"Launching PinchPy Streamlit App at: {app_file.resolve()}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])


def main():
    parser = argparse.ArgumentParser(
        prog="pinchpy",
        description="PinchPy: Cost-Aware, ML-Accelerated Heat Exchanger Network Design & Retrofit Tool",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Calculate pinch points and minimum utility targets")
    p_analyze.add_argument("--file", "-f", type=str, help="Path to streams CSV file")
    p_analyze.add_argument("--benchmark", "-b", type=str, default="kemp_2007_4stream", help="Literature benchmark name")
    p_analyze.add_argument("--dtmin", "-d", type=float, default=10.0, help="Minimum approach temperature (C)")
    p_analyze.set_defaults(func=cmd_analyze)

    # design
    p_design = subparsers.add_parser("design", help="Synthesize a heat exchanger network")
    p_design.add_argument("--file", "-f", type=str, help="Path to streams CSV file")
    p_design.add_argument("--benchmark", "-b", type=str, default="kemp_2007_4stream", help="Literature benchmark name")
    p_design.add_argument("--dtmin", "-d", type=float, default=10.0, help="Minimum approach temperature (C)")
    p_design.add_argument("-u", "-U", type=float, default=0.5, help="Overall heat transfer coefficient U (kW/m2*C)")
    p_design.set_defaults(func=cmd_design)

    # report
    p_report = subparsers.add_parser("report", help="Generate and export an audit report (html, md, or json)")
    p_report.add_argument("--file", "-f", type=str, help="Path to streams CSV file")
    p_report.add_argument("--benchmark", "-b", type=str, default="kemp_2007_4stream", help="Literature benchmark name")
    p_report.add_argument("--dtmin", "-d", type=float, default=10.0, help="Minimum approach temperature (C)")
    p_report.add_argument("-u", "-U", type=float, default=0.5, help="Overall heat transfer coefficient U (kW/m2*C)")
    p_report.add_argument("--out", "-o", type=str, default="pinchpy_report.html", help="Output report file path")
    p_report.set_defaults(func=cmd_report)

    # app
    p_app = subparsers.add_parser("app", help="Launch interactive Streamlit web application")
    p_app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    if getattr(args, "func", None):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

