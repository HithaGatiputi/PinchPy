import io
import json
import pytest
from pathlib import Path

from pinchpy.models.stream import Stream
from pinchpy.analysis.benchmarks import get_benchmark
from pinchpy.synthesis.network_design import design_network
from pinchpy.utils.io import (
    load_streams_from_csv,
    save_streams_to_csv,
    load_streams_from_json,
    save_streams_to_json,
)
from pinchpy.reporting.report import (
    generate_report,
    export_report_markdown,
    export_report_html,
    export_report_json,
)
from pinchpy.cli import cmd_analyze, cmd_design, cmd_report


# =============================================================================
# I/O Utilities Tests
# =============================================================================

def test_streams_csv_roundtrip():
    original = [
        Stream("H1", "hot", 180.0, 80.0, 2.0),
        Stream("C1", "cold", 20.0, 150.0, 1.5),
    ]
    buf = io.StringIO()
    save_streams_to_csv(original, buf)

    buf.seek(0)
    loaded = load_streams_from_csv(buf)

    assert len(loaded) == len(original)
    for orig, read in zip(original, loaded):
        assert orig.id == read.id
        assert orig.type == read.type
        assert orig.supply_temperature == pytest.approx(read.supply_temperature)
        assert orig.target_temperature == pytest.approx(read.target_temperature)
        assert orig.cp == pytest.approx(read.cp)


def test_streams_csv_header_synonyms_and_case():
    csv_text = """# Process stream data
STREAM_ID, TYPE, T_IN, T_OUT, MCP
H1, HOT, 200, 100, 2.5
C1, COLD, 30, 120, 3.0
"""
    streams = load_streams_from_csv(csv_text)
    assert len(streams) == 2
    assert streams[0].id == "H1"
    assert streams[0].type == "hot"
    assert streams[0].supply_temperature == 200.0
    assert streams[0].target_temperature == 100.0
    assert streams[0].cp == 2.5


def test_streams_csv_missing_columns_raises():
    csv_text = "id,supply_temperature,target_temperature\nH1,100,50"
    with pytest.raises(ValueError, match="missing required fields"):
        load_streams_from_csv(csv_text)


def test_streams_csv_invalid_numeric_raises():
    csv_text = "id,type,supply_temperature,target_temperature,cp\nH1,hot,NaN_text,50,2.0"
    with pytest.raises(ValueError, match="non-numeric"):
        load_streams_from_csv(csv_text)


def test_streams_json_roundtrip():
    original = [
        Stream("H1", "hot", 180.0, 80.0, 2.0),
        Stream("C1", "cold", 20.0, 150.0, 1.5),
    ]
    buf = io.StringIO()
    save_streams_to_json(original, buf)

    buf.seek(0)
    loaded = load_streams_from_json(buf)

    assert len(loaded) == len(original)
    assert loaded[0].id == "H1"
    assert loaded[1].id == "C1"


# =============================================================================
# Reporting Engine Tests
# =============================================================================

def test_generate_report_structure():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]

    report = generate_report(
        streams=streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.5,
    )

    assert "metadata" in report
    assert "streams" in report
    assert "pinch_targets" in report
    assert "equipment_schedule" in report
    assert "economics" in report
    assert "emissions" in report

    assert report["streams"]["count"] == len(streams)
    assert report["pinch_targets"]["pinch_hot_temperature"] == 90.0
    assert len(report["equipment_schedule"]) > 0
    assert report["economics"]["annual_operating_cost_usd_per_yr"] > 0
    assert report["emissions"]["annual_co2_reduction_tonnes"] > 0


def test_generate_report_invalid_delta_t_min():
    with pytest.raises(ValueError, match="positive"):
        generate_report(streams=[], delta_t_min=-5.0)


def test_export_report_markdown():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    report = generate_report(streams=streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5)

    md = export_report_markdown(report)
    assert "# PinchPy Engineering Analysis Report" in md
    assert "## 1. Executive Summary" in md
    assert "## 2. Process Stream Inventory" in md
    assert "## 3. Heat Exchanger Equipment Schedule" in md
    assert "H1" in md and "C1" in md


def test_export_report_html():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    report = generate_report(streams=streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5)

    html = export_report_html(report)
    assert "<!DOCTYPE html>" in html
    assert "<title>PinchPy HEN Analysis Report</title>" in html
    assert "Process Stream Inventory" in html
    assert "Heat Exchanger Network Equipment Schedule" in html


def test_export_report_json():
    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    report = generate_report(streams=streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5)

    json_str = export_report_json(report)
    parsed = json.loads(json_str)
    assert parsed["metadata"]["tool"] == "PinchPy"
    assert parsed["pinch_targets"]["pinch_hot_temperature"] == 90.0


# =============================================================================
# CLI Subcommand Execution Tests
# =============================================================================

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cli_analyze_command(capsys):
    args = MockArgs(benchmark="kemp_2007_4stream", dtmin=10.0, file=None)
    cmd_analyze(args)
    captured = capsys.readouterr().out
    assert "PINCH ANALYSIS TARGETS" in captured
    assert "QH,min" in captured
    assert "90.0 °C" in captured


def test_cli_design_command(capsys):
    args = MockArgs(benchmark="kemp_2007_4stream", dtmin=10.0, u=0.5, file=None)
    cmd_design(args)
    captured = capsys.readouterr().out
    assert "SYNTHESIZED HEAT EXCHANGER NETWORK" in captured
    assert "Energy Conserved: YES" in captured


def test_cli_report_command(tmp_path):
    out_file = tmp_path / "test_report.html"
    args = MockArgs(benchmark="kemp_2007_4stream", dtmin=10.0, u=0.5, out=str(out_file), file=None)
    cmd_report(args)
    assert out_file.exists()
    assert "<!DOCTYPE html>" in out_file.read_text(encoding="utf-8")

