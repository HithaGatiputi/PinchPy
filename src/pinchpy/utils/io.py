"""I/O utilities for loading and saving process streams and network configurations (Phase 12)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, TextIO, Union

from ..models.stream import Stream


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "_").replace("-", "_")


def load_streams_from_csv(source: Union[str, Path, TextIO]) -> List[Stream]:
    """Load a list of ``Stream`` objects from a CSV file path, buffer, or string.

    Recognized column names (case-insensitive):
    - Stream ID: 'id', 'name', 'stream', 'stream_id'
    - Stream Type: 'type', 'stream_type' ('hot' or 'cold')
    - Supply Temperature: 'supply_temperature', 'supply_temp', 'ts', 't_in', 'tin', 't_supply'
    - Target Temperature: 'target_temperature', 'target_temp', 'tt', 't_out', 'tout', 't_target'
    - Heat Capacity Flowrate: 'cp', 'mcp', 'heat_capacity_flowrate', 'heat_capacity'

    Returns
    -------
    list[Stream]
        Parsed stream objects.
    """
    if isinstance(source, (str, Path)) and "\n" not in str(source) and Path(source).exists():
        with open(source, mode="r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    elif isinstance(source, (str, Path)):
        lines = str(source).strip().splitlines()
    else:
        lines = source.readlines()

    # Filter out empty lines or comment lines
    clean_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    if not clean_lines:
        return []

    reader = csv.DictReader(clean_lines)
    if reader.fieldnames is None:
        return []

    # Map headers to canonical names
    canonical_map = {}
    for raw in reader.fieldnames:
        norm = _normalize_header(raw)
        if norm in ("id", "name", "stream", "stream_id"):
            canonical_map[raw] = "id"
        elif norm in ("type", "stream_type"):
            canonical_map[raw] = "type"
        elif norm in ("supply_temperature", "supply_temp", "ts", "t_in", "tin", "t_supply", "supply"):
            canonical_map[raw] = "ts"
        elif norm in ("target_temperature", "target_temp", "tt", "t_out", "tout", "t_target", "target"):
            canonical_map[raw] = "tt"
        elif norm in ("cp", "mcp", "heat_capacity_flowrate", "heat_capacity"):
            canonical_map[raw] = "cp"

    streams: List[Stream] = []
    for row_idx, row in enumerate(reader, start=1):
        parsed = {}
        for raw_k, v in row.items():
            if raw_k in canonical_map and v is not None and v.strip() != "":
                parsed[canonical_map[raw_k]] = v.strip()

        required = {"id", "type", "ts", "tt", "cp"}
        missing = required - set(parsed.keys())
        if missing:
            raise ValueError(
                f"Row {row_idx} missing required fields: {sorted(missing)}. "
                f"Row contents: {row}"
            )

        sid = parsed["id"]
        stype = parsed["type"].lower()
        try:
            ts = float(parsed["ts"])
            tt = float(parsed["tt"])
            cp = float(parsed["cp"])
        except ValueError as err:
            raise ValueError(f"Row {row_idx} contains non-numeric numerical value: {err}") from err

        streams.append(Stream(sid, stype, ts, tt, cp))

    return streams


def save_streams_to_csv(streams: List[Stream], target: Union[str, Path, TextIO]) -> None:
    """Save a list of ``Stream`` objects to a CSV file or buffer."""
    fieldnames = ["id", "type", "supply_temperature", "target_temperature", "cp", "duty"]

    def write_rows(writer):
        writer.writeheader()
        for s in streams:
            writer.writerow({
                "id": s.id,
                "type": s.type,
                "supply_temperature": s.supply_temperature,
                "target_temperature": s.target_temperature,
                "cp": s.cp,
                "duty": s.duty(),
            })

    if isinstance(target, (str, Path)):
        with open(target, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            write_rows(writer)
    else:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        write_rows(writer)


def load_streams_from_json(source: Union[str, Path, TextIO]) -> List[Stream]:
    """Load a list of ``Stream`` objects from JSON string or file."""
    if isinstance(source, (str, Path)) and "\n" not in str(source) and Path(source).exists():
        with open(source, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(source, (str, Path)):
        data = json.loads(str(source))
    else:
        data = json.load(source)

    if isinstance(data, dict) and "streams" in data:
        data = data["streams"]

    streams = []
    for item in data:
        streams.append(
            Stream(
                item["id"],
                item["type"],
                float(item["supply_temperature"]),
                float(item["target_temperature"]),
                float(item["cp"]),
            )
        )
    return streams


def save_streams_to_json(
    streams: List[Stream],
    target: Union[str, Path, TextIO],
    indent: int = 2,
) -> None:
    """Save a list of ``Stream`` objects to JSON file or buffer."""
    data = [
        {
            "id": s.id,
            "type": s.type,
            "supply_temperature": s.supply_temperature,
            "target_temperature": s.target_temperature,
            "cp": s.cp,
            "duty": s.duty(),
        }
        for s in streams
    ]
    if isinstance(target, (str, Path)):
        with open(target, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
    else:
        json.dump(data, target, indent=indent)
