"""I/O utilities for PinchPy, including benchmark problem file parsers."""
from __future__ import annotations

import re
from pathlib import Path

from ..models.stream import Stream


def load_furman_sahinidis_dat(filepath: str | Path) -> tuple[list[Stream], float]:
    """Parse a Furman-Sahinidis .dat benchmark problem file.

    Format
    ------
    Free-text header lines, followed by:
    - 'DTmin <value>'
    - 'HSn Tin Tout CP' (hot process streams)
    - 'CSn Tin Tout CP' (cold process streams)
    - 'HUn ...' / 'CUn ...' (utility streams, ignored)
    Any other lines or comments are skipped.

    Parameters
    ----------
    filepath : str or Path
        Path to the .dat file.

    Returns
    -------
    streams : list of Stream
        Parsed hot and cold process streams.
    delta_t_min : float
        Minimum approach temperature specified in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If DTmin is not specified or streams cannot be parsed.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    streams: list[Stream] = []
    delta_t_min: float | None = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            tag = parts[0]

            if tag.lower() == "dtmin":
                if len(parts) >= 2:
                    delta_t_min = float(parts[1])
                continue

            # Process Hot Stream: HSn Tin Tout CP
            if re.match(r"^HS\d+", tag, re.IGNORECASE):
                if len(parts) >= 4:
                    stream_id = parts[0]
                    tin = float(parts[1])
                    tout = float(parts[2])
                    cp = float(parts[3])
                    streams.append(
                        Stream(
                            id=stream_id,
                            type="hot",
                            supply_temperature=tin,
                            target_temperature=tout,
                            cp=cp,
                        )
                    )
                continue

            # Process Cold Stream: CSn Tin Tout CP
            if re.match(r"^CS\d+", tag, re.IGNORECASE):
                if len(parts) >= 4:
                    stream_id = parts[0]
                    tin = float(parts[1])
                    tout = float(parts[2])
                    cp = float(parts[3])
                    streams.append(
                        Stream(
                            id=stream_id,
                            type="cold",
                            supply_temperature=tin,
                            target_temperature=tout,
                            cp=cp,
                        )
                    )
                continue

            # Utility streams (HUn, CUn) or other metadata: ignore

    if delta_t_min is None:
        raise ValueError(f"No DTmin definition found in {path}")

    return streams, delta_t_min


# Backward-compatible alias
load_dat_file = load_furman_sahinidis_dat
