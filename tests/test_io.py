"""Tests for benchmark problem loader and I/O utilities."""
from pathlib import Path
import pytest

from pinchpy.utils.io import load_furman_sahinidis_dat, load_dat_file


def test_load_sample_dat_file():
    sample_path = Path("data/benchmarks/furman_sahinidis/dat_files/4sp1.dat")
    assert sample_path.exists(), "Benchmark 4sp1.dat should exist"

    streams, dtmin = load_furman_sahinidis_dat(sample_path)

    assert dtmin == 10.0
    assert len(streams) == 4

    hot_streams = [s for s in streams if s.stream_type == "hot"]
    cold_streams = [s for s in streams if s.stream_type == "cold"]
    assert len(hot_streams) == 2
    assert len(cold_streams) == 2

    # Assert no utility streams were loaded
    stream_ids = [s.id for s in streams]
    assert "HS1" in stream_ids
    assert "HS2" in stream_ids
    assert "CS1" in stream_ids
    assert "CS2" in stream_ids
    assert "HU1" not in stream_ids
    assert "CU1" not in stream_ids

    # Check stream properties
    hs1 = next(s for s in streams if s.id == "HS1")
    assert hs1.supply_temperature == 320.0
    assert hs1.target_temperature == 200.0
    assert hs1.cp == 16.67


def test_load_dat_ignores_hu_cu_and_comments(tmp_path):
    dat_content = """Free-text header line 1
Free-text header line 2
DTmin 15.5
HS1 180 80 2.5
HU1 250 249 100
CS1 40 140 3.0
CU1 15 25 50
# Comment line
random string
"""
    file_path = tmp_path / "test_case.dat"
    file_path.write_text(dat_content, encoding="utf-8")

    streams, dtmin = load_dat_file(file_path)

    assert dtmin == 15.5
    assert len(streams) == 2
    assert [s.id for s in streams] == ["HS1", "CS1"]


def test_load_dat_missing_dtmin_raises(tmp_path):
    dat_content = """HS1 180 80 2.5
CS1 40 140 3.0
"""
    file_path = tmp_path / "missing_dtmin.dat"
    file_path.write_text(dat_content, encoding="utf-8")

    with pytest.raises(ValueError, match="No DTmin definition found"):
        load_furman_sahinidis_dat(file_path)


def test_load_dat_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_furman_sahinidis_dat("non_existent_file.dat")


def test_load_chen_grossmann_miller_dat_file():
    sample_path = Path("data/benchmarks/chen_grossmann_miller/dat_files/balanced5.dat")
    assert sample_path.exists(), "Benchmark balanced5.dat should exist"

    streams, dtmin = load_dat_file(sample_path)
    assert dtmin == 10.0
    assert len(streams) == 10

    hot_streams = [s for s in streams if s.stream_type == "hot"]
    cold_streams = [s for s in streams if s.stream_type == "cold"]
    assert len(hot_streams) == 5
    assert len(cold_streams) == 5

