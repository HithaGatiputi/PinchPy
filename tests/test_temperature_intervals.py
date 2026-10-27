import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis.temperature_intervals import shifted_temperatures, build_intervals


def _streams():
    return [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]


def test_shifted_temperatures_applies_half_delta_t_min():
    temps = shifted_temperatures(_streams(), 10)
    assert temps == [175.0, 155.0, 75.0, 25.0]


def test_shifted_temperatures_sorted_descending_and_deduplicated():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("H2", "hot", 180, 80, 1.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    temps = shifted_temperatures(streams, 10)
    assert temps == sorted(temps, reverse=True)
    assert len(temps) == len(set(temps))


def test_build_intervals_are_contiguous_and_cover_full_range():
    temps = shifted_temperatures(_streams(), 10)
    intervals = build_intervals(_streams(), 10)

    assert intervals[0][0] == temps[0]
    assert intervals[-1][1] == temps[-1]
    for (high, low), (next_high, _next_low) in zip(intervals, intervals[1:]):
        assert low == next_high
    for high, low in intervals:
        assert high > low


def test_build_intervals_count_matches_breakpoints():
    temps = shifted_temperatures(_streams(), 10)
    intervals = build_intervals(_streams(), 10)
    assert len(intervals) == len(temps) - 1


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
def test_shifted_temperatures_rejects_non_positive_delta_t_min(bad_delta_t_min):
    with pytest.raises(ValueError):
        shifted_temperatures(_streams(), bad_delta_t_min)


@pytest.mark.parametrize("bad_delta_t_min", [0, -5])
def test_build_intervals_rejects_non_positive_delta_t_min(bad_delta_t_min):
    with pytest.raises(ValueError):
        build_intervals(_streams(), bad_delta_t_min)


def test_shifted_temperatures_empty_streams_returns_empty_list():
    assert shifted_temperatures([], 10) == []


def test_build_intervals_empty_streams_returns_empty_list():
    assert build_intervals([], 10) == []


def test_shifted_temperatures_single_stream_gives_two_endpoints():
    streams = [Stream("H1", "hot", 180, 80, 2.0)]
    temps = shifted_temperatures(streams, 10)
    assert temps == [175.0, 75.0]

    intervals = build_intervals(streams, 10)
    assert intervals == [(175.0, 75.0)]


def test_shifted_temperatures_hot_and_cold_collision_deduplicates():
    # Hot stream supply shifted down by 5 (180 - 5 = 175) lands on the same
    # shifted temperature as a cold stream target shifted up by 5 (170 + 5 = 175).
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 150, 170, 1.5),
    ]
    temps = shifted_temperatures(streams, 10)
    assert temps == sorted(set(temps), reverse=True)
    assert 175.0 in temps
    assert temps.count(175.0) == 1
    assert temps == [175.0, 155.0, 75.0]
