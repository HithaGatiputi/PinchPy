from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets

def test_pinch_target_runs():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    result = utility_targets(streams, 10)
    assert "minimum_hot_utility" in result
    assert "minimum_cold_utility" in result
