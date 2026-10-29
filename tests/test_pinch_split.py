from pinchpy.models.stream import Stream
from pinchpy.synthesis.pinch_split import split_streams_at_pinch


def test_split_stream_entirely_above_pinch():
    hot = Stream("H1", "hot", 300, 250, 2.0)
    above, below = split_streams_at_pinch([hot], pinch_hot_temperature=200, pinch_cold_temperature=190)
    assert len(above) == 1 and above[0].id == "H1"
    assert below == []


def test_split_stream_entirely_below_pinch():
    cold = Stream("C1", "cold", 20, 60, 1.0)
    above, below = split_streams_at_pinch([cold], pinch_hot_temperature=200, pinch_cold_temperature=190)
    assert above == []
    assert len(below) == 1 and below[0].id == "C1"


def test_split_hot_stream_spanning_the_pinch():
    hot = Stream("H1", "hot", 250, 50, 2.0)
    above, below = split_streams_at_pinch([hot], pinch_hot_temperature=150, pinch_cold_temperature=140)
    assert len(above) == 1 and len(below) == 1
    assert above[0].supply_temperature == 250
    assert above[0].target_temperature == 150
    assert below[0].supply_temperature == 150
    assert below[0].target_temperature == 50


def test_split_cold_stream_spanning_the_pinch():
    cold = Stream("C1", "cold", 20, 200, 1.0)
    above, below = split_streams_at_pinch([cold], pinch_hot_temperature=150, pinch_cold_temperature=140)
    assert len(above) == 1 and len(below) == 1
    assert above[0].supply_temperature == 140
    assert above[0].target_temperature == 200
    assert below[0].supply_temperature == 20
    assert below[0].target_temperature == 140


def test_split_preserves_total_duty():
    hot = Stream("H1", "hot", 250, 50, 2.0)
    above, below = split_streams_at_pinch([hot], pinch_hot_temperature=150, pinch_cold_temperature=140)
    assert above[0].duty() + below[0].duty() == hot.duty()
