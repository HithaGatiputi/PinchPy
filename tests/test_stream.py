from pinchpy.models.stream import Stream

def test_stream_duty():
    s = Stream("H1", "hot", 180, 80, 2.5)
    assert s.duty() == 250
