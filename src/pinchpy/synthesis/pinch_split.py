"""Splits streams into above-pinch and below-pinch sub-problems (Phase 6).

Heat must never be designed to cross the pinch -- doing so would reintroduce
utility beyond the Phase 3 minimum targets. The standard pinch design method
therefore treats everything above the pinch and everything below the pinch as
two independent matching problems. A stream whose real temperature range
spans the pinch is split into an above-pinch piece and a below-pinch piece,
each a proper ``Stream`` with its own share of the original duty.

Hot streams are split at ``pinch_hot_temperature`` and cold streams at
``pinch_cold_temperature`` (both already on the real, unshifted scale, as
returned by ``pinch_point.utility_targets``).
"""

from ..models.stream import Stream


def _split_stream(stream, pinch_temperature):
    """Return (above_piece, below_piece); either may be None if the stream
    lies entirely on one side of the pinch. "Above" always means hotter than
    ``pinch_temperature``, regardless of stream type."""
    hi, lo = max(stream.supply_temperature, stream.target_temperature), \
        min(stream.supply_temperature, stream.target_temperature)

    if lo >= pinch_temperature:
        return stream, None
    if hi <= pinch_temperature:
        return None, stream

    if stream.type == "hot":
        above = Stream(f"{stream.id}_above", "hot", stream.supply_temperature, pinch_temperature, stream.cp)
        below = Stream(f"{stream.id}_below", "hot", pinch_temperature, stream.target_temperature, stream.cp)
    else:
        above = Stream(f"{stream.id}_above", "cold", pinch_temperature, stream.target_temperature, stream.cp)
        below = Stream(f"{stream.id}_below", "cold", stream.supply_temperature, pinch_temperature, stream.cp)
    return above, below


def split_streams_at_pinch(streams, pinch_hot_temperature, pinch_cold_temperature):
    """Split every stream into its above-pinch and below-pinch pieces.

    Returns ``(above_streams, below_streams)``. Hot streams split at
    ``pinch_hot_temperature``, cold streams at ``pinch_cold_temperature``.
    Streams entirely on one side contribute only to that side, unchanged.
    """
    above_streams = []
    below_streams = []
    for s in streams:
        pinch_temperature = pinch_hot_temperature if s.type == "hot" else pinch_cold_temperature
        above, below = _split_stream(s, pinch_temperature)
        if above is not None:
            above_streams.append(above)
        if below is not None:
            below_streams.append(below)
    return above_streams, below_streams
