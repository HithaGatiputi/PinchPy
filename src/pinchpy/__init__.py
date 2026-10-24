"""PinchPy: Python library for Pinch Analysis and Heat Exchanger Network analysis."""

from .models.stream import Stream, StreamTypeError, StreamValidationError, StreamType

__version__ = "0.1.0"

__all__ = [
    "Stream",
    "StreamType",
    "StreamValidationError",
    "StreamTypeError",
    "__version__",
]
