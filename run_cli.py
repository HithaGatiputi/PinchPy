"""Direct CLI launcher for PinchPy without requiring pip install."""

import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pinchpy.cli import main

if __name__ == "__main__":
    main()

