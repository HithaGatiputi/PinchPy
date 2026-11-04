"""Direct Streamlit App launcher for PinchPy without requiring pip install."""

import subprocess
import sys
from pathlib import Path

app_path = Path(__file__).resolve().parent / "src" / "pinchpy" / "ui" / "app.py"
print(f"Launching PinchPy Streamlit App: {app_path}")

subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

