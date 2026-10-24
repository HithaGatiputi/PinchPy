import matplotlib

matplotlib.use("Agg")  # non-interactive backend: tests never open a window

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")