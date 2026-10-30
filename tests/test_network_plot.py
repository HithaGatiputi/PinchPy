import matplotlib

matplotlib.use("Agg")

from pinchpy.models.stream import Stream
from pinchpy.synthesis.network_design import design_network
from pinchpy.visualization.network_plot import plot_network


def test_plot_network_runs_without_error():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    fig, ax = plot_network(streams, network)
    assert fig is not None
    assert ax is not None


def test_plot_network_empty_streams_shows_note():
    fig, ax = plot_network([], None)
    assert ax.texts


def test_plot_network_handles_split_streams():
    streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    fig, ax = plot_network(streams, network)
    assert fig is not None
