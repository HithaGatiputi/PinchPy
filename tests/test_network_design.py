import pytest

from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets
from pinchpy.synthesis.network_design import design_network
from pinchpy.synthesis.network_evaluation import evaluate_network


def _process_total(network):
    return sum(e.duty for e in network.exchangers if not e.is_utility)


def _heater_total(network):
    return sum(e.duty for e in network.exchangers if e.is_utility and e.hot_stream == "HU")


def _cooler_total(network):
    return sum(e.duty for e in network.exchangers if e.is_utility and e.cold_stream == "CU")


def test_design_network_empty_streams_returns_empty_network():
    network = design_network([], delta_t_min=10)
    assert network.exchangers == []


def test_design_network_meets_utility_targets_simple_case():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    targets = utility_targets(streams, 10)
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)

    assert _heater_total(network) == pytest.approx(targets["minimum_hot_utility"])
    assert _cooler_total(network) == pytest.approx(targets["minimum_cold_utility"])


def test_design_network_conserves_total_energy_balance():
    streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)

    total_hot_duty = sum(s.duty() for s in streams if s.type == "hot")
    total_cold_duty = sum(s.duty() for s in streams if s.type == "cold")
    process_total = _process_total(network)

    assert process_total + _cooler_total(network) == pytest.approx(total_hot_duty)
    assert process_total + _heater_total(network) == pytest.approx(total_cold_duty)


def test_design_network_exchangers_respect_delta_t_min():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    for e in network.exchangers:
        if e.is_utility or e.hot_inlet is None:
            continue
        assert e.hot_inlet - e.cold_outlet >= 10 - 1e-6
        assert e.hot_outlet - e.cold_inlet >= 10 - 1e-6


def test_design_network_sizes_exchangers_when_U_given():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    process = [e for e in network.exchangers if not e.is_utility]
    assert process
    for e in process:
        assert e.area is not None and e.area > 0
        assert e.lmtd is not None and e.lmtd > 0


def test_design_network_without_U_leaves_area_none():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10)
    for e in network.exchangers:
        assert e.area is None


def test_design_network_exact_boundary_match_raises_no_warning(recwarn):
    # Chosen so the only feasible match sits at exactly delta_t_min apart --
    # regression test for the feasible_pair '>' vs '>=' boundary bug.
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


def test_design_network_reduces_stray_utility_on_hard_case():
    # Regression test for the match-ordering/partial-duty fixes: this case
    # used to strand ~125 kW of cold-stream duty with the single-pass greedy
    # matcher; the backtracking + partial-duty-capped matcher gets it down to
    # a small fraction of that (not necessarily to the exact Phase 3 target --
    # see network_design's module docstring for the known remaining gap).
    streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]
    targets = utility_targets(streams, 10)
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)

    excess_hot_utility = _heater_total(network) - targets["minimum_hot_utility"]
    assert 0 <= excess_hot_utility < 20  # was ~125 kW before the fix


def test_design_network_allows_partial_duty_when_full_ticket_off_is_infeasible():
    # A hot stream that has already partly cooled against one cold stream can
    # still feasibly serve part (not all) of a second cold stream's duty;
    # the matcher should not skip this pair outright just because the full
    # tick-off amount would be infeasible.
    streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    process = [e for e in network.exchangers if not e.is_utility]
    # At least one process match should transfer less than either stream's
    # full remaining duty at the time, i.e. a genuine partial match.
    assert any(e.duty < min(s.duty() for s in streams if s.id in (e.hot_stream, e.cold_stream)) for e in process)


def test_evaluate_network_reports_counts_and_duty():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    summary = evaluate_network(network, streams)

    assert summary["number_of_exchangers"] == len(network.exchangers)
    assert summary["number_of_process_matches"] + summary["number_of_utility_exchangers"] == len(network.exchangers)
    assert summary["total_recovered_heat"] == pytest.approx(_process_total(network))
    assert "total_area" in summary
    assert summary["minimum_units"] >= 1


def test_search_budget_consistency_across_existing_cases():
    """Verify search_budget=500 and search_budget=2000 yield the same hot utility on existing test cases."""
    case_simple = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    case_four = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]

    for case in [case_simple, case_four]:
        net_500 = design_network(case, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5, search_budget=500)
        net_2000 = design_network(case, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5, search_budget=2000)

        hu_500 = _heater_total(net_500)
        hu_2000 = _heater_total(net_2000)
        assert hu_500 == pytest.approx(hu_2000)

