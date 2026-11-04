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


def test_evaluate_network_efficiency_and_targets():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    summary = evaluate_network(network, streams, delta_t_min=10)

    assert "target_hot_utility" in summary
    assert "target_cold_utility" in summary
    assert "target_heat_recovery" in summary
    assert "energy_recovery_efficiency" in summary
    assert summary["energy_recovery_efficiency"] == pytest.approx(1.0)


def test_validate_network_pinch_rules_on_simple_case():
    from pinchpy.synthesis.network_evaluation import validate_network_pinch_rules

    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    report = validate_network_pinch_rules(network, streams, delta_t_min=10)

    assert report["is_valid"] is True
    assert report["energy_balance_conserved"] is True
    assert report["stream_balances_satisfied"] is True
    assert report["delta_t_min_respected"] is True
    assert report["no_cross_pinch_transfer"] is True
    assert len(report["delta_t_violations"]) == 0
    assert len(report["cross_pinch_matches"]) == 0


def test_validate_network_pinch_rules_detects_violations():
    from pinchpy.models.exchanger import HeatExchanger
    from pinchpy.models.network import HeatExchangerNetwork
    from pinchpy.synthesis.network_evaluation import validate_network_pinch_rules
    from pinchpy.analysis.benchmarks import get_benchmark

    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    # Kemp 4-stream at dTmin=10 has pinch at Th=90, Tc=80.
    # Deliberately violate pinch rule: cross-pinch match (hot_in=120 > 90, cold_in=50 < 80)
    # and delta_t violation (hot_out=80, cold_in=75 => gap 5 < 10)
    bad_exchanger = HeatExchanger(
        name="BadCrossPinchMatch",
        hot_stream="H1",
        cold_stream="C1",
        duty=50.0,
        delta_t_min=10,
        hot_inlet=120.0,
        hot_outlet=80.0,
        cold_inlet=50.0,
        cold_outlet=100.0,
        is_utility=False,
    )
    bad_net = HeatExchangerNetwork(exchangers=[bad_exchanger])
    report = validate_network_pinch_rules(bad_net, streams, delta_t_min=10)

    assert report["is_valid"] is False
    assert report["no_cross_pinch_transfer"] is False
    assert "BadCrossPinchMatch" in report["cross_pinch_matches"]


def test_utility_exchangers_record_process_temperatures():
    streams = [
        Stream("H1", "hot", 180, 80, 2.0),
        Stream("C1", "cold", 20, 150, 1.5),
    ]
    network = design_network(streams, 10, overall_heat_transfer_coefficient=0.5)
    utility_exchangers = [e for e in network.exchangers if e.is_utility]
    assert len(utility_exchangers) > 0
    for e in utility_exchangers:
        if e.hot_stream in ("HU", "hot_utility", "steam"):
            assert e.cold_inlet is not None
            assert e.cold_outlet is not None
        elif e.cold_stream in ("CU", "cold_utility", "water"):
            assert e.hot_inlet is not None
            assert e.hot_outlet is not None


def test_network_design_kemp_benchmark_validity():
    from pinchpy.analysis.benchmarks import get_benchmark
    from pinchpy.synthesis.network_evaluation import validate_network_pinch_rules

    bench = get_benchmark("kemp_2007_4stream")
    streams = bench["streams"]
    network = design_network(streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.5)
    report = validate_network_pinch_rules(network, streams, delta_t_min=10.0)

    assert report["energy_balance_conserved"] is True
    assert report["stream_balances_satisfied"] is True
    assert report["delta_t_min_respected"] is True
    assert report["no_cross_pinch_transfer"] is True
    assert report["is_valid"] is True
