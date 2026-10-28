from pinchpy.models.stream import Stream
from pinchpy.synthesis.exchanger_matching import feasible_pair, cp_rule_satisfied, feasible_match


def test_feasible_pair_true_when_gap_sufficient():
    hot = Stream("H1", "hot", 180, 80, 2.0)
    cold = Stream("C1", "cold", 20, 150, 1.5)
    assert feasible_pair(hot, cold, delta_t_min=10) is True


def test_feasible_pair_false_when_gap_insufficient():
    hot = Stream("H1", "hot", 100, 90, 1.0)
    cold = Stream("C1", "cold", 85, 95, 5.0)
    assert feasible_pair(hot, cold, delta_t_min=10) is False


def test_cp_rule_above_pinch_requires_hot_cp_not_greater():
    hot = Stream("H1", "hot", 180, 80, 2.0)
    cold_ge = Stream("C1", "cold", 20, 150, 2.0)
    cold_lt = Stream("C2", "cold", 20, 150, 1.0)
    assert cp_rule_satisfied(hot, cold_ge, "above") is True
    assert cp_rule_satisfied(hot, cold_lt, "above") is False


def test_cp_rule_below_pinch_requires_hot_cp_not_less():
    hot = Stream("H1", "hot", 180, 80, 2.0)
    cold_le = Stream("C1", "cold", 20, 150, 2.0)
    cold_gt = Stream("C2", "cold", 20, 150, 3.0)
    assert cp_rule_satisfied(hot, cold_le, "below") is True
    assert cp_rule_satisfied(hot, cold_gt, "below") is False


def test_cp_rule_skipped_when_region_is_none():
    hot = Stream("H1", "hot", 180, 80, 2.0)
    cold = Stream("C1", "cold", 20, 150, 1.0)
    assert cp_rule_satisfied(hot, cold, None) is True


def test_feasible_match_combines_both_rules():
    hot = Stream("H1", "hot", 180, 80, 2.0)
    cold = Stream("C1", "cold", 20, 150, 1.0)
    assert feasible_match(hot, cold, delta_t_min=10, region="above") is False
    assert feasible_match(hot, cold, delta_t_min=10, region="below") is True
