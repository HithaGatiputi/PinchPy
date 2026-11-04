"""Synthesis of Heat Exchanger Networks (HEN)."""

from .network_design import design_network
from .network_evaluation import evaluate_network, validate_network_pinch_rules
from .pinch_split import split_streams_at_pinch
from .exchanger_matching import cp_rule_satisfied, feasible_pair

__all__ = [
    "design_network",
    "evaluate_network",
    "validate_network_pinch_rules",
    "split_streams_at_pinch",
    "cp_rule_satisfied",
    "feasible_pair",
]

