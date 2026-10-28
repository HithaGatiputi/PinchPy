"""Machine learning surrogate and OOD guardrail package for PinchPy."""

from .dataset import generate_case, generate_dataset, save_dataset, split_dataset
from .evaluate import (
    benchmark_batch_screening,
    benchmark_latency,
    compare_models,
    evaluate_area_ranges,
    evaluate_ood_categories,
    evaluate_ood_detection,
    evaluate_regression,
)
from .features import case_features, feature_names, stream_features
from .guardrail import OODGuardrail
from .predict import predict_batch_network_targets, predict_network_targets, predict_targets
from .train import (
    build_ann,
    load_artifacts,
    save_artifacts,
    train_ann,
    train_baselines,
    train_mlp,
)

__all__ = [
    "case_features",
    "feature_names",
    "stream_features",
    "generate_case",
    "generate_dataset",
    "save_dataset",
    "split_dataset",
    "OODGuardrail",
    "build_ann",
    "train_ann",
    "train_mlp",
    "train_baselines",
    "save_artifacts",
    "load_artifacts",
    "evaluate_regression",
    "evaluate_area_ranges",
    "evaluate_ood_categories",
    "compare_models",
    "evaluate_ood_detection",
    "benchmark_latency",
    "benchmark_batch_screening",
    "predict_targets",
    "predict_network_targets",
    "predict_batch_network_targets",
]

