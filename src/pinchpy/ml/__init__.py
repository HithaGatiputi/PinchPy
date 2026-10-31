"""Machine learning surrogate and OOD guardrail package for PinchPy."""

from .dataset import (
    HEN_PREDICTION_TARGETS,
    HEN_TARGET_COLUMNS,
    generate_case,
    generate_dataset,
    generate_hen_case,
    generate_hen_dataset,
    save_dataset,
    split_dataset,
)
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
from .predict import (
    FastANNPredictor,
    predict_ann_fast,
    predict_batch_network_targets,
    predict_hen_network_targets,
    predict_network_targets,
    predict_targets,
    tac_from_predictions,
)
from .train import (
    LogHenTargetTransformer,
    build_ann,
    build_hen_ann,
    load_artifacts,
    save_artifacts,
    train_ann,
    train_baselines,
    train_hen_ann,
    train_mlp,
)

__all__ = [
    "case_features",
    "feature_names",
    "stream_features",
    "generate_case",
    "generate_dataset",
    "generate_hen_case",
    "generate_hen_dataset",
    "HEN_TARGET_COLUMNS",
    "HEN_PREDICTION_TARGETS",
    "save_dataset",
    "split_dataset",
    "OODGuardrail",
    "build_ann",
    "build_hen_ann",
    "train_ann",
    "train_hen_ann",
    "train_mlp",
    "train_baselines",
    "LogHenTargetTransformer",
    "save_artifacts",
    "load_artifacts",
    "evaluate_regression",
    "evaluate_area_ranges",
    "evaluate_ood_categories",
    "compare_models",
    "evaluate_ood_detection",
    "benchmark_latency",
    "benchmark_batch_screening",
    "FastANNPredictor",
    "predict_ann_fast",
    "predict_targets",
    "predict_network_targets",
    "predict_batch_network_targets",
    "predict_hen_network_targets",
    "tac_from_predictions",
]

