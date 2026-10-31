"""Comprehensive evaluation metrics, baseline comparisons, and latency benchmarking."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..analysis.area_target import area_target
from ..analysis.pinch_point import utility_targets
from .features import case_features
from .guardrail import OODGuardrail


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, min_threshold: float = 1.0) -> float:
    """Compute Mean Absolute Percentage Error on samples where y_true >= threshold."""
    y_true_flat = np.asarray(y_true, dtype=float).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true_flat) >= min_threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100.0)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] = ("A_target", "Q_H_min", "Q_C_min"),
) -> dict[str, Any]:
    """Calculate multi-target regression metrics: MAE, RMSE, R², and MAPE.

    Parameters
    ----------
    y_true : array-like
        True ground-truth target values.
    y_pred : array-like
        Predicted target values.
    target_names : tuple of str
        Names of the targets.

    Returns
    -------
    dict[str, Any]
        Dictionary with overall and per-target metrics.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    overall_mae = float(mean_absolute_error(yt, yp))
    overall_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    overall_r2 = float(r2_score(yt, yp))

    per_target = {}
    for i, name in enumerate(target_names[:yt.shape[1]]):
        target_yt = yt[:, i]
        target_yp = yp[:, i]
        per_target[name] = {
            "mae": float(mean_absolute_error(target_yt, target_yp)),
            "rmse": float(np.sqrt(mean_squared_error(target_yt, target_yp))),
            "r2": float(r2_score(target_yt, target_yp)),
            "mape_percent": calculate_mape(target_yt, target_yp),
        }

    return {
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": overall_r2,
        "per_target": per_target,
        # Backward-compatible top-level keys:
        "mae": overall_mae,
        "rmse": overall_rmse,
        "r2": overall_r2,
    }


def compare_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: tuple[str, ...] = ("A_target", "Q_H_min", "Q_C_min"),
) -> dict[str, dict[str, Any]]:
    """Compare multiple models against the ground-truth test partition.

    Returns
    -------
    dict[str, dict[str, Any]]
        Metrics dictionary for each evaluated model name.
    """
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = evaluate_regression(y_test, y_pred, target_names)
    return results


def evaluate_ood_detection(
    guardrail: OODGuardrail,
    in_dist_X: np.ndarray,
    ood_X: np.ndarray,
) -> dict[str, float]:
    """Evaluate OOD detector accuracy, false-alarm rate, and detection recall.

    Parameters
    ----------
    guardrail : OODGuardrail
        Fitted guardrail instance.
    in_dist_X : np.ndarray
        In-distribution test samples (should NOT be flagged as OOD).
    ood_X : np.ndarray
        Out-of-distribution anomaly samples (SHOULD be flagged as OOD).

    Returns
    -------
    dict[str, float]
        Metrics: false_positive_rate, true_positive_rate (recall), accuracy.
    """
    in_flags = [guardrail.check_input(x)[0] for x in in_dist_X]
    ood_flags = [guardrail.check_input(x)[0] for x in ood_X]

    # False positive: in-distribution flagged as OOD
    fpr = float(np.mean(in_flags)) if len(in_flags) > 0 else 0.0
    # True positive: actual OOD flagged as OOD
    tpr = float(np.mean(ood_flags)) if len(ood_flags) > 0 else 0.0

    total_correct = sum(not f for f in in_flags) + sum(f for f in ood_flags)
    total_samples = len(in_flags) + len(ood_flags)
    accuracy = float(total_correct / total_samples) if total_samples > 0 else 0.0

    return {
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "accuracy": accuracy,
        "in_distribution_fallback_frequency": fpr,
        "ood_fallback_frequency": tpr,
    }


def evaluate_area_ranges(
    y_true_area: np.ndarray,
    y_pred_area: np.ndarray,
    bins: list[float] | None = None,
    bin_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute error metrics broken down across different heat exchanger area ranges.

    Parameters
    ----------
    y_true_area : np.ndarray
        True ground-truth area target values (m²).
    y_pred_area : np.ndarray
        Predicted area target values (m²).
    bins : list of float, optional
        Boundary thresholds for area bins (default: [0, 50, 150, 300, inf]).
    bin_labels : list of str, optional
        Descriptive names for each bin range.

    Returns
    -------
    list of dict
        Metrics per bin (bin_label, n_samples, mae, rmse, mape).
    """
    if bins is None:
        bins = [0.0, 50.0, 150.0, 300.0, float("inf")]
    if bin_labels is None:
        bin_labels = ["0 - 50 m²", "50 - 150 m²", "150 - 300 m²", "> 300 m²"]

    yt = np.asarray(y_true_area, dtype=float).ravel()
    yp = np.asarray(y_pred_area, dtype=float).ravel()

    results = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (yt >= low) & (yt < high)
        n_samples = int(np.sum(mask))
        if n_samples > 0:
            b_mae = float(mean_absolute_error(yt[mask], yp[mask]))
            b_rmse = float(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
            b_mape = float(calculate_mape(yt[mask], yp[mask], min_threshold=1.0))
        else:
            b_mae, b_rmse, b_mape = 0.0, 0.0, 0.0

        results.append({
            "range": bin_labels[i],
            "n_samples": n_samples,
            "mae": b_mae,
            "rmse": b_rmse,
            "mape_percent": b_mape,
        })

    return results


def evaluate_ood_categories(
    guardrail: OODGuardrail,
    in_dist_X: np.ndarray,
    categories: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate OOD detector across granular anomaly categories and nominal test data.

    Parameters
    ----------
    guardrail : OODGuardrail
        Fitted guardrail instance.
    in_dist_X : np.ndarray
        In-distribution test samples (negative class, nominal design cases).
    categories : dict of str -> np.ndarray
        Named categories of out-of-distribution anomaly cases (positive class).

    Returns
    -------
    dict[str, Any]
        Per-category metrics, detection tier breakdown, and global confusion matrix.
    """
    in_details = [guardrail.check_input(x) for x in in_dist_X]
    in_flags = [d[0] for d in in_details]
    FP = int(sum(in_flags))
    TN = int(len(in_flags) - FP)
    fpr = float(FP / len(in_flags)) if len(in_flags) > 0 else 0.0

    category_results = {}
    total_tp = 0
    total_ood = 0

    for name, X_cat in categories.items():
        n_samples = len(X_cat)
        total_ood += n_samples
        details = [guardrail.check_input(x) for x in X_cat]
        flags = [d[0] for d in details]
        tp_cat = int(sum(flags))
        total_tp += tp_cat

        tier1_count = 0
        tier2_count = 0
        for is_ood, _, reason in details:
            if is_ood:
                r_str = str(reason).lower()
                if "outside" in r_str or "exceeds" in r_str or "non-physical" in r_str or "supported" in r_str:
                    tier1_count += 1
                else:
                    tier2_count += 1

        category_results[name] = {
            "n_samples": n_samples,
            "flagged": tp_cat,
            "recall": float(tp_cat / n_samples) if n_samples > 0 else 0.0,
            "tier1_bounds_caught": tier1_count,
            "tier2_density_caught": tier2_count,
        }

    TP = total_tp
    FN = total_ood - TP
    recall = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    accuracy = float((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "categories": category_results,
        "confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
        "false_positive_rate": fpr,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "f1_score": f1,
    }


def benchmark_latency(
    model: Any,
    streams: list[Any],
    delta_t_min: float = 10.0,
    overall_heat_transfer_coefficient: float = 0.1,
    guardrail: OODGuardrail | None = None,
    warmup_iterations: int = 20,
    n_rounds: int = 5,
    n_iterations: int = 100,
) -> dict[str, float]:
    """Measure inference latency of deterministic engine vs pure ANN vs guarded inference.

    Includes warm-up iterations and repeated measurement blocks to report mean and
    standard deviations across distinct timing boundaries.

    Returns
    -------
    dict[str, float]
        Average latencies in microseconds (µs), standard deviations, and speedup.
    """
    # 0. Warm-up
    for _ in range(warmup_iterations):
        _ = utility_targets(streams, delta_t_min)
        _ = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
        x_warm = case_features(streams, delta_t_min, overall_heat_transfer_coefficient).reshape(1, -1)
        _ = model.predict(x_warm)
        if guardrail is not None:
            _ = guardrail.check_input(x_warm)

    # 1. Deterministic baseline alone (area_target + utility_targets)
    det_times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter_ns()
        for _ in range(n_iterations):
            _ = utility_targets(streams, delta_t_min)
            _ = area_target(streams, delta_t_min, overall_heat_transfer_coefficient)
        t1 = time.perf_counter_ns()
        det_times.append((t1 - t0) / (n_iterations * 1000.0))

    # 2. Feature Extraction alone
    feat_times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter_ns()
        for _ in range(n_iterations):
            _ = case_features(streams, delta_t_min, overall_heat_transfer_coefficient)
        t1 = time.perf_counter_ns()
        feat_times.append((t1 - t0) / (n_iterations * 1000.0))

    # 3. Pure ANN inference alone (on pre-extracted features using NumPy fast path)
    from .predict import predict_ann_fast

    x_pre = case_features(streams, delta_t_min, overall_heat_transfer_coefficient).reshape(1, -1)
    ann_times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter_ns()
        for _ in range(n_iterations):
            _ = predict_ann_fast(model, x_pre)
        t1 = time.perf_counter_ns()
        ann_times.append((t1 - t0) / (n_iterations * 1000.0))

    # 4. OOD Detection alone (Tier 1 + Tier 2 + Tier 3)
    ood_times = []
    dummy_preds = np.array([120.0, 50.0, 80.0])
    total_hot = sum(float(s.duty()) for s in streams if s.stream_type == "hot")
    total_cold = sum(float(s.duty()) for s in streams if s.stream_type == "cold")
    if guardrail is not None:
        for _ in range(n_rounds):
            t0 = time.perf_counter_ns()
            for _ in range(n_iterations):
                _ = guardrail.check_input(x_pre)
                _ = guardrail.validate_physics(dummy_preds, total_hot, total_cold)
            t1 = time.perf_counter_ns()
            ood_times.append((t1 - t0) / (n_iterations * 1000.0))
    else:
        ood_times = [0.0]

    # 5. Complete Guarded Pipeline End-to-End
    from .predict import predict_network_targets
    guarded_times = []
    for _ in range(n_rounds):
        t0 = time.perf_counter_ns()
        for _ in range(n_iterations):
            _ = predict_network_targets(
                streams,
                delta_t_min=delta_t_min,
                overall_heat_transfer_coefficient=overall_heat_transfer_coefficient,
                model=model,
                guardrail=guardrail,
            )
        t1 = time.perf_counter_ns()
        guarded_times.append((t1 - t0) / (n_iterations * 1000.0))

    det_mean = float(np.mean(det_times))
    ann_mean = float(np.mean(ann_times))
    speedup = det_mean / ann_mean if ann_mean > 0 else 1.0

    return {
        "deterministic_latency_us": det_mean,
        "deterministic_latency_std_us": float(np.std(det_times)),
        "feature_extraction_us": float(np.mean(feat_times)),
        "feature_extraction_std_us": float(np.std(feat_times)),
        "ann_pure_latency_us": ann_mean,
        "ann_pure_latency_std_us": float(np.std(ann_times)),
        "ood_detection_us": float(np.mean(ood_times)),
        "ood_detection_std_us": float(np.std(ood_times)),
        "guarded_latency_us": float(np.mean(guarded_times)),
        "guarded_latency_std_us": float(np.std(guarded_times)),
        "single_case_speedup": float(speedup),
    }


def benchmark_batch_screening(
    model: Any,
    cases: list[tuple[list[Any], float, float]],
    guardrail: OODGuardrail | None = None,
    batch_sizes: list[int] | None = None,
    n_repetitions: int = 3,
) -> list[dict[str, Any]]:
    """Benchmark end-to-end vectorized batch screening against deterministic loop.

    Parameters
    ----------
    model : Pipeline
        Trained ANN surrogate model.
    cases : list of tuple
        Collection of (streams, delta_t_min, U) problem cases.
    guardrail : OODGuardrail, optional
        Fitted guardrail for batch anomaly and physics validation.
    batch_sizes : list of int, optional
        Batch sizes to test (default: [10, 50, 100, 500, 1000]).
    n_repetitions : int, default 3
        Number of timing repetitions per batch size.

    Returns
    -------
    list of dict
        Comparison results for each batch size (times, per-case latency, speedup).
    """
    from .predict import predict_batch_network_targets

    if batch_sizes is None:
        batch_sizes = [10, 50, 100, 500, 1000]

    # Warm-up
    warm_cases = cases[:min(20, len(cases))]
    _ = predict_batch_network_targets(warm_cases, model=None)
    _ = predict_batch_network_targets(warm_cases, model=model, guardrail=guardrail)

    results = []
    for n in batch_sizes:
        if n > len(cases):
            continue
        c = cases[:n]

        # 1. Deterministic Loop
        det_times = []
        for _ in range(n_repetitions):
            t0 = time.perf_counter_ns()
            _ = predict_batch_network_targets(c, model=None)
            det_times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
        t_det_ms = float(np.median(det_times))

        # 2. Vectorized Batch ANN Pipeline
        batch_times = []
        for _ in range(n_repetitions):
            t0 = time.perf_counter_ns()
            batch_res = predict_batch_network_targets(c, model=model, guardrail=guardrail)
            batch_times.append((time.perf_counter_ns() - t0) / 1e6)  # ms
        t_batch_ms = float(np.median(batch_times))

        fallback_count = sum(1 for r in batch_res if r.get("source") == "deterministic_fallback")
        speedup = t_det_ms / t_batch_ms if t_batch_ms > 0 else 0.0

        results.append({
            "batch_size": n,
            "deterministic_time_ms": t_det_ms,
            "deterministic_us_per_case": (t_det_ms / n) * 1000.0,
            "batch_ann_time_ms": t_batch_ms,
            "batch_ann_us_per_case": (t_batch_ms / n) * 1000.0,
            "fallback_count": fallback_count,
            "fallback_percentage": (fallback_count / n) * 100.0,
            "speedup_factor": speedup,
        })

    return results

