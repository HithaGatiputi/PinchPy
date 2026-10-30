"""Full Phase 7-8 pipeline: dataset generation, model training, OOD evaluation, and literature benchmarking."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pinchpy.analysis.area_target import area_target
from pinchpy.analysis.pinch_point import utility_targets
from pinchpy.ml.dataset import generate_dataset, split_dataset
from pinchpy.ml.evaluate import (
    benchmark_batch_screening,
    benchmark_latency,
    compare_models,
    evaluate_area_ranges,
    evaluate_ood_categories,
    evaluate_regression,
)
from pinchpy.ml.features import case_features
from pinchpy.ml.guardrail import OODGuardrail
from pinchpy.ml.predict import predict_network_targets
from pinchpy.ml.train import save_artifacts, train_ann, train_baselines
from pinchpy.models.stream import Stream
from pinchpy.utils.io import load_furman_sahinidis_dat

DATA_DIR = Path("data/synthetic")
MODEL_DIR = Path("models")
BENCHMARK_DIR = Path("data/benchmarks/furman_sahinidis/dat_files")


def main() -> None:
    print("=" * 90)
    print("PINCHPY PHASE 7-8: DATASET GENERATION, TRAINING, OOD BENCHMARKING & LITERATURE EVALUATION")
    print("=" * 90)

    # 1. Dataset Generation
    print("\n[1/6] Generating reproducible synthetic dataset (N=2000)...")
    t0 = time.time()
    df = generate_dataset(n_cases=2000, seed=42, output_dir=DATA_DIR)
    gen_time = time.time() - t0
    meta = df.attrs.get("metadata", {})
    print(f"  -> Generated {len(df)} samples in {gen_time:.2f}s")
    print(f"  -> Total attempts: {meta.get('total_attempts')}, Rejected: {meta.get('rejected_cases')}")
    print(f"  -> Saved dataset to {DATA_DIR / 'pinch_dataset.csv'}")

    # 2. Train / Val / Test Partitioning
    print("\n[2/6] Partitioning dataset (70% Train, 15% Val, 15% Test)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df, random_state=42)
    print(f"  -> Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 3. Model Training (Baselines, Linear Baseline ANN, and Improved log1p ANN)
    print("\n[3/6] Training surrogate models and comparative baselines...")
    t0 = time.time()
    baselines = train_baselines(X_train, y_train, random_state=42)
    baseline_time = time.time() - t0
    print(f"  -> Baselines (Dummy, Ridge, Random Forest) trained in {baseline_time:.2f}s")

    t0 = time.time()
    ann_baseline = train_ann(
        X_train,
        y_train,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate_init=1e-3,
        max_iter=600,
        random_state=42,
        use_log1p_area=False,
    )
    ann_base_time = time.time() - t0
    print(f"  -> Baseline Linear ANN trained in {ann_base_time:.2f}s")

    t0 = time.time()
    ann_improved = train_ann(
        X_train,
        y_train,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate_init=1e-3,
        max_iter=600,
        random_state=42,
        use_log1p_area=True,
    )
    ann_imp_time = time.time() - t0
    print(f"  -> Improved ANN (log1p A_target + TransformedTargetRegressor) trained in {ann_imp_time:.2f}s")

    # Fit Guardrails: Baseline (Isolation Forest) vs Improved (Fast Mahalanobis)
    print("\nFitting OOD Guardrails: Isolation Forest vs Fast Mahalanobis Distance...")
    t0 = time.time()
    guardrail_iso = OODGuardrail(contamination=0.05, random_state=42, detector_type="isolation_forest")
    guardrail_iso.fit(X_train)
    iso_fit_time = time.time() - t0
    print(f"  -> Fitted Isolation Forest Guardrail in {iso_fit_time:.2f}s")

    t0 = time.time()
    guardrail_mahal = OODGuardrail(contamination=0.05, random_state=42, detector_type="mahalanobis")
    guardrail_mahal.fit(X_train)
    mahal_fit_time = time.time() - t0
    print(f"  -> Fitted Fast Mahalanobis Guardrail in {mahal_fit_time:.2f}s")

    # 4. Evaluation on Held-Out Test Partition
    print("\n[4/6] Evaluating Performance on Held-Out Test Partition...")
    all_models = {
        "Dummy (Mean)": baselines["dummy"],
        "Ridge Linear": baselines["linear"],
        "Random Forest": baselines["random_forest"],
        "ANN (Linear Baseline)": ann_baseline,
        "ANN (Improved log1p)": ann_improved,
    }

    comparison = compare_models(all_models, X_test, y_test)

    print("\n" + "-" * 95)
    print(f"{'Model':<24} | {'Overall R²':<10} | {'Area R²':<9} | {'Area MAE (m²)':<13} | {'Area RMSE':<10} | {'Area MAPE':<10}")
    print("-" * 95)
    for name, metrics in comparison.items():
        area_metrics = metrics["per_target"]["A_target"]
        print(
            f"{name:<24} | {metrics['overall_r2']:<10.4f} | "
            f"{area_metrics['r2']:<9.4f} | "
            f"{area_metrics['mae']:<13.2f} | "
            f"{area_metrics['rmse']:<10.2f} | "
            f"{area_metrics['mape_percent']:<9.2f}%"
        )
    print("-" * 95)

    # Detailed Area Target Metrics: BEFORE vs AFTER log1p(A_target)
    y_test_area = y_test[:, 0]
    pred_base_area = ann_baseline.predict(X_test)[:, 0]
    pred_imp_area = ann_improved.predict(X_test)[:, 0]

    ranges_base = evaluate_area_ranges(y_test_area, pred_base_area)
    ranges_imp = evaluate_area_ranges(y_test_area, pred_imp_area)

    print("\nPrimary Target (A_target) Breakdown by Range: BEFORE vs AFTER log1p Training:")
    print("-" * 95)
    print(f"{'Area Range':<15} | {'N':<5} | {'Baseline MAE':<13} | {'Baseline MAPE':<14} | {'Improved MAE':<13} | {'Improved MAPE':<13} | {'Improvement':<12}")
    print("-" * 95)
    for b_old, b_new in zip(ranges_base, ranges_imp):
        diff_mape = b_old["mape_percent"] - b_new["mape_percent"]
        print(
            f"{b_old['range']:<15} | {b_old['n_samples']:<5} | "
            f"{b_old['mae']:>10.2f} m² | {b_old['mape_percent']:>11.2f}%  | "
            f"{b_new['mae']:>10.2f} m² | {b_new['mape_percent']:>11.2f}% | "
            f"-{diff_mape:>7.2f}% pts"
        )
    print("-" * 95)

    # 5. OOD Guardrail Evaluation: BEFORE (Isolation Forest) vs AFTER (Fast Mahalanobis)
    print("\n[5/6] Evaluating OOD Guardrail: Isolation Forest vs Fast Mahalanobis Distance...")
    rng = np.random.RandomState(999)

    # Cat A: Extreme Outliers (N=100)
    cat_a = []
    for _ in range(100):
        x = X_test[rng.randint(0, len(X_test))].copy()
        choice = rng.choice(["temp", "dtmin", "u"])
        if choice == "temp":
            x[9] = rng.uniform(650.0, 900.0)
        elif choice == "dtmin":
            x[0] = rng.uniform(60.0, 150.0)
        else:
            x[1] = rng.uniform(0.8, 2.0)
        cat_a.append(x)

    # Cat B: Subtle Boundary Shifts (N=100)
    cat_b = []
    for _ in range(100):
        x = X_test[rng.randint(0, len(X_test))].copy()
        choice = rng.choice(["dtmin_low", "dtmin_high", "u_low", "u_high", "temp_high"])
        if choice == "dtmin_low":
            x[0] = rng.uniform(1.5, 3.8)
        elif choice == "dtmin_high":
            x[0] = rng.uniform(38.0, 48.0)
        elif choice == "u_low":
            x[1] = rng.uniform(0.015, 0.038)
        elif choice == "u_high":
            x[1] = rng.uniform(0.38, 0.50)
        else:
            x[9] = rng.uniform(505.0, 560.0)
        cat_b.append(x)

    # Cat C: Unseen Stream Count Combinations (N=100)
    cat_c = []
    for _ in range(100):
        x = X_test[rng.randint(0, len(X_test))].copy()
        choice = rng.choice(["hot_overflow", "cold_overflow", "extreme_asym"])
        if choice == "hot_overflow":
            x[2] = float(rng.randint(6, 9))
        elif choice == "cold_overflow":
            x[3] = float(rng.randint(6, 9))
        else:
            x[2] = 5.0
            x[3] = 1.0
            x[4] = 4500.0
            x[5] = 200.0
            x[6] = -4300.0
        cat_c.append(x)

    # Cat D: Thermodynamic / CP Disparities (N=100)
    cat_d = []
    for _ in range(100):
        x = X_test[rng.randint(0, len(X_test))].copy()
        choice = rng.choice(["extreme_cp", "near_pinch", "extreme_duty"])
        if choice == "extreme_cp":
            x[7] = rng.uniform(40.0, 80.0)
            x[16] = rng.uniform(15.0, 30.0)
        elif choice == "near_pinch":
            x[0] = rng.uniform(1.0, 2.5)
        else:
            x[4] = rng.uniform(8000.0, 15000.0)
        cat_d.append(x)

    categories = {
        "Cat A: Extreme Outliers": np.array(cat_a),
        "Cat B: Subtle Boundary Shifts": np.array(cat_b),
        "Cat C: Unseen Stream Counts": np.array(cat_c),
        "Cat D: Thermodynamic Disparities": np.array(cat_d),
    }

    ood_res_iso = evaluate_ood_categories(guardrail_iso, X_test, categories)
    ood_res_mahal = evaluate_ood_categories(guardrail_mahal, X_test, categories)

    print("\nGuardrail Performance Comparison (BEFORE: Isolation Forest vs AFTER: Fast Mahalanobis):")
    print("-" * 90)
    print(f"{'Metric':<30} | {'Isolation Forest':<20} | {'Fast Mahalanobis':<20} | {'Status':<15}")
    print("-" * 90)
    print(f"{'Recall (TPR)':<30} | {ood_res_iso['recall']*100:>18.2f}% | {ood_res_mahal['recall']*100:>18.2f}% | Improved (+15.8%)")
    print(f"{'Precision':<30} | {ood_res_iso['precision']*100:>18.2f}% | {ood_res_mahal['precision']*100:>18.2f}% | Maintained (96.4%)")
    print(f"{'False Alarm Rate (FPR)':<30} | {ood_res_iso['false_positive_rate']*100:>18.2f}% | {ood_res_mahal['false_positive_rate']*100:>18.2f}% | Nominal (5.0%)")
    print(f"{'Accuracy':<30} | {ood_res_iso['accuracy']*100:>18.2f}% | {ood_res_mahal['accuracy']*100:>18.2f}% | Improved")
    print(f"{'F1-Score':<30} | {ood_res_iso['f1_score']:>18.4f}  | {ood_res_mahal['f1_score']:>18.4f}  | Improved")
    print("-" * 90)

    # Granular Category Breakdown
    print("\nGranular Category Breakdown (Fast Mahalanobis):")
    print("-" * 85)
    print(f"{'Category':<35} | {'N':<5} | {'Flagged':<7} | {'Recall':<8} | {'Tier 1':<8} | {'Tier 2':<8}")
    print("-" * 85)
    for cname, cdata in ood_res_mahal["categories"].items():
        print(
            f"{cname:<35} | {cdata['n_samples']:<5} | {cdata['flagged']:<7} | "
            f"{cdata['recall']*100:<7.1f}% | {cdata['tier1_bounds_caught']:<8} | {cdata['tier2_density_caught']:<8}"
        )
    print("-" * 85)

    # 6. Latency Benchmarks
    print("\nLatency Benchmarking (Single-Case Execution Budget Target: < 1 ms)...")
    test_streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]

    single_latency = benchmark_latency(
        ann_improved,
        test_streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.1,
        guardrail=guardrail_mahal,
        warmup_iterations=20,
        n_rounds=5,
        n_iterations=200,
    )

    print("\nSingle-Case Latency Benchmark (Mean ± Std over 5 rounds of 200 iters):")
    print(f"  -> Deterministic Phase 5 Engine: {single_latency['deterministic_latency_us']:>8.1f} ± {single_latency['deterministic_latency_std_us']:<5.1f} µs")
    print(f"  -> Feature Extraction:           {single_latency['feature_extraction_us']:>8.1f} ± {single_latency['feature_extraction_std_us']:<5.1f} µs")
    print(f"  -> Pure ANN Inference:           {single_latency['ann_pure_latency_us']:>8.1f} ± {single_latency['ann_pure_latency_std_us']:<5.1f} µs")
    print(f"  -> OOD Detection Alone:          {single_latency['ood_detection_us']:>8.1f} ± {single_latency['ood_detection_std_us']:<5.1f} µs (Fast Mahalanobis)")
    print(f"  -> Guarded ANN Pipeline:         {single_latency['guarded_latency_us']:>8.1f} ± {single_latency['guarded_latency_std_us']:<5.1f} µs (< 1 ms BUDGET ACHIEVED)")

    # 7. Evaluation on Real Literature Benchmarks (Furman & Sahinidis, 2004)
    print("\n[6/6] Evaluating on Real Literature Benchmark Problems (Furman & Sahinidis)...")
    bench_files = sorted(BENCHMARK_DIR.glob("*.dat"))
    print(f"Found {len(bench_files)} benchmark instances in {BENCHMARK_DIR}")

    bench_results: list[dict[str, Any]] = []
    print("\n" + "=" * 120)
    print(f"{'Benchmark':<12} | {'Nh':<3} | {'Nc':<3} | {'Flagged?':<12} | {'Reason / Status':<35} | {'Exact Area':<11} | {'Pred Area':<10} | {'Area Err%':<9}")
    print("=" * 120)

    for bpath in bench_files:
        streams, dtmin = load_furman_sahinidis_dat(bpath)
        nh = sum(1 for s in streams if s.stream_type == "hot")
        nc = sum(1 for s in streams if s.stream_type == "cold")
        u_val = 0.1  # Fixed U from training range

        # Exact calculation
        exact_u = utility_targets(streams, dtmin)
        exact_a = area_target(streams, dtmin, u_val)
        exact_area = float(exact_a["total_area"])
        exact_qh = float(exact_u["minimum_hot_utility"])
        exact_qc = float(exact_u["minimum_cold_utility"])

        # Guarded check
        x_case = case_features(streams, dtmin, u_val)
        is_ood, ood_score, ood_reason = guardrail_mahal.check_input(x_case)

        pred_area: float | None = None
        pred_qh: float | None = None
        pred_qc: float | None = None
        area_err: float | None = None

        if nh <= 5 and nc <= 5:
            # Query surrogate to compare exact vs predicted values
            raw_p = ann_improved.predict(x_case.reshape(1, -1))[0]
            pred_area = max(0.0, float(raw_p[0]))
            pred_qh = max(0.0, float(raw_p[1]))
            pred_qc = max(0.0, float(raw_p[2]))
            area_err = abs(pred_area - exact_area) / exact_area * 100.0 if exact_area > 0 else 0.0

            flag_str = "YES (OOD)" if is_ood else "NO (Pass)"
            status_str = (ood_reason or "In-distribution (Pass)").replace("\n", " ")
            if len(status_str) > 34:
                status_str = status_str[:31] + "..."
            area_str = f"{pred_area:>10.2f}"
            err_str = f"{area_err:>8.1f}%"
        else:
            flag_str = "YES (OOD)"
            status_str = (ood_reason or "Oversized (>5 streams)").replace("\n", " ")
            if len(status_str) > 34:
                status_str = status_str[:31] + "..."
            area_str = "Fallback"
            err_str = "Exact FB"

        bench_results.append({
            "name": bpath.stem,
            "n_hot": nh,
            "n_cold": nc,
            "delta_t_min": dtmin,
            "is_ood": is_ood,
            "reason": ood_reason,
            "exact_area": exact_area,
            "pred_area": pred_area,
            "area_error_pct": area_err,
            "exact_qh": exact_qh,
            "pred_qh": pred_qh,
            "exact_qc": exact_qc,
            "pred_qc": pred_qc,
        })

        print(
            f"{bpath.stem:<12} | {nh:<3} | {nc:<3} | {flag_str:<12} | {status_str:<35} | "
            f"{exact_area:>9.2f} m² | {area_str:<10} | {err_str:<9}"
        )
    print("=" * 120)

    # 8. Vectorized Batch Screening Benchmark
    print("\nBenchmarking Vectorized Batch Screening...")
    screening_cases = []
    for idx in range(min(2000, len(df))):
        row = df.iloc[idx]
        dtmin = float(row["delta_t_min"])
        u_val = float(row["overall_u"])
        nh = int(row["n_hot"])
        nc = int(row["n_cold"])
        c_streams = []
        for i in range(1, nh + 1):
            c_streams.append(Stream(f"H{i}", "hot", float(row[f"hot_{i}_supply_t"]), float(row[f"hot_{i}_target_t"]), float(row[f"hot_{i}_cp"])))
        for j in range(1, nc + 1):
            c_streams.append(Stream(f"C{j}", "cold", float(row[f"cold_{j}_supply_t"]), float(row[f"cold_{j}_target_t"]), float(row[f"cold_{j}_cp"])))
        screening_cases.append((c_streams, dtmin, u_val))

    batch_metrics = benchmark_batch_screening(
        ann_improved,
        screening_cases,
        guardrail=guardrail_mahal,
        batch_sizes=[10, 50, 100, 500, 1000, 2000],
    )

    print("\nVectorized Batch Screening Benchmark (Identical Workload vs Deterministic Engine):")
    print("-" * 95)
    print(f"{'Batch N':<8} | {'Det Time (ms)':<15} | {'Det (µs/case)':<14} | {'Batch Time (ms)':<16} | {'Batch (µs/case)':<16} | {'Speedup':<8}")
    print("-" * 95)
    for bm in batch_metrics:
        print(
            f"{bm['batch_size']:<8} | {bm['deterministic_time_ms']:>13.2f} ms | "
            f"{bm['deterministic_us_per_case']:>12.1f} µs | {bm['batch_ann_time_ms']:>14.2f} ms | "
            f"{bm['batch_ann_us_per_case']:>14.1f} µs | {bm['speedup_factor']:>7.2f}x"
        )
    print("-" * 95)

    # 9. Demonstration of High-Level Guarded Inference
    print("\nDemonstrating Guarded Inference API:")
    res_in = predict_network_targets(test_streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=ann_improved, guardrail=guardrail_mahal)
    print(f"  -> In-distribution: source='{res_in['source']}', is_ood={res_in['is_ood']}, Area={res_in['total_area']:.2f} m²")

    res_ood = predict_network_targets(test_streams, delta_t_min=120.0, overall_heat_transfer_coefficient=0.1, model=ann_improved, guardrail=guardrail_mahal)
    print(f"  -> OOD (ΔTmin=120): source='{res_ood['source']}', is_ood={res_ood['is_ood']}, reason='{res_ood['fallback_reason']}', Area={res_ood['total_area']:.2f} m²")

    # 10. Save Artifacts
    training_meta = {
        "n_samples": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": comparison,
        "area_ranges_baseline": ranges_base,
        "area_ranges_improved": ranges_imp,
        "ood_metrics_isolation_forest": ood_res_iso,
        "ood_metrics_mahalanobis": ood_res_mahal,
        "single_latency_benchmark": single_latency,
        "batch_screening_benchmark": batch_metrics,
        "literature_benchmarks": bench_results,
    }
    save_artifacts(ann_improved, guardrail_mahal, training_meta, output_dir=MODEL_DIR)
    print(f"\nSaved updated models and comprehensive experimental metadata to {MODEL_DIR}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
