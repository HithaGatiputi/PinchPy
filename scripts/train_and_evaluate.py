"""Full Phase 7-8 pipeline: dataset generation, model training, OOD evaluation, and benchmarking."""
import json
import time
from pathlib import Path

import numpy as np

from pinchpy.ml.dataset import generate_dataset, split_dataset
from pinchpy.ml.evaluate import (
    benchmark_batch_screening,
    benchmark_latency,
    compare_models,
    evaluate_area_ranges,
    evaluate_ood_categories,
    evaluate_ood_detection,
    evaluate_regression,
)
from pinchpy.ml.guardrail import OODGuardrail

from pinchpy.ml.predict import predict_network_targets
from pinchpy.ml.train import save_artifacts, train_ann, train_baselines
from pinchpy.models.stream import Stream

DATA_DIR = Path("data/synthetic")
MODEL_DIR = Path("models")


def main() -> None:
    print("=" * 70)
    print("PINCHPY PHASE 7-8: DATASET GENERATION, TRAINING & OOD BENCHMARKING")
    print("=" * 70)

    # 1. Dataset Generation
    print("\n[1/5] Generating reproducible synthetic dataset (N=2000)...")
    t0 = time.time()
    df = generate_dataset(n_cases=2000, seed=42, output_dir=DATA_DIR)
    gen_time = time.time() - t0
    meta = df.attrs.get("metadata", {})
    print(f"  -> Generated {len(df)} samples in {gen_time:.2f}s")
    print(f"  -> Total attempts: {meta.get('total_attempts')}, Rejected: {meta.get('rejected_cases')}")
    print(f"  -> Saved dataset to {DATA_DIR / 'pinch_dataset.csv'}")

    # 2. Train / Val / Test Partitioning
    print("\n[2/5] Partitioning dataset (70% Train, 15% Val, 15% Test)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(df, random_state=42)
    print(f"  -> Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # 3. Model Training
    print("\n[3/5] Training surrogate ANN and comparative baselines...")
    t0 = time.time()
    baselines = train_baselines(X_train, y_train, random_state=42)
    baseline_time = time.time() - t0
    print(f"  -> Baselines (Dummy, Ridge, Random Forest) trained in {baseline_time:.2f}s")

    t0 = time.time()
    ann_model = train_ann(
        X_train,
        y_train,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=42,
    )
    ann_time = time.time() - t0
    print(f"  -> ANN (MLPRegressor 128->64->32) trained in {ann_time:.2f}s")

    # Fit OOD Guardrail
    print("\n[4/5] Fitting OOD Guardrail on training distribution...")
    guardrail = OODGuardrail(contamination=0.05, random_state=42)
    guardrail.fit(X_train)
    print("  -> OOD Guardrail successfully fitted with Isolation Forest and domain bounds")

    # 4. Evaluation on Held-Out Test Partition
    print("\n[4/6] Evaluating performance on held-out test partition...")
    all_models = {
        "Dummy (Mean)": baselines["dummy"],
        "Ridge Linear": baselines["linear"],
        "Random Forest": baselines["random_forest"],
        "ANN (MLP)": ann_model,
    }

    comparison = compare_models(all_models, X_test, y_test)

    print("\n" + "-" * 85)
    print(f"{'Model':<18} | {'Overall R²':<10} | {'Area R²':<9} | {'Area MAE (m²)':<13} | {'Area RMSE':<10} | {'Area MAPE':<10}")
    print("-" * 85)
    for name, metrics in comparison.items():
        area_metrics = metrics["per_target"]["A_target"]
        print(
            f"{name:<18} | {metrics['overall_r2']:<10.4f} | "
            f"{area_metrics['r2']:<9.4f} | "
            f"{area_metrics['mae']:<13.2f} | "
            f"{area_metrics['rmse']:<10.2f} | "
            f"{area_metrics['mape_percent']:<9.2f}%"
        )
    print("-" * 85)

    # Detailed Area Target Metrics
    y_test_area = y_test[:, 0]
    ann_pred_test = ann_model.predict(X_test)
    ann_pred_area = ann_pred_test[:, 0]
    area_ranges = evaluate_area_ranges(y_test_area, ann_pred_area)

    print("\nPrimary Target (A_target) Error Breakdown by Range:")
    for b in area_ranges:
        print(
            f"  -> Range {b['range']:<12} (n={b['n_samples']:<3}): "
            f"MAE = {b['mae']:<7.2f} m², RMSE = {b['rmse']:<7.2f} m², MAPE = {b['mape_percent']:<6.2f}%"
        )

    print("\nSecondary Targets (Utilities) Performance for ANN:")
    ann_metrics = comparison["ANN (MLP)"]["per_target"]
    for target_name in ["Q_H_min", "Q_C_min"]:
        m = ann_metrics[target_name]
        print(f"  -> {target_name:<10}: R² = {m['r2']:.4f}, MAE = {m['mae']:.2f} kW, RMSE = {m['rmse']:.2f} kW, MAPE = {m['mape_percent']:.2f}%")

    # 5. Expanded OOD Evaluation Across 4 Physical Categories
    print("\n[5/6] Evaluating Multi-Tier OOD Guardrail Across 4 Anomaly Categories...")
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

    ood_results = evaluate_ood_categories(guardrail, X_test, categories)
    print("\n" + "-" * 85)
    print(f"{'Category':<35} | {'N':<5} | {'Flagged':<7} | {'Recall':<8} | {'Tier 1':<8} | {'Tier 2':<8}")
    print("-" * 85)
    for cname, cdata in ood_results["categories"].items():
        print(
            f"{cname:<35} | {cdata['n_samples']:<5} | {cdata['flagged']:<7} | "
            f"{cdata['recall']*100:<7.1f}% | {cdata['tier1_bounds_caught']:<8} | {cdata['tier2_density_caught']:<8}"
        )
    print("-" * 85)

    cm = ood_results["confusion_matrix"]
    print(f"Global OOD Performance (Test Negatives N={cm['TN']+cm['FP']}, OOD Positives N={cm['TP']+cm['FN']}):")
    print(f"  -> Recall (TPR):     {ood_results['recall']*100:.2f}% ({cm['TP']}/{cm['TP']+cm['FN']})")
    print(f"  -> Precision:        {ood_results['precision']*100:.2f}% ({cm['TP']}/{cm['TP']+cm['FP']})")
    print(f"  -> False Alarm Rate: {ood_results['false_positive_rate']*100:.2f}% ({cm['FP']}/{cm['TN']+cm['FP']})")
    print(f"  -> Accuracy:         {ood_results['accuracy']*100:.2f}%")
    print(f"  -> F1-Score:         {ood_results['f1_score']:.4f}")

    # 6. Latency & Batch Screening Benchmarks
    print("\n[6/6] Benchmarking Execution Runtime & Vectorized Batch Screening...")
    test_streams = [
        Stream("H1", "hot", 163.4, 28.7, 3.29),
        Stream("H2", "hot", 125.5, 60.8, 2.35),
        Stream("C1", "cold", 55.6, 184.5, 1.28),
        Stream("C2", "cold", 52.8, 141.3, 2.3),
    ]

    single_latency = benchmark_latency(
        ann_model,
        test_streams,
        delta_t_min=10.0,
        overall_heat_transfer_coefficient=0.1,
        guardrail=guardrail,
        warmup_iterations=20,
        n_rounds=5,
        n_iterations=200,
    )

    print("\nSingle-Case Latency Benchmark (Mean ± Std over 5 rounds of 200 iters):")
    print(f"  -> Deterministic Phase 5 Engine: {single_latency['deterministic_latency_us']:>8.1f} ± {single_latency['deterministic_latency_std_us']:<5.1f} µs")
    print(f"  -> Feature Extraction:           {single_latency['feature_extraction_us']:>8.1f} ± {single_latency['feature_extraction_std_us']:<5.1f} µs")
    print(f"  -> Pure ANN Inference:           {single_latency['ann_pure_latency_us']:>8.1f} ± {single_latency['ann_pure_latency_std_us']:<5.1f} µs")
    print(f"  -> OOD Detection Alone:          {single_latency['ood_detection_us']:>8.1f} ± {single_latency['ood_detection_std_us']:<5.1f} µs")
    print(f"  -> Guarded ANN Pipeline:         {single_latency['guarded_latency_us']:>8.1f} ± {single_latency['guarded_latency_std_us']:<5.1f} µs")

    # Reconstruct cases for batch screening
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
        ann_model,
        screening_cases,
        guardrail=guardrail,
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

    # 7. Demonstration of High-Level Guarded Inference
    print("\nDemonstrating Guarded Inference API:")
    res_in = predict_network_targets(test_streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=ann_model, guardrail=guardrail)
    print(f"  -> In-distribution: source='{res_in['source']}', is_ood={res_in['is_ood']}, Area={res_in['total_area']:.2f} m²")

    res_ood = predict_network_targets(test_streams, delta_t_min=120.0, overall_heat_transfer_coefficient=0.1, model=ann_model, guardrail=guardrail)
    print(f"  -> OOD (ΔTmin=120): source='{res_ood['source']}', is_ood={res_ood['is_ood']}, reason='{res_ood['fallback_reason']}', Area={res_ood['total_area']:.2f} m²")

    # 8. Save Artifacts
    training_meta = {
        "n_samples": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": comparison,
        "area_ranges": area_ranges,
        "ood_metrics": ood_results,
        "single_latency_benchmark": single_latency,
        "batch_screening_benchmark": batch_metrics,
        "hidden_layers": [128, 64, 32],
    }
    save_artifacts(ann_model, guardrail, training_meta, output_dir=MODEL_DIR)
    print(f"\nSaved updated models and comprehensive experimental metadata to {MODEL_DIR}/")
    print("=" * 85)


if __name__ == "__main__":
    main()

