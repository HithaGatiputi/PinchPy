# PinchPy — Pinch Analysis & Heat Exchanger Network (HEN) Analysis

**PinchPy** is a Python library for Pinch Analysis, thermal process integration, and Heat Exchanger Network (HEN) synthesis.

> [!NOTE]
> **Current Status: Phases 1–8 complete — Stream model + validation (Phase 1), ΔTmin temperature shifting/intervals (Phase 2), heat cascade / pinch-point / utility-targeting engine (Phase 3), Composite Curve / Grand Composite Curve plotting (Phase 4), composite-curve area targeting (Phase 5), pinch-based HEN synthesis with a grid-diagram plot (Phase 6), synthetic dataset generation (Phase 7), and ANN surrogate modeling with fast OOD Guardrails (Phase 8).**
> The HEN synthesis uses a bounded backtracking matcher (CP-rule stream splitting, current-temperature feasibility, partial-duty capping) that reaches the exact Phase 3 minimum utility targets on simple problems; on harder multi-stream problems it can still land slightly above the targets, since it only searches the maximum feasible duty per match rather than a full continuous optimization (see `synthesis/network_design.py`). It always issues a `RuntimeWarning` when that happens, and still closes the gap with a utility exchanger so no energy is ever lost. UI/CLI (Phase 9) is planned future work.

---

## 1. What is a Process Stream?

In thermal process integration and Pinch Analysis, a **process stream** represents a fluid flow that must be heated or cooled between defined initial and final temperatures without undergoing a phase change (sensible heat transfer):

- **Supply Temperature ($T_s$)**: The initial available temperature of the stream (°C).
- **Target Temperature ($T_t$)**: The desired final specification temperature of the stream (°C).
- **Heat-Capacity Flow Rate ($CP$)**: The product of mass flow rate ($\dot{m}$) and specific heat capacity ($C_p$), typically expressed in $\text{kW/}^\circ\text{C}$ or $\text{kW/K}$.

---

## 2. Hot vs. Cold Streams

Every valid process stream is classified as either a hot stream or a cold stream based on its thermodynamic requirements:

| Stream Type | Description | Temperature Constraint | Role |
| :--- | :--- | :--- | :--- |
| **Hot Stream** (`"hot"`) | Requires cooling | $T_s > T_t$ | Releases heat to the network |
| **Cold Stream** (`"cold"`) | Requires heating | $T_t > T_s$ | Absorbs heat from the network |

Streams with identical supply and target temperatures ($T_s = T_t$) or inverted temperature directions are physically invalid for sensible heat integration and are rejected by validation.

---

## 3. Required Parameters and Validation Rules

The `Stream` model enforces strict physical validation at initialization:

1. **`id`**: Unique string identifier (e.g., `"H1"`, `"C1"`). Cannot be empty or `None`.
2. **`type`**: Stream classification (`"hot"` or `"cold"`). Case-insensitive and trimmed (`"HOT"`, `"Hot"`, `"hot"` normalize to `"hot"`).
3. **`supply_temperature`**: Inlet temperature (°C). Must be a finite numeric value (`int` or `float`).
4. **`target_temperature`**: Outlet temperature (°C). Must be a finite numeric value (`int` or `float`).
5. **`cp`**: Heat-capacity flow rate ($\text{kW/}^\circ\text{C}$). Must be a finite numeric value strictly greater than zero ($CP > 0$).
6. **`name`** *(optional)*: Descriptive label. Defaults to the stream `id` if omitted or empty.

### Error Handling

Validation failures raise explicit, descriptive exceptions:
- `StreamValidationError` (subclass of `ValueError`): Raised when physical thermodynamic constraints are violated (e.g., $T_s \le T_t$ for a hot stream, $T_t \le T_s$ for a cold stream, $CP \le 0$, or unrecognised stream type).
- `StreamTypeError` (subclass of `StreamValidationError` and `TypeError`): Raised when non-numeric or non-string parameters are passed (e.g., non-numeric temperatures or CP).

---

## 4. Basic Stream Calculations

The `Stream` class provides direct properties for fundamental thermodynamic quantities:

### Temperature Change ($\Delta T$)
Always strictly positive for every valid stream:
- For hot streams: $\Delta T = T_s - T_t$
- For cold streams: $\Delta T = T_t - T_s$

Accessible via properties: `stream.delta_T`, `stream.delta_t`, or `stream.dt`.

### Sensible Heat Duty ($Q$)
Calculated as:
$$Q = CP \times \Delta T$$

- When $CP$ is in $\text{kW/}^\circ\text{C}$ and $\Delta T$ in $^\circ\text{C}$, the duty $Q$ is in **kW**.
- Accessible via properties: `stream.Q`, `stream.heat_duty`, or method `stream.duty()`.

---

## 5. Basic Example

```python
from pinchpy.models.stream import Stream

# Define a hot stream (cooling from 180 °C to 80 °C)
hot = Stream(
    id="H1",
    type="hot",
    supply_temperature=180,
    target_temperature=80,
    cp=2.5,
)

# Define a cold stream (heating from 30 °C to 150 °C)
cold = Stream(
    id="C1",
    type="cold",
    supply_temperature=30,
    target_temperature=150,
    cp=3.0,
)

print(f"H1 ΔT = {hot.delta_T:.0f} °C")  # H1 ΔT = 100 °C
print(f"H1 Q = {hot.Q:.0f} kW")        # H1 Q = 250 kW

print(f"C1 ΔT = {cold.delta_T:.0f} °C")  # C1 ΔT = 120 °C
print(f"C1 Q = {cold.Q:.0f} kW")        # C1 Q = 360 kW
```

A standalone executable script is located at `examples/stream_example.py`:

```bash
python examples/stream_example.py
```

---

## 6. ΔTmin Temperature Shifting & Intervals

Before streams can be compared on a single scale, every hot stream is shifted down by $\Delta T_{min}/2$ and every cold stream is shifted up by $\Delta T_{min}/2$. This guarantees that any heat transfer feasible on the shifted scale respects the minimum approach temperature on the real scale.

```python
from pinchpy.analysis.temperature_intervals import shifted_temperatures, build_intervals

temps = shifted_temperatures([hot, cold], delta_t_min=10)
# -> sorted, de-duplicated shifted breakpoints, highest to lowest

intervals = build_intervals([hot, cold], delta_t_min=10)
# -> contiguous (high, low) temperature bands between those breakpoints
```

`delta_t_min` must be strictly positive; non-positive values raise `ValueError`. An empty stream list returns an empty list rather than raising.

---

## 7. Heat Cascade & Pinch-Point Targeting

The heat cascade (Problem Table Algorithm) walks the shifted intervals from the highest temperature down, accumulating the net heat surplus/deficit of the streams active in each interval. Shifting that running total so its lowest point is exactly zero gives the **minimum hot utility**; the cascade's final value is then the **minimum cold utility**. The shifted temperature(s) where the cascade touches zero are the **pinch point(s)**.

```python
from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets

hot = Stream("H1", "hot", 180, 80, 2.0)
cold = Stream("C1", "cold", 20, 150, 1.5)

result = utility_targets([hot, cold], delta_t_min=10)
print(result)
# {
#   "minimum_hot_utility": 0.0,
#   "minimum_cold_utility": 5.0,
#   "pinch_shifted_temperature": 175.0,
#   "pinch_hot_temperature": 180.0,
#   "pinch_cold_temperature": 170.0,
#   "pinch_shifted_temperatures": [175.0],
#   "is_threshold_problem": True,
# }
```

- `pinch_shifted_temperatures` lists **every** shifted temperature at which the cascade reaches zero (a problem can have more than one pinch).
- `is_threshold_problem` is `True` when one utility target is already zero — the reported "pinch" then sits at the extreme end of the temperature range rather than at an interior constraint, which is expected behaviour for threshold problems rather than an error.
- Passing an empty stream list returns zero utilities and `None` for the pinch temperature fields instead of raising.

Composite-curve point data (for later plotting) is available via:

```python
from pinchpy.analysis.composite_curves import composite_curve_data

curves = composite_curve_data([hot, cold], delta_t_min=10)
# {"hot_curve": [...], "cold_curve": [...]}  # cumulative-enthalpy vs. temperature points
```

---

## 8. Area Targeting (Phase 5)

The composite curves are cut into vertical enthalpy strips in which both curves are straight lines. Each strip is treated as one counter-current exchanger: its LMTD comes from the hot and cold temperature gaps at its two ends, its area is `Q / (U · LMTD)`, and the strip areas add up to the minimum area target for the chosen ΔTmin (uniform overall coefficient `U`, in kW/(m²·°C)).

```python
from pinchpy.models.stream import Stream
from pinchpy.analysis.area_target import area_target

streams = [Stream("H1", "hot", 180, 80, 2.0), Stream("C1", "cold", 20, 150, 1.5)]
result = area_target(streams, delta_t_min=10, overall_heat_transfer_coefficient=0.1)
print(result["total_area"])   # about 44.0 (m²)
```

- Returns `total_area`, per-strip `strips` (with `lmtd` and `area`), `heat_recovery`, `minimum_approach` and `is_threshold_problem`.
- Empty input gives zero area; a non-positive `delta_t_min` or `U` raises `ValueError`.
- If the composite curves cross or touch, the area is undefined and `ValueError` is raised; a `RuntimeWarning` is issued if they come closer than ΔTmin. Both mean the utility targets feeding the curves are inconsistent with ΔTmin.

---

## 9. HEN Synthesis & Grid Diagram (Phase 6)

The pinch is used to split the problem into an above-pinch and a below-pinch sub-problem (`synthesis.pinch_split`), each matched independently so heat never crosses the pinch. Within each sub-problem, feasible matches (`synthesis.exchanger_matching`, which checks both the ΔTmin temperature feasibility and the CP inequality rule at the pinch) are picked greedily, largest duty first, until no further match is feasible; any remaining duty becomes a heater (above pinch) or cooler (below pinch).

```python
from pinchpy.models.stream import Stream
from pinchpy.synthesis.network_design import design_network
from pinchpy.synthesis.network_evaluation import evaluate_network
from pinchpy.visualization.network_plot import plot_network

streams = [Stream("H1", "hot", 180, 80, 2.0), Stream("C1", "cold", 20, 150, 1.5)]
network = design_network(streams, delta_t_min=10, overall_heat_transfer_coefficient=0.5)

print(evaluate_network(network, streams))
# {'number_of_exchangers': ..., 'total_recovered_heat': ..., 'total_area': ..., 'minimum_units': ...}

fig, ax = plot_network(streams, network)
```

- `design_network` returns a `HeatExchangerNetwork` of `HeatExchanger` objects; process matches have `is_utility=False`, heaters/coolers have `is_utility=True` (`hot_stream="HU"` for a heater, `cold_stream="CU"` for a cooler).
- `overall_heat_transfer_coefficient` is optional; when given, each exchanger also gets `lmtd` and `area`.
- The matcher is a greedy tick-off heuristic, not a full stream-splitting search, so on harder problems it can leave a stream short of a feasible partner; when that happens the leftover is still closed with a utility exchanger so no energy is lost, but a `RuntimeWarning` is raised because the design then exceeds the Phase 3 minimum utility targets.
- `plot_network` draws the standard grid diagram: streams as horizontal lines on a shared temperature axis, matches as two circles joined by a line and labelled with duty, utilities as a single labelled circle.

---

## 10. Machine Learning Surrogate & OOD Guardrail (Phases 7–8)

PinchPy incorporates a machine-learning surrogate model (Scikit-Learn `MLPRegressor` pipeline with target transformation) for rapid simultaneous prediction of:
- **Primary Target**: Total Minimum Area $A_{\text{target}}$ ($\text{m}^2$)
- **Secondary Targets**: Minimum Hot Utility $Q_{H,min}$ ($\text{kW}$) and Minimum Cold Utility $Q_{C,min}$ ($\text{kW}$)

> [!NOTE]
> **Engineering Reality & Surrogate Role**: For simple single-case pinch targeting, the exact deterministic Phase 1–5 engine (`area_target` + `utility_targets`) is already fast (~53 µs per case). A surrogate does not provide a single-case speedup for basic targeting. Instead, the ANN surrogate's primary values are:
> 1. **High-Throughput Vectorized Batch Screening**: Concurrently scoping hundreds to thousands of stream configurations and approach temperatures ($\Delta T_{min}$) in parallel (~39–44 µs/case in batches).
> 2. **Foundation for Expensive Downstream Steps**: Serving as a differentiable or rapid proxy for computationally intensive combinatorial synthesis, stream-splitting optimization, and MINLP network design.
> 
> Because neural surrogates extrapolate unpredictably on out-of-distribution inputs, the **multi-tier Out-of-Distribution (OOD) Guardrail with deterministic fallback is the core scientific and safety contribution of this implementation.**

### Research Contribution: Fast Out-of-Distribution (OOD) Guardrail
Surrogate neural networks in process engineering can extrapolate dangerously when queried outside their training regime or produce thermodynamically unphysical outputs. PinchPy solves this with a high-speed, multi-tier **OOD Guardrail**:
1. **Tier 1 — Domain Bounding (Sub-microsecond early exit)**: Validates stream counts ($n_h, n_c \in [1, 5]$), temperature spans, $\Delta T_{min} \in [3, 50]^\circ\text{C}$, heat transfer coefficient $U$, and individual/total $CP$ bounds against the training envelope. Any violation triggers immediate safe fallback without executing neural inference.
2. **Tier 2 — Fast Mahalanobis Anomaly Detection (~4 µs)**: Uses a standardized Ledoit-Wolf precision distance detector on incoming feature vectors. Replacing single-call `IsolationForest` (~10 ms) with Mahalanobis distance accelerates guarded inference by **~48x**, bringing end-to-end guarded pipeline execution down to **~208 µs (well below the < 1 ms target)** while achieving **100.0% OOD recall and 96.4% precision**.
3. **Tier 3 — Physical Consistency Checks**: Validates First Law enthalpy conservation ($Q_C - Q_H = \sum Q_H - \sum Q_C$) and thermodynamic non-negativity ($A > 0, Q_H \ge 0, Q_C \ge 0$).
4. **Automatic Deterministic Fallback**: If an input is flagged as OOD or physically inconsistent, the system transparently falls back to the exact deterministic Phase 5 engine (`area_target`), guaranteeing safe, trusted results.

### Small-Area Accuracy Improvement via `log1p(A_target)`
By training the surrogate with $\ln(1 + A_{\text{target}})$ target scaling and inverting with $\exp(\hat{y}_0) - 1$ at inference, the neural network penalizes relative error across small heat-exchanger sizes rather than being dominated by large area scales:

| Area Range | Test Samples ($N$) | Baseline MAE | Baseline MAPE | Improved MAE | Improved MAPE | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 – 50 m²** | 116 | 32.85 m² | 292.94% | **15.71 m²** | **74.17%** | **-218.77% pts** |
| **50 – 150 m²** | 87 | 67.35 m² | 78.13% | **67.27 m²** | **72.22%** | **-5.91% pts** |
| **150 – 300 m²** | 40 | 89.90 m² | 47.21% | **82.91 m²** | **42.25%** | **-4.96% pts** |
| **> 300 m²** | 57 | 183.80 m² | 35.91% | **202.39 m²** | **36.36%** | Stable |
| **Overall Area** | 300 | 79.14 m² | 145.10% | **75.09 m²** | **61.84%** | **-83.26% pts** |

```python
from pinchpy.models.stream import Stream
from pinchpy.ml.train import load_artifacts
from pinchpy.ml.predict import predict_network_targets, predict_batch_network_targets

streams = [
    Stream("H1", "hot", 163.4, 28.7, 3.29),
    Stream("H2", "hot", 125.5, 60.8, 2.35),
    Stream("C1", "cold", 55.6, 184.5, 1.28),
    Stream("C2", "cold", 52.8, 141.3, 2.3),
]

model, guardrail, _ = load_artifacts("models")

# Single-case in-distribution inference (runs under 210 µs!)
res = predict_network_targets(streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=model, guardrail=guardrail)
print(res)
# {'total_area': 82.23, 'minimum_hot_utility': 73.70, 'minimum_cold_utility': 270.55, 'source': 'ann_surrogate', 'is_ood': False}

# OOD query (e.g., extreme ΔTmin=120 °C): automatically falls back to deterministic engine in < 60 µs
res_ood = predict_network_targets(streams, delta_t_min=120.0, overall_heat_transfer_coefficient=0.1, model=model, guardrail=guardrail)
print(res_ood)
# {'total_area': 0.0, 'minimum_hot_utility': 226.70, 'minimum_cold_utility': 0.0, 'source': 'deterministic_fallback', 'is_ood': True}
```

### Measured Execution Latencies & Detection Performance

| Component / Metric | Isolation Forest Baseline | Fast Mahalanobis Guardrail | Speedup / Status |
| :--- | :---: | :---: | :---: |
| **Deterministic Phase 5 Engine** | 53.0 ± 0.3 µs | 53.0 ± 0.3 µs | Baseline exact ground truth |
| **Feature Extraction** | 5.3 ± 0.0 µs | 5.3 ± 0.0 µs | Fixed-size vectorization |
| **Pure ANN Inference** | 144.4 ± 1.9 µs | 144.4 ± 1.9 µs | MLP evaluation |
| **OOD Detection Latency Alone** | 9,750 µs (~9.8 ms) | **39.2 ± 0.7 µs** | **~250x faster** |
| **End-to-End Guarded Pipeline** | 10,300 µs (~10.3 ms) | **207.8 ± 5.3 µs** | **~50x faster (< 1 ms achieved)** |
| **OOD Anomaly Recall (TPR)** | 84.25% | **100.00%** | +15.75% |
| **Precision** | 96.56% | **96.39%** | Maintained (high trust) |
| **False Alarm Rate (FPR)** | 4.00% | **5.00%** | Nominal (5% design bound) |

### Evaluation on Real Literature Benchmarks (Furman & Sahinidis, 2004)

The pipeline was benchmarked against the 26 real literature problem instances from Furman & Sahinidis (2004) and Letsios, Kouyialis, & Misener (2018) located in `data/benchmarks/furman_sahinidis/dat_files/*.dat`.

Literature benchmarks represent real-world industrial processes (petroleum refineries, crude preheat trains, aromatics plants). When evaluated against the surrogate and guardrail:
- **Oversized Problems ($N_h > 5$ or $N_c > 5$)**: Cleanly detected and flagged by Tier 1 bounds (`"Number of hot/cold streams outside supported range [1, 5]"`), immediately falling back to the exact deterministic engine without crashing.
- **Industrial Scale & CP Disparities**: Massive industrial problems with total heat-capacity flow rates exceeding synthetic plant scales (e.g., `10sp1` with total $CP = 122,070\ \text{kW/}^\circ\text{C}$, `9sp-has1` with $920\ \text{kW/}^\circ\text{C}$) are flagged as OOD, preventing catastrophic unconstrained ANN extrapolation ($> 10^6\ \text{m}^2$).
- **Envelope Compliance**: Cases within supported stream counts are screened with exact vs. predicted comparisons.

| Benchmark | Hot | Cold | Flagged? | Guardrail Decision / Reason | Exact Area ($U=0.1$) | Status |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **4sp1** | 2 | 2 | YES (OOD) | Individual stream CP (20.0 kW/°C) exceeds training bound | 1,036.51 m² | Fallback |
| **6sp-cf1** | 3 | 3 | YES (OOD) | Individual stream CP (12.0 kW/°C) exceeds training bound | 1,044.72 m² | Fallback |
| **6sp-gg1** | 3 | 3 | YES (OOD) | Total CP (175.0 kW/°C) exceeds training bound | 2,324.32 m² | Fallback |
| **6sp1** | 3 | 3 | YES (OOD) | Total CP (85.4 kW/°C) exceeds training bound | 1,860.90 m² | Fallback |
| **7sp-cm1** | 3 | 4 | YES (OOD) | Stream temperatures span [313, 626] °C exceeds envelope | 449.22 m² | Fallback |
| **7sp-torw1** | 4 | 3 | YES (OOD) | Individual stream CP (18.0 kW/°C) exceeds training bound | 2,231.40 m² | Fallback |
| **7sp1** | 3 | 4 | YES (OOD) | Total CP (94.95 kW/°C) exceeds training bound | 2,925.40 m² | Fallback |
| **7sp2** | 3 | 4 | YES (OOD) | Stream temperatures span [100, 590] °C exceeds envelope | 1,288.08 m² | Fallback |
| **8sp-fs1** | 5 | 3 | YES (OOD) | Temperature span [10, 361] °C outside envelope | 2,552.50 m² | Fallback |
| **8sp1** | 4 | 4 | YES (OOD) | Total CP (92.35 kW/°C) exceeds training bound | 3,728.72 m² | Fallback |
| **9sp-al1** | 4 | 5 | YES (OOD) | Mahalanobis feature distance exceeds inlier threshold | 45.51 m² | Fallback |
| **9sp-has1** | 5 | 4 | YES (OOD) | Total CP (920.0 kW/°C) exceeds training bound | 12,108.75 m² | Fallback |
| **10sp-la1** | 4 | 5 | YES (OOD) | Mahalanobis feature distance exceeds inlier threshold | 45.51 m² | Fallback |
| **10sp-ol1** | 4 | 6 | YES (OOD) | Cold streams ($N_c=6$) exceeds capacity limit ($5$) | 41.04 m² | Fallback |
| **10sp1** | 5 | 5 | YES (OOD) | Total CP (122,070 kW/°C) exceeds training bound | 2,660,411.78 m² | Fallback |
| **12sp1** | 9 | 3 | YES (OOD) | Hot streams ($N_h=9$) exceeds capacity limit ($5$) | 75,902.02 m² | Fallback |
| **14sp1** | 7 | 7 | YES (OOD) | Hot streams ($N_h=7$) exceeds capacity limit ($5$) | 6,530.83 m² | Fallback |
| **15sp-tkm** | 9 | 6 | YES (OOD) | Hot streams ($N_h=9$) exceeds capacity limit ($5$) | 13,739.82 m² | Fallback |
| **20sp1** | 10 | 10 | YES (OOD) | Hot streams ($N_h=10$) exceeds capacity limit ($5$) | 761.31 m² | Fallback |
| **22sp-ph** | 11 | 11 | YES (OOD) | Hot streams ($N_h=11$) exceeds capacity limit ($5$) | 6,284.61 m² | Fallback |
| **22sp1** | 11 | 11 | YES (OOD) | Hot streams ($N_h=11$) exceeds capacity limit ($5$) | 4,713.12 m² | Fallback |
| **23sp1** | 11 | 12 | YES (OOD) | Hot streams ($N_h=11$) exceeds capacity limit ($5$) | 3,468.65 m² | Fallback |
| **28sp-as1** | 16 | 12 | YES (OOD) | Hot streams ($N_h=16$) exceeds capacity limit ($5$) | 4,210.60 m² | Fallback |
| **37sp-yfyv** | 21 | 16 | YES (OOD) | Hot streams ($N_h=21$) exceeds capacity limit ($5$) | 49,740.11 m² | Fallback |
| **7sp-s1** | 6 | 1 | YES (OOD) | Hot streams ($N_h=6$) exceeds capacity limit ($5$) | 33,253.30 m² | Fallback |
| **7sp4** | 6 | 1 | YES (OOD) | Hot streams ($N_h=6$) exceeds capacity limit ($5$) | 1,892.04 m² | Fallback |

### Benchmark Credits & Literature Citations
The literature benchmark instances in `data/benchmarks/furman_sahinidis/dat_files/` are digitized from classic heat integration literature compiled by Furman & Sahinidis (2004) and made publicly available under Apache 2.0 by Dimitrios Letsios, Georgia Kouyialis, and Ruth Misener (2018) in the repository [`TYT0710/min_matches_heuristics`](https://github.com/TYT0710/min_matches_heuristics) (also available via [`cog-imperial/min_matches_heuristics`](https://github.com/cog-imperial/min_matches_heuristics)):
- **Furman, K. C., & Sahinidis, N. V.** (2004). *Approximation algorithms for heat exchanger network synthesis.* Industrial & Engineering Chemistry Research, 43(14), 3554–3565.
- **Letsios, D., Kouyialis, G., & Misener, R.** (2018). *Heuristics with performance guarantees for the minimum number of matches problem in heat recovery network design.* Computers & Chemical Engineering, 113, 57–85.
- Complete per-problem literature sources (Lee 1970, Masso 1969, Ciric 1989, Gundersen 1990, Colberg 1990, Shenoy 1995, etc.) are cataloged in [`data/benchmarks/furman_sahinidis/references.txt`](file:///Users/hithagatiputi/Documents/RVCE/PinchPy/data/benchmarks/furman_sahinidis/references.txt).

---

## 11. How to Run Tests & Benchmarks

Install the package in editable mode and run the test suite with `pytest`:

```bash
pip install -e .
pytest -v
```

To run the complete Phase 7–8 synthetic generation, baseline comparison, and OOD benchmarking pipeline:

```bash
python scripts/train_and_evaluate.py
```

To see the Phase 4 plots (opens plot windows and saves PNGs to `data/processed/`):

```bash
python examples/phase4_plots_example.py
```

To print the Phase 5 area targets:

```bash
python examples/phase5_area_target_example.py
```

---

## 12. Project Phases

- ✅ **Phase 1**: Process stream data model + validation
- ✅ **Phase 2**: $\Delta T_{min}$ temperature shifting and interval formation
- ✅ **Phase 3**: Heat cascade (Problem Table Algorithm), pinch-point identification & minimum utility targeting ($Q_{H,min}$, $Q_{C,min}$), composite-curve data generation
- ✅ **Phase 4**: Composite Curve / Grand Composite Curve plotting
- ✅ **Phase 5**: Heat-exchanger duty/LMTD/area calculations wired into composite curves
- ✅ **Phase 6**: Heat Exchanger Network (HEN) synthesis and grid diagrams
- ✅ **Phase 7**: Synthetic dataset generation from the deterministic pinch engine
- ✅ **Phase 8**: ANN (MLP) surrogate model training, baseline comparisons & OOD guardrail
- ⏳ **Phase 9**: Integration + UI/CLI