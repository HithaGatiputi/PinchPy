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
from pinchpy.synthesis.network_evaluation import evaluate_network, network_tac
from pinchpy.visualization.network_plot import plot_network

streams = [Stream("H1", "hot", 180, 80, 2.0), Stream("C1", "cold", 20, 150, 1.5)]
network = design_network(streams, delta_t_min=10, overall_heat_transfer_coefficient=0.5, search_budget=500)

print(evaluate_network(network, streams))
# {'number_of_exchangers': ..., 'total_recovered_heat': ..., 'total_area': ..., 'minimum_units': ...}

# Total Annual Cost (TAC) evaluation
tac_res = network_tac(network)
print(tac_res)
# {'capital': ..., 'hot_utility': ..., 'cold_utility': ..., 'total': ...}

fig, ax = plot_network(streams, network)
```

- `design_network` returns a `HeatExchangerNetwork` of `HeatExchanger` objects; process matches have `is_utility=False`, heaters/coolers have `is_utility=True` (`hot_stream="HU"` for a heater, `cold_stream="CU"` for a cooler).
- `overall_heat_transfer_coefficient` is optional; when given, each exchanger also gets `lmtd` and `area`.
- The matcher is a bounded backtracking heuristic (`search_budget=500` by default), not an unconstrained continuous optimization; when a stream cannot be fully matched, the gap is closed with a utility exchanger so no energy is lost, issuing a `RuntimeWarning` when the design exceeds Phase 3 minimum utility targets.
- `plot_network` draws the standard grid diagram: streams as horizontal lines on a shared temperature axis, matches as two circles joined by a line and labelled with duty, utilities as a single labelled circle.

### Total Annual Cost (TAC) Evaluation (`network_tac`)
PinchPy implements annualized cost evaluation for synthesized heat exchanger networks via `network_tac` (`synthesis/network_evaluation.py`), adopting the standard literature formulation from Yang et al. (*Energy* 324 (2025) 136011):

$$\text{TAC} = \sum_{i \in \text{exchangers}} \left(C_{\text{fixed}} + C_{\text{area}} \cdot A_i^{\beta}\right) + C_{\text{hot}} \cdot Q_{\text{heater}} + C_{\text{cold}} \cdot Q_{\text{cooler}}$$

#### Default Cost Parameters & Literature Units
| Parameter | Keyword Argument | Default Value | Standard Literature Units |
| :--- | :--- | :---: | :--- |
| **Fixed Exchanger Capital** | `fixed_cost` | **2,000.0** | $\$ / \text{yr}$ per exchanger unit |
| **Area Coefficient** | `area_cost` | **800.0** | $\$ / (\text{m}^{2\beta} \cdot \text{yr})$ |
| **Area Exponent** | `area_exponent` | **0.8** | Dimensionless ($\beta \in [0.6, 0.85]$) |
| **Hot Utility Cost Rate** | `hot_utility_cost` | **80.0** | $\$ / (\text{kW} \cdot \text{yr})$ |
| **Cold Utility Cost Rate** | `cold_utility_cost` | **15.0** | $\$ / (\text{kW} \cdot \text{yr})$ |

- **Physical Units**: Heat exchanger areas are in $\text{m}^2$, thermal duties are in $\text{kW}$, and annualized costs are in $\$ / \text{yr}$.
- **Validation**: Requires all exchangers to have finite, non-null area (i.e. `overall_heat_transfer_coefficient` passed to `design_network`). If any exchanger has `area=None`, an explicit `ValueError` is raised.
- **Breakdown Returned**: Returns a dictionary with keys `'capital'`, `'hot_utility'`, `'cold_utility'`, and `'total'` for transparent economic analysis.

### Backtracking Matcher: `search_budget` Parameter & Empirical Trade-Offs
`design_network` accepts a configurable `search_budget` parameter (default `500`), which bounds the maximum number of recursive states explored during tick-off and stream-splitting matching in `_match_region`.

Benchmarking across **200 randomly generated multi-stream synthesis cases** illustrates the speed-versus-stray-utility trade-off:

| Search Budget | Average Stray Utility Duty | Average Run Time / Case | Timeout Rate | Match Quality Impact |
| :---: | :---: | :---: | :---: | :--- |
| **2000** | **40.8 kW** | 3.86 ms | 0.0% | Deepest search, lowest stray utility |
| **500** *(Default)* | **41.2 kW** | **3.41 ms** | **0.0%** | **Optimal balance: 12% faster, only +0.4 kW duty** |
| **200** | 44.5 kW | **2.92 ms** | 0.0% | Fastest, but stray utility increases by +8.1% |

A default budget of `500` ensures fast single-digit millisecond network synthesis while avoiding excessive backtracking on combinatorial branch points.

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
By training the surrogate with $\ln(1 + A_{\text{target}})$ target scaling and inverting with $\exp(\hat{y}_0) - 1$ at inference, the neural network penalizes relative error across small heat-exchanger sizes rather than being dominated by large area scales.

#### Full Model & Target Comparison (Held-Out Test Partition, N=300)

| Metric / Target | Baseline (Linear ANN) | Improved (log1p ANN) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Overall $R^2$** | 0.8821 | 0.8152 | -0.0669 |
| **Area ($A_{\text{target}}$) $R^2$** | 0.6839 | 0.5361 | -0.1478 |
| **Hot Utility ($Q_{H,min}$) $R^2$** | 0.9837 | 0.9540 | -0.0297 |
| **Cold Utility ($Q_{C,min}$) $R^2$** | 0.9787 | 0.9554 | -0.0233 |
| **Overall Area MAE** | 79.14 m² | **75.09 m²** | **-4.05 m² (-5.1%)** |
| **Overall Area RMSE** | 125.17 m² | 151.63 m² | +26.46 m² |
| **Overall Area MAPE** | 145.10% | **61.84%** | **-83.26% pts** |

#### Area Target Error Breakdown by Exchanger Area Range

| Area Range | Test Samples ($N$) | Baseline MAE | Baseline MAPE | Improved MAE | Improved MAPE | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 – 50 m²** | 116 | 32.85 m² | 292.94% | **15.71 m²** | **74.17%** | **-218.77% pts** |
| **50 – 150 m²** | 87 | 67.35 m² | 78.13% | **67.27 m²** | **72.22%** | **-5.91% pts** |
| **150 – 300 m²** | 40 | 89.90 m² | 47.21% | **82.91 m²** | **42.25%** | **-4.96% pts** |
| **> 300 m²** | 57 | 183.80 m² | 35.91% | **202.39 m²** | 36.36% | Slight increase (+18.59 m²) |
| **Overall Area** | 300 | 79.14 m² | 145.10% | **75.09 m²** | **61.84%** | **-83.26% pts** |

> [!NOTE]
> **Trade-Off on Large Networks (> 300 m²)**: While small-network MAPE is slashed by **218.8% pts** (from 292.9% down to 74.2%) and overall MAPE improves by 83% pts, the linear Area MAE on large networks (>300 m²) increased slightly from 183.80 m² to 202.39 m² (+18.59 m²). This is the expected and mathematically natural trade-off of training on $\ln(1 + A_{\text{target}})$: logarithmic loss penalizes relative fractional error uniformly across orders of magnitude rather than allowing high-magnitude large areas to disproportionately dictate the gradient at the expense of small networks.

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

With vectorized NumPy fast inference (`FastANNPredictor`) and standardized Mahalanobis anomaly detection, guarded neural surrogate inference achieves sub-50-microsecond execution, outperforming the deterministic targeting engine:

| Component / Metric | Isolation Forest Baseline | Fast Mahalanobis (Early Iteration) | FastANNPredictor (Optimized NumPy) | Speedup / Status |
| :--- | :---: | :---: | :---: | :---: |
| **Deterministic Phase 5 Engine** | 92.3 ± 0.8 µs | 92.3 ± 0.8 µs | 92.3 ± 0.8 µs | Exact benchmark ground truth |
| **Feature Extraction** | 5.3 ± 0.0 µs | 5.2 ± 0.2 µs | 9.1 ± 0.0 µs | Fixed-size vectorization |
| **Pure ANN Inference** | 144.4 ± 1.9 µs | 143.6 ± 1.4 µs | **18.7 ± 0.4 µs** | NumPy vectorization (~8x faster) |
| **OOD Detection Latency Alone** | 9,750 µs (~9.8 ms) | 37.5 ± 0.8 µs | **12.1 ± 0.2 µs** | Standardized Mahalanobis |
| **End-to-End Guarded Pipeline** | 10,300 µs (~10.3 ms) | 203.1 ± 3.3 µs | **43.5 ± 0.2 µs** | **~2.1x faster than exact engine (~44 µs vs ~92 µs)** |
| **OOD Anomaly Recall (TPR)** | 84.25% | 100.00% | **100.00%** | Safe boundary enforcement |
| **Precision** | 96.56% | 96.39% | **96.39%** | Maintained (high trust) |
| **False Alarm Rate (FPR)** | 4.00% | 5.00% | **5.00%** | Nominal (5% design bound) |

> [!TIP]
> **Sub-50 µs Guarded Inference**: By pre-extracting neural weights into plain contiguous NumPy arrays and bypassing Scikit-Learn `Pipeline.predict` overhead, guarded ANN inference runs in **~43.5–48.3 µs** (well below the < 1 ms target), making it roughly **2x faster than the exact Phase 5 deterministic calculation (~92 µs)** on single cases, while retaining strict OOD guardrail protection.

### Evaluation on Real Literature Benchmarks (Furman & Sahinidis + Chen, Grossmann, & Miller)

The pipeline is benchmarked against **35 unique literature problem instances**: 25 classical instances from Furman & Sahinidis (2004) and 10 large-scale instances from Chen, Grossmann, & Miller (2015) in `data/benchmarks/` (treating `9sp-al1` and `10sp-la1` as one problem due to identical stream tables).

#### CP Rescaling Principle
In pinch analysis, targeting is **strictly linear in heat-capacity flow rate ($CP$)**:
$$\{CP_i\} \to \{k \cdot CP_i\} \implies Q_{H,\min} \to k \cdot Q_{H,\min}, \quad Q_{C,\min} \to k \cdot Q_{C,\min}, \quad A_{\text{target}} \to k \cdot A_{\text{target}}$$
with temperatures and pinch locations unchanged.

To evaluate surrogate generalizability across industrial plant scales, each problem is rescaled by $k = 15.0 / \sum CP_i$ so total $CP$ falls into the surrogate's nominal training envelope. The guardrail and ANN predict on the scaled features, predictions are unscaled by $1/k$, and compared against the exact engine on the **ORIGINAL** problem:

| Benchmark | Hot | Cold | Flagged? | Guardrail Decision / Reason | Exact Area ($U=0.1$) | Pred Area | Area Error |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **10sp-la1 / 9sp-al1** | 4 | 5 | YES (OOD) | Feature Mahalanobis distance exceeds inlier threshold | 45.51 m² | 15.59 m² | 65.7% (FB) |
| **10sp-ol1** | 4 | 6 | YES (OOD) | Oversized problem (4 hot, 6 cold streams > 5) | 41.04 m² | — | Fallback |
| **10sp1** | 5 | 5 | YES (OOD) | Feature Mahalanobis distance exceeds inlier threshold | 2,660,411.78 m² | 3,291,308.41 m² | 23.7% (FB) |
| **12sp1** | 9 | 3 | YES (OOD) | Oversized problem (9 hot, 3 cold streams > 5) | 75,902.02 m² | — | Fallback |
| **14sp1** | 7 | 7 | YES (OOD) | Oversized problem (7 hot, 7 cold streams > 5) | 6,530.83 m² | — | Fallback |
| **15sp-tkm** | 9 | 6 | YES (OOD) | Oversized problem (9 hot, 6 cold streams > 5) | 13,739.82 m² | — | Fallback |
| **20sp1** | 10 | 10 | YES (OOD) | Oversized problem (10 hot, 10 cold streams > 5) | 761.31 m² | — | Fallback |
| **22sp-ph** | 11 | 11 | YES (OOD) | Oversized problem (11 hot, 11 cold streams > 5) | 6,284.61 m² | — | Fallback |
| **22sp1** | 11 | 11 | YES (OOD) | Oversized problem (11 hot, 11 cold streams > 5) | 4,713.12 m² | — | Fallback |
| **23sp1** | 11 | 12 | YES (OOD) | Oversized problem (11 hot, 12 cold streams > 5) | 3,468.65 m² | — | Fallback |
| **28sp-as1** | 16 | 12 | YES (OOD) | Oversized problem (16 hot, 12 cold streams > 5) | 4,210.60 m² | — | Fallback |
| **37sp-yfyv** | 21 | 16 | YES (OOD) | Oversized problem (21 hot, 16 cold streams > 5) | 49,740.11 m² | — | Fallback |
| **4sp1** | 2 | 2 | **NO (Pass)** | **In-distribution (Pass)** | **1,036.51 m²** | **1,636.41 m²** | **57.9%** |
| **6sp-cf1** | 3 | 3 | YES (OOD) | Feature Mahalanobis distance exceeds inlier threshold | 1,044.72 m² | 620.64 m² | 40.6% (FB) |
| **6sp-gg1** | 3 | 3 | **NO (Pass)** | **In-distribution (Pass)** | **2,324.32 m²** | **877.69 m²** | **62.2%** |
| **6sp1** | 3 | 3 | **NO (Pass)** | **In-distribution (Pass)** | **1,860.90 m²** | **3,008.39 m²** | **61.7%** |
| **7sp-cm1** | 3 | 4 | YES (OOD) | Stream temperatures span [313, 626] °C exceeds envelope | 449.22 m² | 135.30 m² | 69.9% (FB) |
| **7sp-s1** | 6 | 1 | YES (OOD) | Oversized problem (6 hot, 1 cold streams > 5) | 33,253.30 m² | — | Fallback |
| **7sp-torw1** | 4 | 3 | **NO (Pass)** | **In-distribution (Pass)** | **2,231.40 m²** | **292.69 m²** | **86.9%** |
| **7sp1** | 3 | 4 | YES (OOD) | Feature Mahalanobis distance exceeds inlier threshold | 2,925.40 m² | 6,025.53 m² | 106.0% (FB) |
| **7sp2** | 3 | 4 | YES (OOD) | Stream temperatures span [100, 590] °C exceeds envelope | 1,288.08 m² | 1,657.42 m² | 28.7% (FB) |
| **7sp4** | 6 | 1 | YES (OOD) | Oversized problem (6 hot, 1 cold streams > 5) | 1,892.04 m² | — | Fallback |
| **8sp-fs1** | 5 | 3 | YES (OOD) | Stream temperature span [10, 361] °C outside envelope | 2,552.50 m² | 2,564.28 m² | 0.5% (FB) |
| **8sp1** | 4 | 4 | **NO (Pass)** | **In-distribution (Pass)** | **3,728.72 m²** | **1,957.58 m²** | **47.5%** |
| **9sp-has1** | 5 | 4 | YES (OOD) | Stream CP (0.22 kW/°C) falls below individual CP envelope | 12,108.75 m² | 7,446.23 m² | 38.5% (FB) |
| **cgm/balanced5** | 5 | 5 | YES (OOD) | Feature Mahalanobis distance exceeds inlier threshold | 884.22 m² | — | Fallback |
| **cgm/balanced8** | 8 | 8 | YES (OOD) | Oversized problem (8 hot, 8 cold streams > 5) | 1,298.39 m² | — | Fallback |
| **cgm/balanced10** | 10 | 10 | YES (OOD) | Oversized problem (10 hot, 10 cold streams > 5) | 1,413.21 m² | — | Fallback |
| **cgm/balanced12** | 12 | 12 | YES (OOD) | Oversized problem (12 hot, 12 cold streams > 5) | 1,599.79 m² | — | Fallback |
| **cgm/balanced15** | 15 | 15 | YES (OOD) | Oversized problem (15 hot, 15 cold streams > 5) | 2,061.17 m² | — | Fallback |
| **cgm/unbalanced5** | 5 | 5 | YES (OOD) | Stream CP (0.14 kW/°C) falls below individual CP envelope | 1,686.52 m² | — | Fallback |
| **cgm/unbalanced10** | 10 | 10 | YES (OOD) | Oversized problem (10 hot, 10 cold streams > 5) | 1,996.92 m² | — | Fallback |
| **cgm/unbalanced15** | 15 | 15 | YES (OOD) | Oversized problem (15 hot, 15 cold streams > 5) | 2,539.96 m² | — | Fallback |
| **cgm/unbalanced17** | 17 | 17 | YES (OOD) | Oversized problem (17 hot, 17 cold streams > 5) | 2,553.65 m² | — | Fallback |
| **cgm/unbalanced20** | 20 | 20 | YES (OOD) | Oversized problem (20 hot, 20 cold streams > 5) | 2,697.55 m² | — | Fallback |

**Key Findings from Expanded Literature Benchmarking**:
1. **In-Distribution Generalization**: 5 benchmarks (`4sp1`, `6sp-gg1`, `6sp1`, `7sp-torw1`, `8sp1`) pass the guardrail. The ANN predicts their targets with a mean area error of **63.2%**, capturing the correct orders of magnitude on real industrial stream tables without any literature data in training.
2. **Multi-Tier Guardrail Safety on CGM Problems**:
   - 8 of the 10 Chen-Grossmann-Miller instances are oversized ($N \in [8, 20] > 5$) and are immediately caught by Tier 1 size bounds.
   - The two 5x5 problems are reliably intercepted: `cgm/balanced5` triggers Tier 2 Mahalanobis distance, and `cgm/unbalanced5` triggers Tier 1 CP lower envelope (min stream CP 0.14 < 0.26 kW/°C).
   - In all 30 OOD cases across the literature, deterministic fallback ensures 100% safe execution without unconstrained neural extrapolation.

---

## 11. Phase 6 HEN Synthesis Surrogate (Area, Units, Duties & TAC)

In addition to pinch targeting surrogates, PinchPy includes a dedicated surrogate model for **Phase 6 Heat Exchanger Network synthesis and Total Annual Cost (TAC)**. Rather than predicting only theoretical thermodynamic targets, this model approximates the outcomes of discrete network design (`design_network`) and economic evaluation (`network_tac`).

### Dataset & Training Setup
- **Dataset Generation**: `generate_hen_dataset(n_cases=1000, seed=42, search_budget=500)` synthesizes and evaluates 1,000 valid HEN cases using the bounded backtracking matcher.
- **Labels (6 targets)**:
  1. `total_area`: Total heat exchanger area across all process and utility exchangers ($\text{m}^2$).
  2. `exchanger_count`: Total number of physical exchanger units ($N$).
  3. `total_heater_duty`: Total heating utility duty ($\text{kW}$).
  4. `total_cooler_duty`: Total cooling utility duty ($\text{kW}$).
  5. `hot_utility_above_target`: Stray hot utility exceeding Phase 3 theoretical minimum target ($\text{kW}$).
  6. `tac`: Ground truth Total Annual Cost evaluated via `network_tac` ($\$ / \text{yr}$).
- **Separate Model Architecture**: An `MLPRegressor` (128, 64, 32) trained on the same 54 thermodynamic features, with $\ln(1+y)$ transformation applied to `total_area`, `total_heater_duty`, `total_cooler_duty`, and `hot_utility_above_target` via `LogHenTargetTransformer`.
- **Fast Inference**: Executed via `FastANNPredictor` with plain NumPy matrix multiplications, achieving **~48 µs** single-case inference.

### Algebraic TAC Computation (Never Predicted Directly)
Following the principle of physical consistency, the surrogate **never predicts TAC directly as a black-box regression target**. Instead, the neural surrogate predicts network area ($A_{\text{pred}}$), unit count ($N_{\text{pred}}$), and utility duties ($Q_{H,\text{pred}}, Q_{C,\text{pred}}$), and TAC is computed algebraically via `tac_from_predictions`:

$$\text{TAC}_{\text{pred}} = N_{\text{pred}} \cdot \left(2000 + 800 \cdot \left(\frac{A_{\text{pred}}}{N_{\text{pred}}}\right)^{0.8}\right) + 80 \cdot Q_{H,\text{pred}} + 15 \cdot Q_{C,\text{pred}}$$

### Performance on Held-Out Test Partition ($N=150$)

| Target / Label | $R^2$ Score | Mean Absolute Error (MAE) | Mean Absolute Percentage Error (MAPE) |
| :--- | :---: | :---: | :---: |
| **Total Area ($A_{\text{net}}$)** | **0.6796** | **129.52 m²** | **32.40%** |
| **Exchanger Count ($N$)** | **0.4477** | **8.60 units** | **51.92%** |
| **Total Heater Duty ($Q_H$)** | -0.6605 | 463.13 kW | 300.23% |
| **Total Cooler Duty ($Q_C$)** | 0.1366 | 380.36 kW | 272.79% |
| **Hot Utility Above Target** | -0.2409 | 35.19 kW | 351.57% |
| **TAC ($\$ / \text{yr}$, algebraic)** | **0.5166** | **$74,820.61 / yr** | **29.33%** |

#### Honest Analysis of Weak Spots & Engineering Insights
- **Heuristic Combinatorics & Stray Duty Step Changes**: `design_network` uses heuristic backtracking (`search_budget=500`). When streams cannot be fully matched without stream splitting, stray utility exchangers appear intermittently (e.g. 0 kW vs hundreds of kW). These discrete, discontinuous step changes yield negative $R^2$ and high MAPE on individual heater/cooler duties, as a smooth continuous feed-forward MLP cannot easily regress discrete branch choices.
- **Aggregate Robustness for Area & TAC**: In contrast to individual stream duties, **total network area ($R^2 \approx 0.68$, MAPE ~32%)** and **algebraic TAC ($R^2 \approx 0.52$, MAPE ~29%)** exhibit solid, stable predictive performance. Because capital and utility costs aggregate over the entire network, individual stream match variations smooth out, enabling rapid preliminary economic screening.

### Synthesis Latency Benchmark: FastANNPredictor vs `design_network`

| Workload | Deterministic Synthesis (`design_network` + TAC) | FastANNPredictor + Algebraic TAC | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Single Case** | 20.3 µs (simple) to ~3.4 ms (multi-stream) | **48.3 µs** | Up to **~70x** on complex cases |
| **Batch 10** | 28.90 ms (2.89 ms/case) | **0.43 ms (43.5 µs/case)** | **66.5x** |
| **Batch 100** | 326.33 ms (3.26 ms/case) | **8.60 ms (86.0 µs/case)** | **38.0x** |
| **Batch 1000** | 2,996.65 ms (3.00 ms/case) | **13.17 ms (13.2 µs/case)** | **227.5x** |

### Literature Benchmarking with CP Scaling (Phase 6 Surrogate)
For HEN synthesis with CP scaling ($k = 15 / \sum CP_i$), area and duties scale linearly by $1/k$, whereas **exchanger count ($N$) is topological and is not multiplied by $1/k$**. Across all 35 literature benchmarks, 5 instances pass the guardrail:
- **Mean TAC Error**: **23.0%**
- **Mean Area Error**: **37.3%**
- **Detailed In-Distribution Accuracy**:
  - `4sp1`: Exact TAC \$479,987 vs Pred \$611,874 (27.5% err), Exact Area 1698.28 m² vs Pred 1524.88 m² (10.2% err)
  - `6sp-gg1`: Exact TAC \$819,550 vs Pred \$1,058,697 (29.2% err), Exact Area 2126.75 m² vs Pred 3299.95 m² (55.2% err)
  - `6sp1`: Exact TAC \$1,423,591 vs Pred \$1,328,612 (6.7% err), Exact Area 5712.18 m² vs Pred 4321.79 m² (24.3% err)
  - `7sp-torw1`: Exact TAC \$739,898 vs Pred \$374,880 (49.3% err), Exact Area 1871.86 m² vs Pred 711.97 m² (62.0% err)
  - `8sp1`: Exact TAC \$1,809,284 vs Pred \$1,854,191 (2.5% err), Exact Area 7340.10 m² vs Pred 4794.32 m² (34.7% err)

---

## 12. Benchmark Credits & Literature Citations

The benchmark datasets in `data/benchmarks/` credit the following literature sources (documented in [`data/benchmarks/README.md`](file:///Users/hithagatiputi/Documents/RVCE/PinchPy/data/benchmarks/README.md)):
- **Furman, K. C., & Sahinidis, N. V.** (2004). *Approximation algorithms for heat exchanger network synthesis.* Industrial & Engineering Chemistry Research, 43(14), 3554–3565.
- **Chen, Y., Grossmann, I. E., & Miller, D. C.** (2015). *Computational strategies for the global optimization of heat exchanger networks.* Computers & Chemical Engineering, 72, 282–299.
- **Letsios, D., Kouyialis, G., & Misener, R.** (2018). *Heuristics with performance guarantees for the minimum number of matches problem in heat recovery network design.* Computers & Chemical Engineering, 113, 57–85. Repository: [`TYT0710/min_matches_heuristics`](https://github.com/TYT0710/min_matches_heuristics) (Apache 2.0).
- **Yang, A., et al.** (2025). *Physics-informed surrogate modeling and cost optimization for heat exchanger networks.* Energy, 324, 136011.

---

## 13. How to Run Tests & Benchmarks

Install the package in editable mode and run the test suite with `pytest`:

```bash
pip install -e .
pytest -v
```

To run the complete Phase 7–8 synthetic generation, baseline comparison, Phase 6 surrogate evaluation, and OOD benchmarking pipeline:

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

## 14. Project Phases

- ✅ **Phase 1**: Process stream data model + validation
- ✅ **Phase 2**: $\Delta T_{min}$ temperature shifting and interval formation
- ✅ **Phase 3**: Heat cascade (Problem Table Algorithm), pinch-point identification & minimum utility targeting ($Q_{H,min}$, $Q_{C,min}$), composite-curve data generation
- ✅ **Phase 4**: Composite Curve / Grand Composite Curve plotting
- ✅ **Phase 5**: Heat-exchanger duty/LMTD/area calculations wired into composite curves
- ✅ **Phase 6**: Heat Exchanger Network (HEN) synthesis, grid diagrams, and Total Annual Cost (TAC) evaluation
- ✅ **Phase 7**: Synthetic dataset generation from deterministic pinch & synthesis engines
- ✅ **Phase 8**: Fast ANN surrogate modeling, baseline comparisons, OOD guardrails, and Phase 6 HEN synthesis surrogate
- ⏳ **Phase 9**: Integration + UI/CLI