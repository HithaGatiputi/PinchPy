# PinchPy — Pinch Analysis & Heat Exchanger Network (HEN) Analysis

**PinchPy** is a Python library for Pinch Analysis, thermal process integration, and Heat Exchanger Network (HEN) synthesis.

> [!NOTE]
> **Current Status: Phases 1–5 and 7–8 complete — Stream model + validation, ΔTmin temperature shifting/intervals, heat cascade / pinch-point / utility-targeting engine, Composite Curve / Grand Composite Curve plotting, composite-curve area targeting, and ANN surrogate modeling with OOD Guardrails.**
> HEN synthesis (Phase 6) and UI/CLI (Phase 9) are planned future work.

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

---

## 9. Machine Learning Surrogate & OOD Guardrail (Phases 7–8)

PinchPy incorporates a machine-learning surrogate model (Scikit-Learn `MLPRegressor` pipeline) for rapid simultaneous prediction of:
- **Primary Target**: Total Minimum Area $A_{\text{target}}$ ($\text{m}^2$)
- **Secondary Targets**: Minimum Hot Utility $Q_{H,min}$ ($\text{kW}$) and Minimum Cold Utility $Q_{C,min}$ ($\text{kW}$)

### Research Contribution: Out-of-Distribution (OOD) Guardrail
Surrogate neural networks in process engineering can extrapolate dangerously when queried outside their training regime or produce thermodynamically unphysical outputs. PinchPy solves this with a multi-tier **OOD Guardrail**:
1. **Domain Bounding**: Validates stream counts, temperature spans, $\Delta T_{min}$, and heat transfer coefficient $U$ against the training envelope.
2. **Feature Density Anomaly Detection**: Uses a trained `IsolationForest` to assess the likelihood of incoming feature representations.
3. **Physical Consistency Checks**: Validates First Law energy conservation ($Q_C - Q_H = \sum Q_H - \sum Q_C$) and non-negativity ($A > 0, Q_H \ge 0, Q_C \ge 0$).
4. **Automatic Deterministic Fallback**: If an input is flagged as OOD or physically inconsistent, the system transparently falls back to the exact deterministic Phase 5 engine (`area_target`), guaranteeing safe, trusted results.

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

# Single-case in-distribution inference
res = predict_network_targets(streams, delta_t_min=10.0, overall_heat_transfer_coefficient=0.1, model=model, guardrail=guardrail)
print(res)
# {'total_area': 162.89, 'minimum_hot_utility': 73.70, 'minimum_cold_utility': 270.55, 'source': 'ann_surrogate', 'is_ood': False}

# OOD query (e.g., extreme ΔTmin=120 °C): automatically falls back to deterministic engine
res_ood = predict_network_targets(streams, delta_t_min=120.0, overall_heat_transfer_coefficient=0.1, model=model, guardrail=guardrail)
print(res_ood)
# {'total_area': 0.0, 'minimum_hot_utility': 226.70, 'minimum_cold_utility': 0.0, 'source': 'deterministic_fallback', 'is_ood': True}

# Vectorized batch screening (high-throughput plant scoping)
cases = [(streams, dt, 0.1) for dt in [10.0, 15.0, 20.0, 25.0]]
batch_results = predict_batch_network_targets(cases, model=model, guardrail=guardrail)
print(f"Screened {len(batch_results)} network candidates concurrently.")
```

### Measured Performance & Batch Speedup

| Metric | Deterministic Phase 5 | Pure ANN Inference | Guarded ANN Pipeline | Batch ANN (N=2,000) |
| :--- | :---: | :---: | :---: | :---: |
| **Single-Case Latency** | 145.4 µs | 245.2 µs | 10.3 ms | **83.2 µs / case** |
| **Throughput** | ~6,880 / s | ~4,080 / s | ~97 / s | **~12,020 / s** |
| **Speedup vs Deterministic** | 1.00x | 0.59x | 0.014x | **1.95x** |


---

## 10. How to Run Tests & Benchmarks

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

## 11. Project Phases

- ✅ **Phase 1**: Process stream data model + validation
- ✅ **Phase 2**: $\Delta T_{min}$ temperature shifting and interval formation
- ✅ **Phase 3**: Heat cascade (Problem Table Algorithm), pinch-point identification & minimum utility targeting ($Q_{H,min}$, $Q_{C,min}$), composite-curve data generation
- ✅ **Phase 4**: Composite Curve / Grand Composite Curve plotting
- ✅ **Phase 5**: Heat-exchanger duty/LMTD/area calculations wired into composite curves
- ⏳ **Phase 6**: Heat Exchanger Network (HEN) synthesis and grid diagrams
- ✅ **Phase 7**: Synthetic dataset generation from the deterministic pinch engine
- ✅ **Phase 8**: ANN (MLP) surrogate model training, baseline comparisons & OOD guardrail
- ⏳ **Phase 9**: Integration + UI/CLI