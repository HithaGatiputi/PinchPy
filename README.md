# PinchPy — Pinch Analysis & Heat Exchanger Network (HEN) Analysis

**PinchPy** is a Python library for Pinch Analysis, thermal process integration, and Heat Exchanger Network (HEN) synthesis.

> [!NOTE]
> **Current Status: All Phases 1–12 Complete** — Process stream model, ΔTmin shifting & heat cascade, pinch targets, literature benchmark validation (Kemp, Towler & Sinnott, Smith, 4SP1), dataset generation, ANN surrogate targeting, Heat Exchanger Network synthesis & pinch-rule validation, Guthrie/Hall CAPEX & OPEX economics, decarbonization & CO2 tracking, industrial retrofit diagnosis & planning, and interactive Streamlit UI / CLI / downloadable engineering reports.

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
- `StreamValidationError` (subclass of `ValueError`): Raised when physical thermodynamic constraints are violated.
- `StreamTypeError` (subclass of `StreamValidationError` and `TypeError`): Raised when non-numeric or non-string parameters are passed.

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

hot = Stream(id="H1", type="hot", supply_temperature=180, target_temperature=80, cp=2.5)
cold = Stream(id="C1", type="cold", supply_temperature=30, target_temperature=150, cp=3.0)

print(f"H1 ΔT = {hot.delta_T:.0f} °C, Q = {hot.Q:.0f} kW")
print(f"C1 ΔT = {cold.delta_T:.0f} °C, Q = {cold.Q:.0f} kW")
```

---

## 6. ΔTmin Temperature Shifting & Intervals

Before streams can be compared on a single scale, every hot stream is shifted down by $\Delta T_{min}/2$ and every cold stream is shifted up by $\Delta T_{min}/2$.

```python
from pinchpy.analysis.temperature_intervals import shifted_temperatures, build_intervals

temps = shifted_temperatures([hot, cold], delta_t_min=10)
intervals = build_intervals([hot, cold], delta_t_min=10)
```

---

## 7. Heat Cascade & Pinch-Point Targeting (Phase 5)

The heat cascade (Problem Table Algorithm) walks the shifted intervals from the highest temperature down, accumulating net heat balances. Shifting that running total so its lowest point is exactly zero yields the **minimum hot utility** ($Q_{H,min}$); the cascade's final value is the **minimum cold utility** ($Q_{C,min}$), and the zero-point is the **pinch point**.

```python
from pinchpy.models.stream import Stream
from pinchpy.analysis.pinch_point import utility_targets

streams = [
    Stream("H1", "hot", 170, 60, 3.0),
    Stream("H2", "hot", 150, 30, 1.5),
    Stream("C1", "cold", 20, 135, 2.0),
    Stream("C2", "cold", 80, 140, 4.0),
]

result = utility_targets(streams, delta_t_min=10)
print(result)
# {
#   "minimum_hot_utility": 20.0,
#   "minimum_cold_utility": 60.0,
#   "pinch_shifted_temperature": 85.0,
#   "pinch_hot_temperature": 90.0,
#   "pinch_cold_temperature": 80.0,
#   "pinch_shifted_temperatures": [85.0],
#   "heat_recovery": 450.0,
#   "is_threshold_problem": False,
# }
```

---

## 8. Benchmark Validation (Phase 6)

PinchPy is strictly validated against published literature benchmarks in [`docs/benchmarks.md`](docs/benchmarks.md) and [`pinchpy.analysis.benchmarks`](src/pinchpy/analysis/benchmarks.py):
- **Ian C. Kemp (2007)**: 4-stream benchmark at $\Delta T_{min} = 10\,^\circ\text{C}$ and $20\,^\circ\text{C}$.
- **G. Towler & R. Sinnott / B. Linnhoff et al. (1982)**: 4-stream benchmark at $\Delta T_{min} = 20\,^\circ\text{C}$.
- **Robin Smith (2005)**: 2-stream benchmark (Table 16.1) at $\Delta T_{min} = 10\,^\circ\text{C}$.
- **Lee, Masso & Rudd (1970) / Papoulias & Grossmann (1983)**: 4SP1 benchmark at $\Delta T_{min} = 20\,^\circ\text{C}$ and $10\,^\circ\text{C}$.

All benchmark tests pass with exact numerical agreement (`tests/test_benchmarks.py`).

---

## 9. Dataset Generation (Phase 7)

High-quality synthetic datasets obeying all thermodynamic laws can be generated directly using the exact pinch engine:

```python
from pinchpy.ml.dataset import generate_dataset

df = generate_dataset(n_cases=1000, seed=42, output_path="data/raw/pinch_dataset_1000.csv")
print(df.shape)  # (1000, 46)
```

See [`notebooks/05_generate_ml_dataset.ipynb`](notebooks/05_generate_ml_dataset.ipynb) for an interactive walkthrough.

---

## 10. ANN Surrogate Model Training & Inference (Phase 8)

Train a Multi-Layer Perceptron (ANN) to surrogate the pinch targeting engine for real-time target estimation:

```python
from sklearn.model_selection import train_test_split
from pinchpy.ml.train import train_ann
from pinchpy.ml.predict import predict_case, predict_targets
from pinchpy.ml.evaluate import evaluate_regression

X, y = df.iloc[:, :-3], df.iloc[:, -3:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_ann(X_train, y_train, hidden_layer_sizes=(128, 64), random_state=42)

# Evaluate on test set (achieves R² > 0.90)
y_pred = predict_targets(model, X_test)
metrics = evaluate_regression(y_test, y_pred)
print(f"R² Score: {metrics['r2']:.4f}")

# Direct inference from Stream objects
predicted = predict_case(model, streams, delta_t_min=10.0)
print(predicted)
```

See [`notebooks/06_train_ml_model.ipynb`](notebooks/06_train_ml_model.ipynb) for model training, metrics breakdown, and parity plots.

---

## 11. Area Targeting & HEN Synthesis

- **Area Targeting**: `pinchpy.analysis.area_target.area_target(streams, delta_t_min, U)`
- **HEN Design**: `pinchpy.synthesis.network_design.design_network(streams, delta_t_min)`
- **Grid Plot**: `pinchpy.visualization.network_plot.plot_network(streams, network)`

---

## 12. Running the User Interface & CLI

### Interactive Web Application (Streamlit)
Launch the interactive web application to input streams, inspect composite curves, view HEN grid diagrams, evaluate economics, run retrofit diagnoses, and export reports:
```bash
pinchpy app
# or: streamlit run src/pinchpy/ui/app.py
```

### Command-Line Interface (CLI)
```bash
# Calculate pinch points and minimum utility targets
pinchpy analyze --benchmark kemp_2007_4stream --dtmin 10

# Synthesize a heat exchanger network
pinchpy design --benchmark kemp_2007_4stream --dtmin 10 -U 0.5

# Generate downloadable audit report (HTML, Markdown, or JSON)
pinchpy report --benchmark kemp_2007_4stream --dtmin 10 --out report.html
```

---

## 13. Running Tests

```bash
pip install -e .
pytest -v
```

All **314 tests** pass (313 passed, 1 xpassed) across unit calculations, targeting, benchmarks, ML surrogate models, network synthesis, economics, retrofit diagnostics, and reporting interfaces.

---

## 14. Project Roadmap (All Completed)

- ✅ **Phase 1**: Process stream data model + physical validation
- ✅ **Phase 2**: $\Delta T_{min}$ temperature shifting and interval formation
- ✅ **Phase 3**: Heat cascade (Problem Table Algorithm)
- ✅ **Phase 4**: Composite Curve & Grand Composite Curve plotting
- ✅ **Phase 5**: Pinch-point identification and minimum utility targeting ($Q_{H,min}$, $Q_{C,min}$)
- ✅ **Phase 6**: Validation against published literature benchmarks (Kemp, Towler & Sinnott, Robin Smith, 4SP1)
- ✅ **Phase 7**: Synthetic dataset generation using the validated engine (1,000 cases, 43 features)
- ✅ **Phase 8**: ANN surrogate targeting with MLP training & evaluation ($R^2 > 0.91$)
- ✅ **Phase 9**: Heat Exchanger Network (HEN) synthesis, stream splitting, and pinch-rule validation
- ✅ **Phase 10**: Cost and $	ext{CO}_2$ analysis (Guthrie/Hall CAPEX, OPEX, payback, decarbonization)
- ✅ **Phase 11**: Industrial retrofit analysis (cross-pinch detection, utility penalties, modification plans)
- ✅ **Phase 12**: Interactive Streamlit UI, CLI, and downloadable engineering reports (HTML, MD, JSON, CSV)
