# PinchPy — Pinch Analysis & Heat Exchanger Network (HEN) Analysis

**PinchPy** is a Python library for Pinch Analysis, thermal process integration, and Heat Exchanger Network (HEN) synthesis.

> [!NOTE]
> **Current Status: Phase 1 — Process Stream Data Model + Validation.**
> Future modules (heat cascade, utility targeting, composite curves, HEN synthesis, and ANN approximation) are currently planned future work and have not yet been released.

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

## 6. How to Run Tests

Install the package in editable mode and run the test suite with `pytest`:

```bash
pip install -e .
pytest -v
```

---

## 7. Planned Future Phases (Upcoming)

The following capabilities are planned for subsequent phases and will build on top of the Phase 1 Stream model:

- **Phase 2**: $\Delta T_{min}$ temperature shifting and interval formation
- **Phase 3**: Heat cascade & Problem Table Algorithm
- **Phase 4**: Pinch temperature & minimum utility targeting ($Q_{H,min}$, $Q_{C,min}$)
- **Phase 5**: Composite Curves (CC) and Grand Composite Curve (GCC)
- **Phase 6**: Heat Exchanger Network (HEN) synthesis and grid diagrams
- **Phase 7**: Surrogate modelling & ANN prediction
