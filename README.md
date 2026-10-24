# PinchPy — Heat Exchanger / Pinch Analysis + ANN

PinchPy is the project codebase for analysing process streams, calculating pinch-analysis utility targets, supporting heat-exchanger-network work, and applying a **feed-forward Artificial Neural Network (ANN / MLP)** as the primary ML model.

## Core project flow

**Process stream data → validation → pinch analysis → heat cascade → utility targets / pinch → HEN calculations → ANN prediction**

The ANN is trained on cases whose labels come from the deterministic pinch-analysis engine. This keeps the ML layer tied to physically meaningful ground-truth calculations.

## ANN objective

The primary model is a scikit-learn `MLPRegressor` with:

- input features: stream supply/target temperatures, heat-capacity flow rates, stream duties, ΔTmin, and stream counts;
- hidden layers: dense ReLU layers (default `64 → 32`);
- optimizer: Adam;
- outputs: minimum hot utility, minimum cold utility, and shifted pinch temperature.

XGBoost is **not** a required model in this project. The ANN/MLP is the main model.

## Repository structure

- `src/pinchpy/models/` — stream, exchanger and network data models
- `src/pinchpy/analysis/` — temperature intervals, heat cascade, pinch and utility targets
- `src/pinchpy/calculations/` — duty, LMTD, heat-transfer and area calculations
- `src/pinchpy/synthesis/` — HEN matching/design/evaluation scaffolding
- `src/pinchpy/ml/` — dataset generation, ANN training, prediction, evaluation and features
- `src/pinchpy/visualization/` — plotting scaffolding
- `tests/` — automated tests
- `notebooks/` — step-by-step project demonstrations
- `data/` — raw/processed/synthetic datasets
- `models/` — trained ANN artifacts
- `docs/` — methodology, equations and architecture

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

Generate training data from the pinch engine:

```python
from pinchpy.ml.dataset import generate_dataset

df = generate_dataset(n_cases=2000, seed=42)
```

Train the ANN:

```python
from pinchpy.ml.train import train_ann

X = df.iloc[:, :-3]
y = df.iloc[:, -3:]
model = train_ann(X, y)
```

## Important modelling principle

The deterministic pinch-analysis calculation is the source of truth. The ANN is an approximation layer intended to learn the relationship between process-stream inputs and pinch-analysis targets; it should be validated against exact calculations before being used for engineering decisions.
