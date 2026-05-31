# ADR-012: Target Scale and Prefix Convention

**Status:** Proposed
**Date:** 2026-05-30
**Deciders:** Simon, VIEWS platform team

---

## Context

The VIEWS platform has historically used column-name prefixes to encode the mathematical transformation applied to a target variable: `ln_` for natural logarithm, `lx_` for offset logarithm, `lr_` for linear (untransformed). Downstream consumers — evaluation, ensembles, reporting — would inspect these prefixes to infer the data's scale.

This convention is being retired across the platform. views-pipeline-core ADR-040 forbids inferring semantics from data content ("the no-sniffing rule"). views-reporting ADR-011 states that data must arrive on its original measurement scale. What remains undocumented is the producer side: what this repo is responsible for, what the modeling libraries are responsible for, and what prefixes mean in current practice.

---

## Decision

### 1. Active target prefixes

Two target prefixes are in active use:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `lr_` | **Linear.** The value is on its original measurement scale (e.g., event counts). | `lr_sb_best` |
| `by_` | **Binary.** A classification target derived from count data by the modeling library (e.g., `lr_sb_best > 0 → by_sb_best`). | `by_sb_best` |

All other prefixes (`ln_`, `lx_`, etc.) are **deprecated** and must not appear as targets in new model configurations.

### 2. Transform responsibility belongs to the modeling library

Each modeling library (views-hydranet, views-stepshifter, views-baseline, views-r2darts2) owns the full transform lifecycle for its models:

- The library applies transforms on ingestion (e.g., `log1p`).
- The library inverts transforms before emitting predictions.
- The library may use any transform or chain of transforms internally — this is opaque to the rest of the pipeline.

This repo declares *which* transforms to apply (via the `transformations` dict in `config_hyperparameters.py`), but the declaration is an instruction to the modeling library, not to pipeline-core or to this repo's infrastructure.

Neither views-pipeline-core nor any other infrastructure component applies or reverses target transformations. This is a platform-wide design invariant, not merely current behavior. If a contribution to any repo introduces transform logic outside the modeling library, that is a bug.

### 3. Predictions leave this repo on measurement scale

Every prediction emitted by a model or ensemble in this repo is on its original measurement scale. This is a precondition for:

- Evaluation (views-pipeline-core evaluation stages)
- Ensemble aggregation (concat, mean, or any other method)
- Downstream consumers (views-reporting, prediction store, views-faoapi)

### 4. Ensembles do not handle transforms

Ensembles receive measurement-scale predictions from their constituent models and emit measurement-scale aggregated predictions. They have no transform awareness and must not need any. If a future ensemble design requires internal transformations, it must invert them before output — same principle as individual models.

### 5. Binary targets are not a scale transformation

The `by_` prefix denotes a different view of the data (binary: did an event occur?), not a different scale of the same quantity. Binary targets are derived from count data by the modeling library (e.g., in HydraNet's model class) because evaluation requires both regression and classification outputs. This is a modeling concern, not a pipeline concern.

---

## Rationale

The alternative — propagating transform metadata through the pipeline so that downstream consumers can invert — was the legacy approach. It failed because:

- It coupled every consumer to the producer's internal transform choices.
- It required every aggregation and evaluation function to handle all possible transforms.
- It encoded ephemeral implementation details (which transform was used) into column names, which then leaked into data schemas, file formats, and APIs.

Placing the full lifecycle in the modeling library keeps the contract simple: data in, measurement-scale predictions out. The modeling library is the only component that needs to know what transforms it uses.

---

## Consequences

### Positive

- Model contributors know exactly what they configure here (target names, transform declarations) and what they don't need to worry about (inversion — that's the library's job).
- Ensembles and evaluation can treat all predictions uniformly.
- No transform metadata needs to propagate through prediction files, stores, or APIs.

### Negative

- If a modeling library has a bug in its inverse transform, the error propagates silently as wrong-scale predictions. There is no downstream check that can catch this without domain knowledge of expected value ranges.
- The `by_` convention is a pragmatic compromise, not a clean abstraction. It exists because evaluation needs classification targets and the modeling library derives them internally rather than fetching them from the feature store.

---

## Implementation Notes

- The `transformations` dict in `config_hyperparameters.py` (e.g., `{'log1p': ['lr_sb_best', ...]}`) is passed to the modeling library. This repo does not execute it.
- The `derivations` dict in `config_hyperparameters.py` (e.g., `{'binary': [{'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0}]}`) instructs the modeling library to derive binary targets. This repo does not execute it.
- New models must use `lr_` for regression targets and `by_` for classification targets. No other prefixes are permitted.

---

## Validation & Monitoring

- Existing tests (`test_config_parity.py`, `test_model_configs.py`) can be extended to assert that all `regression_targets` use the `lr_` prefix and all `classification_targets` use the `by_` prefix.
- Any model producing predictions on a non-measurement scale is a bug in the modeling library, not in this repo. Such bugs would manifest as anomalous evaluation metrics (e.g., CRPS or MSE orders of magnitude off expected ranges).

---

## References

- views-pipeline-core ADR-040: Authority of declarations over inference (the no-sniffing rule)
- views-reporting ADR-011: Data arrives on its original measurement scale
- views-pipeline-core ADR-014: Model definition and structure
