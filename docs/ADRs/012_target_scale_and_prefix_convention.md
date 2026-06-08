# ADR-012: Target Scale and Prefix Convention

**Status:** Active
**Date:** 2026-05-30 (updated 2026-06-08)
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** views-pipeline-core ADR-055 (Raw-Space Model I/O Contract)

---

## Context

The VIEWS platform has historically used column-name prefixes to encode the mathematical transformation applied to a target variable: `ln_` for natural logarithm, `lx_` for offset logarithm, `lr_` for linear (untransformed). Downstream consumers — evaluation, ensembles, reporting — would inspect these prefixes to infer the data's scale.

This convention is retired across the platform. views-pipeline-core ADR-055 (Raw-Space Model I/O Contract) ratifies the binding rule: **models return predictions in raw target space, transforms are model-internal, config-declared, and inverted before output.** views-pipeline-core ADR-003 (Authority of Declarations over Inference) forbids inferring semantics from data content. This ADR documents what those platform-wide decisions mean for this repo specifically — what views-models is responsible for, what it delegates to modeling libraries, and what must never happen in config files.

### The motivating incident

Commit `5fcfe43` (2025-11-20) silently added `np.log1p`/`np.expm1` inside views-stepshifter, forcing log-space training with no config declaration. Commit `08ee2eb` (2026-04-11) reverted it. During the 5-month window between those commits, trained artifacts may have been serialized in log-space. This is the kind of ambiguity this ADR — and ADR-055 — exists to eliminate.

---

## Decision

### 1. Active target prefixes

Two target prefixes are in active use:

| Prefix | Meaning | Example |
|--------|---------|---------|
| `lr_` | **Linear.** The value is on its original measurement scale (e.g., event counts). The prefix is an identity convention — it does not indicate that any transform has been applied or needs to be undone. | `lr_sb_best` |
| `by_` | **Binary.** A classification target derived from count data by the modeling library (e.g., `lr_sb_best > 0 → by_sb_best`). | `by_sb_best` |

All other prefixes (`ln_`, `lx_`, etc.) are **deprecated** and must not appear as targets in new model configurations.

Per ADR-055 clause 5: a column named `ln_ged_sb` is **not evidence** that the values are in log-space. The prefix is part of the column's identity, not a scale signal. The model's config declaration (the `transformations` dict in `config_hyperparameters.py`) is the sole source of truth for what transform was applied.

### 2. Transform responsibility belongs to the modeling library

Each modeling library (views-hydranet, views-stepshifter, views-baseline, views-r2darts2) owns the full transform lifecycle for its models:

- The library applies transforms on ingestion (e.g., `log1p`).
- The library inverts transforms before emitting predictions.
- The library may use any transform or chain of transforms internally — this is opaque to the rest of the pipeline.

This repo declares *which* transforms to apply (via the `transformations` dict in `config_hyperparameters.py`), but the declaration is an instruction to the modeling library, not to pipeline-core or to this repo's infrastructure. Per ADR-055 clause 3, this config declaration is the **sole source of truth** for numerical scale.

Neither views-pipeline-core nor any other infrastructure component applies or reverses target transformations. Per ADR-055 clause 4, **the model library is the sole owner of inversion.** If a contribution to any repo introduces transform logic outside the modeling library, that is a contract violation.

### 3. Predictions leave this repo on measurement scale

Every prediction emitted by a model or ensemble in this repo is on its original measurement scale. This is a precondition for:

- Evaluation (views-pipeline-core evaluation stages)
- Ensemble aggregation (concat, mean, or any other method)
- Downstream consumers (views-reporting, prediction store, views-faoapi)

Per ADR-055 clause 8: models declaring `output_scale: "natural"` in their config are compliant. Models declaring `output_scale: "log"` are self-declaring non-compliance — this is a transitional state, not permission to remain non-compliant.

### 4. Ensembles do not handle transforms

Ensembles receive measurement-scale predictions from their constituent models and emit measurement-scale aggregated predictions. They have no transform awareness and must not need any. If a future ensemble design requires internal transformations, it must invert them before output — same principle as individual models.

### 5. Queryset-level target transforms are non-compliant

Per ADR-055 clause 6: querysets must deliver target columns in their natural scale. A queryset that applies `.transform.ops.ln()` to a target column (e.g., delivering `ln_ged_sb` instead of raw `ged_sb_best_sum_nokgi`) shifts the scale ambiguity upstream rather than eliminating it.

**Current status (verified 2026-06-08):** Zero models in this repo actively apply `.transform.ops.ln()` to the three target columns (`lr_ged_sb`, `lr_ns_best`, `lr_os_best`). Many models have the transform commented out — confirming it was deliberately removed. Feature columns may still have active log transforms at the queryset level; this is permitted (features are not governed by ADR-055).

Verification tool: `python tools/audit_queryset_transforms.py`

### 6. Binary targets are not a scale transformation

The `by_` prefix denotes a different view of the data (binary: did an event occur?), not a different scale of the same quantity. Binary targets are derived from count data by the modeling library (e.g., in HydraNet's `derivations` config) because evaluation requires both regression and classification outputs. Per views-hydranet ADR-046, this is a **Feature Derivation** (additive, no inversion needed), not a **Value Transformation** (in-place, must invert).

---

## Rationale

The alternative — propagating transform metadata through the pipeline so that downstream consumers can invert — was the legacy approach. It failed because:

- It coupled every consumer to the producer's internal transform choices.
- It required every aggregation and evaluation function to handle all possible transforms.
- It encoded ephemeral implementation details (which transform was used) into column names, which then leaked into data schemas, file formats, and APIs.
- PredictionFrame (ADR-042) has no column names and no scale metadata — it cannot carry prefix-based scale signals. This is by design, not a limitation.

Placing the full lifecycle in the modeling library keeps the contract simple: data in, measurement-scale predictions out. The modeling library is the only component that needs to know what transforms it uses.

---

## Consequences

### Positive

- Model contributors know exactly what they configure here (target names, transform declarations) and what they don't need to worry about (inversion — that's the library's job).
- Ensembles and evaluation can treat all predictions uniformly.
- No transform metadata needs to propagate through prediction files, stores, or APIs.
- The audit tool (`tools/audit_queryset_transforms.py`) provides programmatic verification that no queryset-level target transforms are active.

### Negative

- If a modeling library has a bug in its inverse transform, the error propagates silently as wrong-scale predictions. ADR-055 clause 8 notes that ensemble-level enforcement exists (`validate_output_scale_consistency()`), but single-model enforcement relies on per-repo discipline.
- Frozen artifacts from the `5fcfe43` window (2025-11 → 2026-04) in views-stepshifter may emit log-space predictions when loaded under current raw-output code. These must be audited and retrained.
- The `by_` convention is a pragmatic compromise, not a clean abstraction. It exists because evaluation needs classification targets and the modeling library derives them internally.

---

## Implementation Notes

- The `transformations` dict in `config_hyperparameters.py` (e.g., `{'log1p': ['lr_sb_best', ...]}`) is passed to the modeling library. This repo does not execute it.
- The `derivations` dict in `config_hyperparameters.py` (e.g., `{'binary': [{'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0}]}`) instructs the modeling library to derive binary targets. This repo does not execute it.
- New models must use `lr_` for regression targets and `by_` for classification targets. No other prefixes are permitted.

---

## Validation & Monitoring

- `tools/audit_queryset_transforms.py` — programmatically verifies no queryset-level log transforms are applied to target columns. Run periodically or before release.
- Existing tests (`test_config_completeness.py`) can be extended to assert that all `regression_targets` use the `lr_` prefix and all `classification_targets` use the `by_` prefix.
- Any model producing predictions on a non-measurement scale is a bug in the modeling library, not in this repo. Such bugs would manifest as anomalous evaluation metrics (e.g., CRPS or MSE orders of magnitude off expected ranges).

---

## References

### Platform-wide authority
- [views-pipeline-core ADR-055: Raw-Space Model I/O Contract](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/055_raw_space_model_io_contract.md) — the binding platform-wide contract. This ADR is the views-models expression of ADR-055.
- [views-pipeline-core ADR-003: Authority of Declarations over Inference](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/003_authority_of_declarations_over_inference.md) — the no-sniffing rule. Scale is declared in config, not inferred from column names.
- [views-pipeline-core ADR-042: PredictionFrame Adoption](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/042_prediction_frame_adoption.md) — column-name-free, scale-metadata-free transport.

### Per-repo implementations
- [views-hydranet ADR-046: Symmetric Feature Lifecycle](https://github.com/views-platform/views-hydranet/blob/main/docs/ADRs/active/046_symmetric_feature_lifecycle.md) — transformations (must invert) vs. derivations (no inversion).
- [views-hydranet ADR-003: Philosophy of Engineering](https://github.com/views-platform/views-hydranet/blob/main/docs/ADRs/active/003_philosophy_of_engineering_and_semantic_authority.md) — Law 5 (Explicit Transformation), Law 6 (Prefix-Purity).
- [views-stepshifter ADR-003: Raw Target Space I/O Contract](https://github.com/views-platform/views-stepshifter/blob/main/docs/ADRs/003_raw_target_space_io_contract.md) — 8 decision clauses, enforcement guards, TRANSFORMS registry.
- [views-r2darts2 ADR-012: Scaling Pipeline and Calibration Integrity](https://github.com/views-platform/views-r2darts2/blob/main/docs/ADRs/012_scaling_pipeline_and_calibration_integrity.md) — Darts native Pipeline, global_fit=True.

### This repo
- [ADR-011: Partition Boundary Semantics](011_partition_semantics.md) — `output_scale` optional config key positioned under ADR-055 clause 8.
- `tools/audit_queryset_transforms.py` — programmatic verification of queryset-level transform compliance.
