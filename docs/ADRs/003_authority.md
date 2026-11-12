# ADR-003: Authority of Declarations Over Inference

**Status:** Accepted — amended 2026-09-17 (maturity declaration: `config_maturity.py` → `maturity`, ADR-017 Phase 2, #449)
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

In a monorepo with ~66 models, it is tempting to infer model properties from directory names, file patterns, or import statements. This leads to fragile conventions that break silently when someone deviates from the expected pattern.

The system has already experienced this: `create_catalogs.py` originally used `exec()` to load configs, and the `heat_waves`/`hot_stream` models silently diverged from the standard forecasting offset without anyone noticing.

---

## Decision

**All meaningful model properties must be explicitly declared in configuration files, not inferred from structure.**

Specifically:
- Model algorithm, level of analysis, targets, and creator are declared in `config_meta.py`
- Maturity (`candidate | graduate | retired`, ADR-017 §3) is declared in `config_maturity.py`; sources whose engine is still on pipeline-core 2.x declare the legacy `deployment_status` in `config_deployment.py`, translated by ADR-017 §3 — one file per source, never both
- Hyperparameters and temporal settings are declared in `config_hyperparameters.py`
- Partition boundaries are declared in each model's self-contained `config_partitions.py` (consistency enforced by tests)
- Model name must match the directory name (enforced by `test_config_completeness.py`)

### Fail-Loud Invariant

When a required declaration is missing or invalid:
- The system must fail explicitly, not infer a default
- `config_meta.py` must contain all required keys: `name`, `algorithm`, `level`, `creator`, `prediction_format`, `rolling_origin_stride`
- `config_hyperparameters.py` must contain: `steps`, `time_steps`
- `config_maturity.py` must contain: `maturity` (one of: `candidate`, `graduate`, `retired`); the legacy `config_deployment.py` must contain: `deployment_status` (one of: `shadow`, `deployed`, `baseline`, `deprecated`)

### Forbidden Behaviors

- Inferring model level (cm/pgm) from directory name or queryset
- Inferring algorithm type from import statements
- Using filename-based logic to determine model behavior
- Silently defaulting missing config keys

---

## Consequences

### Positive
- Config completeness is testable (see `tests/test_config_completeness.py`)
- Catalog generation reads from declarations, not heuristics
- New required keys can be added and enforced via tests

### Negative
- Adding a new required key requires updating all ~66 models
- Config files contain some redundancy (e.g., `time_steps` duplicates `len(steps)`)

---

## References

- ADR-001 (Ontology)
- `tests/test_config_completeness.py` — enforces required keys
- `create_catalogs.py` — reads declarations to build catalogs
