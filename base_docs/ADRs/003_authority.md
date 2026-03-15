# ADR-003: Authority of Declarations Over Inference

**Status:** Accepted
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
- Deployment status is declared in `config_deployment.py`
- Hyperparameters and temporal settings are declared in `config_hyperparameters.py`
- Partition boundaries are declared in `common/partitions.py` (single source of truth)
- Model name must match the directory name (enforced by `test_config_completeness.py`)

### Fail-Loud Invariant

When a required declaration is missing or invalid:
- The system must fail explicitly, not infer a default
- `config_meta.py` must contain all required keys: `name`, `algorithm`, `level`, `creator`, `prediction_format`, `rolling_origin_stride`
- `config_hyperparameters.py` must contain: `steps`, `time_steps`
- `config_deployment.py` must contain: `deployment_status` (one of: `shadow`, `deployed`, `baseline`, `deprecated`)

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
