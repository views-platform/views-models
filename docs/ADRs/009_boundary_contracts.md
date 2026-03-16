# ADR-009: Boundary Contracts and Configuration Validation

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models has several important boundaries where data enters or exits the system:
1. Config files → framework managers (config loading)
2. viewser querysets → data backend (data fetching)
3. Model predictions → ensemble aggregation → reconciliation → output
4. Config files → catalog generation scripts (documentation)

At each boundary, assumptions about data format, required keys, and valid values must be enforced.

---

## Decision

### Configuration Boundaries

Every config file boundary must enforce:

| Config File | Required Keys | Validated By |
|------------|---------------|-------------|
| `config_meta.py` | `name`, `algorithm`, `level`, `creator`, `prediction_format`, `rolling_origin_stride` | `tests/test_config_completeness.py` |
| `config_deployment.py` | `deployment_status` (enum: shadow/deployed/baseline/deprecated) | `tests/test_config_completeness.py` |
| `config_hyperparameters.py` | `steps`, `time_steps` | `tests/test_config_completeness.py` |
| `config_partitions.py` | Self-contained `generate()` function; boundaries must match canonical values; offset must be `-1` | `tests/test_config_partitions.py` |

### Structural Boundaries

| Convention | Rule | Validated By |
|-----------|------|-------------|
| Model naming | `^[a-z]+_[a-z]+$` | `tests/test_model_structure.py` |
| Required files | `main.py`, `run.sh`, `configs/` with 6 config files | `tests/test_model_structure.py` |
| CLI pattern | Import from `views_pipeline_core.cli`, no `wandb.login()` | `tests/test_cli_pattern.py` |

### Ensemble Boundaries

| Convention | Rule | Validated By |
|-----------|------|-------------|
| Reconciliation | PGM ensembles declare `reconcile_with` referencing a CM ensemble | Not yet enforced by tests |
| Ordering | CM ensemble must complete before PGM ensemble | Documented in `docs/monthly_run_guide.md`, not programmatically enforced |

### Catalog Generation Boundary

- Config files are loaded via `importlib.util` (not `exec()`)
- Validated by `tests/test_catalogs.py::TestNoExecUsage`

---

## Known Gaps

- Ensemble `reconcile_with` targets are not validated against existing ensemble directories
- No boundary validation for `config_queryset.py` (requires external packages)
- No validation that `config_meta.algorithm` matches the manager import in `main.py`

---

## Consequences

### Positive
- Configuration errors are caught by tests before they reach production
- New required keys can be added to the test suite and enforced across all models
- Structural conventions are machine-verifiable

### Negative
- Boundary validation requires keeping test expectations in sync with evolving config requirements
- Some boundaries (queryset, ensemble reconciliation) cannot be tested without external packages

---

## References

- ADR-003 (Authority of Declarations)
- ADR-005 (Testing)
- `tests/` — all boundary validation tests
