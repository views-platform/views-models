# ADR-005: Testing as Mandatory Critical Infrastructure

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models had zero test coverage until 2026-03-15. Configuration drift, partition divergence, and CLI pattern inconsistency accumulated undetected. The heat_waves/hot_stream offset bug and the 13 old-pattern CLI models were only discovered during a manual assimilation review.

---

## Decision

Tests are mandatory critical infrastructure, not optional documentation.

### Test Categories

We adopt a three-team testing taxonomy:

| Team | Purpose | Examples in this repo |
|------|---------|----------------------|
| **Green** (Correctness) | Verify the system works as intended | `test_config_completeness.py` — required keys exist, values are valid |
| **Beige** (Convention) | Catch configuration drift and convention violations | `test_model_structure.py` — naming, file presence; `test_config_partitions.py` — delegation to shared module; `test_cli_pattern.py` — CLI import consistency |
| **Red** (Adversarial) | Expose failure modes by testing edge cases | Not yet implemented — future work |

### Test Design Principles

1. **Tests must run without ML dependencies** — Tests parse source code and use `importlib.util` to load config modules, avoiding dependency on `views_pipeline_core`, `ingester3`, or algorithm packages.
2. **Tests are parametrized over all models** — Every test runs against all ~66 models, catching drift immediately.
3. **Tests run fast** — The full suite completes in ~2 seconds.

### Current Test Suite

| File | Category | What it validates |
|------|----------|-------------------|
| `tests/test_config_completeness.py` | Green | Required config keys, valid values, `time_steps == len(steps)` |
| `tests/test_config_partitions.py` | Beige | Shared partition module correctness, model delegation |
| `tests/test_model_structure.py` | Beige | Naming convention, required files, config directory structure |
| `tests/test_cli_pattern.py` | Beige | New CLI import pattern, no explicit `wandb.login()` |
| `tests/test_catalogs.py` | Green | No `exec()` usage, markdown generation correctness |

### Test Requirements for Changes

- Adding a new required config key: add the key to `REQUIRED_META_KEYS` or `REQUIRED_HP_KEYS` in `test_config_completeness.py`
- Adding a new model: it must pass all existing tests (enforced by parametrization)
- Changing partition boundaries: update each model's `config_partitions.py` and the expected values in `test_config_partitions.py`

---

## Known Gaps

- No red-team (adversarial) tests yet
- Catalog generation function tests require `views_pipeline_core` (skipped in most dev environments)
- No cross-validation between `config_meta.algorithm` and `main.py` manager import
- No ensemble config tests
- Tests are not wired into CI (`.github/workflows/`)

---

## Consequences

### Positive
- Config drift is now detectable before merge
- Partition consistency is enforced across all models
- New models automatically inherit all convention tests

### Negative
- Source-based tests cannot validate runtime behavior
- Tests must be kept in sync with evolving config requirements

---

## References

- `tests/conftest.py` — test infrastructure and fixtures
- ADR-003 (Authority of Declarations)
- ADR-008 (Observability)
