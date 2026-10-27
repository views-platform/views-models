# ADR-005: Testing as Mandatory Critical Infrastructure

**Status:** Accepted (amended 2026-07-19: fourth category `live`)
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
| **Red** (Adversarial) | Expose failure modes by testing edge cases | `test_failure_modes.py` — config loading error paths |
| **Live** (Real-world) | Probe real external services (network, credentials); must `pytest.skip` truthfully when the environment lacks access — never a false red | `tests/test_liveness_*.py` — one live probe per surface (amendment 2026-07-19) |

**Amendment 2026-07-19 (falsify audit, register C-103):** green/beige/red all
describe *offline* tests distinguished by the question they ask; tests that
touch the real world are a fourth axis, not a kind of red. The liveness
suites had mislabeled live network probes as `red` (which means
adversarial/error-path); those are now `live`, and `red` marks the actual
error-path tests. Rule: a `live` test asserts only invariants, never
freshness values, and skips with the reason on any environment failure.

### Test Design Principles

1. **Tests must run without ML dependencies** — Tests parse source code and use `importlib.util` to load config modules, avoiding dependency on `views_pipeline_core`, `ingester3`, or algorithm packages.
2. **Tests are parametrized over all models** — Every test runs against all 66 models, catching drift immediately.
3. **Tests run fast** — The full suite completes in ~2 seconds.

### Current Test Suite

| File | Category | What it validates |
|------|----------|-------------------|
| `tests/test_config_completeness.py` | Green | Required config keys, valid values, `time_steps == len(steps)` |
| `tests/test_config_partitions.py` | Beige | Shared partition module correctness, model delegation |
| `tests/test_model_structure.py` | Beige | Naming convention, required files, config directory structure |
| `tests/test_cli_pattern.py` | Beige | New CLI import pattern, no explicit `wandb.login()` |
| `tests/test_catalogs.py` | Green | No `exec()` usage, markdown generation correctness |
| `tests/test_ensemble_configs.py` | Green | Ensemble structure, required keys, constituent model existence and level consistency |
| `tests/test_darts_reproducibility.py` | Green | DARTS reproducibility gate parameter completeness (skipped without `views_r2darts2`) |
| `tests/test_algorithm_coherence.py` | Beige | Algorithm-to-package mapping, requirements.txt consistency with main.py imports |
| `tests/test_failure_modes.py` | Red | Config loading error paths: syntax errors, missing functions, non-existent files |

### Test Requirements for Changes

- Adding a new required config key: add the key to `REQUIRED_META_KEYS` or `REQUIRED_HP_KEYS` in `test_config_completeness.py`
- Adding a new model: it must pass all existing tests (enforced by parametrization)
- Changing partition boundaries: update each model's `config_partitions.py` and the expected values in `test_config_partitions.py`

---

## Known Gaps

- Catalog generation function tests require `views_pipeline_core` (skipped in most dev environments).
  ⚠️ **An instance, not an accepted gap** — a skip standing in for a verdict about the repository is
  what [ADR-025](025_a_check_that_cannot_see_must_refuse.md) §2.5 forbids. The skip is truthful
  where it reports the *environment* (the package is genuinely absent); it is a defect where the
  census it guards derives to empty and the suite still reads green. Converting these is epic #608
  story **S5 (#613)**, which classifies all nine sites one by one.
- DARTS reproducibility tests require `views_r2darts2` (skipped without it); no equivalent for
  stepshifter or baseline. ⚠️ Same distinction, same story (**#613**) — see
  [ADR-025](025_a_check_that_cannot_see_must_refuse.md) §2.5.
- ~~Tests are not wired into CI (`.github/workflows/`) — see Risk Register C-03~~ **Corrected
  2026-10-27:** they are. `run_tests.yml` runs the suite on **push and pull_request to `main` and
  `development`**, and `roster_configs_load.yml`, `runtime_smoke.yml`, `cic_sync_check.yml` and
  `secret_scan.yml` add four more gates. ⚠️ **What C-03 should now say is two narrower residuals,
  and they are different things:**
    - ⛔ **CI cannot tell a quiet test family from a passing one.** `run_tests.yml:41` is a bare
      `pytest`; `pyproject.toml` declares only `markers` — no `testpaths`, no `--strict-markers`,
      no `addopts`, no minimum-collected assertion and no skip budget — and **pytest exits 0 on
      skips.** So a family that silently empties is indistinguishable from one that passed. Both
      extra workflows exist *because this already happened*: *"two production ensemble members sat
      unloadable from August to September while every test passed"*
      (`roster_configs_load.yml:7-11`, #404/#458). Tracked as **#611 (S3)**; the rule is
      [ADR-025](025_a_check_that_cannot_see_must_refuse.md).
    - `run_tests.yml:38` installs only **published** `views_pipeline_core==3.3.0`, so CI cannot
      execute anything depending on an unreleased symbol — which is why
      `roster_configs_load.yml` exists at all: the HydraNet load test `importorskip`s a package
      `run_tests.yml` never installs, so in the main job it has **always** skipped.
- No static validation of queryset correctness — see Risk Register C-02
- Red-team coverage is limited to config loading infrastructure; no adversarial tests for runtime
  behavior. ⚠️ This is the gap [ADR-025](025_a_check_that_cannot_see_must_refuse.md) §1 names as
  **C-106**'s — *"the test architecture verifies declarations exhaustively but runtime behavior
  nowhere in CI"* — self-declared a strategic root, Mitigated with a live residual, and the CI-layer
  statement of ADR-025's own rule.

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
