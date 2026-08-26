# ADR-013: Regression-Target Name Agnosticism (Config Is the Single Source of Truth)

**Status:** Accepted
**Date:** 2026-06-24
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-003](003_authority.md) (Authority of Declarations Over Inference), [ADR-012](012_target_scale_and_prefix_convention.md) (Target Scale and Prefix Convention)

---

## Context

The regression-target variable name (`lr_ged_sb`, `lr_sb_best`, `lr_ged_ns`, …) had become an **implicit, hardcoded contract** woven through views-models: tests asserted specific literals, tooling embedded the trio names, and a model's targets could be declared in `config_meta.py` *or* `config_hyperparameters.py` with no enforced single location. Any naming change became a cross-cutting, breakage-prone event.

A first attempt (issues #151/#153) tried to fix this by **forcing every model to one canonical name** (`lr_ged_sb`) and adding a guard test that **hardcoded** `{lr_ged_sb, lr_ged_ns, lr_ged_os}`. That treated the symptom (non-uniformity) rather than the cause (name-coupling), and the guard itself was the anti-pattern — it silently skipped the 11 DL models that declare targets in `config_hyperparameters`.

Investigation (EPIC #154) established: **views-pipeline-core already consumes the target name agnostically** (it reads `config["regression_targets"]` via the merged config); the hardcoding lived entirely in views-models tests/tooling. This ADR ratifies the principle and what it requires of this repo.

This is the name-level corollary of **ADR-003** (semantics are *declared*, never *inferred from data content*): if scale must not be inferred from a column name (ADR-012), neither may *any* behavior depend on the literal target name. The name is an arbitrary, model-local label.

---

## Decision

### 1. A single source of truth, accessed one way

A model's regression targets are obtained **only** through `tests/conftest.py::get_regression_targets(model_dir)`, which reads `config_meta.py` then `config_hyperparameters.py` (config_meta precedence, hp fallback — mirroring `ConfigurationManager.get_combined_config`). No test, tool, or script may read targets any other way or assume a declaration location.

### 2. No code hardcodes a target-name literal

Tests, tooling, investigation, and audit scripts **derive** target names from config — never compare against a string literal. Parity/consistency tests assert *relationships* (constituents agree; an ensemble matches its constituents; loss keys match a model's own targets), not specific names.

### 3. The `lr_`/`by_` prefix is a human convention, not a code guard

ADR-012's prefix convention (`lr_` = linear scale, `by_` = binary) remains a **readability** convention for contributors. It is **not** enforced by a code assertion. This deliberately supersedes ADR-012's suggestion (its §Validation) to add a test asserting `regression_targets` use the `lr_` prefix — such a guard is exactly the hardcoded-name anti-pattern this ADR removes. Prefix adherence is checked by humans in review, not by tests.

### 4. Cross-location agreement is enforced

If a model declares `regression_targets` in **both** `config_meta.py` and `config_hyperparameters.py`, the two must be equal (`tests/test_regression_targets.py`). This keeps the single accessor unambiguous.

### 5. Intra-ensemble agreement is the one structurally-required exception

The PFE ensemble pools constituent predictions **by target name** (it reads each declared target from each constituent's output). So within an ensemble, every constituent must declare all of the ensemble's targets — enforced as a **config-derived** contract (`tests/test_ensemble_configs.py::test_constituents_cover_ensemble_targets`), with no literal names. This is the *only* place name-agreement is required; making ensembles name-*heterogeneous* is a pipeline-core change (views-pipeline-core#203), out of this repo's scope.

### 6. Presentation constants are permitted, if documented and graceful

Human-facing display lookups (plot colours, short labels) may be keyed by target name **if** they are documented as presentation-only and degrade gracefully (`.get(target, default)`) for unknown names. They must not drive logic. (Sole current instance: `investigations/plot_sanity_checks.py`.)

---

## Rationale

Coupling consumers to the producer's name choice is what made naming load-bearing. Deriving from a single declared source decouples them: **renaming or varying a model's target requires touching only that model's config** (and regenerating that model's own artifacts) — never tests, tooling, or other models. The modeling library and pipeline-core were already agnostic; this brings views-models' own surfaces in line.

---

## Consequences

### Positive
- A target rename no longer ripples into tests/tooling. Proven by the **rename probe** (below).
- The two-location declaration ambiguity is resolved by one accessor; the DL-model blind spot is closed.
- No mass model renames are needed: each ensemble is already internally consistent; differing names across *unrelated* models are fine.

### Negative
- The target name is no longer a single uniform value across the repo. That is intended — uniformity is not the goal, agnosticism is.
- Intra-ensemble name-agreement (§5) cannot be removed in-repo; it is enforced, not eliminated, pending views-pipeline-core#203.
- Prediction artifacts still embed the literal target name in filenames/dirs (so a rename requires regenerating *that model's* predictions). Tracked as views-pipeline-core#204.

---

## Implementation Notes

- Accessor + helper: `tests/conftest.py` (`get_regression_targets`, `regression_targets_by_location`).
- Contracts: `tests/test_regression_targets.py` (agreement + resolution), `tests/test_config_completeness.py` (metrics⇒targets), `tests/test_datafactory_parity.py` (relational parity), `tests/test_ensemble_configs.py` (constituent coverage), `tests/test_datafactory_source_names.py` (descriptors source raw `ged_*_best`, never a renamed/bridge name — the input-source corollary).
- Disposition of the prior attempt: the 11 standardized models (#153) are kept (harmless consistency); the hardcoded guard was removed (#156); the ranger rename (#152) is **moot** under agnosticism and was closed; #151's views-models portion is superseded by this epic.
- Pipeline-core dependencies: views-pipeline-core#203 (heterogeneous pooling), #204 (artifact-name decoupling), #205 (DL training reads merged config → enables a single physical declaration location).

---

## Validation & Monitoring

- **Rename probe (epic acceptance):** temporarily renaming one model's declared target to an arbitrary string leaves the entire `tests/` agnostic-contract suite green. Verified 2026-06-24 (adolecent_slob → `probe_xyz_target`: 1588 passed, 99 skipped, 0 failed).
- **Literal sweep:** `grep -rnE "[\"'](lr_(ged_)?(sb|ns|os)|by_(sb|ns|os))" tests/ tools/ investigations/ models/*/scripts/` returns only the documented presentation constants in `plot_sanity_checks.py` — zero load-bearing literals.
- The contracts above run in the standard suite (`pytest`), beige/green markers per ADR-005.

---

## References

- EPIC [#154](https://github.com/views-platform/views-models/issues/154); tracking [#162](https://github.com/views-platform/views-models/issues/162); stories #155–#161, #163.
- [ADR-003](003_authority.md) — declarations over inference (this ADR extends it from scale to the name itself).
- [ADR-012](012_target_scale_and_prefix_convention.md) — the `lr_`/`by_` prefix convention (now human-only per §3).
- [ADR-005](005_testing.md) — testing as critical infrastructure.
- Pipeline-core dependencies: views-pipeline-core#203 / #204 / #205.
