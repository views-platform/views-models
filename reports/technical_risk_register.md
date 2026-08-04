# Technical Risk Register — views-models

**Last updated:** 2026-08-04  
**Governing ADR:** [ADR-010](../docs/ADRs/010_technical_risk_register.md)  
**Total entries:** 140 (131 concerns + 9 disagreements)  
**Concerns:** Open 61 | Mitigated 20 | Resolved 41 | Accepted 3 | Partially Resolved 1 | Subsumed 1 | Merged 4  
**Concerns by tier:** T1 5 | T2 43 | T3 54 | T4 25 (4 merge stubs carry no tier)  
**Disagreements:** Open 7 | Resolved 1 | Subsumed 1  
**Last curated:** 2026-07-31 (`review-rr strategic`, first full pass — tier recalibration, 4 merges, 6 causal clusters identified)

---

## Concerns

> Status is a field, not a section — Open, Mitigated, Accepted, Resolved, Merged and Subsumed entries are interleaved in ID order (ADR-010 §Concern Format). Merged entries are retained as ID-preserving stubs so cross-references never dangle.

### C-01 — Partition boundary updates require atomic edits to 73 files

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A decision is made to change calibration, validation, or forecasting partition boundaries |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `meta/partitions.json` is the single source of truth. `tools/partitions/bump.py` (replaces deleted `scripts/update_partitions.py`) rewrites all 100 files with invariant validation, temporal plausibility (val test end ≤ Dec previous year), post-write verification, atomic writes, and JSONL lockfile with git state. `test_config_partitions.py` enforces consistency via shared parser from `tools.partitions.fileops`. Override mechanism (`# PARTITION_OVERRIDE:`) permits declared deviations — see C-56 for staleness risk. **2026-06-06:** ADR-011 migration procedure still references the deleted `scripts/update_partitions.py` — must be updated to reference `python -m tools.partitions.bump`. See ADR-011. **Tier recalibrated 1 → 3 during review-rr (2026-07-31):** the original Tier 1 was impact-only. `tools/partitions/bump.py` now rewrites all 100 files with invariant validation, temporal plausibility checks, post-write verification, atomic writes and a JSONL lockfile, and C-56 (the override-staleness residual) is Resolved — so no silent-corruption path remains. What survives is annual coordination cost under a validated tool, which is Tier 3. Member of **Cluster A** (declared-but-unenforced). |

---

### C-02 — No static validation of queryset correctness

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A VIEWS database column is renamed or removed, or a queryset references a non-existent column |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `config_queryset.py` is the most complex config file (up to 734 lines) with zero test coverage. Failures are runtime-only (data fetch phase). Validation would require access to the VIEWS database schema. **2026-04-22 (test-review):** This gap was the root cause of the bright_starship `dict.publish()` crash — `generate()` returns a plain dict for datafactory models but no test validates return type or shape. Minimum viable test: verify `generate()` exists, returns correct type, and that datafactory descriptors contain required keys (`source`, `zarr_url`, `features`). See C-40 (return type contract mismatch). **Tier recalibrated 1 → 2 during review-rr (2026-07-31):** the failure mode as written is a *loud* runtime crash at the data-fetch phase, not silent corruption — Tier 2 under the register's own tier table. **Re-promote to Tier 1 if** a queryset can reference a wrong-but-*existent* column and train silently on the wrong variable; that variant is not currently claimed by this entry and has never been demonstrated. Member of **Cluster A** (declared-but-unenforced). |

---

### C-03 — Integration tests are manual-only, not in CI

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A model breaks at training time but all CI checks pass |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `run_integration_tests.sh` is the only mechanism testing actual model training/evaluation. It runs locally and takes hours. The CI pytest workflow (`run_tests.yml`) only runs fast structural tests. A model can be merged broken. **2026-04-10:** Incident confirms this risk — a `df.applymap()` → `df.map()` change in views-stepshifter (commit `06e73a9`) broke all stepshifter model evaluation. Surfaced only by manual integration test, not by CI. See C-31. |

---

### C-04 — Algorithm label / implementation drift

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `config_meta["algorithm"]` is changed without updating `main.py` imports, or vice versa |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_algorithm_coherence.py::TestAlgorithmManagerCoherence` validates that `config_meta["algorithm"]` belongs to the correct package family and that the package matches `main.py` imports. Uses a hardcoded `ALGORITHM_TO_PACKAGE` mapping that must be updated when new algorithms are added. |

---

### C-05 — Incomplete hyperparameter validation for non-DARTS models

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A stepshifter or baseline model is created with missing hyperparameters |
| **Source** | repo-assimilation |
| **Status** | Open (cross-repo, pending releases) |
| **Notes** | **Baseline: done.** `views-baseline` has a `ReproducibilityGate` (ADR-014, CIC, 13 tests) on `feature/reproducibilitygate` — covers all 5 algorithms with `CORE_GENOME` + `ALGORITHM_GENOMES`, runtime enforcement in `BaselineForecastingModelManager`, importable contract. **Stepshifter: done.** `views-stepshifter` has a `ReproducibilityGate` (ADR-001, CIC, 17 tests) on `feature/reproducibilitygate` — covers all 5 algorithms with `CORE_GENOME` + `ALGORITHM_GENOMES` (split into `parameter_keys`/`config_keys` for nested params), runtime enforcement in `StepshifterManager._train_model_artifact()`, importable contract. **Remaining:** Both branches pending merge and package release. Once released, views-models can add validation tests following the `test_darts_reproducibility.py` pattern. All three algorithm packages (r2darts2, stepshifter, baseline) will then expose canonical HP contracts. |

---

### C-06 — `config_queryset.py` has unique external dependencies among config files

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Tooling or tests attempt to load `config_queryset.py` without `viewser` and `views_pipeline_core` installed |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted as intentional deviation per ADR-002. The `viewser` DSL is essential for queryset definition. Testing gap addressed separately via C-02. |

---

### C-07 — Scaffold builder is untested

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A `views_pipeline_core` template update changes scaffold output, causing newly created models to fail existing tests |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_scaffold_builders.py` added (2026-04-06) with 7 AST-based tests verifying injection seams and 2 functional tests (skipped without `views_pipeline_core`). Injection seams (`input_fn`, `get_version_fn`, `pipeline_config`) allow mocked testing of `build_model_scripts()`. Remaining gap: no test validates that generated scaffold output satisfies structural tests. |

---

### C-08 — `requirements.txt` / `main.py` coherence is unvalidated

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A model's `requirements.txt` specifies one algorithm package but `main.py` imports a different one |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `test_algorithm_coherence.py::TestRequirementsCoherence` validates that `requirements.txt` package name (normalized hyphens to underscores) matches the package imported in `main.py`. **Scope limit found 2026-08-02 (expert-code-review):** that check covers only the *algorithm* package. It does not detect a file declaring an **additional** dependency the model never imports. Two instances existed — `ensembles/skinny_love` and `ensembles/white_mustang` both declared `views-frames>=1.7.0,<2.0.0`, and the sole mention of `views_frames` in either directory was that line. Removed in PR #325; neither environment had the package installed and skinny_love had completed a run without it. The general rule — **declare what you import** — is unenforced in the extra-dependency direction, and closing that is part of the proposed requirements-hygiene test (**D-06**). Cross-refs: **C-116**, **D-06**. |

---

### C-09 — Two placeholder models with no implementation

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Tooling or documentation includes `thousand_miles` or `thrift_shop` as active models |
| **Source** | repo-assimilation |
| **Status** | Resolved |
| **Notes** | Both directories deleted (2026-04-06). |

---

### C-10 — Conda environments committed to repository tree

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A contributor mistakes `envs/` contents for tracked repository state |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted. `envs/` contents are gitignored. The directories exist as local convenience for contributors. |

---

### C-11 — Single deployed model limits deployment path testing

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A second model moves to `deployed` status and encounters untested deployment-path behavior |
| **Source** | repo-assimilation |
| **Status** | Accepted |
| **Notes** | Accepted as current deployment posture. Deployment gating tested via `test_config_completeness.py::test_deployment_status_is_valid`. Additional models will be deployed as they mature. |

---

### C-12 — Global warning suppression in all model entrypoints

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A numerical instability, deprecation, or data quality issue produces a Python warning that is silently swallowed |
| **Source** | expert-code-review (Martin, Nygard, Hickey) |
| **Status** | Resolved |
| **Notes** | `warnings.filterwarnings("ignore")` removed from all 74 `main.py` files (2026-04-06). Enforcement test added: `test_cli_pattern.py::test_no_global_warning_suppression` (AST-based, parametrized across all models and ensembles). |

---

### C-13 — No prediction quality validation before ensemble aggregation

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A constituent is added to a deployed ensemble's `config_modelset.py`, or an existing constituent's loss / scaler / sample count changes — verify a NaN/Inf/range gate runs before aggregation (C-72 is the realized instance: 46–63% `Inf` cells reached a deployed ensemble's input) |
| **Source** | expert-code-review (Nygard, Kleppmann) |
| **Status** | Open |
| **Notes** | `white_mustang` (deployed ensemble) aggregates via median. No NaN/Inf check or range validation occurs before aggregation. If multiple constituent models produce garbage, the ensemble output degrades silently. Downstream consumers (UN FAO API) receive degraded data. |

---

### C-14 — Training artifacts have no run identity, so they silently overwrite (concurrent *and* sequential)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Two training runs for the same model execute simultaneously against the same `artifacts/` directory, **or** a model is re-trained and the previous artifact set is replaced in place — in both cases verify whether any prior artifact was needed for reproduction before the run |
| **Source** | expert-code-review (Kleppmann); merged with C-22 during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `models/*/artifacts/`, `models/*/wandb/`; DARTS `force_reset: true` in `config_hyperparameters.py` |
| **Notes** | Artifacts have no run ID or timestamp in filenames. Concurrent case: the second writer silently overwrites the first — low probability, but it destroys reproducibility when it occurs. Sequential case: re-running a model overwrites the previous artifacts with no versioning or deduplication; `force_reset: true` in DARTS hyperparameters acknowledges this but does not solve it. W&B logs exist but are not cross-referenced with artifact files, so there is no way to ask "which artifact produced this logged run?". **Merged with C-22 during review-rr (2026-07-31)** — one root cause (no run identity on artifacts), two trigger paths; separate entries invited fixing one and thinking the class was closed. **Tier recalibrated 4 → 3:** silent loss of reproducible state is not a Tier-4 code-quality observation, and the merged partner C-22 was already Tier 3. See also C-85 (the *consumer* side of the same missing identity: the ensemble resolves a cached prediction by artifact timestamp and cannot tell stale from current), C-110 (config reproducibility). |

---

### C-15 — Zero CIC failure mode test coverage

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A CIC's failure-modes table gains or changes a row — verify `tests/test_failure_modes.py` covers the new/changed mode in the same PR |
| **Source** | test-review (Nygard) |
| **Status** | Mitigated |
| **Notes** | `test_failure_modes.py` expanded from 4 to 9 tests (2026-04-06). New tests cover: empty config files, import errors, runtime errors, integration test runner exit codes. Remaining gap: no tests for scaffold builder `FileExistsError`, no tests for ensemble aggregation failure. 9 of 21 CIC failure modes now covered. **2026-05-20 (test expansion):** `test_failure_modes.py` expanded to ~30 tests with new red-team classes: `TestPartitionBoundaryValidation` (steps=0/−1/default across all models), `TestEnsembleConstituentIntegrity` (config loadability, partition alignment, malformed model lists), `TestMalformedQuerysetDescriptor` (missing keys, None return, circular import). Scaffold builder `FileNotFoundError` now tested in `test_scaffold_builders.py`. Estimated 15 of 21 CIC failure modes covered. **Tier recalibrated 1 → 2 during review-rr (2026-07-31):** this is a test-coverage gap, not a demonstrated silent-corruption path, and both its peers — C-16 (zero direct unit tests on CIC classes) and C-23 (beige-heavy suite) — sit at Tier 2. Holding it at Tier 1 above its own siblings was a peer inconsistency, not a severity judgment. Trigger also rewritten from symptomatic ("a failure mode occurs in production") to actionable. Member of **Cluster A** (declared-but-unenforced). |

---

### C-16 — Zero direct unit tests for any CIC class

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A PR changes a CIC-governed class's public method signature, return shape, or exception behaviour — verify a direct unit test exercises the changed method (not just its downstream output) |
| **Source** | test-review (Beck, Feathers) |
| **Status** | Mitigated |
| **Notes** | All 5 CIC-documented classes (`ModelScaffoldBuilder`, `EnsembleScaffoldBuilder`, `PackageScaffoldBuilder`, `CatalogExtractor`, `IntegrationTestRunner`) have zero direct unit tests. Tests validate their *outputs* (model directory structure) but never instantiate or exercise the classes. 33 CIC guarantees total, only 2 directly tested (6%), 6 indirectly tested (18%), 25 untested (76%). **2026-05-20 (test expansion):** Direct functional tests added for `ModelScaffoldBuilder` (5 tests: dir creation, README generation, subdirs, gitkeep, missing-dir error), `EnsembleScaffoldBuilder` (3 tests: inheritance, dir creation, missing-dir error), `PackageScaffoldBuilder` (8 AST-based tests: class/method existence, create+validate call chain, exception propagation, name validation), `CatalogExtractor` (8 tests: `replace_table_in_section` edge cases, `generate_markdown_table` structure, `create_link` format), `IntegrationTestRunner` (5 tests: help exit 0, nonexistent model warning, unknown flag error). CIC guarantee coverage improved from 6% to ~45%. Remaining gap: runtime behavioral tests for scaffold output satisfying structural tests, ensemble aggregation failure modes. |

---

### C-17 — Tooling scripts are untested "edit and pray" zones

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer modifies `create_catalogs.py`, `update_readme.py`, or `generate_features_catalog.py` and introduces a regression |
| **Source** | test-review (Feathers) |
| **Status** | Mitigated |
| **Notes** | 16 characterization tests added in `test_tooling_scripts.py` (2026-04-06) covering: `replace_table_in_section`, `generate_markdown_table`, `generate_repo_structure`, "Created on" regex, and Column extraction regex. Scripts cannot be imported directly (top-level `views_pipeline_core` imports) so pure function logic is duplicated in tests. Remaining gap: orchestration logic (`__main__` blocks, `extract_models()`) untestable without `views_pipeline_core` runtime. |

---

### C-18 — `build_model_scaffold.py` I/O coupling prevents testability

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any attempt to write automated tests for `ModelScaffoldBuilder` or `EnsembleScaffoldBuilder` |
| **Source** | test-review (Beck), expert-code-review (Martin, Ousterhout) |
| **Status** | Resolved |
| **Notes** | `build_model_scripts()` now accepts optional `input_fn` and `get_version_fn` keyword arguments (2026-04-06). Defaults to `input()` and `PackageManager.get_latest_release_version_from_github()` — backward compatible. `EnsembleScaffoldBuilder.build_model_scripts()` accepts optional `pipeline_config`. Tests pass mock callables to avoid stdin/network. Also fixed `== False` to `not` in package validation. CICs updated. |

---

### C-19 — `create_catalogs.py` has no transactional file write safety

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `create_catalogs.py` crashes between reading and writing `README.md` |
| **Source** | test-review (Feathers), expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | `update_readme_with_tables()` now writes to a `NamedTemporaryFile` in the same directory, then calls `os.replace()` for an atomic rename (2026-04-06). A crash mid-write leaves only the temp file; the original README is untouched. |

---

### C-20 — No timeout or circuit breaker in data fetch path

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | The VIEWS database is slow or unreachable during model training |
| **Source** | expert-code-review (Nygard) |
| **Status** | Open (cross-repo) |
| **Notes** | Models fetch data via `viewser.Queryset.publish()` with no timeout, retry limit, or fallback. A database outage hangs every model indefinitely during normal operation. The only timeout is the external `timeout` command in `run_integration_tests.sh` (1800s), which only applies during integration testing. **Cross-repo location (verified 2026-04-11):** `views-models` only *defines* querysets in each model's `config_queryset.py`. The actual `publish()` calls live in `views-pipeline-core/views_pipeline_core/modules/dataloaders/dataloaders.py:1027,1052` (`get_data()` and the no-drift backup path) plus two metadata-cache call sites in `handlers.py:1691,2124`. Fix must be implemented in `views-pipeline-core` — either by passing a `timeout` parameter to `publish()` (if `viewser.Queryset` accepts one) or by wrapping `get_data()` in a `concurrent.futures` timeout context. Escalate as a `views-pipeline-core` task. |

---

### C-21 — Partition boundary semantics undocumented

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor or auditor asks "why 121? why 444?" and finds no answer |
| **Source** | expert-code-review (Kleppmann) |
| **Status** | Resolved |
| **Notes** | ADR-011 documents ViewsMonth-to-date mapping, split point rationale, invariants, override mechanism, and migration procedure (2026-04-06). `meta/partitions.json` serves as the canonical reference. |

---

### C-22 — No idempotency guarantee in model training artifacts *(merged into C-14)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-14** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-22 (sequential re-run overwrites artifacts without versioning) and C-14 (concurrent runs overwrite each other) were two trigger paths on one root cause — **training artifacts carry no run identity** — with the same fix. Tracked together at C-14, Tier 3. |

---

### C-23 — Test suite is overwhelmingly beige; red coverage is low

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A model family with no existing red coverage is added (anything beyond the distributional-baseline set the runtime smoke covers) — verify at least one red test exercises its runtime failure path before merge |
| **Source** | test-review (category distribution analysis) |
| **Status** | Mitigated |
| **Notes** | Red coverage improved from 4 to 9 tests (2026-04-06). New tests cover config loading edge cases and integration test runner failure modes. Distribution still heavily beige (~64%) but red category is no longer negligible. Further improvement requires testing scaffold builder and ensemble aggregation failure modes. **2026-05-20 (test expansion):** ADR-005 pytest markers (`@pytest.mark.red/beige/green`) added to all test files and registered in `pyproject.toml`. Red tests expanded to 285 (from 9): partition boundary validation, ensemble constituent integrity checks, malformed queryset descriptors, integration runner CIC coverage. Distribution: 285 red (7%), 2726 beige (67%), 1038 green (25%), 34 unmarked (1%). Suite total: 3775 passed, 308 skipped. |

---

### C-24 — DARTS model `main.py` duplicates manager instantiation

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A DARTS model's manager constructor signature changes and only one of the two instantiations is updated |
| **Source** | expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | All 15 DARTS models and `purple_alien` (HydraNet) refactored to single-instantiation pattern (2026-04-06). Manager is now assigned to local variable before the `args.sweep` branch, matching the stepshifter/baseline convention. |

---

### C-25 — `white_mustang` ensemble uses deprecated CLI import pattern

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `views_pipeline_core` removes the deprecated `cli.utils` module |
| **Source** | expert-code-review (Martin) |
| **Status** | Resolved |
| **Notes** | `white_mustang/main.py` rewritten to match `cruel_summer` pattern (2026-04-06): uses `ForecastingModelArgs`, no `wandb.login()`, no `LoggingManager`. `test_cli_pattern.py` extended to cover all ensembles via `any_model_dir` fixture, plus new `test_no_global_warning_suppression` test. |

---

### C-26 — `IntegrationTestRunner` `--level` filter silently excludes broken models

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A model's `config_meta.py` has a syntax error; `--level cm` filtering silently skips it |
| **Source** | test-review (Leveson) |
| **Status** | Resolved |
| **Notes** | Fixed in `run_integration_tests.sh:109-153` (2026-04-11). The `--level` filter loop now captures Python stderr to a temp file, checks the subprocess exit code, and on failure: (1) prints `ERROR classifying <model>: config_meta.py failed to load` plus the last line of the traceback to stderr, (2) collects the model in a `CLASSIFICATION_ERRORS` array, (3) **fails fast with `exit 2`** before running any integration tests, listing every unclassifiable model. Manually verified with a synthetic broken `config_meta.py` (`SyntaxError: '(' was never closed`) — script aborts at exit 2 with the model name and traceback line surfaced. Real models still classify cleanly with no regression. The `--library` filter (lines 128-137) uses `grep -q` on `requirements.txt` and does not have the same silent-failure mode. |

---

### C-27 — Missing `requirements.txt` for `rude_boy` ensemble

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Dependency tooling or tests assume all ensembles have a `requirements.txt` |
| **Source** | tech-debt-cleanup |
| **Status** | Resolved |
| **Notes** | `ensembles/rude_boy/` was the only ensemble missing `requirements.txt`. Created with `views-pipeline-core>=2.0.0,<3.0.0` matching all other ensembles (2026-04-06). |

---

### C-28 — CI workflow only checks last script exit code

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `create_catalogs.py` fails but `update_readme.py` succeeds; CI auto-commits corrupted output |
| **Source** | tech-debt-cleanup |
| **Status** | Resolved |
| **Notes** | `.github/workflows/update_catalogs.yml` used `$?` which only captured `update_readme.py` exit code. A `create_catalogs.py` crash was silently ignored. Fixed by adding `set -e` to the run block and removing the redundant `$?` check (2026-04-06). Also removed stale `create_catalogs_01` test branch from triggers. |

---

### C-29 — Dead root-level `config_partitions.py` in `rude_boy` with stale boundaries

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A contributor or tool reads the wrong `config_partitions.py` and assumes `(121, 396)` boundaries are correct |
| **Source** | tech-debt-cleanup (C-01 investigation) |
| **Status** | Resolved |
| **Notes** | `ensembles/rude_boy/config_partitions.py` (root level) had boundaries `(121, 396)/(397, 444)` — 4.5 years shorter calibration window than standard. Framework always loads from `configs/` (which had correct values). Root file was dead code from copy-paste chain (Dylan Aug 2025 → xiaolongsun revert Oct 2025). Deleted (2026-04-06). |

---

### C-30 — `ucdp_extractor` had non-standard partition boundaries from copy-paste error

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `ucdp_extractor` runs with boundaries `(121, 396)` and forecasting offset `-2`, training on a 4.5-year shorter window than all other models |
| **Source** | tech-debt-cleanup (C-01 investigation) |
| **Status** | Resolved |
| **Notes** | `extractors/ucdp_extractor/configs/config_partitions.py` used `(121, 396)/(397, 444)` boundaries and offset `-2`. Root cause: smellycloud (Nov 2025, commit `901ec1e`) copied from `rude_boy`'s deviant root file instead of using the standard template. Extractor was in `shadow` status, excluded from all CI/testing, so the deviation was never caught. Fixed to standard values (2026-04-06). Test coverage extended to include extractors and postprocessors. |

---

### C-31 — Upstream algorithm package API changes break views-models silently

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A views-stepshifter, views-r2darts2, views-baseline, or views-hydranet maintainer "modernizes" a deprecated API call (e.g., pandas, numpy, sklearn) and merges to development without verifying the views-models environment supports the new API |
| **Source** | incident response (2026-04-10) |
| **Status** | Mitigated |
| **Notes** | views-models has no contract test that validates upstream packages still work in its installation environment. **Concrete incident:** views-stepshifter commit `06e73a9` (`chore: clean tech debt`) changed `df.applymap()` → `df.map()` claiming "deprecated API fix (pandas 2.0+)". `DataFrame.map()` was actually only added in **pandas 2.1.0**, and the production environment runs **pandas 1.5.3**. All stepshifter model evaluation broke at the `_get_standardized_df` boundary. Caught only by a manual integration test for `bittersweet_symphony`. Fix: revert to `applymap()` (works in all versions). The deeper problem: **views-models is installed against a frozen environment, but its dependencies are continuously developed against newer environments.** A boundary contract test (e.g., a smoke test that imports the manager and runs a 1-step prediction on a tiny synthetic dataset) would catch this in CI. Related: C-03 (no integration tests in CI), C-08 (requirements coherence — but that's package name, not API surface). |

---

### C-32 — Scaffold builder does not persist empty standard directories in Git

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A new model is scaffolded via `build_model_scaffold.py` and committed before all standard subdirectories contain files, or an existing model with latent gaps is cloned to a fresh server environment |
| **Source** | manual (2026-04-11) |
| **Status** | Mitigated |
| **Notes** | **Original framing was incomplete.** `update_gitkeep_empty_directories()` already existed at `build_model_scaffold.py:283` and was already called in `__main__`. Investigation on 2026-04-11 revealed the actual root cause: `.gitignore` line 10 (`logs/`) — a repo-wide rule for "Integration test logs" — silently swallowed `models/*/logs/.gitkeep` files even when the scaffold created them. The ranger_* hotfix worked only because the .gitkeep files were force-added (`git add -f`). Downstream managers crashed with `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` from `ModelPathManager` path resolution. **Mitigation v1 (2026-04-11 morning):** (1) `.gitignore` changed from `logs/` to `logs/*` + `!logs/.gitkeep` to allow the directory placeholder through while preserving the "ignore log file contents" intent; (2) `build_model_directory()` now creates `.gitkeep` inline immediately after each `subdir.mkdir()` call, so the invariant holds from the moment the directory is created; (3) `update_gitkeep_empty_directories(delete_gitkeep=False)` default flipped — the previous `True` default removed `.gitkeep` from "non-empty" dirs, but gitignored data files (`*.parquet`, `*.pkl`) count as non-empty, so the deletion behavior was a latent footgun that re-introduced the bug; (4) backfilled `logs/.gitkeep` for 4 affected models: `old_money`, `orange_pasta`, `wildest_dream`, `yellow_pikachu`. **Regression recurrence (2026-04-11 evening, commit `cd668ea`):** v1 backfill covered only 4 of 37 affected models. A fresh clone on `/home/simmaa/` running `models/invisible_string/main.py` crashed at `ModelPathManager._initialize_model_specific_directories` (`views-pipeline-core/.../model_path.py:458`) with `FileNotFoundError` on `notebooks/`. Scan revealed **9 models** missing tracked `notebooks/.gitkeep` and **28 models** missing tracked `logs/.gitkeep` — all latent failures on any non-dev checkout. The ranger-incident hotfix patched only the four models that had already been reported; every other pre-existing model remained un-backfilled. **Mitigation v2 (commit `cd668ea`):** backfilled the remaining 37 `.gitkeep` placeholders; additionally tightened `test_model_structure.py` to close the C-33 gap (see that entry). Related to C-07 (scaffold builder testing gap), C-33 (test contract now git-index based). |

---

### C-33 — No CI gate for model directory completeness

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A PR adds or modifies a model such that one of the standard subdirectories (`artifacts/`, `data/raw/`, `data/generated/`, `logs/`) is absent on fresh clone, and the PR merges without the hollow state being flagged |
| **Source** | manual (2026-04-11) |
| **Status** | Resolved |
| **Notes** | **v1 test (2026-04-11 morning):** `TestModelDirectoryStructure` added to `tests/test_model_structure.py`. The class uses the existing `model_dir` fixture (`tests/conftest.py:72`, parametrized over `ALL_MODEL_DIRS`) and asserted every model contained four runtime-critical subdirectories: `artifacts/`, `data/raw/`, `data/generated/`, `logs/`. **Regressed (2026-04-11 evening):** the v1 test had two structural gaps that let C-32 recur unnoticed. (1) `REQUIRED_SUBDIRS` omitted `notebooks/` and `reports/` even though `ModelPathManager._initialize_directories` validates both at runtime (`views-pipeline-core/.../model_path.py:442,458`); a model missing either directory would pass the test and crash on first instantiation. (2) The check used `path.is_dir()` on the local filesystem, so any developer who had ever run a model locally would see the test pass regardless of whether the directory was tracked in git — the exact failure mode C-33 was meant to prevent (fresh-clone absence). C-32's `/home/simmaa/` recurrence was a direct consequence: `invisible_string` passed C-33 locally but had no tracked `notebooks/.gitkeep`. **v2 test (commit `cd668ea`):** `REQUIRED_SUBDIRS` extended to the full set `[artifacts, data/raw, data/generated, data/processed, logs, notebooks, reports]` — parity with `ModelPathManager` runtime validation. The assertion replaced `path.is_dir()` with a `git ls-files` probe via a helper `_git_tracks_path()`, so "pass" means "tracked in the git index" — fresh-clone state, not working-tree state. Coverage now 74 models × 7 subdirs = 518 tracked-path assertions; full suite 3243 passing. See also C-32 (now re-mitigated with 37 backfilled .gitkeeps), C-07 (scaffold builder testing), C-16 (CIC class testing gaps). **2026-06-26 (postprocessor gap — PR #210):** the v2 contract covered `models/` only (`model_dir` fixture over `ALL_MODEL_DIRS`); **`postprocessors/` was uncovered**. `un_fao` — the only postprocessor, run through `PostprocessorPathManager` (same `model_path.py` directory validation) — shipped with **7 missing scaffold dirs** (only `configs/`+`logs/` existed) and crashed at `_initialize_directories` during the vpp#24 `africa_me_legacy` smoke test, before any Appwrite/datafactory work. The contract's filesystem-vs-index lesson held but its *scope* didn't include postprocessors. **Fix (PR #210):** backfilled the 7 `.gitkeep`s (incl. `logs/.gitkeep`, which the `logs/*` gitignore had swallowed — same footgun as C-32) and extended the contract with `TestPostprocessorDirectoryStructure` over a new `postprocessor_dir` fixture (`ALL_POSTPROCESSOR_DIRS`), so the same 7-subdir tracked-path assertion now guards postprocessors. `apis/` (different manager) intentionally out of scope. |

---

### C-34 — `--library` filter in `run_integration_tests.sh` silently excludes models lacking `requirements.txt`

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A user runs `bash run_integration_tests.sh --library baseline` and one of the eligible models is missing `requirements.txt` (or the file is unreadable); the model is excluded from the run with no warning |
| **Source** | code-review (2026-04-11) — discovered during C-26 fix in Sprint 2 |
| **Status** | Open |
| **Notes** | `run_integration_tests.sh:128-137` uses `if [ -f "$req_file" ] && grep -q "views-${FILTER_LIBRARY}" "$req_file"`. A missing or unreadable `requirements.txt` causes silent exclusion — the same class of bug C-26 had in the `--level` filter, but in the `--library` filter. C-26's Sprint 2 fix added the `CLASSIFICATION_ERRORS` fail-fast pattern (lines 109-153) for level classification only; the library filter was left untouched because it does not crash and the legitimate "model declares no matching library" case must remain a silent skip. The remaining gap: a model that lacks `requirements.txt` entirely cannot be distinguished from one that declares a different library. **Recommended fix:** when `requirements.txt` does not exist for a model in the candidate set, emit a `WARNING: cannot classify <model> by library: missing requirements.txt` to stderr and exclude it explicitly (don't fail fast — this is milder than C-26 because it doesn't indicate a broken file). After C-08 (requirements coherence test) and C-27 (rude_boy backfill), this gap is mostly future-protection — it would re-emerge if a new model is added without `requirements.txt` and `--library` filtering is used before C-08 catches the omission. See also C-26 (same pattern, resolved 2026-04-11), C-08 (requirements coherence — mitigated), C-27 (rude_boy `requirements.txt` — resolved). |

---

### C-35 — No CI gate for CIC ↔ code synchronization

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A PR modifies behavior of a CIC-governed class (anything in `docs/CICs/*.md`) — new guarantees, new failure modes, new inputs, new exit codes, new outputs — without updating the corresponding CIC file in the same PR, and merges without the drift being flagged |
| **Source** | review-diff (2026-04-11) — discovered during PR review of `fix/hydranet_loss_hp` |
| **Status** | Resolved |
| **Notes** | ADR-006 requires CIC updates to follow behavioral changes ("Changes to intent must update this contract," quoted at the bottom of every CIC). The repo enforces this via social review, not automation: nothing in `.github/workflows/` or `tests/` verifies that CIC-governed files have not drifted from their CIC. **Concrete evidence (this PR):** three commits to `run_integration_tests.sh` (`97aeb38` added DEPRECATED skip + exit code 130; `cd668ea` unrelated but didn't touch the CIC; `1ea564c` added `--foreground` changing signal semantics) shipped before review-diff flagged that `docs/CICs/IntegrationTestRunner.md` sections 3 (guarantees), 6 (failure modes table), and 7 (boundaries) still described the pre-change behavior. Each commit passed all pytest checks and was individually reviewed, yet the CIC drift went uncaught for three iterations. The test suite (3312 passing) has zero cross-references between CIC content and code behavior. **Why this matters beyond this PR:** CICs are load-bearing documentation for onboarding, incident response, and upstream contract negotiation (e.g., the C-31 pandas incident relied on CICs to understand the boundary between views-models and views-stepshifter). Stale CICs give readers a confidently wrong mental model. The bigger the drift, the worse the misdirection. **Recommended fix (not in scope for this concern):** a CI check that, for every file under `docs/CICs/`, enforces "if the target code file(s) changed in this PR, the CIC must also have changed in this PR." The challenge is mapping CIC → target files; the CIC filename already names the class, and a one-line frontmatter field (e.g., `target: run_integration_tests.sh`) plus a 30-line `.github/workflows/cic_sync_check.yml` would suffice. Related: C-15 (zero CIC failure mode test coverage — specifically about testing declared failure modes), C-16 (zero direct unit tests on CIC classes — specifically about behavior coverage), C-07 (scaffold builder testing gap). This concern is distinct: it's about documentation drift, not test coverage. |

---

### C-36 — `create_catalogs.py` uses fixed module names in `importlib` loading, risking stale module cache

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A Python runtime or future code change registers importlib-loaded modules in `sys.modules`; subsequent `extract_models()` calls return config data from the wrong model |
| **Source** | review-diff (2026-04-20) |
| **Status** | Resolved |
| **Location** | `create_catalogs.py:48,57` |
| **Notes** | `spec_from_file_location("config_meta", config_meta)` reused the literal name `"config_meta"` for every model's config file. Fixed (2026-04-21): module names now include the model directory name (`f"config_meta_{model_dir_name}"`), matching the `conftest.py:load_config_module` pattern. See also C-17, C-19. |

---

### C-37 — bright_starship `config_partitions.py` uses `_current_month_id()` instead of `ViewsMonth`, creating test blind spot

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | The `ViewsMonth` epoch or convention diverges from `(year - 1980) * 12 + month`, or a developer relies on `test_config_partitions.py` passing as proof that bright_starship's forecasting offset is correct |
| **Source** | review-diff (2026-04-20) |
| **Status** | Mitigated |
| **Location** | `models/bright_starship/configs/config_partitions.py:17-20,35` |
| **Notes** | bright_starship reimplements `ViewsMonth.now().id` as `_current_month_id()` to avoid `ingester3` dependency. The test regex finds zero matches, so the offset check vacuously passes. **Mitigated (2026-04-21):** added `# PARTITION_OVERRIDE:` comment so the test framework explicitly skips with a warning rather than silently passing. **2026-05-20 (fix):** Removed `_current_month_id()` from all 4 synthetic entries (vertical_dream, horizontal_dream, diagonal_dream, synthetic_chorus) by replacing dynamic forecasting ranges with fixed boundaries — train (121, 540), test (541, 541 + steps). Synthetic data has no external data availability constraint so fixed ranges are sufficient. These files no longer carry the epoch-divergence risk. Residual risk applies only to bright_starship, heavy_strider, heavy_freighter, light_strider, and shining_codex (all carry `# PARTITION_OVERRIDE:` comments). **2026-05-26 (ensemble parity dimension):** bold_comet, blazing_meteor, and stellar_horizon also use `_current_month_id()`. golden_hour (viewser ensemble) uses `ViewsMonth`. When comparing golden_hour ↔ stellar_horizon forecasting parity, the two implementations may disagree by ±1 month at month boundaries, silently shifting the forecasting train/test partition and invalidating the comparison. `test_datafactory_parity.py` only checks calibration/validation boundaries (static, identical) — it does not catch forecasting divergence. Forecasting parity comparisons in the runbook (Phase 7) must account for this. See also C-01, D-01. |

---

### C-38 — `datafactory_query` not installed in any environment that can run bright_starship

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer runs `python main.py -r calibration` in `views-hydranet-env` (or any env with `views_hydranet` + `views_pipeline_core`) without `datafactory_query` installed, and `calibration_viewser_df.parquet` is not cached |
| **Source** | falsify (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/main.py:33` (`from configs.config_queryset import fetch_data`), `models/bright_starship/configs/config_queryset.py:115` (`from datafactory_query import load_dataset`), `models/shining_codex/main.py:27` (same pattern), `models/shining_codex/configs/config_queryset.py:90` (same pattern) |
| **Notes** | **Falsification audit F-1/F-2 chain.** `views-datafactory` (which provides `datafactory_query`) is declared in `requirements.txt` but not installed in `views-hydranet-env` — the only conda environment that has both `views_hydranet` and `views_pipeline_core`. When `_ensure_data()` encounters a cache miss, it imports `datafactory_query` at line 96 and crashes with `ModuleNotFoundError`. Two of three run_types (`validation`, `forecasting`) have cached parquets from a prior session, masking the missing dependency. `calibration` has no cache — the standard first run (`-r calibration -t -e`) fails immediately. The local `envs/views-hydranet` directory expected by `run.sh` also does not exist; `run.sh` would create it and install deps from `requirements.txt` (which includes the git+https datafactory dep), but that's a ~10 min bootstrap, not "ready to run." **Fix:** `conda run -n views-hydranet-env pip install "views-datafactory>=1.9.0"` (on PyPI since 2026-07-27). See also C-06 (config_queryset external deps — accepted for viewser; this is the datafactory equivalent), C-37 (bright_starship partition deviation), C-40 (generate() contract mismatch). **Cross-repo (IDs below belong to the *views-pipeline-core* register, NOT this one — the same numbers exist here with unrelated content):** `vpc C-51` (`get_data()` hardcodes viewser), `vpc C-52` (drift detection loss), `vpc C-53` (`use_saved` overload). **2026-06-12:** the bright_starship half is fixed on this workstation — `views-hydranet-env` now has datafactory_query and the readiness probe passes locally. Still open for shining_codex (`views-r2darts2` env unprovisioned; its probe skips) and for any fresh machine — keep Open until the env story (run.sh bootstrap or release-pinned install) is settled. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the failure is a loud `ModuleNotFoundError` at first run — provisioning friction, not structural fragility with a silent consequence. Demoted alongside C-42, C-50 and C-73 so the Tier-2 band means "silent or stakeholder-visible", not "annoying on a fresh clone". Member of **Cluster C** (cross-repo dependencies have no released contract). **Root cause registered 2026-08-02:** this is a specific instance of **C-116** — 131 `requirements.txt` resolve into 11 shared environments, so a package a model needs can be absent because a co-tenant's run shaped the environment. Fix the class from C-116; this entry stays as the concrete instance that surfaced it. |

---

### C-39 — All 70 `run.sh` scripts use `#!/bin/zsh` — will fail on Linux servers and CI

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any `run.sh` is executed on a Linux server, Docker container, or CI runner where zsh is not installed (i.e., most deployment targets) |
| **Source** | review-diff (2026-04-21); **regression measured 2026-08-02** while scoping #310 |
| **Status** | Resolved (2026-08-02, second time) — see the regression note |
| **Location** | `models/*/run.sh`, `ensembles/*/run.sh`, `apis/*/run.sh`, `extractors/*/run.sh`, `postprocessors/*/run.sh`, `models/execute_all.sh` (82 scripts total) |
| **Notes** | **REGRESSED between 2026-05-04 and 2026-06-28, while marked Resolved; re-fixed 2026-08-02 (#310).** 24 `run.sh` carrying `#!/bin/zsh` were found. Every one was created *after* the April fix — 2026-05-04 (`first_love`, `bad_romance`, `smol_cat`, others), 2026-05-19 (`fake_model`), 2026-06-28 (the 12 r2darts models). They are not stragglers a sweep missed. **Cause: the fix was applied to the output, never to the generator.** `run.sh` is emitted by `views_pipeline_core/templates/model/template_run_sh.py` in **views-pipeline-core**, which still emits `#!/bin/zsh` and was last modified 2026-04-03 — eighteen days *before* the April fix landed here, by the same author. Every model scaffolded since has been born with the defect. **Impact, measured:** `models/execute_all.sh:10` invokes `"$script"` directly and every ensemble README documents `./run.sh`, so on Linux 11 of the 24 (zsh *and* executable) fail with `bad interpreter`. One is `ensembles/first_love`, which `monthly_run.sh` runs in production — undetected because `monthly_run.sh` calls `bash run.sh`, which ignores the shebang. **Exit:** `tests/test_run_sh_portability.py` now fails on any tracked `.sh` declaring zsh, so the next generator-sourced regression is caught on the day it lands rather than eight weeks later. The generator itself is **not fixed here** — that is another repository (**views-pipeline-core#384**), so this entry stays exposed to re-regression until that lands; the test is what makes that visible. **The executable bit, the second half of the same failure (fixed 2026-08-02 on maintainer decision):** 18 tracked `run.sh` were committed non-executable (mode 100644), breaking the same two entry points with `Permission denied`; 13 overlapped the zsh set, so those went from one Linux failure straight to another. Now mode 100755 and pinned by `test_every_run_sh_is_executable`. One deliberate exception, also pinned: `tools/credentials/platform_env.sh` is `source`d and never executed (ADR-018), so an executable bit there would advertise an entry point it does not have. **The pattern:** third entry this quarter marked Resolved while a generator or a scope kept producing the defect (C-60 tools layout, C-112 shell-vs-exported scope). Cross-refs: **C-60**, **C-112**, **C-113**. Member of **Cluster A** (declared-but-unenforced). |
| **Original notes (2026-04-21)** | **Resolved (2026-04-21).** All 79 `#!/bin/zsh` shebangs changed to `#!/usr/bin/env bash`. `models/execute_all.sh` line 10 changed from `zsh "$script"` to `"$script"` (delegates to shebang). 35 missing trailing newlines and 23 missing executable permissions also fixed. `scripts/audit_shell_health.sh` added to verify: 82 scripts, 490 checks, CLEAN verdict. No zsh-specific syntax was found in any script — all were plain POSIX/bash. |

---

### C-40 — `generate()` return type contract mismatch — dict vs Queryset, no validation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new model migrates to views-datafactory and its `config_queryset.generate()` returns a dict descriptor; `views-pipeline-core` calls `.publish()` on it and crashes |
| **Source** | expert-code-review (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/configs/config_queryset.py` (returns dict), `models/shining_codex/configs/config_queryset.py` (returns dict), `views-pipeline-core/views_pipeline_core/data/model_path.py:691-692` (`get_queryset()` returns raw `generate()` output with no type checking) |
| **Notes** | Standard viewser models return a `Queryset` object from `generate()`. bright_starship and shining_codex (datafactory models) return a plain dict with `"source": "views-datafactory"`, `"zarr_url"`, `"features"` keys. `get_queryset()` in views-pipeline-core performs no type checking — it calls `generate()` and returns whatever it gets. Downstream, `_fetch_data_from_viewser()` calls `.publish()` on the result, crashing with `AttributeError: 'dict' object has no attribute 'publish'`. The contract between views-models (config producer) and views-pipeline-core (config consumer) is entirely implicit. **Phase 1 workaround:** `args.saved = True` in bright_starship's `main.py` routes around the viewser path. **Phase 2 fix (views-pipeline-core):** type dispatch in `get_data()` based on descriptor type + `generate()` return type validation in `get_queryset()`. **Cross-repo (IDs below belong to the *views-pipeline-core* register, NOT this one):** `vpc C-51` (root cause — `get_data()` hardcodes viewser), `vpc C-42` (missing ViewsDataLoader CIC). See also C-06 (config_queryset external deps), C-38 (datafactory_query not installed). |

---

### C-41 — shining_codex has no readiness tests

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer clones the repo and runs `python main.py -r calibration` for shining_codex without the `views-r2darts2` environment and `datafactory_query` installed |
| **Source** | tech-debt-cleanup (2026-04-21) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `models/shining_codex/` (no `tests/` directory or test files) |
| **Notes** | bright_starship has readiness tests (`test_bright_starship_readiness.py`) that verify environment prerequisites (conda env, `datafactory_query`, `DartsForecastingModelManager` import) and config structural validity. shining_codex, cloned from bright_starship, has no equivalent tests. Without readiness tests, failures will surface only at runtime with opaque error messages (e.g., `ModuleNotFoundError` for `datafactory_query` or `views_r2darts2`). See C-38 (datafactory_query not installed), C-03 (integration tests manual-only). **2026-06-12: Resolved** (issue #122, with C-75): `test_bright_starship_readiness.py` is parametrized over both datafactory models — shining_codex gets the same env pre-flight probe (skips while `views-r2darts2` is unprovisioned, which is truthful) and the same static dependency-contract checks (requirements / queryset import / generate()). Single parametrized file avoids the copy-paste drift this entry complained about. |

---

### C-42 — Synthetic models depend on unreleased `views-pipeline-core` branch

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | The `feature/hydranet_ensamble_africa_me` branch of views-pipeline-core changes its synthetic data API (pattern names, queryset descriptor keys, or `DataFrameEnsembleManager`/`PredictionFrameEnsembleManager` constructor) before merge, breaking synthetic models and ensembles |
| **Source** | pr-review (2026-05-20) |
| **Status** | Open |
| **Location** | `models/vertical_dream/configs/config_queryset.py`, `models/horizontal_dream/configs/config_queryset.py`, `models/diagonal_dream/configs/config_queryset.py`, `ensembles/synthetic_chorus/main.py`, `models/lucid_dream/configs/config_queryset.py`, `models/vivid_dream/configs/config_queryset.py`, `models/waking_dream/configs/config_queryset.py`, `ensembles/synthetic_chant/main.py` |
| **Notes** | PR #56 adds `vertical_dream`, `horizontal_dream`, `diagonal_dream`, and `synthetic_chorus` — all four depend on the `"source": "synthetic"` queryset descriptor and `DataFrameEnsembleManager`, which exist only on the `feature/hydranet_ensamble_africa_me` branch of `views-pipeline-core`. If that branch renames pattern values (e.g., `"vertical_stripe"` → `"v_stripe"`), changes required descriptor keys, or alters the `EnsembleManager` import path, the synthetic models will fail at data-load time with no structural test catching the mismatch — `test_model_structure.py` validates directory layout but not queryset descriptor validity against pipeline-core. This is the same class of cross-repo coupling as C-31 and C-38 but with a sharper trigger: the dependency is on an unreleased, in-flux branch rather than a released package. Risk resolves naturally once the pipeline-core branch merges and the API stabilizes. **2026-05-24 (PR #58):** Three additional PredictionFrame synthetic models (`lucid_dream`, `vivid_dream`, `waking_dream`) and one ensemble (`synthetic_chant`) added. These extend the dependency surface to `PredictionFrameEnsembleManager`, `ConflictologyModel`, and `MixtureBaseline` distributional outputs. All run successfully against `views-pipeline-core v2.3.0` — if that version is released, this risk may be resolved. **2026-05-26 (confirmed):** `envs/views_ensemble` created by ensemble `run.sh` installs `views-pipeline-core` from PyPI, which lacks `PredictionFrameEnsembleManager`. `synthetic_chant` ensemble failed with `ImportError: cannot import name 'PredictionFrameEnsembleManager'` until local editable install replaced the PyPI version. This confirms the trigger: any fresh clone or CI environment that creates `views_ensemble` from `requirements.txt` will fail for PredictionFrame ensembles. See also C-31 (upstream API breaks), C-38 (datafactory_query not installed), C-40 (generate() return type contract mismatch), C-50 (views-baseline version spec mismatch — same class of fresh-clone failure). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the confirmed failure is a loud `ImportError` at ensemble start on any fresh env — provisioning friction, not silent fragility. Member of **Cluster C** (cross-repo dependencies have no released contract). |

---

### C-43 — Ensemble ground truth is order-dependent on `config_meta.models` list

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer reorders the `models` list in `ensembles/synthetic_chorus/configs/config_meta.py` |
| **Source** | falsify audit (2026-05-20) |
| **Status** | Open |
| **Location** | `ensembles/synthetic_chorus/configs/config_meta.py:4` |
| **Notes** | The ensemble evaluation loads prediction files from constituent models in list order. The actual `synth_target` values (ground truth) come from the first model's predictions — currently `vertical_dream`. The analytically derived expected MSE (4.34444) depends on this ordering. Reordering the list silently changes which model supplies the ground truth, producing a different MSE with no error signal. Mitigated by `tests/test_falsification_synthetic.py::test_synthetic_chorus_first_model_is_vertical_dream` which asserts vertical_dream is first, and by the README which documents the order-dependency. This is a test-internal concern with no production impact — synthetic models are not deployed. |

---

### C-44 — Quality-blind ensemble aggregation (concat *and* mean) degrades the ensemble below its best constituents

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Building a `concat` **or** `mean` ensemble whose constituents have heterogeneous quality on a target (one constituent materially worse than others), with `aggregation` set as a bare config string and no constituent-quality gate |
| **Source** | golden_hour calibration run (2026-05-25); broadened by repo-assimilation + chunky_bunny (2026-06-13) |
| **Status** | Open |
| **Location** | `ensembles/*/configs/config_meta.py` (`aggregation`); `views-pipeline-core` ensemble aggregation paths (PredictionFrameEnsembleManager concat; mean) |
| **Notes** | Observed 53% CRPS degradation on `lr_sb_best` vs best individual model (golden_hour: 0.233 vs purple_alien: 0.152). blue_stranger (0.223) contributed 64 poor-quality samples that diluted the 128 better samples from purple_alien and violet_visitor. Concat treats all posterior samples equally — no mechanism to down-weight poor contributors. For future ensembles, consider weighted aggregation or model selection for targets where constituent quality varies significantly. Models were uncalibrated so this finding may not hold after hyperparameter optimization. **2026-06-13 (repo-assimilation R7 — merged here): the same quality-blindness affects `aggregation: "mean"`.** chunky_bunny (equal-weight mean of 23 constituents) scored MSLE **0.590**, worse than 14 of its own constituents and Pareto-dominated by smol_cat alone (0.503 MSLE / 0.872 MCR vs the ensemble's 0.590 / 0.584): the mean blends timid stepshifters (MCR 0.2–0.3) with honest DL/Hurdle models and lands in a mediocre middle. `aggregation` is a bare string in `config_meta.py` with no quality gate, for either method. See also C-13 (no prediction quality validation before aggregation), C-86 (constituent feature incoherence), [[project-mcr-timid-prophet]]. |

---

### C-45 — Ensemble `-t` flag causes full retraining cascade when models are pre-trained

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Running any PredictionFrameEnsembleManager or DataFrameEnsembleManager with `-t` when constituent models already have trained artifacts in their `artifacts/` directories |
| **Source** | golden_hour calibration run (2026-05-25) |
| **Status** | Open |
| **Location** | `views-pipeline-core` EnsembleManager train path (invokes constituent `run.sh` subprocesses) |
| **Notes** | Running `python main.py -r calibration -t -e` on a pre-trained ensemble causes: (1) retrain all constituent models via run.sh subprocess (~2h), (2) create new model artifacts with new timestamps, (3) discover no predictions exist for those new timestamps, (4) re-evaluate all constituent models via run.sh subprocess (~3h), (5) finally perform the actual aggregation (~30 min). This wasted ~6 hours on golden_hour. The correct command when models are already trained: `python main.py -r calibration -e --saved`. The `-t` flag on ensembles should either warn when artifacts already exist, or detect and reuse existing timestamps rather than creating new ones. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the consequence is ~6 hours of wasted compute — fully observable, recoverable, and it produces no wrong output. Costly, not fragile. See also C-14 (artifacts have no run identity — the timestamp churn this entry describes is the same missing identity; absorbed C-22). |

---

### C-46 — Classification targets not evaluable at PredictionFrame ensemble level

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Adding `classification_targets` to any PredictionFrame ensemble's `config_meta.py` |
| **Source** | golden_hour design review (2026-05-24) |
| **Status** | Open |
| **Location** | `views-pipeline-core` `PredictionFrameEnsembleManager.prepare_actuals_df` (identity lambda), `ensembles/golden_hour/configs/config_meta.py` (regression-only by design) |
| **Notes** | `PredictionFrameEnsembleManager.prepare_actuals_df` is a no-op identity lambda. Classification targets (`by_sb_best`, `by_ns_best`, `by_os_best`) are derived signals not present in raw viewser data. Individual HydraNet models derive them via `DataFetcher.apply_blueprint()`, but the ensemble doesn't inherit that derivation logic. Including `classification_targets` in ensemble `config_meta` causes `KeyError` when `EvaluationStage._load_actuals()` looks for the derived columns in raw actuals. Workaround: exclude `classification_targets` from ensemble config; evaluate classification at individual model level only. golden_hour correctly implements this workaround. Fix would require `PredictionFrameEnsembleManager` to implement target derivation or delegate to constituent model blueprints. See also C-15 (CIC failure mode coverage — ensemble aggregation failure modes listed as remaining gap). |

---

### C-47 — Track A/B dual output produces redundant predictions with contradictory documentation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer sets `skip_predictions_delivery` back to `False` to re-enable Track B parquets without verifying the PyArrow memory fix is in place |
| **Source** | golden_hour investigation (2026-05-25) |
| **Status** | Mitigated |
| **Location** | `views-pipeline-core` config (`skip_predictions_delivery` flag), `models/*/data/generated/` (both `.npy` and `.parquet` outputs coexist) |
| **Notes** | HydraNet models produce both Track A (`.npy` PredictionFrame, 64 posterior samples) and Track B (`.parquet` DataFrame delivery, point predictions) simultaneously. **Mitigated (2026-05-26):** All 19 PredictionFrame models now have `skip_predictions_delivery: True`, suppressing Track B parquet generation. The contradictory `False, #True,` comment pattern has been removed. `test_track_parity.py` (40 tests) verified Track A and Track B produce identical values before Track B was disabled. `CoreConfigSniffer` (views-pipeline-core PR #87) now enforces the key as mandatory — models without it crash at config validation. Residual risk: if Track B is re-enabled without the PyArrow memory fix, the 5.5M Python float object allocation (~4.8–6.4 GB peak) will recur. See also C-40 (generate() return type contract mismatch). |

---

### C-48 — Viewser vs datafactory variable variant mismatch confounds parity comparison

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | CRPS or forecast parity comparison between golden_hour (viewser) and stellar_horizon (datafactory) shows divergence; root cause is data input differences, not pipeline differences |
| **Source** | config diff investigation (2026-05-26) |
| **Status** | Resolved |
| **Location** | `models/purple_alien/configs/config_queryset.py` (`ged_sb_best_sum_nokgi`), `models/bright_starship/configs/config_queryset.py` (`ged_sb_best`), same pattern for `ged_ns_best` and `ged_os_best` |
| **Notes** | The viewser trio (purple_alien, blue_stranger, violet_visitor) trains on `ged_*_best_sum_nokgi` — the summed, no-known-geographical-imprecision variant of UCDP fatality counts. The datafactory trio (bright_starship, bold_comet, blazing_meteor) trains on `ged_*_best` — the base variant. Despite different variable names, both deliver functionally identical values. **Resolved (2026-05-26):** Direct cell-by-cell comparison of cached training parquets (4,876,920 rows × 6 columns) showed 99.99% exact match for all three target variables: lr_sb_best (614 differing rows of 4.9M), lr_ns_best (138), lr_os_best (182). Correlations all >0.999. The `_sum_nokgi` suffix does not indicate a different aggregation — both sources deliver the same fatality sums per PRIO-GRID cell-month. The ~600 differing rows have small absolute differences reflecting timing differences in UCDP data ingestion. This concern is fully disproven as a source of prediction divergence. See `reports/parity_investigation_20260526.md` for full analysis. See also C-02 (queryset validation), C-40 (generate() contract mismatch). **2026-06-24 (un_fao delivery consumer, #94):** The same variant pair now feeds the **un_fao postprocessor**, which switched from viewser `*_sum_nokgi` to datafactory `*_best` (commit 2518335) and *delivers per-cell actuals to FAO* — a stricter consumer than model training. A row-level oracle over `africa_me_legacy`, months 480–485 (`tests/test_un_fao_datafactory_equivalence.py`) confirms C-48's finding at the delivery level: coverage identical (78,660 cells, none added/dropped), values 99.97% identical (~21 cells differ across the 3 targets, net +15/+14/+39 fatalities) — consistent with ingestion-timing skew, not a structural aggregation change. The test enforces this as a **bounded-divergence guard** (cell-fraction ≤1%, net ≤10% of total per target) that tolerates the skew but fails loud on a real aggregation/region change; a strict bit-equality test is kept `xfail` to document the non-identity. Net-positive drift (datafactory slightly higher) is within noise for this window but worth re-checking before #127 goes global to `land_gaul` (5× more cells → 5× the absolute skew in delivered numbers). |

---

### C-49 — Feature set divergence between viewser and datafactory model configs

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Parity comparison between golden_hour and stellar_horizon produces unexplained spatial or regional bias differences |
| **Source** | config diff investigation (2026-05-26) |
| **Status** | Partially Resolved |
| **Location** | `models/purple_alien/configs/config_queryset.py` (lines 22-23: `col`, `row` columns), `models/bright_starship/configs/config_queryset.py` (no spatial features) |
| **Notes** | Originally three concerns. **Partially resolved (2026-05-26):** **(1) Spatial features: DISPROVEN.** Raw data comparison confirmed `col` and `row` are 100% identical between viewser and datafactory parquets. Both data loading paths provide them. **(2) Country encoding: CONFIRMED but metadata-only.** viewser uses VIEWS-internal `country_id` (e.g., 192); datafactory uses FAO `gaul0_code` (e.g., 159, or -1 for unassigned). 0% cell-level match. However, `c_id` is in `identity_cols`, NOT in `features` — HydraNet uses only 3 input channels (lr_sb_best, lr_ns_best, lr_os_best). Unless curriculum sampling or stratified evaluation uses `c_id` values downstream, this is a metadata-only divergence with no model impact. Downgraded from Tier 2 to Tier 4. **(3) NA handling:** Not yet investigated. See `reports/parity_investigation_20260526.md` for full analysis. See also C-48 (resolved — variable variant not a divergence source), C-02 (queryset correctness). |

---

### C-50 — `views-baseline` not published to PyPI; `requirements.txt` version spec unresolvable on fresh clone

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer clones the repo on a new machine and runs any baseline model's `run.sh`, which creates `envs/views-baseline` and fails at `pip install -r requirements.txt` because `views-baseline>=1.0.0,<2.0.0` has no matching distribution on PyPI |
| **Source** | synthetic ensemble run (2026-05-26) |
| **Status** | Open |
| **Location** | `models/lucid_dream/requirements.txt`, `models/vivid_dream/requirements.txt`, `models/waking_dream/requirements.txt`, `models/vertical_dream/requirements.txt`, `models/horizontal_dream/requirements.txt`, `models/diagonal_dream/requirements.txt`, `models/red_ranger/requirements.txt`, `models/green_ranger/requirements.txt`, `models/blue_ranger/requirements.txt`, `models/black_ranger/requirements.txt`, `models/pink_ranger/requirements.txt`, `models/yellow_ranger/requirements.txt`, `models/white_ranger/requirements.txt`, `models/light_strider/requirements.txt`, `models/heavy_strider/requirements.txt`, `models/average_cmbaseline/requirements.txt`, `models/average_pgmbaseline/requirements.txt`, `models/zero_cmbaseline/requirements.txt`, `models/zero_pgmbaseline/requirements.txt`, `models/locf_cmbaseline/requirements.txt`, `models/locf_pgmbaseline/requirements.txt` (21 models total) |
| **Notes** | All 21 baseline models declare `views-baseline>=1.0.0,<2.0.0` in `requirements.txt`. The `views-baseline` package is not published to PyPI at all — it is only available as a local editable install from `~/Documents/scripts/views_platform/views-baseline` at version `0.1.0`. On existing developer machines with the pre-existing `envs/views-baseline` env, the pip dry-run check succeeds because the package is already installed, and `run.sh` proceeds normally. On a fresh clone (new machine, CI, new contributor), `run.sh` creates the conda env, `pip install` fails with `No matching distribution found for views-baseline`, and the model crashes with `ModuleNotFoundError: No module named 'views_baseline'`. **Observed (2026-05-26):** All 6 synthetic model runs showed `ERROR: No matching distribution found for views-baseline<2.0.0,>=1.0.0` but succeeded because the env already had the local install. **Fix options:** (1) publish `views-baseline` to PyPI at version `>=1.0.0`, (2) change `requirements.txt` to use a git+https URL (matching the `views-datafactory` pattern in HydraNet models), (3) update `run.sh` to install from local path if available (but run.sh must not be modified — see feedback constraint). See also C-38 (same class: `datafactory_query` not installed), C-42 (same class: `views-pipeline-core` from PyPI lacks features), C-08 (requirements coherence). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** `No matching distribution found` is a loud, immediate, self-describing failure — provisioning friction, not silent fragility. Member of **Cluster C** (cross-repo dependencies have no released contract); the cluster fix is a release/pin discipline, of which publishing `views-baseline` is one instance. |

---

### C-51 — Datafactory trio missing `sampling_strategy` — ADR-049 required field, runtime crash

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer runs `bash models/bold_comet/run.sh -r calibration` (or bright_starship, blazing_meteor) and views-hydranet rejects the config with `'sampling_strategy' is required (ADR-049)` |
| **Source** | review (PR #59, 2026-05-31) |
| **Status** | Resolved |
| **Location** | `models/bright_starship/configs/config_hyperparameters.py`, `models/bold_comet/configs/config_hyperparameters.py`, `models/blazing_meteor/configs/config_hyperparameters.py`, `models/heavy_freighter/configs/config_hyperparameters.py` |
| **Notes** | The viewser trio (purple_alien, blue_stranger, violet_visitor) received `sampling_strategy` in this PR cycle (threshold/boltzmann/sigmoid respectively). The datafactory trio and heavy_freighter were not updated — bold_comet and blazing_meteor were cloned from bright_starship, which also lacked the field. views-hydranet's curriculum learner validates the key at config load time and raises `KeyError` on absence. All four models would fail immediately on any training run. The parity test (`test_datafactory_parity.py::TestDatafactoryTrioConfigParity::test_identical_shared_hyperparameters`) does not catch this because it strips loss keys and compares models pairwise — since all three are equally missing the field, they match each other. **Resolved (2026-06-01):** Added `'sampling_strategy': 'threshold'` to all four affected models (3 datafactory + heavy_freighter). Added `test_hydranet_has_sampling_strategy` to `test_config_completeness.py` to catch this class of omission for all HydraNet models (scoped via `meta_config["algorithm"] == "HydraNet"`) — this test is what caught heavy_freighter. See also C-05 (incomplete HP validation — covers stepshifter/baseline, not HydraNet), C-38 (datafactory_query not installed — same models, different dependency class), C-42 (unreleased pipeline-core branch — different: import availability, not config completeness). |

---

### C-52 — 12 PF models missing config keys required for PFE ensemble participation

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds any of the 12 affected models as a constituent of a PredictionFrameEnsembleManager ensemble — the ensemble will crash or produce wrong sample counts because constituent configs lack `n_posterior_samples` and/or `regression_targets` |
| **Source** | test_pfe_production_readiness.py (TDD green tests, 2026-06-01) |
| **Status** | Resolved |
| **Location** | `models/{black_ranger,blue_ranger,green_ranger,lucid_dream,pink_ranger,red_ranger,vivid_dream,waking_dream,yellow_ranger}/configs/config_hyperparameters.py` (missing both `n_posterior_samples` and `regression_targets`), `models/{heavy_strider,light_strider,white_ranger}/configs/config_hyperparameters.py` (missing `n_posterior_samples` only) |
| **Notes** | All 21 models declare `prediction_format: "prediction_frame"` in `config_meta.py`, meaning they produce PredictionFrame outputs. But 12 of them lack `n_posterior_samples` (needed by PFE to verify aggregated sample counts) and 9 of those also lack `regression_targets` (needed to know which target directories to validate). The 9 models with fully compliant configs (purple_alien, blue_stranger, violet_visitor, bright_starship, bold_comet, blazing_meteor, heavy_freighter, pink_pirate, heavy_strider partially) are the only ones eligible for PFE ensembles today. This blocks the PFE production roadmap: Steps 2-5 require running constituent models through PFE, and any model without these keys cannot participate. The ranger models (7 of 12) use an older config convention with `n_samples` instead of `n_posterior_samples` and no explicit `regression_targets` — they predate the HydraNet multi-target architecture. The dream models (lucid_dream, vivid_dream, waking_dream) are synthetic test models that also predate the convention. **Resolved (2026-06-02):** Added `n_posterior_samples` and `regression_targets` to all 12 affected `config_hyperparameters.py` files. Values derived from each model's `config_meta.py` (regression_targets) and existing `n_samples` (n_posterior_samples). xfail markers removed from `test_pfe_production_readiness.py` — all 21 PF models now pass config-level readiness tests unconditionally. See #70. |

---

### C-53 — Config value regression during cross-branch merges

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer merges `development` into a feature branch (or vice versa) when both branches have modified the same model `config_hyperparameters.py` with different values for the same key — git auto-resolves by picking one side, silently dropping the other's intentional change |
| **Source** | tech-debt-cleanup (2026-06-02) |
| **Status** | Open |
| **Location** | `models/blue_stranger/configs/config_hyperparameters.py`, `models/violet_visitor/configs/config_hyperparameters.py` (observed); any model config modified on both branches (systemic) |
| **Notes** | Observed during merge of `development` into `feature/golden_hour_ensemble`: blue_stranger and violet_visitor had `skip_predictions_delivery` changed to `True` on the feature branch (intentional), while development still had `False` (pre-existing). Git auto-merged without conflict markers, silently regressing the value to `False`. Also introduced a stray `prediction_format` key in hyperparameters (belongs only in config_meta). Caught during tech-debt-cleanup verification; would have caused Track B parquet generation and potential OOM in ensemble runs. **Mitigated (2026-06-02):** Fixed in this session. No automated guard exists — mitigation is manual post-merge review of config diffs. See also C-01 (73 duplicated config files amplify this risk), C-52 (same files, different keys). |

---

### C-54 — Experimental model (heavy_freighter) in production model directory without marker

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | **(a) Original —** a developer adds heavy_freighter to a production ensemble's `config_meta.models` list without realizing it uses a global grid (360×720 vs regional 180×180), producing incompatible spatial dimensions. **(b) Broadened (2026-07-20) —** a cross-cutting sweep (partition bump, config-key migration, catalog regeneration, an all-models parametrized test) runs over `models/` and silently includes the ~30 experimental/placeholder directories, because no marker distinguishes them |
| **Source** | tech-debt-cleanup (2026-06-02) |
| **Status** | Open |
| **Location** | `models/heavy_freighter/configs/config_hyperparameters.py` (`height: 360`, `width: 720` — global grid vs regional 180×180) |
| **Notes** | heavy_freighter uses global grid coverage (360×720) vs the regional Africa-ME grid (180×180) used by all ensemble-eligible models. Its training params (tobit, 200 lessons, 16 samples, scheduled sampling) now match the production models — only the grid differs. It is correctly excluded from golden_hour and stellar_horizon ensembles. The risk is that no directory convention, marker file, or test distinguishes global-grid models from regional models — the only signal is reading the config. Low severity because incompatible spatial dimensions would cause a shape mismatch error at ensemble aggregation time. **Broadened (repo-assimilation 2026-07-20):** heavy_freighter is one instance of a wider gap — `models/` now holds **120 directories**, a large fraction experimental/placeholder (8 `temporary_*`, the `*_dream`/synthetic families, `*_ranger` families, `*_dwarf` experiment set) with **no lifecycle, retirement, or marker mechanism** separating production models from scaffolds/experiments. The only signals remain per-config reads. Consequences: the all-models parametrized test fleet (conftest `ALL_MODEL_DIRS`) is inflated by non-production dirs; "what is real vs scaffold" is cognitive load with no machine answer; and cross-cutting changes (partition bumps, config-key migrations) touch experimental dirs indiscriminately. Still Tier 4 — no correctness impact, but a growing maintenance/comprehension cost. Exit direction (not applied): a maturity/marker convention (ties to ADR-017's proposed `maturity` axis) or a `models/experimental/` separation. |

---

### C-55 — Stale `xfail` marker on `test_datafactory_query_importable` produces xpass noise

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer reviews CI output and sees an xpass warning for `test_datafactory_query_importable`, masking real xpass regressions |
| **Source** | falsify Round 3 (2026-06-04) |
| **Status** | Resolved |
| **Location** | `tests/test_bright_starship_readiness.py:29` |
| **Notes** | The `@pytest.mark.xfail` decorator on `TestF1_DatafactoryQueryDependency` was stale — `datafactory_query` is now installed. Removed the xfail; the test is environment-gated by the class-level `skipif(not shutil.which("conda"))`. See C-38. **Resolved 2026-06-04.** |

---

### C-56 — Override partition files become silently stale after annual bump

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | After an annual partition bump, the 8 PARTITION_OVERRIDE HydraNet models continue using pre-bump partition values |
| **Source** | falsify: bump completeness (2026-06-06) |
| **Status** | Resolved |
| **Notes** | **Resolved 2026-06-06:** Root cause was the ingester3 dependency — all 8 override files existed solely to avoid importing `ViewsMonth`. Removed ingester3 from all 83 files, replaced with inline `_current_month_id()`. Removed all `# PARTITION_OVERRIDE:` comment markers. Replaced with a programmatic `PARTITION_OVERRIDE = True` flag for legitimate research overrides (currently unused). The bump tool now updates all 100 files uniformly. See C-01. |

---

### C-57 — Regex parser matches comments instead of real dict in config_partitions.py

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds a comment like `# Old values: "calibration": {"train": (100, 200), "test": (201, 250)}` to a config_partitions.py file; the next bump silently writes new values into the comment and leaves the actual partition dict unchanged |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:extract_values()` and `rewrite_values()` — regex `"calibration":\s*\{(.*?)\}` matches first occurrence |
| **Notes** | The regex matches the first occurrence of `"calibration": {` in the file. If that's in a comment, docstring, or dead code, `extract_values` reads wrong values and `rewrite_values` modifies the wrong location. No current file triggers this, but a single comment addition would cause silent corruption. **Tier 2 justification:** silent data corruption — the tool reports success while leaving the actual partition values unchanged. **2026-06-28 (pattern recurrence, config_meta):** the same comment-vs-real-dict regex hazard recurred *outside* the partition tooling — an ad-hoc model-cloning script (12 CM datafactory models) patched `config_meta.py` `regression_targets` with a `count=1` regex that matched a **commented** template line (`# "regression_targets": [...]`) ahead of the real key, leaving the real target wrong. Caught by per-model spot-check before commit (not shipped → C-57 stays **Resolved**). Confirms the pattern is general to **any regex-based config edit**: source models carry commented template key-lines in `config_meta.py`, so config-patching tooling must skip commented lines. Logged so future model-cloning/config tooling guards against it. |

---

### C-58 — `_load_canonical()` has no error handling for missing/corrupt partitions.json

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `meta/partitions.json` is deleted, moved, or edited with invalid JSON; the bump tool prints a raw Python traceback instead of a helpful error message |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/bump.py:_load_canonical()` |
| **Notes** | The function is two lines: `open()` + `json.load()` with no try/except. Missing file → `FileNotFoundError`. Corrupt JSON → `JSONDecodeError`. Missing keys → `KeyError` from `PartitionBoundaries.from_json()`. For annual critical infrastructure run by a maintainer, a raw traceback is a robustness failure. |

---

### C-59 — `write_atomic()` does not clean up temp files on `os.replace()` failure

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | `os.replace()` fails during a bump (permission error, disk full) after the temp file has been written; orphaned `.tmp` files remain in config directories |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:write_atomic()` |
| **Notes** | Creates `NamedTemporaryFile(delete=False)` and calls `os.replace()`. No try/finally to clean up the temp file if replace raises. A failed run touching 100 files could leave up to 100 orphaned `.tmp` files. Low probability in practice (os.replace rarely fails on same-filesystem renames) but easy to fix with try/except around os.replace. |

---

### C-60 — Repo root and scripts/ mix operational tooling, scaffolding, and investigations with no structural separation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor tries to understand the tooling layout and must read 8+ filenames at the root and 14+ files in scripts/ to distinguish operational tools from scaffold builders from investigation scripts |
| **Source** | falsify: tools organization (2026-06-07); **regression measured by falsify 2026-07-31** |
| **Status** | **Re-opened 2026-07-31, then re-closed with a guard** — it had been marked Resolved while regressing |
| **Trigger (added 2026-07-31)** | A tool is added to `tools/` without a directory named for its responsibility. The rule is documented in `tools/README.md`'s own opening line and nothing checks it, so the decay is invisible until someone counts. |
| **Location** | Originally: repo root (6 Python, 2 shell), `scripts/`, `tools/` (partitions only). **Regression:** `tools/` root — 2 loose files at the 2026-06-07 resolution (`__init__.py`, `audit_shell_health.sh`), **6** by 2026-07-31. |
| **Notes** | Violates CCP (catalog scripts change together but aren't grouped), CRP (4 unrelated responsibilities in one directory), and screaming architecture (flat layout requires reading every filename). Fix: organize into `tools/catalogs/`, `tools/scaffold/`, `tools/partitions/`; move investigation scripts to `investigations/`; move root shell scripts to appropriate locations. See ADR-011, C-01. **REGRESSED, AND THE REGISTER SAID OTHERWISE (falsify probe P4, 2026-07-31).** Counted at the resolving commit vs today: **2 loose files → 6**. All four additions post-date the resolution, three of them within five days — `audit_queryset_transforms.py` (2026-06-08), `check_credentials.py` (2026-07-27), `registry_to_env.py` (2026-07-28), `verify_committed.sh` (2026-07-31, added by the assistant during this very work). The entry read `Resolved` throughout. **The generalisable finding is not the mess but the measurement:** a structural rule stated in prose (here, `tools/README.md`'s "each subdirectory handles one responsibility") decays silently, and a register status is a claim about the past that nothing re-checks. **Re-closed 2026-07-31** by grouping on responsibility — `tools/credentials/` (`check_credentials.py`, `registry_to_env.py` — they change together, CCP) and `tools/audit/` (`shell_health.sh`, `verify_committed.sh`, `queryset_transforms.py` — read-only verification passes) — leaving only `__init__.py` at the root. **WET was preserved deliberately:** the three credential readers (`tools/credentials/*`, `tools/liveness/appwrite_api.py`) share zero code and were **not** unified; co-location is not consolidation. Two moved files carried `parents.parent` repo-root constants that silently rebased one directory deeper — caught by the test suite, fixed to `parents[2]`, and worth remembering as the standing cost of depth-counted paths. **Residual:** the count is still not machine-checked; a structural test asserting "no loose executable files at `tools/` root" would convert this entry from prose to a tripwire. Until that exists, expect the same drift. |

---

### C-61 — Fixture exclusion lists diverge across 3 locations

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new fixture model is added to `_FIXTURE_ENTRIES` in `create_catalogs.py` but not to `_FIXTURE_NAMES` in `fileops.py` or `_FIXTURE_MODELS` in `conftest.py` — causing inconsistent catalog output, bump coverage, and test discovery |
| **Source** | repo-assimilation (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tools/partitions/fileops.py:_FIXTURE_NAMES` (12 entries), `tools/catalogs/create_catalogs.py:_FIXTURE_ENTRIES` (12 entries), `tests/conftest.py:_FIXTURE_MODELS` (1 entry) |
| **Notes** | Three independent fixture exclusion sets. `_FIXTURE_MODELS` in conftest has only `fake_model` while the other two have 12 entries. The sets happen to not conflict currently because conftest uses `main.py` presence (not name) to discover models, so the extra 11 fixture names in the other lists are redundant there. But the naming inconsistency (`_FIXTURE_MODELS` vs `_FIXTURE_NAMES` vs `_FIXTURE_ENTRIES`) and the different cardinalities create confusion. Should be unified into a single source of truth. **2026-06-27 (#99 — completes a partial resolution):** the earlier fix introduced the canonical `meta/fixtures.json` for `fileops.py` + `create_catalogs.py`, but **`update_readme.py` was missed** — it still hardcoded `{fake_model, test_model, test_ensemble}` (3 of 12), so it would emit READMEs for the 9 synthetic fixture models it should skip. #99 repoints `update_readme.py` to load `meta/fixtures.json` (all three catalog/partition consumers now derive from one file) and hardens `test_bump_partitions.TestFixtureSetConsistency` — the prior check AST-matched a *set literal* and went **vacuous** once create_catalogs switched to JSON; it now source-checks that create_catalogs + update_readme load `fixtures.json` and hardcode no literal (negative-tested to fail loud). `conftest._FIXTURE_MODELS` stays `{fake_model}` **by design** — name-based exclusion for a different concern (test discovery uses `main.py` presence). |

---

### C-62 — No CIC for tools/partitions/ (partition bump tool)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer modifies `tools/partitions/domain.py` or `fileops.py` behavioral guarantees without a contract to verify against |
| **Source** | repo-assimilation (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tools/partitions/` (3 modules, 37 tests, 3 falsification audits, but no CIC) |
| **Notes** | The partition tooling is the most thoroughly tested and audited component in the repo (37 unit tests, 3 falsification rounds, expert code review). But it has no Class Intent Contract documenting its guarantees, failure modes, or boundaries. The CIC sync check workflow (`cic_sync_check.yml`) cannot flag changes to this tool. Low urgency since the test coverage is strong, but the contract gap creates a documentation asymmetry with the other tools (all have CICs). |

---

### C-63 — Partition bump test files missing ADR-005 category markers

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Test category analysis (red/beige/green distribution) reports inaccurate numbers because 4 test files (test_bump_partitions.py, test_falsify_bump_completeness.py, test_falsify_bump_edge_cases.py, test_falsify_bump_robustness.py) have no ADR-005 markers |
| **Source** | test-review (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tests/test_bump_partitions.py`, `tests/test_falsify_bump_*.py` (3 files) |
| **Notes** | ADR-005 defines the red/beige/green taxonomy for test classification. The 4 partition bump test files (37 tests total) were written without category markers. Most are green (functional correctness) with some beige (structural compliance). The falsification verification tests could be marked green (they verify fixes). Low priority but creates a documentation gap in test distribution reporting. |

---

### C-64 — Zero red (adversarial) tests for all 9 tool modules

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer introduces a bug in tools/partitions/ or tools/catalogs/ that only manifests with adversarial input (corrupt file, permission error, concurrent execution); no red test catches it |
| **Source** | falsify: test category completeness (2026-06-07) |
| **Status** | Resolved |
| **Location** | `tests/test_bump_partitions.py`, `tests/test_catalogs.py`, `tests/test_scaffold_builders.py`, `tests/test_tooling_scripts.py` |
| **Notes** | **Resolved 2026-06-07:** 30 red tests now cover 8 of 9 tool modules. Partition tooling: 9 red (garbage input, partial structure, negative month_ids, missing return, missing section, negative bump, missing JSON key, non-iterable value, permission error cleanup). Scaffold: 3 red (github failure, without_directory_raises x2). Catalogs: 11 red (malformed markers, empty content, missing keys, non-list targets, empty model list, adversarial regex input). `build_package_scaffold.py` cannot be tested without `views_pipeline_core` — accepted gap. Also found: `_format_targets` crashes on non-string non-list input (TypeError) — characterized as red test. |

---

### C-65 — generate_features_catalog.py and update_readme.py have no core-functionality tests

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer modifies the main loop of `generate_features_catalog.py` or `update_readme.py` (model discovery, config loading, output generation); no test catches a regression in core behavior |
| **Source** | falsify: test category completeness (2026-06-07) |
| **Status** | Open |
| **Location** | `tools/catalogs/generate_features_catalog.py` (115 lines, 4 regex characterization tests only), `tools/catalogs/update_readme.py` (276 lines, 6 helper characterization tests only) |
| **Notes** | **Partially resolved 2026-06-07:** Added 11 functional tests for `generate_features_catalog.py`: 5 for `extract_columns_from_querysets()` (single file, dedup, loa extraction, empty dir crash, non-Python ignored) and 6 for `generate_markdown_table()` (valid markdown, headers, placeholders, row count, empty crash, queryset preserved). Found 2 bugs: empty dir crashes groupby (C-66), empty DataFrame crashes tabulate (C-67). `update_readme.py` orchestration remains untestable without views_pipeline_core — accepted. |

---

### C-66 — `generate_features_catalog.py` crashes on empty input (both functions)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | `generate_features_catalog.py` is run against a model set that yields no queryset columns — an empty or non-Python directory, a filtered-to-nothing model list, or a fresh scaffold before any `config_queryset.py` exists |
| **Source** | test: catalog core tests (2026-06-07); merged with C-67 during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `tools/catalogs/generate_features_catalog.py:72` (`extract_columns_from_querysets`), `:97` (`generate_markdown_table`) |
| **Notes** | Two crashes on the same empty-input path, one immediately downstream of the other. **(1) `extract_columns_from_querysets():72`** — builds an empty `columns_info` list, converts it to a column-less DataFrame, then calls `df.groupby(['column_name','loa'])`, which raises `KeyError` because those columns don't exist. Fix: early return when `columns_info` is empty. **(2) `generate_markdown_table():97`** — `tabulate(..., colalign=("center",))` assumes at least one data column; an empty DataFrame has zero, so it raises `IndexError`. Fix: skip `colalign` when `table_data` is empty, or return a header-only table. Both characterized as red tests (`test_empty_directory_crashes`, `test_empty_dataframe_crashes_tabulate`). **Merged with C-67 during review-rr (2026-07-31)** — same file, same session, same input condition, adjacent lines, and the second is only reachable through the first; two entries overstated the count without adding information. Member of **Cluster D** (catalog tooling is a script, not a program). **Backlog candidate:** mechanical fix, single file, Tier 4 — see the review-rr demotion list. |

---

### C-67 — `generate_markdown_table()` crashes on empty DataFrame *(merged into C-66)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-66** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. Same file (`tools/catalogs/generate_features_catalog.py`), same empty-input condition, and reachable only through the C-66 crash path. Tracked together at C-66, Tier 4. |

---

### C-68 — `config_meta.py` fields duplicate operational config keys with no enforcement of doc-only status

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer edits `regression_targets` (or `level`, `algorithm`, `prediction_format`) in a model's `config_meta.py` expecting it to change training/evaluation behavior, unaware the file is documentation-only |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/configs/config_meta.py`, `models/*/configs/config_hyperparameters.py` |
| **Notes** | `config_meta.py`'s docstring states "modifying it will not affect the model, the training, or the evaluation." Yet several keys it declares — notably `regression_targets` — are also required as *operational* keys in `config_hyperparameters.py` (C-52 added `regression_targets` to 9 hyperparameter files for PFE participation). The same logical field thus lives in two files with opposite semantics: inert in meta, behavioral in hyperparameters. No test asserts the two copies agree, and no warning fires when a developer edits the inert copy. A change to the meta copy is silently ignored; a stale meta copy also misleads readers and the generated catalogs (`tools/catalogs/create_catalogs.py` reads `config_meta.py`). Low severity — no model-output corruption — but a maintainability footgun amplified across 90 models. See also C-52 (regression_targets added to hyperparameters), C-53 (stray `prediction_format` key leaked into hyperparameters during merge). **Tier recalibrated 4 → 3 during review-rr (2026-07-31):** "a change to the meta copy is silently ignored" across ~90 models is the *same defect class* as C-104 (Tier 2 — "the CI contract validates a key the runtime may not read"). A silently-ignored config edit is not a Tier-4 code-quality observation; it is the repo's signature failure mode. Held at 3 rather than 2 only because the wrong-edit here produces stale documentation rather than a wrong forecast. Member of **Cluster A** (declared-but-unenforced), with C-104 and C-85. |

---

### C-69 — `config_sweep.py` has zero test coverage and no validation of swept-parameter structure

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer edits a model's `config_sweep.py` and mistypes a swept parameter — e.g., `'values': [...]` written as `'value': [...]`, or a parameter name that does not match `config_hyperparameters.py` — then launches `--sweep`; the sweep runs but silently pins or ignores the parameter |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/configs/config_sweep.py` (observed: `models/violet_visitor/configs/config_sweep.py`) |
| **Notes** | Unlike `config_meta.py` (`test_config_completeness.py`), `config_partitions.py` (`test_config_partitions.py`), and `config_hyperparameters.py` (C-05 ReproducibilityGate), `config_sweep.py` has no structural or semantic test. The current working-tree rewrite of `models/violet_visitor/configs/config_sweep.py` (a 128-line hand edit mixing `{'value': ...}` and `{'values': [...]}` entries) illustrates the exposure: a `values`→`value` typo silently converts a swept dimension into a fixed constant, and a parameter key that does not correspond to a hyperparameter is silently ignored by W&B. Failures are not loud — the sweep completes but explores the wrong space, wasting GPU/compute and surfacing a misleading "best" run. Affects anyone running sweeps. See also C-05 (HP presence validation — does not cover sweep configs), D-04 (static-analysis vs behavioral-execution test gap). |

---

### C-70 — `run.sh` environment-bootstrap logic duplicated across ~90 protected scripts

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | The planned conda→uv migration (see `reports/conda_to_uv_migration_*`) or any change to env-bootstrap logic requires editing the near-identical `run.sh` in every model/ensemble/api/extractor/postprocessor directory |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Open |
| **Location** | `models/*/run.sh`, `ensembles/*/run.sh`, `apis/*/run.sh`, `extractors/*/run.sh`, `postprocessors/*/run.sh` (~90+ scripts) |
| **Notes** | Every model carries a near-identical `run.sh` that bootstraps a conda env, dry-run-checks `requirements.txt`, and invokes `main.py`. The bootstrap logic is duplicated rather than sourced from a shared script, so a change must fan out across all ~90 files — and these files are production infrastructure that must not be casually modified (operating constraint). C-39 already demonstrated the fan-out cost (79 shebangs corrected in one sweep); C-50 notes `run.sh` cannot be edited to fix the local-install path. The duplication is consistent with the project's accepted self-containment stance for configs (D-01), but unlike partition configs there is no `meta/`-style single source of truth or bump tool for `run.sh` — it is accepted-by-default rather than deliberately governed. Low severity (failures are loud, at bootstrap time), but a coordination cost that recurs on every infra change. See also D-01 (intentional config duplication is load-bearing), C-39 (shebang fan-out — resolved), C-50 (`run.sh` modification constraint). **2026-08-02 (expert-code-review):** the duplication now has a measured correctness cost, not only a migration cost — **C-119** records that the duplicated install gate decides a production install from a line count of pip's log, and **C-115** records a version boundary encoded in the duplicated `env_path` line. The fix belongs in the generator (`template_run_sh.py`, views-pipeline-core#384), not in ~131 copies; fixing copies is what let C-39 regress 24 times. |

---

### C-71 — violet_visitor diverged from trio parity on two axes at once (regression loss *and* posterior sample count)

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Someone runs or interprets a golden_hour↔stellar_horizon parity comparison assuming the viewser and datafactory trios are matched — but violet_visitor differs on **both** the regression loss (`hurdle_nb`, formerly `lognormal_nll`, vs `tobit` on the other five) **and** `n_posterior_samples` (**8** vs **16** on the other five) |
| **Source** | review (PR #116, 2026-06-09); merged with C-87 (review-diff 2026-06-18) during review-rr (2026-07-31) |
| **Status** | Open |
| **Location** | `models/violet_visitor/configs/config_hyperparameters.py` (`EXPERIMENT_IN_PROGRESS = True`; `loss_reg`, `n_posterior_samples`); the skip + documenting assertion at `tests/test_datafactory_parity.py::test_both_trios_use_same_loss`, `::test_constituent_posterior_samples_match`, `::test_violet_visitor_is_experiment_in_progress` |
| **Notes** | violet_visitor's regression loss was intentionally changed from `tobit` to `lognormal_nll` (Arm-1 hurdle experiment, magnitude_calibration dossier 2026-06-08, issue #85; commit 908d383). The viewser trio (pink_pirate, blue_stranger, violet_visitor) and datafactory trio (bright_starship, bold_comet, blazing_meteor) were designed to be loss-identical so golden_hour (viewser ensemble) and stellar_horizon (datafactory ensemble) could be compared apples-to-apples (the parity programme behind C-48). violet_visitor's divergence breaks that: a golden_hour↔stellar_horizon comparison now confounds the loss change with the data-source change. `test_both_trios_use_same_loss` previously asserted strict uniformity (`{"tobit"}`); it was updated (PR #116) to pin the expected diverged state (five `tobit` + violet_visitor `lognormal_nll`), so the divergence is explicit and any *further* drift is still caught. The risk is interpretive, not silent — but a reader unaware of the experiment could draw wrong parity conclusions. Revisit when Arm-1 concludes: either restore `tobit`, or promote the hurdle loss across the whole trio. See also C-48 (variable-variant parity — resolved), C-37 (forecasting parity divergence), C-44 (concat aggregation quality), C-69 (sweep config untested). **2026-06-12:** the divergence persists but the loss moved again: `lognormal_nll` → `hurdle_nb` (TruncatedNB body + weighted-BCE gate; ZINB epic views-hydranet#102, decision A). `test_both_trios_use_same_loss` pin updated in the same changeset. The parity caveat is unchanged: golden_hour↔stellar_horizon comparisons still confound the loss change with the data-source change. **Second divergence axis, absorbed from C-87 (review-rr, 2026-07-31):** violet_visitor's `n_posterior_samples` was cut **16 → 8** on 2026-06-16 as an **interim OOM workaround** — the eval stage OOMs at 16; 8 is gated by a "one run completes without the eval-stage OOM" check and is to be **restored to 16 once the OOM is fixed** (tracked as `views-hydranet C-116` / views-hydranet#124, outside this repo's register). Consequence: the trio sample-count parity invariant is broken on the same model that already broke the loss invariant, and golden_hour's expected concat total shifts 48 → **40** (16+16+8, see C-74). `test_constituent_posterior_samples_match` was updated in the same change to **pin** the intentional divergence rather than assert uniformity — mirroring the C-71 loss pin — so both divergences are explicit and any *further* drift still fails. **Why merged:** one model, one cause (deliberate single-model experiments run against a trio designed for parity), one closing action, and both axes already pinned by tests — two entries doubled the apparent risk without doubling the information. **Revisit when both experiments conclude:** restore `tobit` + `n_posterior_samples: 16`, revert both test pins, or promote the changes across the whole trio. Member of **Cluster E** (parity-programme drift). See also C-74 (golden_hour sample count), C-72 (the numerical fallout of the loss experiment), C-37 (forecasting-boundary parity), C-48/C-49 (trio parity). **2026-08-03 — mechanism changed from exact-value pin to truthful skip (Epic #242 S1.5, views-hydranet#255; resolves views-platform/views-models#254 + #297).** The root problem surfaced by #254/#297: violet_visitor is an **actively churning** R&D model — its committed `loss_reg` is a moving target (the committed value was `mse`, the parity pin expected `hurdle_nb`, and the working tree flickers), so pinning an exact value flickers red/green and was the **last red test blocking the dev→main release**. Fix: violet_visitor's config now declares `EXPERIMENT_IN_PROGRESS = True`; `test_both_trios_use_same_loss` and `test_constituent_posterior_samples_match` **skip** it while still pinning the other five (`tobit` / `16`), and a new `test_violet_visitor_is_experiment_in_progress` asserts the marker so the skip is never silent (fails loud with re-pin instructions if the marker is removed). This unblocks the release without committing a transitional value; the exact settled value is pinned at Epic #242 S3 #246 when the fleet moves to the `gated_NB` roster. Same change removed the **retired `body_mask` knob** from violet's committed config (→ `body_supervision: 'all'`, ADR-065 migration). **2026-08-03:** the parity pins now **skip** violet_visitor via an `EXPERIMENT_IN_PROGRESS` marker in its config rather than pinning a value that churns while the experiment runs; re-pin when the roster lands (Epic #242 S3 #246). This note lives in Notes and not in Status because the Status field is a controlled vocabulary that `tests/test_falsification_merge_readiness.py::test_open_count_accurate` parses — prose after `Open` silently drops the entry from the header count. |

---

### C-72 — violet_visitor predictions overflow to `Inf` under the Arm-1 `lognormal_nll` loss

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | golden_hour (or any consumer) is next run/aggregated against violet_visitor's calibration predictions while it runs the Arm-1 `lognormal_nll` loss — 46–63% of regression cells are `Inf` |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `models/violet_visitor/configs/config_hyperparameters.py` (`loss_reg: lognormal_nll`, `loss_reg_sigma: 0.9`, `hurdle_threshold: 0`); artifact `models/violet_visitor/data/generated/predictions_calibration_20260609_051916/` |
| **Notes** | Verified directly: the 2026-06-09 calibration run has `Inf` in **63.5% / 59.4% / 46.4%** of `lr_sb_best / lr_ns_best / lr_os_best` cells (finite max 3.4e38 = float32 ceiling); the prior `tobit` run (2026-06-08) was clean (0 Inf, max ≈ 4365). Root cause: the lognormal inverse `exp(µ)` overflows float32. **Classification targets (`by_*`) are sane** — the breakage is regression-only. There **is** a signal (`tests/test_pfe_production_readiness.py::TestTransformUndoScale::test_no_inf[violet_visitor_calibration]` catches it) → Tier 2, not Tier 1. **Accepted as an active experiment**: the user has chosen to leave violet_visitor's loss as-is (issue #85, magnitude_calibration dossier, commit `908d383`); this entry documents the known state — it is **not** a request to change the model. `lognormal_nll` is a registered, valid loss in views_hydranet (`utils/utils.py:66`); this is purely numerical, not a registration issue. To make the experiment usable, tame the overflow (clamp/bound `µ` in `views_hydranet` `LogNormalFixedSigmaLoss`). Downstream: a fresh golden_hour run would ingest the Inf. See also C-71 (same change's parity impact), C-74 (golden_hour sample count), C-44 (concat aggregation). **2026-06-12:** Arm-1 (`lognormal_nll`) is superseded — violet_visitor switched to `hurdle_nb` (ZINB epic views-hydranet#102, decision A), removing the overflow-prone lognormal inverse from the active config. The Inf-bearing 2026-06-09 artifact remains on disk until a fresh hurdle-NB calibration run replaces it; keep Open until a clean artifact exists (the `test_no_inf` guard stays armed). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the overflow-prone `lognormal_nll` configuration is **superseded** — the active config is `hurdle_nb`, so no future run can reproduce the `Inf`. What remains is one stale on-disk artifact, guarded by an armed test that fails loud if it is consumed. Residual cleanup, not live fragility. Member of **Cluster E** (parity-programme drift). |

---

### C-73 — Ensemble scaffold builder imports an unreleased pipeline-core symbol; CI installs core unpinned

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | CI (or any fresh `pip install views_pipeline_core`) resolves a released pipeline-core (PyPI 2.3.0 / tag 2.3.1) that lacks `template_config_modelset` — the 3 `EnsembleScaffoldBuilder` tests fail at import and the builder is unusable |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `tools/scaffold/build_ensemble_scaffold.py:8`; `.github/workflows/run_tests.yml:19` (`pip install views_pipeline_core`, unpinned); `tests/test_scaffold_builders.py::TestEnsembleScaffoldBuilderDirectoryCreation` |
| **Notes** | `build_ensemble_scaffold.py` imports `template_config_modelset` from `views_pipeline_core.templates.ensemble`. That symbol exists only on pipeline-core `development` — in **no released/tagged version**: PyPI latest is 2.3.0; git tag 2.3.1 is malformed (its `pyproject` still says `version = "2.3.0"` and it also lacks the symbol). CI installs the package **unpinned**, resolving to 2.3.0, so the 3 scaffold tests `ImportError` and the builder is broken against any release. Real fix (no skip): cut a properly-versioned pipeline-core release shipping the symbol — HEAD is **137 commits ahead of 2.3.1** (dependency removals, signature/exception changes) → likely **minor/major, not patch**; run a cross-consumer smoke-import first; prefer a minimal release branch over 2.3.0 — then pin views-models CI + the scaffold path narrowly to it. (Templates already package via poetry-core — no `packages` directive — so adding `templates/{model,ensemble,package}/__init__.py` is robustness, not the blocker.) See also C-31 (upstream API breakage), C-42 (synthetic models on unreleased core branch). **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the failure is a loud `ImportError` at collection time in CI, not a silent wrong result. Held prominent despite the demotion because it is the **named concrete blocker inside Cluster C** and a constituent of C-80's standing red CI — fixing it is a prerequisite for the green-CI baseline, which is itself the precondition for every other signal in this register being trustworthy. Member of **Cluster C** (cross-repo dependencies have no released contract). |

---

### C-74 — golden_hour `concat` yields 12 posterior samples instead of 48

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A `PredictionFrameEnsembleManager` `concat` ensemble (golden_hour: 3 constituents × 16 samples) is aggregated and the output carries fewer samples than the sum of its constituents |
| **Source** | falsify (2026-06-09) |
| **Status** | Open |
| **Location** | `ensembles/golden_hour` (`aggregation: concat`); views-pipeline-core `PredictionFrameEnsembleManager` concat path; `tests/test_pfe_production_readiness.py::TestPFEEnsembleAggregation::test_aggregated_sample_count[golden_hour_calibration]` |
| **Notes** | golden_hour (concat, 3×16) should aggregate to **48** posterior samples; its calibration artifact (`predictions_calibration_20260603_135314`, June 3 — **predates** the violet_visitor Inf, so NOT caused by C-72) has only **12**. 12 is not a clean multiple of 48, so this is unlikely to be mere staleness of one constituent (that would give 16/32) — it points to a real defect in the concat path (samples dropped/sub-sampled rather than concatenated), which would **silently understate ensemble uncertainty**. Verify with a fresh run: 48 → it was staleness; still 12 → real concat bug to fix in views-pipeline-core. See also C-44 (concat CRPS quality), C-45 (ensemble `-t` cascade), C-46 (PFE classification targets). **2026-06-18:** the expected total changed from 48 to **40** — violet_visitor dropped to 8 samples (C-87), so golden_hour's constituents are now 16+16+8; factor this into the fresh-run check. **2026-06-26 (rusty_bucket work):** a control falsifies the "concat-path-wide" hypothesis — `synthetic_chant` (3×64, equal constituents) aggregates to exactly **192 = sum**, confirming PFE concat *does* concatenate the sample axis correctly (`prediction_frame_ensemble.py:99`). So golden_hour's **12** is a golden_hour-specific defect (its unequal 16/16/8 constituents + a stale June-3 artifact), **not** a views-pipeline-core concat bug and **not** a test mis-encoding: `_expected_ensemble_samples` correctly expects `sum(samples)` (verified and kept). The `test_aggregated_sample_count` failure is artifact-dependent — it **skips in CI** (fresh clone has no artifacts) and fails only on stale local artifacts. Fresh-run check still owed: rerun golden_hour → 40 = staleness; still 12 = a real golden_hour aggregation defect. **2026-07-31 — investigated; the fresh run is now the ONLY way to settle it, and the artifact was never self-consistent.** Two facts recovered from git and disk: (1) at the artifact's own timestamp (`predictions_calibration_20260603_135314`) all three constituents were configured at **`n_posterior_samples: 64`** — sum **192**, not 48 and not 40 — so the artifact's **12** did not match its *contemporaneous* config either; config drift since (64/64/64 → 16/16/8) explains none of it. (2) The inputs are gone: `pink_pirate` and `blue_stranger` have **no prediction artifacts on this machine at all**, and violet_visitor's are from 2026-07-27 — so the pooled result cannot be reconstructed or diagnosed from artifacts, only re-run. The likeliest mechanism given (1) and (2) is **C-85** — the ensemble loads each constituent's cached `y_pred.npy` by artifact timestamp with no config fingerprint, so golden_hour pooled whatever stale per-constituent npys existed on 2026-06-03, not what the configs declared. That makes this a probable *instance* of C-85 rather than an independent concat defect, consistent with the `synthetic_chant` control (3×64 → exactly 192) still passing today. **Test made truthful the same day:** `test_aggregated_sample_count` compared a frozen artifact against *live* configs and so failed permanently until someone re-ran; it now skips with the drift spelled out (`pink_pirate: 64 at artifact time -> 16 now; …`) and asserts normally whenever the compared value has **not** moved — deliberately keyed on the *value*, not on "the config file changed", so `synthetic_chant` (the C-74 control) stays live. Guard pinned by `TestArtifactStalenessGuard` against going vacuous. **Tier recalibrated 2 → 3 during review-rr (2026-07-31):** the original Tier 2 was set while the "PFE concat drops samples platform-wide" hypothesis was live. The 2026-06-26 `synthetic_chant` control **falsified** it (3×64 → exactly 192), narrowing this to one stale June-3 artifact on one ensemble, in a test that skips in CI and is not on any delivery path. Re-promote to Tier 2 **if** the owed fresh run still returns 12 — that would restore a real, silent uncertainty-understatement defect. See also C-90 (degenerate-mixture stand-ins), C-91 (sample-count adequacy), C-71 (the 16→8 change that moved the expected total to 40). Member of **Cluster E** (parity-programme drift). |

---

### C-75 — bright_starship datafactory readiness test is mis-scoped for CI (false red)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | CI runs `test_bright_starship_readiness.py::TestF1` — it shells `conda run -n views-hydranet-env` for a workstation-only env absent in CI, erroring (`EnvironmentLocationNotFound`) instead of testing a CI-checkable contract |
| **Source** | repo-assimilation (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_bright_starship_readiness.py::TestF1_DatafactoryQueryDependency` (class `skipif` only checks `shutil.which("conda")`, truthy in CI) |
| **Notes** | The test is a local pre-flight probe (per its docstring) but executes in CI because the `skipif(not shutil.which("conda"))` guard passes (CI has miniconda) while the named env `views-hydranet-env` does not exist → false red. Real fix (no skip): provision `views-datafactory` in the CI job and assert a real `import datafactory_query` in the CI interpreter, plus static contract checks (requirements declares it; descriptor shape; spec resolvable). Add the equivalent for shining_codex (closes C-41). See also C-38 (datafactory_query availability), C-55 (prior stale-xfail on this test — resolved). **2026-06-12: Resolved** (issue #122, HYBRID design decided with maintainer): the conda probe is now a workstation pre-flight that skips truthfully when the target env is absent (`_conda_env_path` basename-matches `conda env list --json`, probes via `conda run -p`); CI-meaningful coverage moved to static contract checks (requirements declares views-datafactory; queryset imports datafactory_query; generate() exists via AST) — static because the queryset imports datafactory at module level and no pinned views-datafactory release exists (C-73 lesson: no unpinned git deps in CI). Real-install CI check deferred to a tracked follow-up issue, conditional on a datafactory release. Guard sanity itself is pinned by `TestEnvGuardSanity`. |

---

### C-76 — `test_values_not_log_compressed` applies a false invariant to `ZeroModel`

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | The PFE log-compression test runs against a zero/constant baseline (e.g. zero_cmbaseline) and asserts `max>10`, which a correct all-zeros prediction can never satisfy |
| **Source** | repo-assimilation + expert-review (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_pfe_production_readiness.py::TestTransformUndoScale::test_values_not_log_compressed` |
| **Notes** | The `max>10` heuristic (guarding against predictions left on `log1p` scale) is valid for learned-magnitude models but FALSE for `ZeroModel`, which correctly emits all-zeros (zero_cmbaseline max=0.0 → perpetual fail). Verified `locf_cmbaseline` (max 17412) and `average_cmbaseline` (max 4743) legitimately pass and MUST keep the guard — so the fix is to exclude **`ZeroModel` only** (keyed off `config_meta["algorithm"]`) and, better, assert `max==0 and min==0` for ZeroModel (a ZeroModel emitting nonzero is itself a bug). Local-only (CI has no prediction artifacts). A test-design correction, not a coverage skip. **2026-06-12: Resolved** exactly as described (issue #129) — ZeroModel branch asserts all-zeros, all other models keep the `max>10` guard. |

---

### C-77 — synthetic_chant README omits cross-pattern CRPS-inflation semantics

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A reader interprets synthetic_chant's ensemble CRPS as prediction quality, unaware it reflects cross-pattern disagreement measured against models[0]'s actuals |
| **Source** | repo-assimilation + falsify (2026-06-09) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `ensembles/synthetic_chant/README.md`; `tests/test_falsification_synthetic_runs.py::test_falsify_01_synthetic_chant_readme_documents_crps_inflation` |
| **Notes** | Genuine documentation gap (TDD-red test). Constituents use different synthetic patterns — `lucid_dream`=`vertical_stripe` (models[0] → supplies ground-truth actuals), `vivid_dream`=`horizontal_stripe`, `waking_dream`=`diagonal_gradient`; the ensemble evaluates all predictions against models[0]'s actuals, so CRPS (constituent 0.000/0.002/0.043 → ensemble 1.044) measures cross-pattern disagreement, not prediction quality. Real fix: document these facts in the README (mirror `ensembles/synthetic_chorus/README.md`). See also C-43 (synthetic_chorus order-dependency), C-42 (synthetic models on unreleased core). **2026-06-12 (root cause):** the documentation EXISTED — added 2026-05-26 (`8af868e`, the same commit that added the test) — and was deleted by the 2026-06-04 README regeneration (`243873a`); `tools/catalogs/update_readme.py` rebuilds READMEs from the scaffold, preserving only the `## Created on…` tail. Re-writing the docs without fixing the generator (C-78) just re-arms the failure — sequence with C-78. **2026-06-12: Resolved together with C-78** (issues #123/#130) — semantics restored inside a `<!-- manual -->` block, which the fixed generator now preserves. |

---

### C-78 — README regeneration silently destroys hand-written documentation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `tools/catalogs/update_readme.py` is run (manually or via `update_catalogs.yml`) against any model/ensemble README carrying manual content outside the preserved `## Created on…` tail |
| **Source** | session investigation (2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tools/catalogs/update_readme.py:125-135` (scaffold rebuild, `## Created on` regex tail-preserve); `.github/workflows/update_catalogs.yml` (automated path) |
| **Notes** | Verified incident: the synthetic_chant CRPS-semantics documentation added 2026-05-26 (`8af868e`) was deleted by the 2026-06-04 regeneration (`243873a`, "docs: regenerate model catalog tables and per-model READMEs") — the direct cause of the C-77 test failure and the first June 4 CI red. The generator rebuilds each README from `README_scaffold.md` and preserves only the `## Created on…` tail, so ANY hand-written section in any of the ~100 model/ensemble READMEs is silently destroyed on every regeneration — no diff review gate on the automated path, no error signal. Tier 3 (silent destruction of committed work product; affects every contributor who documents a model). Real fix: preserve-markers (e.g. a `<!-- manual -->` block) or regenerate only the generated tables, plus a regression test that a marked manual section survives regeneration (C-65: the tool currently has zero tests). See also C-77 (the wiped instance), C-65, C-36. **2026-06-12: Resolved** (issue #130) — `tools/catalogs/readme_preserve.py` extracts `<!-- manual -->…<!-- /manual -->` blocks from the old README and re-appends them after regeneration (wired into both loops of `update_readme.py`); regression tests in `tests/test_readme_preserve.py` (chips at C-65). |

---

### C-79 — Stale strict-xfail on fired chunky_bunny readiness tripwire keeps suite red

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Anyone runs the local suite (or reads its output) while `test_target_transform_fix_is_released` still carries `@pytest.mark.xfail(strict=True)` — the XPASS registers as a hard failure and noise-trains readers to ignore red |
| **Source** | session investigation (2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tests/test_chunky_bunny_readiness.py::test_target_transform_fix_is_released` |
| **Notes** | The tripwire worked exactly as designed: it was armed 2026-06-09 against "published views-stepshifter lacks `target_transform`" and fired when views-stepshifter merged the mechanism to main on 2026-06-08/09 (`261ef6c`, PR #74 → main merge #76, released as 1.3.0). The strict-xfail marker is now stale and produces a permanent suite failure (same genre as resolved C-55). Fix: flip to a plain assertion. The two sibling tripwires remain LEGITIMATELY red and must stay armed: `test_per_model_envs_exist` (envs/views_stepshifter, envs/views_r2darts2 unprovisioned on this box) and `test_ensemble_uses_the_fixed_code_path` (validation env ≠ execution env, placeholder). I.e., the release precondition is met but chunky_bunny is NOT yet runnable via run.sh envs — the #117 dev-mode run tracker sidesteps this. See also C-55 (genre), issues #117, #114, views-stepshifter#55. **2026-06-12: Resolved** (issue #128) — xfail removed; the test is now a plain regression guard with a `skipif` when the sibling views-stepshifter checkout is absent (CI-safe, the C-75 lesson applied proactively). The two sibling tripwires remain armed. |

---

### C-80 — No green CI baseline since 2026-06-04 — new failures arrive invisible, merges proceed unvalidated

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any PR is merged to development while run_tests.yml is red — the merge is structurally unvalidated and any NEW breakage it introduces is indistinguishable from the standing red |
| **Source** | session investigation (2026-06-12) |
| **Status** | Open |
| **Location** | `.github/workflows/run_tests.yml`; GitHub Actions history (last green: 2026-06-04 01:21) |
| **Notes** | Every run_tests.yml run since 2026-06-04 01:21 has failed (40/40 checked). The standing red is the union of C-73 (scaffold/pipeline-core skew, since June 5), C-75 (bright_starship env probe, structural), and C-77/C-78 (README wipe, June 4). Consequence observed this week: three independent NEW breakages (June 5 scaffold skew, June 8 chunky_bunny tripwire fire, June 9 zero_cmbaseline false invariant) accumulated unnoticed because red-on-red signals nothing, and PRs #116–#126 were all merged on red CI. Tier 2: structural fragility with a realistic, recurring trigger — every merge until CI is green again. Exit: resolve C-73 + C-75 + C-77/C-78 (tracked as the CI-green umbrella issue), then adopt the policy that development merges require green CI. See also C-28 (CI only checks last exit code), C-03 (integration tests not in CI). **The LOCAL half, found and fixed 2026-07-31:** the same harm existed on the developer's machine for a different reason — two tests returned a verdict on **local workspace state** rather than on code, so a normal `pytest` was red by construction. `TestF1_UncommittedWork` asserted the working tree was clean (perpetual trigger: any work in progress failed it) on the false premise that "uncommitted changes will be lost on merge" — a GitHub merge does not touch a local tree. `test_aggregated_sample_count` compared a frozen artifact against live configs (C-74). Both were **green in CI and red locally** — the C-75 class inverted, and invisible in CI precisely because a fresh clone has neither dirty files nor artifacts. Rewritten to their real invariants (dirty files that *overlap the incoming diff*; artifacts whose compared *value* has drifted), each with negative tests pinning the guard against going vacuous. Local suite is now **7416 passed, 0 failed**. Lesson for this entry's exit: "green CI" is a necessary but insufficient target — a suite can be honest in CI and useless on the machine where the work happens. |

---

### C-81 — `update_readme.py`'s top-level write-as-you-iterate loops crash mid-run and fire on import

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `update_catalogs.yml` runs on a fresh checkout (where `ensembles/cruel_summer` and `ensembles/white_mustang` have no tracked `artifacts/`), or a local regeneration runs while a stray partial model dir sits in `models/` — the script crashes after rewriting an arbitrary prefix of READMEs |
| **Source** | session verification of PR #133 (2026-06-12) |
| **Status** | Open |
| **Location** | `tools/catalogs/update_readme.py` (both loops construct `ModelPathManager`/`EnsemblePathManager` with default `validate=True`; writes happen per-directory as iteration proceeds) |
| **Notes** | Observed live, three separate crash points: stray untracked `models/teenage_dirtbag` and `models/cool_cat` (partial dirs, no `artifacts/`), then tracked `ensembles/white_mustang` (no `artifacts/` in git; `cruel_summer` same gap — the C-32 `.gitkeep` backfill covered models, not these ensembles). `ModelPathManager` raises `FileNotFoundError` on a missing standard dir, killing the whole run. Because the script writes each README as it iterates (`iterdir()`, unsorted), a crash leaves an arbitrary subset regenerated — locally confusing; in the workflow the step fails (post-C-28 `set -e`), so catalogs go silently stale rather than partially committed. Fix directions: (a) construct path managers with `validate=False` (catalog generation is read-only on the dir structure) or per-entry try/except + end-of-run failure summary; (b) backfill `artifacts/.gitkeep` for cruel_summer/white_mustang (C-32 extension to ensembles); (c) iterate only git-tracked dirs so workstation strays can't break tooling. See also C-32 (root cause for the tracked gaps — Mitigated, recurrence here), C-28 (exit-code masking in this workflow — Resolved), C-65 (catalog tools untested), C-78 (manual-block preservation — Resolved; orthogonal fix in the same script). **Second failure path, absorbed from C-93 (review-rr, 2026-07-31): the same two loops sit at module top level with no `if __name__ == "__main__"` guard** (`update_readme.py:84` models, `:215` ensembles), so merely *importing* the module to reuse a helper or unit-test a function executes the entire regeneration as a side effect — and crashes on the first incomplete dir. Demonstrated during #99: `python -c "from tools.catalogs.update_readme import _FIXTURE_ENTRIES"` raised `FileNotFoundError` on `models/teenage_dirtbag/artifacts`. This is why #99's `TestFixtureSetConsistency` had to **source-read** the file with a regex instead of importing it — the canonical fixture value is not assertable by import, which in turn is why that check went vacuous once create_catalogs switched to JSON (see C-61). **Why merged:** identical location (the two top-level loops), one refactor closes both — wrap each loop in a `def main()` behind `__main__`, construct the path managers with `validate=False`, and collect-then-write instead of writing as you iterate. Splitting "it crashes when run" from "it runs when imported" implied two fixes where there is one. Member of **Cluster D** (catalog tooling is a script, not a program), with C-65, C-66, C-83. |

---

### C-82 — Manual blocks duplicate when a README also carries a `## Created on` section

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A README processed by `update_readme.py` carries BOTH a `## Created on` section and a `<!-- manual -->` block, and a regeneration runs — the block is emitted twice and multiplies on every subsequent run |
| **Source** | falsify (PR #133 audit, probe P1, 2026-06-12) |
| **Status** | Resolved (2026-06-12) |
| **Location** | `tools/catalogs/update_readme.py` (Created-on capture `re.search(r"(## Created on.*)", …, re.DOTALL)`, both loops); interaction with `readme_preserve.merge_manual_blocks` |
| **Notes** | The Created-on regex captures from the heading to END OF FILE; merged manual blocks live at the end of the file, so they get swallowed into the captured created-section (re-inserted via `{{CREATED_SECTION}}`) AND re-appended by the merge → duplication, compounding per regeneration. Latent when found: no README the script processes had a Created section (test_model/test_ensemble are fixture-skipped; apis/ and postprocessors/ are not iterated). Wrong-output is duplication, not loss → Tier 4. **Resolved same day:** `readme_preserve.strip_manual_blocks()` added; both loops now run the Created-on capture on the stripped text (blocks extracted from the original first). Falsification stub `tests/test_falsification_readme_preserve.py` un-xfailed to a plain regression guard. See also C-78 (sibling failure mode — loss), C-81 (sibling failure mode — crash), C-65 (catalog tools untested — now partially chipped). |

---

### C-83 — `## Created on` sections are lost on the second regeneration (heading rename breaks recapture)

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A README gains a `## Created on` section and `update_readme.py` runs twice — the first run renames the heading to `## Model Created on`, which the capture regex `(## Created on.*)` no longer matches, so the second run drops the section entirely |
| **Source** | falsify (PR #133 audit, bonus discovery while pinning C-82, 2026-06-12) |
| **Status** | Open |
| **Location** | `tools/catalogs/update_readme.py` (both loops: `re.search(r"(## Created on.*)" …)` followed by the `[:2] + " Model"` heading rewrite) |
| **Notes** | Pre-existing, unrelated to the C-78/C-82 fixes. The rename-then-recapture mismatch means any created-section survives exactly one regeneration — which likely explains why NO currently-processed README has one (they were silently eaten by successive catalog runs over time; only fixture/non-iterated READMEs retain theirs). Same content-loss family as C-78 but a different mechanism. Fix directions: match both headings (`(## (?:Model )?Created on.*)`) and stop re-prefixing if already prefixed, or stop renaming the heading altogether. Alternatively: deprecate the special-cased created-section in favor of the `<!-- manual -->` mechanism (C-78), which is rename-proof. See also C-78, C-82, C-65. |

---

### C-84 — Constituent wandb partition metadata diverges after a partial re-run, blocking the ensemble evaluation report

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A subset of an ensemble's constituents is re-run after a partition bump (or any config change) while the rest keep older runs — their *latest* wandb run configs then disagree, and `EnsembleManager … -e -re` aborts the report with `Partition metadata mismatch between models` |
| **Source** | execution incident (chunky_bunny re-aggregate, 2026-06-13) |
| **Status** | Open |
| **Location** | `views-reporting/views_reporting/templates/reports/evaluation.py:189-211` (reads `get_latest_run(...).config` per constituent and requires `{run_type: {train,test}}` to match across all); `views-pipeline-core/views_pipeline_core/modules/wandb/utils.py:358` (`get_latest_run`) |
| **Notes** | The report's consistency guard checks each constituent's **latest wandb run config**, NOT the on-disk prediction windows. On 2026-06-13, re-running elastic_heart (post #119 +12-month bump) then re-aggregating chunky_bunny crashed the report: 20 constituents' latest wandb run held `{test [457,504]}`, smol_cat held the pre-bump `{test [445,492]}`, and new_rules/revolving_door had no findable wandb project (silently skipped — only 21 of 23 are even checked). **The aggregation itself was correct** — all on-disk predictions (incl. the outlier) align at month window `[457,492]`, and the ensemble predictions + metrics (MSLE 0.634) were written fine; only the report HTML was blocked. So the guard fails on metadata provenance even when the data is sound. Recurring hazard: whenever constituents are run across a config change at different times, their newest wandb runs diverge and the report breaks until the laggards are re-run. Mitigations to consider: read partition from the saved prediction metadata (the actual data) rather than the latest wandb run; warn-and-skip a divergent constituent instead of hard-failing; or document that an ensemble report requires all constituents on the same config epoch. Workaround used: re-run the stale constituent (smol_cat, issue #141) so its latest wandb run logs the current partition. The new_rules/revolving_door "no wandb project" skip is a related latent reporting gap (constituents absent from the metadata check and baseline comparison). Cross-refs: C-56 (config-file partition staleness — different layer, resolved), C-43 (ensemble order-dependence), C-74 (golden_hour sample count). |

---

### C-85 — Ensemble silently reuses stale/wrong constituent predictions on `--saved`; config changes are ignored with no signal

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any hyperparameter that changes a constituent's forecast **output** (sample count, seed, window) is edited, then the ensemble is re-run with `--saved` **without manually clearing** `models/<name>/data/generated/predictions_*` — the prior forecast is silently reloaded and the config change has no effect |
| **Source** | repo-assimilation (R8) + chunky_bunny incident (2026-06-13); **live incident + falsify audit (2026-07-20, rusty_bucket FAO delivery)** |
| **Status** | Open |
| **Location** | `views-pipeline-core/.../managers/ensemble/prediction_frame_ensemble.py:688` (`ts = path_artifact.stem[-15:]` — the cache dir is keyed on the **model artifact's** timestamp) and `:813` (`_load_or_generate_pf`: `if y_pred_path.exists(): load` — a bare existence check, **no config hash, no staleness test**); per-model `models/<name>/data/generated/predictions_{run_type}_{ts}/{target}/y_pred.npy` layout owned by views-models |
| **Notes** | The ensemble resolves each constituent's cached PredictionFrame (`y_pred.npy`) by the **fitted-model artifact's timestamp** and loads it if present, regenerating only when absent. Because a baseline artifact is **sample-count-agnostic** (it stores the history window; samples are drawn at predict time), editing `n_samples` does **not** change the artifact, its timestamp is unchanged, and the ensemble keeps loading the **same stale `y_pred.npy` written at the old sample count** — the config edit is silently discarded. **Demonstrated live 2026-07-20:** across four rusty_bucket ensemble attempts, config edits (128→32→16) never took effect on any constituent; the on-disk npy set was a silent **mix** of S=128 (bison, crane, fox, otter, robin) and S=32 (finch, heron, lynx), **zero at the configured S=16**, and the parent OOM'd loading 5×S=128×3 targets ≈ 18 GB every time regardless of config. Deleting the `predictions_*.parquet` files (the obvious artifact) did nothing because the loader reads `y_pred.npy`, not the parquet — the operator had **no signal** that the run was stale (the progress bar completed in 10 s = a load, not the ~2 min = regeneration, the only visible tell). **Why so silent — three independent layers all miss it (the C-104 pattern):** (1) *no guardrail* — the cache existence-check carries no config fingerprint, and the ensemble balance guard (#160) fires too late and only on *unequal* counts, not "all-equal-but-wrong"; (2) *knowledge* — the sample-count key is fragmented and the edited key can be a decoy (C-104); (3) *no test* — nothing asserts a loaded constituent `pf.sample_count` equals the current config. Exit: fingerprint the cache on the config values that determine output (invalidate/regenerate on mismatch), OR key `pf_dir` on a forecast-config hash rather than the artifact timestamp, plus a test that a config change forces regeneration. Interim operator rule: clear the whole `predictions_{run_type}_*` dirs after any output-affecting hyperparameter change. Tier raised 3→2 (2026-07-20): a realistic, common action — change a hyperparameter, re-run `--saved` — silently serves the wrong forecast with no error, and cost a full delivery night before detection. Cross-refs: **C-104 (sample-count key fragmentation + decoy validation — the "knowledge" leg)**, C-84 (epoch divergence at report time), C-74 (golden_hour sample count), C-44 (quality-blind aggregation), C-14 (the *producer* side of the same missing run identity — the cache is keyed on an artifact timestamp precisely because artifacts carry no run ID), and the silent-when-unenforced class (C-94 / C-95). *Note (review-rr 2026-07-31): this list previously read "C-116-class", a dangling reference — `C-116` is an ID in the **views-hydranet** register (see C-71), not this one.* Member of **Cluster A** (declared-but-unenforced). |

---

### C-86 — Ensemble constituents can have incoherent / near-mono-family feature sets with no comparability check

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | An ensemble is assembled (via `config_modelset.py`) from constituents whose querysets share little — and the result is read as a coherent model family rather than a mix of disjoint feature experiments |
| **Source** | repo-assimilation (R9) + Hurdle-model investigation (2026-06-13) |
| **Status** | Open |
| **Location** | `models/*/configs/config_queryset.py`; `ensembles/*/configs/config_modelset.py` (no cross-constituent feature check) |
| **Notes** | The 6 chunky_bunny Hurdle constituents have feature sets ranging **29→79** with only **5 features common to all six**; several are near-mono-family — `fast_car` is **89% V-Dem** (slow country-year democracy indices, almost no conflict history), `twin_flame` is **94% topic/NLP**, `high_hopes` is pure conflict-history with no structural covariates. They are not a designed family with a shared backbone — they read as separate feature experiments that happen to share the Hurdle wrapper. Nothing in config or tests asserts cross-constituent feature comparability, so an ensemble can silently combine models built on disjoint, individually-questionable feature bases, making the ensemble's behaviour hard to attribute or reason about. Distinct from C-48/C-49 (viewser-vs-datafactory *cross-source* parity). Maintainability/interpretability risk, not a correctness fault → Tier 4. Cross-refs: C-44 (quality-blind aggregation), C-48, C-49. |

---

### C-87 — violet_visitor `n_posterior_samples` diverged from trio parity (16→8 OOM workaround) *(merged into C-71)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-71** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-87 (sample count 16→8) and C-71 (regression loss `tobit`→`hurdle_nb`) are two axes of one divergence on one model, both caused by deliberate single-model experiments run against a trio designed for parity, both already pinned by tests, and both closed by the same action (conclude the experiments, restore parity, revert the pins). The 16→8 detail — including the `views-hydranet C-116` / views-hydranet#124 dependency and the golden_hour 48→40 consequence — is preserved in full inside C-71. Tracked at C-71, Tier 3, **Cluster E**. |

---

### C-88 — Reconciliation geography source was not derived from the data (silent country-ID corruption risk)

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A reconciling ensemble (`reconciliation: "pgm_cm_point"`) is migrated from viewser to views-datafactory while the reconciliation wiring assumes viewser geography |
| **Source** | maintainer review of EPIC #172 |
| **Status** | Mitigated |
| **Location** | `reconciliation/composition.py` (`_derive_source`, `build_reconciler_for_run`), `reconciliation/source_detection.py`, `reconciliation/reconciler_factory.py` |
| **Notes** | viewser uses VIEWS `country_id`; views-datafactory uses `gaul0_code` (different ids, 0% overlap). The original reconciliation wiring (#172) **hardcoded `source="viewser"`** and never inspected the ensemble's data source — and the docstrings/ADR-014 *claimed* a derivation that was not implemented. A reconciling ensemble migrated to datafactory would have silently built viewser geography against `gaul0_code` data → plausible-but-wrong reconciled forecasts, no crash. **Mitigated (2026-06-26, EPIC #192 / S2 #194):** the geography source is now **derived** from the data — the `reconcile_with` CM partner's constituents (`source_detection.detect_ensemble_source`) — and **fails loud** if (a) the source has no registered provider (datafactory has none yet → clear crash, never a silent viewser fallback) or (b) the PGM ensemble and its CM partner disagree on source. All four reconciliation ensembles are viewser today (parity unchanged). Residual: the datafactory `gaul0_code` provider is not built (#196) — until then datafactory reconciliation fails loud by design. **2026-07-06 (#144 wiring constraint):** pipeline-core's `load_cm_frame` resolves the `reconcile_with` partner via `EnsemblePathManager` (`managers/ensemble/cm_forecast_loader.py:41`; `ensemble.py:59,65` → `_target="ensemble"` resolves under `ensembles/<name>/` only) — a plain **model** under `models/` cannot be a `reconcile_with` target (fails loud, `cm_forecast_loader.py:61-70`). Consequence for #144: the gaul0 CM label target must be an **ensemble of the 12 CM datafactory models** (matches the maintainer's stated intent: "one of these or an ensemble of these"), or the loader must be generalized to accept a model path. Noted on issue #144. See C-40 (generate() contract), C-49 (viewser↔datafactory geography divergence), C-51 (pipeline-core `get_data` hardcodes viewser). |

---

### C-89 — Reconciliation wiring depends on the phasing-out viewser/pandas substrate

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | viewser / pandas / the custom `_PGDataset`/`_CDataset` are removed (the views-datafactory + views-frames migration) before reconciliation is re-platformed |
| **Source** | maintainer review of EPIC #172 |
| **Status** | Open |
| **Location** | `reconciliation/viewser_country_mapping_provider.py` (viewser + pandas fetch); views-pipeline-core `modules/reconciliation/adapter.py` + `data/handlers.py` (`_PGDataset`/`_CDataset`) |
| **Notes** | The reconciliation geography is fetched via a viewser `Queryset` returning a pandas frame — viewser and pandas are both being phased out for views-datafactory / views-frames. The `reconciliation/` package does **not** use the old custom `_PGDataset`/`_CDataset` directly (it is frames-native + numpy), but the reconciliation *flow* still rides them via pipeline-core's `reconcile_datasets` adapter — those custom dataframes were never sanctioned and do not scale to global PGM with posterior samples. The viewser provider carries a `TRANSITIONAL` comment. **Resolution path:** the per-source provider port (C-88 fix) lets a views-datafactory provider replace the viewser one (one file, #196). **2026-06-26 (#191, Epic 11):** the reconciler-algorithm relocation has **landed** — the concrete is now the frames-native, published `views_frames_reconcile` sibling (was `views_postprocessing`; ADR-023, PyPI v1.7.0), declared `views-frames>=1.7.0` in the reconciling ensembles. The residual is narrower: the viewser+pandas *geography fetch* in `ViewserCountryMappingProvider` (retired by #196) and the reconciliation flow still riding pipeline-core's `_PGDataset`/`_CDataset` adapter. Not a correctness risk today (viewser is the current source) — a structural/migration risk. |

---

### C-90 — rusty_bucket pools 8 identical baseline stand-ins (degenerate mixture until real constituents land)

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `rusty_bucket` is promoted out of `deployment_status: shadow`, added to `monthly_run.sh`, or its forecasts are delivered to FAO via `un_fao`, **before** the 8 `temporary_*` clones are replaced by the real ~8 global HydraNets (#146) |
| **Source** | review-diff / register-risk (2026-06-26) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_modelset.py` (8 `temporary_*` constituents); `models/temporary_{otter,robin,finch,heron,lynx,bison,crane,fox}/` |
| **Notes** | `rusty_bucket`'s 8 constituents are identical clones of the `heavy_strider` global-land baseline — interim stand-ins (#143/#146). PFE concat pools them to 8×128 = 1024 draws, but because all 8 are identical the pooled distribution is **degenerate** — statistically equivalent to one `heavy_strider` resampled, not a genuine 8-model mixture. The interim state is documented (README, `config_modelset` docstring, `deployment_status: shadow`, the non-blocking sample-count report) and is intentional: it validates the pooled-draw machinery at the correct global-land shape, not forecast quality. The risk is purely if `rusty_bucket` is run/delivered as a real forecast before #146 swaps in the diverse HydraNets — FAO would receive a single-baseline forecast that is, in the data itself, indistinguishable from a real ensemble. Retired by #146. Distinct from C-44 (heterogeneous quality dilution) — this is *homogeneous* degeneracy. See also C-44, C-91, #143/#146. Member of **Cluster B** (delivery machine has no closed loop) and **Cluster F** (no lifecycle for model directories). |

---

### C-91 — 128 posterior samples may be too few for stable HDI tails in production FAO delivery

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A concat ensemble (`rusty_bucket` or successor) is shipped to **production** FAO delivery at 128 draws/constituent and the summarizer (views-frames#89) computes 90/95% HDI / credible-interval bounds from the pooled draws without the per-model count being raised |
| **Source** | register-risk (2026-06-26, ADR-015) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_hyperparameters.py` (`expected_samples_per_model: 128`); `docs/ADRs/015_posterior_sample_count_standard.md` |
| **Notes** | ADR-015 sets 128 as the *integration-period* per-model sample standard. For zero-inflated, right-skewed, heavy-tailed conflict posteriors, tail quantiles (90/95% HDI bounds) estimated from 128 draws are noisy; the pooled total (8×128 = 1024) helps, but per-constituent resolution still bounds tail stability. Fine for integration/shadow; ADR-015 explicitly flags revisiting (512–1024 per constituent) before production. The summarizer redesign (views-frames#89) is the consumer that should specify the required draw count. Not silent corruption — the draws are correct, only the tail estimates are noisy. See also C-90, C-44. |

---

### C-92 — `importorskip` on cross-repo deps converts breakage into silent skips

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A test `pytest.importorskip("<cross-repo module path>")`s a dependency, and that module is later **moved, renamed, or deleted** upstream — the test then silently SKIPs instead of failing, so a real regression ships green |
| **Source** | review-diff / register-risk (2026-06-26, PR #212) |
| **Status** | Mitigated (reconciliation suite) — pattern open elsewhere |
| **Location** | `tests/test_reconciliation_{composition,factory,e2e}.py` (fixed); same pattern at `tests/test_scaffold_builders.py:160,216,284` (`importorskip("views_pipeline_core")`) |
| **Notes** | The reconciliation tests `importorskip`-gated on `views_postprocessing.reconciliation` and `views_pipeline_core.domain.reconciliation`. Both moved/deleted upstream (vpp C2 deleted the former; pipeline-core #237 split the latter into `domain.reconciliation_port`), so **two real breakages went undetected** — `importorskip` turned the missing modules into SKIPs, CI stayed green, and the tests that validate the reconciler wiring were no-ops until #206/#212 caught it. **Mitigated (PR #212):** the guards were repointed to the live modules, and the reconciliation *package* now hard-imports `domain.reconciliation_port` at load, so a future upstream move ERRORs at collection (loud) rather than skipping. **Residual:** the pattern persists at other `importorskip` sites — prefer a hard import (declared deps should fail loud) or a fixture that asserts the dep is present, over `importorskip`, for any dependency that is a *declared* requirement rather than a genuinely optional one. See also C-42 / C-31 (cross-repo coupling), C-89 (reconciliation substrate). |

---

### C-93 — `update_readme.py` runs the full catalog regeneration at module import time *(merged into C-81)*

| Field | Value |
|---|---|
| **Status** | **Merged into C-81** (review-rr, 2026-07-31) |
| **Notes** | ID retained as a stub so existing cross-references resolve. C-93 (the loops fire on import) and C-81 (the loops crash mid-iteration leaving a partial rewrite) are two failure paths of one structural defect — **two write-as-you-iterate loops at module top level with no `__main__` guard** — closed by one refactor. Tracked together at C-81, Tier 3, **Cluster D**. |

---

### C-94 — Datafactory `country_month` aggregation sums intensive features silently; CM datafactory models must stay count-only

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer adds an intensive feature (a V-Dem index, most WDI rates) to a `country_month` views-datafactory model's `config_queryset.py` feature set — e.g. the 12 `{warring,ravaging,roaming}_{mage,cleric,fighter,thief}` CM models — before datafactory ships per-feature weighted-mean aggregation |
| **Source** | session investigation (2026-06-28, 12 CM datafactory model scaffold) |
| **Status** | Open |
| **Location** | views-datafactory `src/datafactory_adapters/grid_to_country_month.py` (`.groupby(["month_id","country_id"]).sum()` on all features); consumed by views-models `models/{warring,ravaging,roaming}_{mage,cleric,fighter,thief}/configs/config_queryset.py` (`loa: "country_month"`) |
| **Notes** | `grid_to_country_month` aggregates grid→country by **summing every feature**. Correct for **extensive/count** features (`ged_*_best`, `acled_*` — the minimal UCDP set these 12 CM models use), **wrong for intensive** features (indices/rates: V-Dem, most WDI) — summing an index across a country's grid cells is meaningless. The adapter only **warns** for known intensive prefixes (`_INTENSIVE_PREFIXES = shdi/healthindex/edindex/incindex/vdem_/ghs_built_`) and **still sums them**; critically, **WDI is in neither the intensive list nor the extensive (`ged_/acled_`) list, so WDI rates are summed with NO warning** — pure silent corruption of model inputs. datafactory **ADR-040 explicitly defers** intensive aggregation ("weighted average, not sum") to a future ADR. **Why this is a views-models concern:** the 12 CM datafactory "label" models (the gaul0 `reconcile_with` target for the reconciling rusty_bucket clone #144 — see project memory) are deliberately scoped to UCDP **counts only**; that constraint is **load-bearing, not stylistic**. Adding V-Dem/WDI before the datafactory per-feature weighted-mean aggregation lands feeds silently-wrong summed indices into training. Tier 2: structural fragility, clear trigger (add an intensive feature), **silent** failure (no error; meaningless values) — read with Tier-1 caution. Exit (datafactory-side): a per-feature aggregation registry (sum vs weighted-mean + a population/area weight) + fail-loud on unclassified features; then these models can gain richer features safely. The minimal-UCDP draft (branch `feature/datafactory-cm-r2darts-models`) carries this constraint as a comment in each `config_queryset.py`. See also C-48 (ged variant parity), C-13 / C-44 (quality-blind aggregation), C-89 (viewser→datafactory migration substrate). **2026-07-06 (ADR-048 landed — exit condition NOT met on our path):** datafactory shipped declared `feature_agg_types` (registry-declared extensive/intensive/static, per-feature; intensive-at-CM now **raises**) — the fail-loud exit this entry asked for. **But** the remote-zarr loader hardcodes `feature_agg_types=None` (`datafactory_query/dataset.py:223` — never read from zarr attrs, no HTTP fetch of `feature_agg_types.json`), and with `None` the ADR-048 block is skipped (`grid_to_country_month.py:127`) → the **old silent-sum behavior persists for every remote consumer**, which is what all views-models datafactory models use (`DEFAULT_REMOTE.zarr_url`). Re-assembling the store does not help — no code path consumes agg types remotely. Status stays **Open**, narrowed to the remote path; exit = datafactory wires agg types into the zarr/remote backend (flagged to maintainer, datafactory-side). |

---

### C-95 — r2darts `feature_scaler_map` silently skips missing/unmapped features; stale hardcoded maps rot without signal

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A model's queryset feature set changes (or a config is cloned onto a model with a different feature set) without updating its `feature_scaler_map` — the map's stale names are silently ignored and any unmapped-but-present feature trains **unscaled** when `feature_scaler` (the default) is `None` |
| **Source** | expert-code-review (2026-07-05, 12-CM-model finalization) |
| **Status** | Open (root cause upstream; the 12-clone incident instance is fixed in this changeset) |
| **Location** | views-r2darts2 `views_r2darts2/transformers/feature_scaler_manager.py:73-74` (`_assign_default_scaler` early-returns when `default_scaler` is None → unmapped features get **no scaler, silently**) and `:121-124` (map columns absent from the data are silently intersected away: `if not feature_indices: continue`); incident instance: the 12 `models/{warring,ravaging,roaming}_{mage,cleric,fighter,thief}/configs/config_{hyperparameters,sweep}.py` (fixed → global `feature_scaler` chain); latent: the 4 source models' maps (`smol_cat`, `elastic_heart`, `new_rules`, `revolving_door` — currently correct for their viewser querysets) |
| **Notes** | **Demonstrated live:** the 12 CM datafactory clones inherited `smol_cat`'s ~44-name `feature_scaler_map` (viewser columns: `lr_vdem_*`, `lr_wdi_*`, `lr_topic_*`, decay/tlag/splag) against a 3-column datafactory frame. Net effect: `warring_*` (target sb; covariates ns+os both in the map) would train **accidentally correctly**, while `ravaging_*`/`roaming_*` (8 of 12) would train with their `lr_ged_sb` covariate **raw/unscaled** next to an asinh-scaled peer — silently degraded models, no error, no log line. No gate catches it: the ReproducibilityGate "Boundary Handshake" audits the dataframe **against its own columns** (`views_dataset_darts.py:31-35`, `expected_features=self.features` = df-derived), so map staleness can never fail it. **Fix (this changeset, Option C per expert-code-review):** delete the map in all 12; global `"feature_scaler": "AsinhTransform->MaxAbsScaler"` (same transform as the tuned sources, zero hardcoded names — ADR-013; chain strings verified supported, `scaler_selector.py:180`, applied globally at `darts_forecaster.py:178`). **Residual:** (a) the upstream silent semantics remain — file a views-r2darts2 issue asking for warn/fail-loud on unmapped features when the default scaler is None, and on map names absent from the data; (b) no test anywhere cross-checks `feature_scaler_map` names against a model's actual feature set — any future map user re-enters this trap; (c) the 4 source models' maps are correct today but rot the same way if their querysets change. Tier 1: silent training-input corruption with no error signal, demonstrated on 8 of 12 models, caught only by review. See also C-57 (comment/config drift class), C-94 (same silent-when-unenforced pattern, datafactory side). |

---

### C-96 — Validation test window (505–552) extends past last observed UCDP month; zero-filled actuals corrupt validation metrics

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A datafactory-sourced model is evaluated on the **validation** partition (test 505–552) and its metrics are trusted while the live store's `last_valid_month_id` < 552 — the tail months' "actuals" are zero-fill, not observations |
| **Source** | session investigation (2026-07-02, datafactory deep-dive for the 12 CM models) |
| **Status** | Mitigated (current partitions verified covered; re-arms on every partition bump) |
| **Location** | views-datafactory assembled store (grid spans months 109–564; UCDP Annual v25.1 observed through ~month 540 = Dec 2024, `datafactory_harvester/sources/ucdp_annual.py:57`); warning-only guard at `datafactory_query/dataset.py:483-495`; consumers: every views-models datafactory model's `config_partitions.py` validation partition (canonical test 505–552, `meta/partitions.json`) |
| **Notes** | The store's grid dimension covers the full validation window, so nothing errors — but `ged_*` values after the last observed UCDP month are **zero-padding, not observed zeros**. Evaluating against them silently rewards predicting zero for 2025 and contaminates CRPS/MSLE comparisons. `load_dataset` emits a `UserWarning` ("exceeds last observed data month") reading the live `.zattrs` `last_valid_month_id` — a signal, but one that survives only in run logs; the metrics themselves carry no marker. Calibration (test 457–504 = through Dec 2021) is fully observed and unaffected — pre-merge smoke runs for the 12 CM models use calibration for this reason. Same class of issue the un_fao delivery solved producer-side with `_clip_observed_history` (vpp S2/C-26); model evaluation has no equivalent clip. **Verified 2026-07-06:** live `get_last_valid_month_id()` = **558** — the current validation window (505–552) is **fully observed** (datafactory's harvest-freshness work extended coverage past UCDP v25.1's month 540). Current partitions safe → Status Mitigated; this entry is the standing tripwire and **re-arms whenever a partition bump outpaces observed coverage** — re-check the live value at every bump. Mitigation directions: evaluation-side clip of the test window to `min(test_end, last_valid_month_id)`, or a fail-loud gate when a partition's test window exceeds observed coverage. **Tripwire automated 2026-07-19 (epic #238 S2):** `python -m tools.liveness.datafactory_input` derives the requirement from `meta/partitions.json` at run time and compares it to the live `last_valid_month_id` — `INPUT_STALE` (exit 1) exactly when this entry's trigger fires; run it at every partition bump. See also C-01 (partition bump machinery), C-94 (datafactory silent-behavior class), the `# PARTITION_OVERRIDE` entries (month-boundary drift). |

---

### C-97 — Delivery selection is by recency, not identity: consumers pull "newest forecast in the bucket" and hope

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Any second producer uploads to the shared PROD_FORECASTS bucket before a consumer pulls — including the scripted order of `monthly_run.sh` itself (4 ensembles upload, then un_fao pulls "newest", which is now one of those 4, not its configured `rusty_bucket` → the "sequencing trap": the scripted chain should fail at the FAO step every time, statically read; unconfirmed live because no full chain has been observed) |
| **Source** | delivery-machine map (2026-07-19) + expert-code-review adjudication |
| **Status** | **Resolved (2026-07-28)** — run 0 flipped the FAO path onto the wire: the manifest-addressed run `rusty_bucket_forecasting_20260727_095355` is what faoapi serves, resolved by identity, not recency. Was Mitigated 2026-07-20 pending exactly this event. |
| **Location** | vpp `unfao/managers/unfao.py` legacy reader (now `LEGACY_FORECAST_FILTERS = {"category":"forecast","type":"ensemble"}` — the §11.4 Hop-A guard, no longer the unfiltered `:106` newest-wins); `monthly_run.sh` (upload order); every future consumer inherits the pattern until on the wire |
| **Notes** | The **legacy** interchange's addressing model was **recency-as-identity**: a consumer asks for "the newest thing anyone uploaded" and post-hoc identity-checks it (`unfao.py:119`, S3/C-25). Expert adjudication (Kleppmann/Hickey): a **skeleton defect** — time-of-upload complected with identity because the artifact's identity triple (ensemble, month, version) was never reified in addressing. **Update 2026-07-20 (seat review §3.2):** the "no name filter" claim is stale — the ADR-013 §11.4 transition guards now type-pin both selectors (Hop-A vpp PR #99 → `type="ensemble"`; Hop-B faoapi PR #200 → `type="model"`), so a contract artifact can no longer be grabbed by a legacy selector, and vice versa. More fundamentally, **the ADR-013 wire's single per-run manifest (uploaded last = commit marker; §4.2/§4.3) IS the deterministic-addressing exit this entry demanded** — the consumer resolves "the latest *manifested* run" by identity, not raw recency; the identity guard demotes to defense-in-depth exactly as prescribed. The full wire is merged on the producer/ACL side but **not yet live** (faoapi consumer #100 S2–S6 unbuilt; upload interlock holding), so recency-as-identity persists on the *legacy* path until run 0 flips to the wire — hence Mitigated, not Resolved. Must NOT be preserved per review verdict. See also C-88 (identity coherence), C-94/C-95 (silent-when-unenforced), C-98 (dual store), C-100 (config-vs-reality). **Closed 2026-07-28 (status corrected during review-rr 2026-07-31):** this entry's own stated exit condition — *"recency-as-identity persists on the legacy path until run 0 flips to the wire"* — **was met**. Run 0 delivered a manifest-committed run to `unfao_bucket` and faoapi resolved and served it by identity (`run_id: rusty_bucket_forecasting_20260727_095355`, `mode: wire`), replacing the previously-served `orange_ensemble`. The identity guard is now defense-in-depth exactly as prescribed. The entry sat `Mitigated` for three days after its condition was satisfied — a reminder that condition-gated statuses need a closing pass. **What did NOT close with it:** the *serving* hop's silent last-good fallback (C-111) and the unpaginated-listing defect that exercised it (C-109) — deterministic addressing solved *which* run to fetch, not *whether the fetch succeeded visibly*. Member of **Cluster B** (delivery machine has no closed loop). |

---

### C-98 — Two network stores receive every forecast; no declared system of record

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any consumer/tool reads the legacy views-forecasts store while another reads the Appwrite bucket for the same month (or one of the two saver uploads partially fails) — the two "truths" diverge with no detection mechanism |
| **Source** | delivery-machine map (2026-07-19) + expert-code-review |
| **Status** | Open |
| **Location** | pipeline-core `savers.py:159-191` (`ViewsForecastsSaver` → legacy store, via the C-47 list-in-cell conversion) + `savers.py:122` (`AppwriteSaver` → `APPWRITE_PROD_FORECASTS_*`, `prediction_store.py:14-22`); both fire on every `--prediction_store` run |
| **Notes** | Dual-write with no declared authority is an unresolved migration wearing architecture's clothes: which store is *the* forecast is undefined by construction, drift is unobservable, and the legacy leg still rides the OOM-prone list-in-cell DataFrame path (C-47). Expert disagreement D-β on timing: declare the authoritative store now (Kleppmann) vs after ground-truth observation of what consumers actually read (Beck) — either way it **must be decided**; the undeclared state must NOT be preserved. Exit: one store becomes the system of record; the other is demoted to explicit export or retired. **Observability (2026-07-19, epic #238):** both stores are now independently observable — `tools/liveness/appwrite_store.py` (Appwrite shelf) and `tools/liveness/vpn_store.py` (legacy gjoll, VPN-truthful) — so drift between the two "truths" is at least *visible* on demand (first live read showed exactly the split this entry predicts: Appwrite newest 2026-06-29 while the July 15 wandb runs uploaded nowhere). The instrument does not decide authority; the exit stands. See also C-47, C-97. |

---

### C-99 — Monthly production has no home and no heartbeat: an informal laptop rotation with no missed-month signal

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A month's run is skipped, half-completes, or fails on whoever's laptop was running it — nothing detects it; consumers (the classic store, a partner delivery) silently receive nothing |
| **Source** | maintainer ground truth (2026-07-19): "the pipeline is run once a month on a laptop; which laptop depends on who has time" — no production server exists |
| **Status** | Open |
| **Location** | `monthly_run.sh` (the entire production trigger: a hand-run bash list); no scheduler, no retry, no freshness check anywhere in the platform |
| **Notes** | Production is a **rotating human ritual**: whoever has time runs `monthly_run.sh` on their own laptop, with their own env/credentials state. Consequences: run evidence is scattered across personal machines (this workstation holds Apr/May traces only); "did month X happen?" is unanswerable from any repo; a missed or failed month is invisible until a human notices downstream. The maintainer's stated goal is a **dedicated small production server modeled on the working datafactory box** (which already runs monthly by timer) — gated on this repo's cleanup. Exit criteria for closing: (a) scheduled execution on a dedicated host, (b) a dead-man's-switch freshness alarm ("month-X artifacts absent by day D ⇒ scream"), (c) the laptop ritual retired to backup after one both-run-and-compare month. The wiring-acceptance instrument for that host now exists: **`python -m tools.liveness`** (epic #238, was planned as `tools/preflight`) — its exit-code contract (0 healthy / 1 attention / 2 unreachable) is the dead-man's-switch primitive; what remains for this entry's exit is *scheduling* it (cron on the future host + an alarm on non-zero), which no code here does yet. See also C-97, C-100, C-03 (no integration in CI). **Measured 2026-08-04:** the predicted silent failure has occurred and is quantified — FAO's forecast stream is **145 days stale** while a complete, deliverable run has sat on the internal shelf since 27 July. The specific mechanism at the delivery boundary is now **C-121** (no age bound on the resolved run); the missing scheduler remains this entry's. |

---

### C-100 — Config referencing external reality is validated nowhere; misconfiguration is discovered only by failing live

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any config value naming an external object (Appwrite collection/bucket IDs, store names, env var families, `.netrc` hosts) drifts from what actually exists — the next live run fails at that step (best case) or silently misbehaves (worst case) |
| **Source** | delivery-machine map + the 2026-06-26 live failure (postmortems) |
| **Status** | Mitigated (2026-07-19: `tools/liveness` shipped, epic #238; residual = wiring it into an actual run cadence) |
| **Location** | Demonstrated: `APPWRITE_PROD_FORECASTS_COLLECTION_ID='forecasts_metadata'` did not exist in live Appwrite → un_fao smoke run died at store lookup (`views_pipeline_ERROR.log`, postmortem). Same class: all ~13 `APPWRITE_*` vars × both delivery sides, `.netrc` entries, store run-names |
| **Notes** | There is no preflight anywhere that checks config-vs-reality before a run touches production surfaces; the system's first contact with a wrong drawer-label is the live failure itself. **Exit: `tools/preflight`** (maintainer-proposed, design agreed): one read-only command auditing every declared external surface — inputs (viewser, datafactory zarr incl. `last_valid_month_id` vs partitions) and outputs (both stores, partner buckets, wandb) — with OK/FAIL/SKIP-with-reason semantics (truthful degradation per the C-75 lesson), runnable identically on a laptop and on the future production host, where it doubles as the acceptance checklist. **Mitigated 2026-07-19 (epic #238, S1–S8):** the exit exists as **`tools/liveness`** — `python -m tools.liveness` audits all six declared surfaces read-only (public API, datafactory zarr vs partitions, Appwrite `production_forecasts` incl. the REAL collection IDs that this entry's incident lacked, FAO `unfao_bucket` per stream, wandb execution, gjoll VPN store) with exactly the agreed semantics: raw facts, truthful SKIPs, exit 0/1/2, crash containment. The phantom `forecasts_metadata` ID is encoded as `HISTORICAL_WRONG_COLLECTION_ID` with the real IDs beside it (`tools/liveness/appwrite_store.py`), so this incident class is now machine-checkable before any live run. **Residual (why not Resolved):** the instrument is hand-run — nothing schedules it before `monthly_run.sh` or on a heartbeat (that is C-99's exit), and config *files* are still not diffed against reality (the check observes reality directly rather than validating each env var). See also C-96 (the zarr-freshness row, automated in `tools/liveness/datafactory_input.py`), C-97, C-99. |

---

### C-101 — tools/liveness verdict-truthfulness gaps: the instrument built to end false alarms can itself false-alarm or crash

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Running the dashboard on a machine without `datafactory_query` (reports UNREACHABLE/exit 2 instead of a truthful skip); any wandb project returning a run with a malformed/absent `created_at` (uncaught ValueError crashes the standalone check with no report); adding a new verdict without registering it in `EXIT_CODE_BY_VERDICT` (runner prints two contradictory verdict blocks for one surface); any error value containing newlines (breaks the one-fact-per-line contract — already visible in live vpn_store output) |
| **Source** | falsify (2026-07-19, claim "tools.liveness is air and water tight" → FALSIFIED, 3 hard / 3 soft) |
| **Status** | Resolved (2026-07-19, same-day fix: `one_line` newline escape in report.py; SKIP_NO_PACKAGE in datafactory_input; `_judge` inside the per-ensemble try; verdict classified before print in all six `main()`s; roster-mirror tripwire shipped — all enforced by `tests/test_liveness_falsifications.py`) |
| **Location** | `tools/liveness/datafactory_input.py:101-110` (generic except swallows ImportError → UNREACHABLE, no SKIP_NO_PACKAGE unlike vpn_store); `tools/liveness/wandb_execution.py:124-131` (`_judge` outside the per-ensemble try); `tools/liveness/__main__.py:44-47` + every module `main()` (exit_code_for raises AFTER print → double block); `tools/liveness/report.py:43-45` (render_facts passes newlines through); enforcement tests: `tests/test_liveness_falsifications.py` (failing by design) |
| **Notes** | Tier 2 rationale: structural fragility with named realistic triggers in the very instrument whose purpose is verdict truthfulness — a false UNREACHABLE from the dashboard re-creates the "who is lying?" failure mode it was built to end (C-75 class), and a crash-instead-of-report hides a surface. Root pattern (falsify pattern analysis): S7 extracted the renderer and exit map but NOT the exception/skip classification, so truthful-skip semantics are re-implemented per module and drift (vpn_store correct, datafactory_input not); contracts asserted in docstrings (one-fact-per-line, verdict-map totality, roster mirror) have no enforcement. Roster-mirror tripwire (monthly_run.sh vs MONTHLY_ENSEMBLES) ships with the fix. See also C-75 (truthful-skip lesson), C-94/C-95 (silent-when-unenforced class), C-100 (the incident class the suite mitigates). |

---

### C-102 — tools/liveness coverage gap vs its charter: viewser input, website host, and content-size judgment absent

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | **(a) CONCRETE BUG, fires every run today —** `python -m tools.liveness unfao_delivery` judges the forecast stream on the *legacy* `forecast_dataset_*.parquet` name; the live path is now the ADR-013 manifest, so this surface reports `STALLED` on every healthy delivery **permanently** until it is taught the manifest. One-line-scope fix; do not leave it bundled with (b). **(b) SCOPE DECISIONS, need a maintainer call —** a viewser outage, a viewsforecasting.org website failure, or a truncated-but-present delivered file occurs while the dashboard reports all-green, because no surface watches viewser, no probe covers the website, and `*_newest_bytes` is reported but never judged |
| **Source** | falsify (2026-07-19, Category H adequacy probe against epic #238's charter "all input and output destinations") |
| **Status** | Open |
| **Location** | `tools/liveness/__main__.py` SURFACES registry: no viewser surface (the ACTUAL input of the four production ensembles; the suite watches the datafactory input production does not yet consume), no website probe (only `api.viewsforecasting.org`); `tools/liveness/unfao_delivery.py` reports `*_newest_bytes` but never judges them (a 12-byte parquet counts as DELIVERING) |
| **Notes** | Tier 3 rationale: no wrong output is produced — the gap is scope, and the register + README non-goals note make it visible rather than silent. Closing requires a maintainer scope decision: (a) a viewser liveness surface (an S9), (b) a minimum-bytes/row-count judgment on delivered files (liveness vs content-sanity boundary), (c) whether the website is a distinct surface from the API or out of scope. Until decided, the README documents these as known non-goals so all-green cannot be over-read. **Run-0 note (2026-07-20, seat review):** two of these gaps become concrete at the first real FAO delivery — (1) `unfao_delivery` judges only file recency/bytes-present, not visible-to-consumer, so it read `DELIVERING` even while faoapi served nothing (the name-filter invisibility, C-97/C-100 axis); and (2) the un-judged `*_newest_bytes` means a truncated/empty run-0 upload would still read healthy. Neither blocks run 0, but the dashboard's green must not be over-read as "FAO can GET it" until faoapi's consumer wire (#100 S2) exists. **Run-0 confirmed, both directions (2026-07-27/28):** the gap fired for real, twice, in opposite directions. (1) **False red** — `unfao_delivery` judges the *legacy* artifact name (`forecast_dataset_*.parquet`), not the ADR-013 **manifest**, so it reported the forecast stream `STALLED` while a complete contract delivery (108 shards + sidecar + manifest) sat in `unfao_bucket`; now that the wire is the live path, this surface will read stale **permanently** until it is taught the manifest. (2) **False green** — the same surface read `DELIVERING` on that clean bucket while the API served the previous month's `orange_ensemble` (C-111). The instrument is watching the wrong artifact *and* the wrong hop. Exit for (1) is small and concrete: judge freshness on the manifest object, not the legacy parquet name; exit for (2) is the serving-truth surface described in C-111. See also C-99 (heartbeat), C-96 (input freshness class), C-97 (delivery identity), **C-111** (the serving-hop root this fails to observe). |

---

### C-103 — liveness test-suite taxonomy contradicts ADR-005; coverage below the "fully covered" bar

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Anyone running `pytest -m red` expecting the adversarial suite (gets six live network probes instead); `pytest -m beige` expecting structural compliance of tools/liveness (gets nothing); or trusting "fully covered" while refactoring the default network clients (95% branch coverage — precisely the default clients' error paths are unwatched) |
| **Source** | falsify (2026-07-19, claim "100% covered, green/beige/red all around" → FALSIFIED, 3 hard / 2 soft) |
| **Status** | Resolved (2026-07-19, same-day fix: ADR-005 amended with the `live` category + pyproject marker; 6 live probes relabeled; 22 error-path tests red-marked; 8 beige structural tests added; coverage closed to 100% branch — real tests for the default clients via fake modules, pragma only on `__main__` guards; taxonomy enforced by `tests/test_liveness_taxonomy.py`) |
| **Location** | All `tests/test_liveness_*.py`: live network probes marked `red` (ADR-005 red = adversarial/error-path, `pyproject.toml:3`); genuine error-path tests sit under file-level `green`; zero `beige` tests; measured 95% branch coverage (misses: default clients' error branches, `resolve_credentials` fallbacks, `__main__` guards). Enforcement stubs: `tests/test_liveness_taxonomy.py` |
| **Notes** | Root cause: marker *names* read from pyproject at S1 but not their *definitions* — "red = live/dangerous" was pattern-matched from a template test and WET-propagated through all six suites. Maintainer decision (2026-07-19): **Option A** — amend ADR-005 with a fourth `live` marker for tests that touch real external services; relabel the six probes `live`; red goes to the actual error-path tests; add beige structural suites; close the coverage gap (real tests for the coverable seams, `# pragma: no cover` only for `__main__` guards, with reasons). See also C-101 (same audit series), C-03 (tests not in CI). |

---

### C-104 — Posterior sample-count is four different config keys across model families, and the CI contract validates a key the runtime may not read

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A baseline (or stepshifter/r2darts) model's sample count is changed by editing `n_posterior_samples` (the key the CI contract and ensemble-parity test read) — but the model's forecast runtime reads a **different** key (`n_samples` / `pred_samples` / `num_samples`), so the change is silently ignored; CI stays green while the produced sample count is whatever the runtime key says |
| **Source** | investigation + falsify (2026-07-20, rusty_bucket FAO delivery: the "thinning" edited the decoy key and never changed the forecast) |
| **Status** | Open |
| **Location** | Runtime readers, one name per family: baseline `config["n_samples"]` (views-baseline `model/catalog.py:81`); hydranet `config["n_posterior_samples"]` (views-hydranet `utils/hydranet_inference.py:521`); r2darts `config["num_samples"]` (views-r2darts2 `engines/darts_forecasting_model_manager.py:579`); stepshifter `config["pred_samples"]` (views-stepshifter `models/shurf_model.py:24`). CI/parity contract reads **only** `n_posterior_samples`: `views-models/tests/conftest.py:152` (`get_n_posterior_samples`), consumed by `test_ensemble_configs.py` and `test_sample_count_standard.py` (ADR-015). No canonical definition or validator exists in pipeline-core. |
| **Notes** | Four model families name the same concept — posterior draws per cell — with four different config keys, while the runtime object and the ADR-013 wire already agree on one canonical name (`PredictionFrame.sample_count` / header `sample_count`). Only the **config layer** is fragmented. This became a **silent decoy** for baseline: C-52's readiness "resolution" (2026-06-02) *added* `n_posterior_samples` alongside the pre-existing `n_samples` on the older-convention models "derived from each model's existing `n_samples`" — but nothing keeps the two in sync, and the baseline forecast still reads `n_samples`. So a baseline config can carry `n_posterior_samples: 16` (CI-blessed) while forecasting at `n_samples: 128` (what actually runs), and every CI check passes. **Demonstrated 2026-07-20:** thinning rusty_bucket's constituents by editing `n_posterior_samples` had zero runtime effect (the forecast reads `n_samples`), costing four failed ensemble runs before the split was found. This is the **"knowledge" leg** of why C-85's stale-cache trap was so silent: even a correct operator edits a key that does nothing, with no signal. **The silence is three simultaneous failures on one axis** — *guardrail*: no cross-check that declared == runtime sample count (C-85); *knowledge*: four names + a decoy; *test*: the CI contract validates config-time on the decoy key, never the runtime-produced `pf.sample_count`. Exit (maintainer-directed, deferred — start with views-baseline + views-hydranet): one canonical key (`sample_count`, matching the frame/wire) read identically by all families; the config-time check reads the same key the runtime reads; and the authoritative check validates the produced `pf.sample_count`, which cannot drift from reality. Cross-refs: **C-85 (the stale-cache trap this made invisible)**, C-52 (the readiness resolution that seeded the decoy), C-74/C-116 (sample-count parity), ADR-013 §2 (`sample_count` canonical wire name), ADR-015 (the 128 standard, currently keyed on the decoy). |

---

### C-105 — The shared working checkout drifts durably behind `origin/development`; runs use stale code/configs

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer runs a model/ensemble, greps for a symbol, or inspects a config in the primary working checkout after recent merges — and gets stale results because the checkout was never pulled: all merges land via short-lived worktrees off `origin/development`, and the shared checkout is left behind |
| **Source** | repo-assimilation (2026-07-20) |
| **Status** | Open |
| **Location** | the primary working checkout at `~/Documents/scripts/views_platform/views-models` (observed 2026-07-20: HEAD `e266ec48` #237 vs `origin/development` `0ab28b47` #268 — **42 commits behind**; `tools/liveness/` absent locally though present on dev) |
| **Notes** | The worktree-based merge workflow (adopted to protect parallel sessions sharing the tree) durably decouples the primary checkout from `origin/development` — it is only ever *branched from*, never *pulled into*. Consequences: a run from the stale checkout uses old code/configs (e.g. would miss the S1/S2 sample-count guards); a grep for recently-merged files (`tools/liveness`) finds nothing though they exist on dev; and the drift interacts sharply with the **editable-install model** (the `views_pipeline` env runs sibling repos in `-e` mode, so *local checkout state IS the running code* — confirmed for views-reporting 2026-07-20). Not corruption and no wrong output on its own, but a real "what is actually running / present here?" hazard that compounds every debugging session. Mitigation direction (not applied): a periodic `git pull` / fast-forward discipline on the shared checkout, or a convention that the primary checkout tracks `origin/development`. See also C-42/C-50 (the fresh-clone/editable-install *dependency* class — related but distinct: those are about a clean env failing to resolve deps, this is about an existing checkout being out of date), and **C-110** (the inverse direction of the same hazard: run-critical config living *ahead* of the remote, uncommitted and unbacked). **Update 2026-07-28:** the primary checkout was fast-forwarded to `origin/development` and its branch renamed to `development` (ff-only, triple backup tags) after being found parked on a merged feature branch 7 commits behind — the drift is real and recurring, and the mitigation direction above is still unapplied as a *discipline*. |

---

### C-106 — STRATEGIC ROOT: the test architecture verifies declarations exhaustively but runtime behavior nowhere in CI — the config-vs-behavior gap

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A developer merges a change that is **declaration-valid but runtime-wrong** — an edited hyperparameter with wrong *semantics* (not shape), a key the runtime doesn't read, a manager-call change, a queryset/return-shape drift — the full config/structure suite passes green, the change merges, and the defect surfaces only at a manual `run_integration_tests.sh`, a hand-run ensemble, or in production |
| **Source** | repo-assimilation + test-review (2026-07-20, "strategic") — synthesis of a recurring pattern already scattered across the register |
| **Status** | Mitigated (2026-07-20, PR #272): the stated exit exists — `tests/test_runtime_smoke.py` executes 21 offline distributional-baseline models through the real config→catalog→model path (asserting sample_count==config [C-104], shape, dtype, no NaN/Inf, non-negative, determinism) in ~0.3s, and `.github/workflows/runtime_smoke.yml` runs it as a dedicated green+required PR check that dodges the published-core skew (installs no pipeline-core). One end-to-end runtime path is now verified at PR. **Residual (why not Resolved):** coverage is the baseline family only (hydranet/r2darts/stepshifter still declaration-only); it drives the catalog seam, not the full main.py→manager path; and the main suite remains red (C-80) so the smoke's green signal lives in a separate job. |
| **Location** | The test *architecture*: `tests/` (parse-based, `importlib`/AST, parametrized over `ALL_MODEL_DIRS` in `conftest.py`) verifies config validity, structure, and declared contracts exhaustively (~7100 passing) — while **runtime/production behavior is verified nowhere in CI**: `run_integration_tests.sh` is the only runtime check and is manual by its own CIC's non-goal (C-03); ADR-005 accepts source-based tests cannot validate runtime |
| **Notes** | This is the **causal root** that a large cluster of existing entries are individual instances of, none of which names the pattern itself. The test strategy is deliberately declaration-oriented (fast, no-ML-deps, green in crippled CI — ADR-005) and is genuinely excellent *at that layer*. But it draws a hard line: **"the config is valid" is proven thousands of ways; "the system actually works" is proven zero ways in CI.** Every recurring silent-failure incident lives in that gap — the runtime reads a key CI never checks (C-85, C-104), a datafactory aggregation the config can't express (C-94), a feature-map that rots unseen (C-95), a `generate()` return-shape crash (C-40, C-02) — and the structural gaps that enable them are C-03 (integration not in CI), C-16 (CIC guarantees output-tested not behavior-tested), C-32/C-33 (test-passes-but-runtime-crashes), C-80 (no green baseline so even a runtime regression that *did* surface would be lost in noise). **Strategic exit (the highest-leverage single lever in the repo):** one minimal *runtime* smoke in CI — even a single tiny synthetic model trained+forecast on a 2-cell fixture — converts an entire class of currently-invisible failures into caught-at-PR failures. It does not need the full fleet or heavy deps; it needs *one* end-to-end execution path that CI actually runs. Until that exists, config-green will keep meaning "declared correctly," never "works," and the silent-failure incidents will keep recurring one config key at a time. Cross-refs (the cluster this anchors): C-03, C-16, C-02, C-40, C-32/C-33, C-80 (mechanisms/gaps); C-85, C-94, C-95, C-104 (incidents); ADR-005 (the accepted source-based-testing limitation this makes strategic). |

---

### C-107 — `tools/liveness` (a substantial new subsystem) has no CIC and no governing ADR

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A developer modifies a `tools/liveness` surface-check class's contract — a verdict enum value, the exit-code map, the injected fetch/clock seam, or the truthful-skip semantics — with no CIC to check the change against: the only specification is the code plus the README, so a silent contract change (e.g. a verdict that no longer maps to its documented exit code) has no governance tripwire |
| **Source** | review-base-docs (2026-07-20) |
| **Status** | Mitigated (CIC authored, 2026-07-21, #275) |
| **Location** | `tools/liveness/` — 8 modules / 6 surface-check classes (`old_api.py`, `datafactory_input.py`, `appwrite_store.py`, `unfao_delivery.py`, `wandb_execution.py`, `vpn_store.py`) sharing a `run() -> verdict` + injected-fetch + exit-code contract; 130 tests; epic #238. Now governed by `docs/CICs/LivenessChecks.md` |
| **Notes** | Under **ADR-006** (intent contracts for non-trivial classes), the six liveness check classes — each a cohesive contract with an injected-dependency seam, a verdict enum, and an exit-code mapping — warrant a CIC, and the suite as a whole (an operational instrument used platform-wide) arguably warrants an ADR. It had neither. **Well-mitigated**, which is why Tier 4 not 3: a thorough `tools/liveness/README.md` documents every surface, verdict, exit code, and the encoded conventions with receipts; and ADR-005 (§ the `live` category) + ADR-017 (§ the observability instrument for derived state) both reference it. So the *contract existed in prose* — the gap was that it was not in the machine-checkable CIC form the repo's own convention prescribes, so `validate_docs.sh` and the CIC-audit tools could not guard it. **Resolution (#275, 2026-07-21):** authored `docs/CICs/LivenessChecks.md` — one subsystem-level CIC (the `ReconciliationWiring.md` precedent) covering the shared check-class contract, referencing the README as the human companion; wired `tools/liveness/report.py` and `tools/liveness/__main__.py` into `.github/workflows/cic_sync_check.yml` so a change to the verdict/exit-code map or the runner now forces the CIC to move in the same PR. A dedicated **ADR** for the suite remains optional (ADR-017 already leans on it) — deliberately not built. Cross-refs: C-103 (same subsystem, the test-taxonomy aspect — resolved), C-16 (CIC coverage of non-trivial classes), C-106 (the runtime-smoke work, similarly under-governed — no ADR yet). |

---

### C-108 — No durable, access-controlled source of truth for platform secrets

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A laptop holding the only copy of the production Appwrite datastore API key dies / is reimaged / the colleague leaves; or a fresh checkout must run a production step. Today there is no durable backup to restore from and no documented provisioning — recovery means scrambling to regenerate keys in the Appwrite console (a rotation), and every consumer must be re-provisioned by hand. |
| **Source** | credential audit 2026-07-27 (`reports/security/appwrite_credentials_audit.md`) |
| **Status** | **Subsumed by þing-01 (The Appwrite Seam Contract, then named PLATFORM-001), 2026-07-28** — the secrets-management architecture is decided by the ratified cross-repo verdict; the risk is now owned by that contract + its per-repo follow-through, not by this entry (was: Open, architecture deferred). |
| **Location** | Platform-wide. Producer/postprocessor read via `load_dotenv` → `views-models/.env` (pipeline-core `configs/prediction_store.py` `_ENV_MAP`; postprocessing `unfao/managers/unfao.py`). Server: `views-faoapi/deployment/bootstrap.sh:21` hardcodes `SOURCE_ENV=/home/sonja/.../views-models/.env`. |
| **Notes** | The 15 Appwrite keys (+ `ACLED_PASSWORD`, `GDL_API_TOKEN`, `UCDP_API_TOKEN`, `VIEWS_DATAFACTORY`) that a UN-facing production service depends on exist **only** in personal, gitignored `.env` files on ≤2 laptops plus one derived server copy (`.env.faoapi`). **No durable, backed-up, access-controlled source of truth; no independent per-person revocation; no rotation runbook; the deploy bootstrap is pinned to one individual's home directory.** This is the root cause of credentials repeatedly being re-supplied across sessions ("the mystery"). *Not a leak:* the audit verified **no secret is in git, in any repo, across full history**, and none is pasted into any tracked file — so this is a **fragility / recoverability** risk, not exposure. **Vocabulary is consistent** (one canonical naming across all repos); the problem is *provisioning architecture*, not sprawl. **Immediate hygiene shipped (this entry's partial):** `views-models/.env.example` (the documented schema), `tools/check_credentials.py` (self-diagnosing "which keys am I missing?"), and `tests/test_credentials_presence.py`. **Deferred by maintainer decision (2026-07-27), informed by an external assessment:** the real secrets-management architecture is a deliberate investigation, NOT chosen now — options are SOPS+age with individual keys (interim/low-complexity, per-person revocation), a secrets manager / OIDC short-lived creds (production automation), or a team password manager (human-accessed); GPG-shared-passphrase-in-a-repo is explicitly *interim bootstrap only, not the target*. Tracked in views-models#280. Cross-refs: the monthly-run-ritual entry (same "scattered on personal laptops" family), C-79 (infra/IP-in-config hygiene). **Subsumed (2026-07-28):** the deferred investigation (views-models#280) converged at þing-01 into **The Appwrite Seam Contract** (named `PLATFORM-001` at the time) — one identity/secrets/config contract homed in views-appwrite, an owned coordinate registry, a declared secret/coordinate split, three credential tiers, and raise-by-default failure semantics. The risk persists until the contract is fully implemented, but it is now owned by that contract, not this entry. views-models follow-through: #285 (this retarget), #286 (harvest-token deletion), #287 (launcher registry read), #288 (curation-list re-homing). Contract: https://github.com/views-platform/views-appwrite/blob/856d617/docs/ADRs/platform/appwrite_seam_contract.md — **v1.3.0**, pinned at `856d617`. (This citation read `60674b2` until 2026-08-02, which is **v1.0.0** — two ratified versions behind (v1.2.0, v1.3.0; v1.1.0 was proposed and never ratified); nothing signalled that, which is the argument ADR-011 makes for readable identifiers over opaque ones.) |

---

### C-109 — Unpaginated Appwrite `list_documents` silently truncates at 25; "the whole set" is a 25-item lie

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any code that enumerates Appwrite documents to build a *complete* set — resolving a wire manifest's shard list, listing every delivered file for a month, counting artifacts to decide "is the run finished?" — and calls `databases.list_documents(...)` (or the REST equivalent) without an explicit `Query.limit()` + offset loop. The set silently caps at the SDK default of 25 and the caller treats the truncated slice as the whole. |
| **Source** | run-0 serving incident (2026-07-28), root-caused live during the FAO delivery |
| **Status** | Open (the *incident* is fixed in views-faoapi; the *pattern* is unguarded platform-wide) |
| **Location** | Incident: views-faoapi `src/views_faoapi/managers/appwrite.py:838-870` `search_files_by_metadata` — `list_documents(db_id, coll_id, queries=queries)` with no limit/offset → resolved **25 of 108** shards for the run-0 manifest → ingest refused → the API served the previous month's `orange_ensemble` for ~24h while a clean run-0 sat in `unfao_bucket`. Fixed in views-faoapi#287 (offset loop, `DEFAULT_PAGE_LIMIT=100`), deployed v1.3.2. views-models exposure: `tools/liveness/appwrite_api.py:90` `newest_first_query(limit=…)` — always passes an explicit limit and only ever asks "newest N", so it does not currently manifest, but nothing enforces that. |
| **Notes** | Tier 2 rationale: structural fragility with a named, already-realized trigger — a client-library **default** silently redefines "all" as "the first 25", and the caller has no signal. It cost the first-ever UN-facing global-land delivery a day of serving a stale forecast. What converted silent-partial into loud-refuse in this instance was luck of design: the ADR-013 manifest **declares its shard count**, so the consumer could compare 25 against 108 and refuse. Any listing that is *not* cross-checked against a declared expected count would have produced a **silently partial dataset** instead — the same defect one hop away from Tier 1. Exits (either or both): (a) a shared paginated-list helper as the only sanctioned way to enumerate Appwrite documents in every repo, so the default is never reachable; (b) the ADR-013 discipline generalized — every enumeration that must be complete carries a declared expected count and refuses on mismatch. Cross-refs: C-97 (the addressing/identity axis of the same interchange), C-100 (config-vs-reality discovered only by failing live), C-111 (the fallback that hid this failure from the producer). |

---

### C-110 — The configuration that produced the live UN-facing forecast exists only in an uncommitted working tree

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Anyone re-runs `postprocessors/un_fao` from a clean checkout, a fresh clone, or after a `git checkout -- .` / stash-drop in this tree — the committed configs still say `REGION = "africa_me_legacy"` (13,110 cells, pandas path), so the re-run **succeeds** and delivers a Middle-East+Africa forecast into `unfao_bucket` under the same product name as the global-land one now being served. No error, no warning, wrong extent. |
| **Source** | run-0 delivery (2026-07-27/28); confirmed still uncommitted 2026-07-31 (`git status`) |
| **Status** | Open |
| **Location** | `postprocessors/un_fao/configs/config_queryset.py` (`REGION` `africa_me_legacy`→`land_gaul`; `"data_format": "feature_frame"`; datafactory pin `>=1.9.0,<2.0.0`), `postprocessors/un_fao/configs/config_meta.py` (`wire_contract`, `wire_upload_enabled`, `region: land_gaul`), `postprocessors/un_fao/README.md`, `postprocessors/un_fao/requirements.txt` — all `M` in the working tree, none on `origin/development` |
| **Notes** | Tier 2 rationale: not silent corruption *today* (the served artifact is correct and its provenance is verifiable on the shelf), but a structurally fragile state with a realistic, one-command trigger and a wrong-output consequence — the delivered product is **not reproducible from any committed state**, and the reproduction attempt fails *quietly and plausibly* rather than loudly. The working tree is acting as the system of record for a UN-facing deliverable. Aggravating factor: the same tree also holds parallel-session-owned `violet_visitor` edits, so it cannot simply be committed wholesale — the un_fao paths must be staged by name. **Exit: commit the four un_fao paths via the merge ritual** (small, ready, blocked on nothing). Related cross-repo state, unverified from this repo and therefore not registered separately: the views-postprocessing bug-#1 name-scoping fix (`_prod_forecasts_datastore(name_scoped=False)`) was reported uncommitted in that checkout and may have the same exposure — worth a check in that repo. Cross-refs: **C-105** (the *inverse* direction of the same "working tree is the system of record" hazard — that entry is the checkout drifting *behind* the remote, this one is run-critical state living *ahead* of it and unbacked); C-53 (config value regression across merges). **Second variable found 2026-08-04, same file, same shape:** `postprocessors/un_fao/configs/config_meta.py` carries `wire_upload_enabled: True` in the working tree and **nowhere in git**. That key is the ADR-013 §11.4 upload interlock — with it absent the sink stages locally and makes ZERO store calls; with it present the run publishes to the UN FAO's bucket. So **whether this platform delivers to a partner is decided by an uncommitted edit on one laptop**, and two identical checkouts behave differently toward an external party. Registered here rather than as a new entry because it is the same defect at the same location: the configuration governing a UN-facing delivery exists only in a working tree. **Exit:** commit the key with whatever value is intended, so the deployed behaviour is derivable from a commit. Cross-ref: **C-117** (the dependency half, mitigated). |

---

### C-111 — The serving hop degrades silently: a failed ingest falls back to last-good, and the producer sees nothing

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A delivered run is malformed, partial, or hits any consumer-side ingest bug — `views-faoapi` logs the failure server-side, keeps serving the previous month's forecast as if current, and **no signal reaches the producing side**: not the postprocessor (already exited 0), not `tools/liveness` (which judges bucket recency, not what the API actually serves), not a human. The stale forecast is discovered only when someone thinks to ask. |
| **Source** | run-0 serving incident (2026-07-27/28) — Hop 2 delivered clean, faoapi served `orange_ensemble` (March) for ~24h, discovered only by direct maintainer question |
| **Status** | Open |
| **Location** | views-faoapi `src/views_faoapi/managers/dataset_service.py:621-720` `_load_wire_run` (lazy on-request ingest; refuse → fall back to last-good) — by design, and correct as a *availability* policy; the gap is that the degradation is invisible upstream. views-models side: `tools/liveness/unfao_delivery.py` observes the FAO bucket, not the served response; there is no surface for "what is the API serving right now, and is it the run we shipped?". |
| **Notes** | Tier 2 rationale: structural fragility across a repo boundary with a realized trigger and a stakeholder-visible consequence — UN consumers received a month-old forecast presented as current, with no error anywhere in the producer's view. Serve-last-good is the *right* availability choice; the defect is that it is a **silent** downgrade, so "delivered" and "served" are two different truths with nothing comparing them. Compounding observability defects seen the same session: the API's `/version` endpoint lagged the deployed git tag by one release (reported `1.3.4` on `deployed_tag v1.3.5`), so "which build is live?" could not be answered from the API itself; and `/provenance/forecast`'s top-level fields carried the *previous* run's name/filename/`run_id` (views-faoapi#290, fixed v1.3.5) so even a correct serve looked wrong. Exits: (a) a **serving-truth liveness surface** in `tools/liveness` that reads the live API's provenance and compares `run_id` against the newest manifest in `unfao_bucket` — this is the check that would have caught run-0 in minutes and is squarely a views-models deliverable; (b) upstream, a degraded/stale flag in the served response or an ingest-failure signal the producer can poll. Cross-refs: **C-102** (the detection half — `unfao_delivery` reads `DELIVERING` while the API serves nothing; this entry is the *root* it fails to see), C-97 (delivery identity), C-99 (no heartbeat), C-109 (the ingest bug this fallback concealed). |

---

### C-112 — Presence checks read SHELL scope while consumers need EXPORTED scope; the same blind spot has now shipped twice in four days

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any shell code that decides "is this credential/coordinate available?" with `[ -n "$VAR" ]` or `[ -n "${!NAME}" ]`, in a script that has previously `source`d a `.env`. Sourcing sets the variable in **shell** scope; the check passes; the child process — which is what actually needs it — receives nothing. |
| **Source** | code-review max on PR #315 (2026-08-02), reproduced directly; earlier instance found during þing-02 (`orð_09 §2H`) |
| **Status** | Open |
| **Location** | Instance 1 (2026-07-31, fixed in #314): `postprocessors/un_fao/run.sh` `_platform001_coordinate_state()` — announced *"Coordinates ARE present in the environment (exported outside this script)"* on the strength of an unexported shell variable. Instance 2 (2026-08-02, caught pre-merge in #315): `tools/credentials/platform_env.sh` `platform_env_export_secret()` guard `[ -n "${!PLATFORM_ENV_SECRET_NAME:-}" ]` — returns early because `run.sh` sourced `.env` for `GITHUB_TOKEN` two dozen lines above, so the `export` is never reached; `platform_env_validate()` shares the blind spot and reports the environment complete. |
| **Notes** | **The recurrence is the finding, not either instance.** Both were written by an author who had just read #293 — the incident whose entire content is *"`source` without `export` does not reach the child"* — and both reproduced it anyway, because bash makes the two scopes indistinguishable at the point of test. `[ -n "$VAR" ]` cannot tell them apart; only `[ -n "${VAR+x}" ]` combined with an `export -p` lookup, or a probe of an actual child process, can. **Reproduced on the PR branch:** with `.env` sourced first, `platform_env_export_secret` returns 0, `platform_env_validate` passes, and `python -c 'os.environ.get(...)'` returns `None` — a run that reports a complete environment and hands the child nothing. Instance 2 was additionally *protected by a test*: `test_the_launcher_does_not_export_the_secret_itself` asserts the launcher must not export the secret directly, which is correct as a one-writer rule and, combined with the defective guard, enforced the bug. **Exit:** a single sanctioned way to answer "will the child see this?" — probe the exported environment (`env` / a real child), never the shell's. Everything else in this class is a rediscovery. Tier 2: silent, produces a successful-looking run that delivers nothing, and the demonstrated recurrence rate is twice in four days. Cross-refs: **C-111** (silent serving degradation — same family: success reported, nothing delivered), C-94/C-95 (silent-when-unenforced), C-57 (the sibling class where a *regex* cannot distinguish a comment from code, which also recurred twice this week). Member of **Cluster A** (declared-but-unenforced). **A sibling shape, found in the same review and fixed rather than registered** (one instance, now pinned by a test): a guard written for the **steady state** was reused during **setup**, where the condition it treats as fatal is the normal starting state — `bootstrap.sh` called the fatal `platform_env_export_secret` before prompting, so a first-ever machine's very first output was *"FATAL: … does not exist. Run ./bootstrap.sh"* addressed to the person running `./bootstrap.sh`. Not the same defect as this entry, but the same question badly posed: **a check must be asked in the context it will answer for.** |

### C-113 — A feature branch was committed with no parent, and every gate in the merge ritual passed it

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any `git commit` that is interrupted (timeout, Ctrl-C, a hung prompt) and then re-run. If the interruption left `HEAD` unborn, the retry produces a **root commit** and the branch silently becomes an orphan with no common ancestor. Every subsequent commit extends the orphan. |
| **Source** | Pre-merge safety gate on PR #315 (2026-08-02) — caught by hand, not by any check |
| **Status** | Mitigated |
| **Location** | `feat/309-311-platform-env-and-bootstrap`, commit `0651448f` (`parents: []`). Five commits, 5 reachable ancestors against `development`'s 877. Repaired by re-parenting each tree with `git commit-tree` onto `2554d731` and moving the branch with `git reset --soft` (no working-tree write — the maintainer had six dirty files including two in `models/violet_visitor/`, which was running experiments). |
| **Notes** | **What made it dangerous is that nothing reported it.** `git status`, `git log`, `ruff`, the 7,268-test committed-state suite, four rounds of code review, and `gh pr view` all passed: the *tree* was correct (development's content plus exactly nine intended files), only the *history* was disjoint. GitHub reported `mergeable=MERGEABLE` and offered the merge button. The only signal was `git diff development...branch` failing with `fatal: no merge base`, rc=128 — and on the first attempt **that failure was itself swallowed**: the command's error text went to a variable, the empty result was grepped for `violet_visitor`, found nothing, and the gate printed *"violet_visitor: NOT touched ✓"*. A gate that cannot distinguish "clean" from "did not run" is not a gate. **Probable cause:** a `git commit -m` whose message contained backticked shell (`` `read -rs` ``) — bash executed `read`, which blocked on stdin for two minutes until killed. **Exits, in order of value:** (1) every safety gate must check the exit status of the command it reasons about and refuse to report a verdict on empty output; (2) always `git commit -F <file>`, never `-m` with a message containing backticks; (3) a cheap pre-merge assertion — `git merge-base --is-ancestor origin/development HEAD` — would have caught this in one line. Tier 2: silent, survived every existing control, and the failure mode is a merge of disjoint history into `development`. Cross-refs: C-112 and C-57 (same family — a check that cannot distinguish the state it claims to test), **C-94/C-95** (silent-when-unenforced). Member of **Cluster A** (declared-but-unenforced). |

### C-114 — The detector built for the November key expiry reported a rejected key as mild staleness

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | The Appwrite datastore key expiring, being revoked, or being replaced with a wrong value — expected around **2026-11-30** for both current keys. Also any code that judges Appwrite reachability from a **file-listing** response alone. |
| **Source** | Building the #302 preflight (2026-08-02); found because the preflight was tested against a simulated dead key before being trusted |
| **Status** | Resolved |
| **Location** | `tools/liveness/appwrite_store.py` and `tools/liveness/unfao_delivery.py` — both read only `GET /storage/buckets/{id}/files`. Fixed by `assert_bucket_reachable` in `tools/liveness/appwrite_api.py`, called first inside each surface's existing `try`. |
| **Notes** | **Appwrite answers the file-listing endpoint with HTTP 200 and `total: 0` when the key is rejected.** Measured three ways against the live server (Appwrite 1.9.5, 2026-08-02): real key → 200, `total=461`; garbage key → 200, `total=0`; empty key → 200, `total=0`. Listing files was the *only* call either surface made, so a dead credential was indistinguishable from an empty bucket. `appwrite_store` returned `STORE_IDLE` with `error: bucket contains no files`; `unfao_delivery` would have returned `DELIVERY_STALLED`. Both are **exit 1, "attention"** — a verdict a human reads as "nothing landed lately", which is unremarkable for a monthly cadence. **Why Tier 1 rather than 2.** This is not a check that might mislead in principle; it is the check this platform designated as the detector for a *known, dated* silent failure. C-99 records that the write path logs *"Forecasts uploaded successfully"* while uploading nothing once the key dies; #302 exists to schedule this detector against that date; and the detector renders exactly that failure as ordinary staleness. Both the alarm and the thing it watches were silent in the same way, so the platform would have concluded "quiet month" through a full delivery cycle to an external partner. **The fix, and why the bucket GET.** Every other endpoint tested returns 401 for the same rejected key — bucket get, bucket list, database get, collection list, `/health`. `assert_bucket_reachable` GETs the **bucket itself** because it settles two questions in one call: the key is accepted (401 if not) and the bucket coordinate still resolves (404 if not) — a wrong bucket id would otherwise also have surfaced as emptiness, for the same reason. Verified live afterwards: real key → `STORE_ACTIVE`, 461 files; garbage key → `UNREACHABLE`, `HTTP Error 401`, exit 2. **What generalises.** *An empty result and a refused request are the same bytes unless something distinguishes them.* Cross-refs: **C-99** (the alarm is built and nothing runs it — this entry is why scheduling it was not yet sufficient), **C-111** (silent serving degradation), **C-112** (a check asked in a scope that cannot answer it), **C-113** (a gate that cannot tell "clean" from "did not run"). Member of **Cluster A** (declared-but-unenforced). Contract updated in `docs/CICs/LivenessChecks.md` §6. |

### C-115 — A version boundary is encoded as a hyphen: merging two look-alike env directories silently gives 31 models the wrong package

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Anyone normalising the inconsistent `env_path` names (`envs/views_r2darts2` vs `envs/views-r2darts2`, `envs/views_stepshifter` vs `envs/views-stepshifter`), or a `run.sh` regenerated from views-pipeline-core with a normalised path. It reads as fixing a typo. |
| **Source** | expert-code-review (2026-08-02), env x declared-spec cross-tab |
| **Status** | Open |
| **Location** | `envs/views_r2darts2` (22 tenants) vs `envs/views-r2darts2` (9 tenants); the 31 r2darts models' `requirements.txt`; `env_path=` line 18 of each `run.sh` |
| **Notes** | **The two directory names differ by one character and that character is load-bearing.** `envs/views_r2darts2` holds 12 models declaring `views-r2darts2==0.1.0` and 10 declaring `>=0.1.0`; `envs/views-r2darts2` holds 9 declaring `>=1.0.0,<2.0.0`. `==0.1.0` and `>=1.0.0,<2.0.0` are **mutually unsatisfiable**. They do not collide today only because they resolve to separately-named directories. Merge the names — the obvious tidy-up — and `run.sh` installs each tenant's own file into one shared prefix with no uninstall, so the resolved version becomes whatever ran last. Half the models then run a version they did not ask for. **Tier 1, not 2:** there is no error. pip reports success for each individual install; the models train and emit forecasts; the forecasts are computed by the wrong algorithm version. The counterpart split `envs/views_stepshifter` (32) vs `envs/views-stepshifter` (7) declares **identical** specs and is pure duplication (~9G each on disk) — so the naming inconsistency is meaningful in one place and meaningless in the other, with nothing distinguishing them. **Exit:** rename to state the constraint (`envs/views_r2darts2_v0` / `_v1`) or pin the split with a test whose failure message explains it. A five-line test is the cheapest high-value change available. Cross-refs: **C-116** (the shared-environment root cause), **C-70** (`run.sh` duplication), C-39 (generator-sourced regression). Member of **Cluster A** (declared-but-unenforced). |

---

### C-116 — 131 requirements.txt resolve into 11 shared environments, so a model's dependencies are decided by its co-tenants' run order

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Two tenants of one environment needing different versions of the same package — or anyone reading a `requirements.txt` to answer "what will this model run with?" |
| **Source** | expert-code-review (2026-08-02); measured cross-tab of `env_path` against declared specs |
| **Status** | Open |
| **Location** | 131 `requirements.txt` -> 11 `envs/*` values. `envs/views-baseline` 37 tenants, `envs/views_stepshifter` 32, `envs/views_r2darts2` 22, `envs/views_ensemble` 13, `envs/views-r2darts2` 9, `envs/views-hydranet` 8, `envs/views-stepshifter` 7, plus 4 singletons. Install logic: `run.sh:22-39`. |
| **Notes** | **Per-model *declaration* is real; per-model *isolation* is not, and the repo reads as though both were.** Each `run.sh` pip-installs only its own `requirements.txt` into the shared prefix and never uninstalls, so environment contents depend on which tenant last ran. **Proven in both directions.** *Declared but absent:* `ensembles/skinny_love/requirements.txt` declared `views-frames>=1.7.0,<2.0.0` while `envs/views_ensemble` has views-frames not installed — and skinny_love completed a run in that state on 2026-07-22 (wandb `atomic-jazz-101`, `state=finished`). *Present but undeclared:* 27 models receive views-datafactory because a co-tenant declares it; in `envs/views-baseline` only 10 of 37 tenants declare it. To answer what a model will run with you need four facts — its own file, its `env_path`, its co-tenants' files, and the order they last ran — and three of them are not in the file you are reading. **Not a call to centralise:** the `config_partitions.py` precedent stands and the files must stay per-model. What is missing is that the *shared* thing has no declaration at all. **Exit (smallest honest):** (a) `pip freeze` per run stored with the artifact — see C-117; (b) a generated comment in each `requirements.txt` naming its environment and co-tenant count, which pulls the three hidden facts into the file being read. Cross-refs: **C-38** (a specific instance — `datafactory_query` absent from an env that must run bright_starship), **C-115** (the unsatisfiable case this enables), **C-70**, C-08. Member of **Cluster A**. |

---

### C-117 — The dependency closure that produces a UN-facing forecast exists only on one laptop's disk

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Anyone asking which package versions produced a specific delivered forecast — or a forecast being questioned by the partner. |
| **Source** | expert-code-review (2026-08-02, Nygard/Kleppmann) |
| **Status** | Mitigated |
| **Location** | `envs/` (gitignored); `run.sh` (no resolved-manifest capture); `monthly_run.sh` |
| **Notes** | Environments are gitignored, so the versions a run used are a property of one machine's disk and of no commit. Measured on the maintainer's laptop: **3 of the 11 environments exist at all**, totalling 22G (`envs/views-baseline` alone is 9.0G) — full provisioning would be roughly 100-200G, which is also why 11 shared environments rather than 131 is the correct resource decision and not laziness. The consequence is that the repo's own stated rule — *"if it is not committed to git, you cannot assume it exists"* — is violated by the dependency closure of a UN-facing deliverable. **This is the dependency twin of C-110**, which records the same failure for *configuration*; registered separately because the artifact, the owner and the fix differ, but they should be closed together. **Exit:** one line in `run.sh` — `pip freeze > logs/<run_id>_env.txt` — kept with the forecast artifact. This converts "which versions produced this?" from unanswerable to a lookup and is the highest-value change surfaced by this review, ahead of any hygiene work. Cross-refs: **C-110** (configuration provenance), **C-116** (why the environment is not derivable), C-10 (`envs/` in the tree, Accepted), C-97/C-98 (delivery identity and system of record). Member of **Cluster A**. **Mitigated 2026-08-03 (#327):** `monthly_run.sh` now writes a `pip freeze` per environment into `reports/env_snapshots/`, after each folder runs (before would describe the environment as it was, not as it was used), deduplicated so the four ensembles sharing `envs/views_ensemble` produce one file. The header carries the run id, the environment and **the commit** — neither half is sufficient alone. Written to a TRACKED directory: `logs/` is gitignored, so a snapshot there would be exactly as ephemeral as the thing it describes, and `.gitignore`'s blanket `*.txt` needed an explicit negation, pinned by `test_environment_snapshots_are_not_gitignored` (this repo has lost a file to a blanket rule before — `*.yml` silently swallowed a new workflow). Not fully Resolved: the operator must commit the snapshots, and nothing enforces that. **The first snapshot ever taken immediately diagnosed a live production breakage** — it showed `views-pipeline-core` installed `-e` (editable) with no `views-frames` line, which is why all four production ensembles fail at import; see C-116. |

---

### C-118 — 27 models accept any future views-datafactory major

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | views-datafactory publishing a 2.0. |
| **Source** | expert-code-review (2026-08-02) |
| **Status** | Open |
| **Location** | 27 `requirements.txt` carrying `views-datafactory>=1.9.0` (10 in `envs/views-baseline`, 12 in `envs/views_r2darts2`, 4 in `envs/views-hydranet`, 1 in `envs/views-r2darts2`); the sibling `postprocessors/un_fao/requirements.txt` already carries `views-datafactory>=1.9.0,<2.0.0` |
| **Notes** | An unbounded upper spec on 27 models, installing itself during a monthly hand-run on whichever laptop is free. The 28th file already carries the ceiling, so closing this is making 27 files match a decision the repo has already made rather than making a new one. **Tier 3 and not 2 deliberately:** this is a specific, measured instance of **C-31** (*"upstream algorithm package API changes break views-models silently"*, Tier 2), and the family severity is already carried there — double-counting it would inflate the register rather than inform it. Escalate only if the same pattern is found on a package with no Tier-2 parent. Cross-refs: **C-31** (parent), C-50 (spec unresolvable on fresh clone), C-116. |

---

### C-119 — The install gate decides a production install from a line count of pip's log

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | pip emitting any unexpected line — a deprecation warning, an index warning, a proxy notice — or a pip older than 22.2, which has no `--dry-run` and errors instead. |
| **Source** | expert-code-review (2026-08-02, Feathers/Nygard) |
| **Status** | Open |
| **Location** | `run.sh:27-33` in ~131 scripts, e.g. `models/bad_blood/run.sh:27` |
| **Notes** | The gate is `missing_packages=$(pip install --dry-run -r requirements.txt 2>&1 \| grep -v "Requirement already satisfied" \| wc -l)`, then install when `>0`. It counts **lines of merged stdout and stderr**, not missing packages, so its answer depends on pip's log formatting and on stderr being quiet. Any warning triggers a full `pip install` — which, in a shared environment (**C-116**), can *mutate other models' dependencies as a side effect of a log message*. There is no seam to test it: the logic is inline and duplicated ~131 times, which is **C-70**'s cost made concrete. The fix belongs in the generator, not here — `template_run_sh.py` in views-pipeline-core, already open as **views-pipeline-core#384** for the shebang — otherwise it follows the C-39 pattern of fixing copies while the generator keeps producing the defect. Cross-refs: **C-70** (duplication), **C-116** (shared mutable environment), **C-39** (fix-the-copies-not-the-generator, regressed 24x), views-pipeline-core#384. |

### C-120 — A general test-exemption mechanism with a model-specific guard: any model could opt out of parity enforcement silently

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding `EXPERIMENT_IN_PROGRESS = True` to any model's `config_hyperparameters.py` — a one-line, plausible-looking edit that exempts that model from the cross-ensemble parity pins. Also: any future exemption mechanism whose *escape hatch* is general while its *alarm* names one subject. |
| **Source** | code-review of `chore/s1_5-fleet-config-hygiene` (2026-08-03), reproduced |
| **Status** | Resolved |
| **Location** | `tests/test_datafactory_parity.py` — `_experiment_in_progress()`, `test_both_trios_use_same_loss`, `test_constituent_posterior_samples_match`, and the guard now named `test_the_experiment_in_progress_roster_is_exactly_as_declared` |
| **Notes** | The branch introduced a correct idea — an experiment whose values churn should not be pinned to an exact value that flickers red/green — and guarded it with `test_violet_visitor_is_experiment_in_progress`, which asserted only that **violet_visitor** carries the marker. **But `_experiment_in_progress()` applies to any model.** So the exemption was general and the alarm was specific. **Reproduced, not theorised:** adding one line to `models/pink_pirate/configs/config_hyperparameters.py` removed it from *both* parity pins and the suite reported **51 passed** with nothing to indicate a model had stopped being checked. A second probe marked all six trio models: both pins then compared `{} == {}` and passed — a test asserting nothing at all. **Fix:** pin the exemption **roster as a set** (`EXPERIMENTS_IN_PROGRESS = {"violet_visitor"}`), which fails in both directions — a model gaining the marker and a model losing it — plus an explicit non-empty assertion in each pin so neither can ever pass vacuously. Verified by re-running all three probes against the fix. **The generalisable rule:** *an escape hatch must be guarded at the same scope it operates.* A guard that names one subject cannot cover a mechanism that accepts any. Cross-refs: **C-112** (a check asked in a scope that cannot answer it), **C-113** (a gate that cannot tell "clean" from "did not run"), **C-115** (an invariant guarded by name rather than by the invariant). Member of **Cluster A** (declared-but-unenforced). **Near-miss recorded on C-71:** the same branch put prose after `Open` in a Status field, which silently dropped the entry from the header count — caught by `test_open_count_accurate`, which is what that test is for. |

### C-121 — The delivery boundary accepts a forecast of unbounded age

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | `un_fao` running in a month where no fresh `rusty_bucket` run was produced first — which is every month, because `rusty_bucket` is not in `monthly_run.sh`. |
| **Source** | expert-code-review of the delivery composition (2026-08-04), traced against the live store |
| **Status** | Open |
| **Location** | `views-postprocessing/views_postprocessing/unfao/managers/unfao.py::_read_forecast_data_contract`; `contract/wire/source_selection.py::resolve_run` |
| **Notes** | `resolve_run` selects **the newest fully-manifested run for a named ensemble** — identity plus completeness, which is right, and is what **C-97** records as resolved. What it does not do is bound the run's **age**. So the delivery step will silently republish an arbitrarily old forecast as the current one. **Measured 2026-08-04:** FAO's forecast stream is 145 days stale (#320, newest `forecast_dataset` 2026-03-10) while `production_forecasts` holds exactly one complete run, `rusty_bucket_forecasting_20260727_095355` (all three `lr_ged_*` targets), untouched since 27 July. The forecast existed; nothing carried it across, and nothing would have objected if it had carried the March one instead. The method **already logs** `"Contract inbound resolved: run %s"` — the fact needed for the assertion is in hand and simply not asserted on. **Exit:** a declared freshness budget and a refusal, at the boundary that already knows the answer. This is the one change that addresses the failure that actually occurred. Cross-refs: **C-99** (no missed-month signal — this is its specific, now-measured instance at the delivery boundary), **C-97** (selection, resolved), **C-114** (a detector that reported its own failure as mild staleness). Member of **Cluster A** (declared-but-unenforced). |

---

### C-122 — The production pipeline's assembly is five ordered strings, and the order is an unstated data dependency

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Adding a producer *below* its consumer in `monthly_run.sh`, or reordering the existing five lines. Concretely: adding `rusty_bucket` after the `un_fao` line rather than before it. |
| **Source** | expert-code-review of the delivery composition (2026-08-04) |
| **Status** | Open |
| **Location** | `monthly_run.sh` — the five `run_folder` lines; `postprocessors/un_fao/configs/config_meta.py` (`"ensemble"`); `views-postprocessing/unfao/product.py` (`UPLOAD_ENABLED`) |
| **Notes** | `un_fao` consumes what the ensembles produce, and that dependency is encoded **only** as line order in a shell script. Getting it wrong raises nothing: the consumer simply delivers a previous run. **Everything beneath this point injects its dependencies** — `_ContractStorePort` is an explicit DIP port, `contract/` and `delivery/` are partner-neutral and a test proves it — while the composition root hard-codes five concrete paths in a fixed sequence. The three files that must agree to deliver a forecast live in two repositories with no reference between them. Related: **13 ensembles exist and only 4 are in `monthly_run.sh`**; the one `un_fao` is configured to consume, `rusty_bucket`, is **not among them**, so "run everything monthly" produces four forecasts nobody delivers and delivers one forecast nobody just made. **Deliberately not fixed yet.** A declared composition is an abstraction over exactly one instance in this repo, and the maintainer's rule is to extract on a second incident behind a named trigger. **The named trigger: when a second partner delivery needs a production run in views-models.** The scar already exists one repo away — views-postprocessing #211, *"every partner-scoped guard was scoped to ONE partner"*, fixed there with a declared list asserted against the filesystem (`tests/conftest.py::PARTNER_PACKAGES`), not a framework. That is the shape to copy when the trigger fires. Cross-refs: **C-99**, **C-97**, **C-121**, **C-120** (same bug class: a general mechanism guarded for one subject). |

---

### C-123 — `rusty_bucket`'s config does not describe what it emits

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any tool or person deriving a delivery's target names from model config rather than from the manifests in the bucket. |
| **Source** | expert-code-review (2026-08-04); confirmed against the live store |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_meta.py:4` |
| **Notes** | Declares `regression_targets: ["lr_sb_best", "lr_ns_best", "lr_os_best"]`. The run it actually produced emitted `lr_ged_sb`, `lr_ged_ns`, `lr_ged_os` — verified by listing `production_forecasts`, where the manifests are named `rusty_bucket_forecasting_20260727_095355__lr_ged_*__manifest.json`. Delivery works **only** because `resolve_run` matches manifest filenames rather than the config. So the config is decorative at precisely the point a reader would trust it, and the only reliable source of truth for a delivery's contents is a live bucket query — which is what this review had to do. Cross-refs: views-models#151 (target-name standardisation), **C-104** (one quantity, four config keys). |

---

### C-124 — Coverage is validated after the expensive path, not at resolution

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A run resolving successfully for a region whose cell coverage it cannot satisfy — e.g. an africa-only run against `region: land_gaul`. |
| **Source** | expert-code-review (2026-08-04) |
| **Status** | Open |
| **Location** | `views-postprocessing/contract/wire/source_selection.py` — `resolve_run` (manifests) vs `TargetLease.load()` (`assert_complete_coverage`, `assert_no_excluded_cells`) |
| **Notes** | `resolve_run` checks that every expected target has a content-verified manifest; `expected_cells` and `excluded_gids` are passed to the lease and only enforced when frames materialise. Resolution succeeding therefore does not mean delivery will succeed — the failure arrives after shard downloads, on a monthly hand-run on a laptop. The lazy design is correct for memory (it is the run-0 OOM fix); what is missing is a cheap precheck so an unsatisfiable run is rejected before the heavy fetch. Cross-refs: **C-121**. |

### C-125 — A target-name gate would fail correct delivery files, because a source's config does not describe what it emits

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Enabling an edit-time `targets` check in a delivery declaration (ADR-019 §1 `REQUIRE.targets`) while any source's config still misdescribes its own output. |
| **Source** | expert-code-review of the delivery declaration design (2026-08-04) |
| **Status** | Open |
| **Location** | `ensembles/rusty_bucket/configs/config_meta.py:4`; the proposed `deliveries/*.py` `REQUIRE.targets` (ADR-019) |
| **Notes** | `rusty_bucket` declares `regression_targets: ["lr_sb_best", "lr_ns_best", "lr_os_best"]` and emits `lr_ged_sb/ns/os` (**C-123**). A `targets` gate checked against the source config would therefore reject a *correct* delivery file for the repo's own FAO ensemble. **The cost is pedagogical, which is why it is worth an entry of its own:** ADR-020 makes error messages load-bearing — the design's value is that a newcomer is guided down one level at a time — and the first lesson this would teach is that the repo's errors are wrong. Nothing recovers from that. **Exit:** fix C-123, then promote `targets` from a run-time assertion (checked against the run's manifests, which are truthful) to an edit-time one. Until then ADR-020 §4 records `targets` as a stair that ends outside the repo. Cross-refs: **C-123**, ADR-019 §1, ADR-020 §4. |

---

### C-126 — The delivery design makes dormancy visible but not absence of execution

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A delivery declared `intent = live()` that no runner picks up, or that ships a run far older than the consumer expects. |
| **Source** | expert-code-review of the delivery declaration design (2026-08-04, Nygard) |
| **Status** | Open |
| **Location** | proposed `deliveries/*.py` — `DELIVERY.intent`, `REQUIRE.max_age`; ADR-019 §4 freshness rule, ADR-020 §4 "where the stairs end" |
| **Notes** | The design solves the *paused* case well: `paused(reason, since=...)` cannot be set silently, so a dormant edge carries an explanation and an age. It does **not** solve the *live-but-never-run* case — a `live()` delivery that nothing executes produces no error, because nothing failed. That is precisely the 145-day FAO silence (**C-121**, **C-99**), surviving the redesign. ADR-020 §4 names it honestly as *"not a locked door — a hole in the floor"*. **Exit, two parts, both already written into the amendment:** `REQUIRE.max_age` is mandatory whenever `intent = live()` (ADR-019 §4), and `tools/liveness` reports **derived status beside declared intent** (ADR-017 §7), so `live()` + "never delivered" is a visible contradiction rather than an absence. Registered separately from C-121 because C-121 is the defect in today's code and this is the residual risk in tomorrow's design — closing one does not close the other. Cross-refs: **C-121**, **C-99**, **C-110**. |

---

### C-127 — Fleet-wide config values are written in two literal styles, so a naive grep silently under-counts

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Executing the `deployment_status` → `maturity` migration (ADR-017 §11), or any future statement of the form "measured across all N sources" produced by grepping the fleet. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P1 |
| **Status** | Open |
| **Location** | 128 × `{models,ensembles}/*/configs/config_deployment.py` — 47 use `{"deployment_status": "shadow"}`, 81 use `{'deployment_status': 'shadow'}` (e.g. `models/brown_cheese/configs/config_deployment.py:19`); the false figures were at `docs/ADRs/017_source_composition_delivery.md:74` and `docs/forecast_delivery_map.md:158` |
| **Notes** | ADR-017 §2 and the delivery map both stated *"Measured across all 131 sources: 120 `shadow`, 6 `baseline`, 4 `deprecated`, 1 `deployed`"*. The true distribution is **117 / 6 / 4 / 1 across 128 files**, and there are **132** source directories — four (`models/cool_cat`, `models/teenage_dirtbag`, `models/test_model`, `ensembles/test_ensemble`) carry no `config_deployment.py` at all. The measurement had been taken with a double-quote pattern that matched 47 of 128 files and silently reported the rest as absent. **The wrong number is now corrected in both documents; the durable risk is the heterogeneity that caused it.** ADR-017 §2's numbers are its evidence base, so this was not cosmetic — a rule whose stated justification is false is the kind a future contributor overturns. Same bug class as **C-114** (a detector blind to one spelling of the thing it was built to see), in a different subsystem: there a rejected key read as mild staleness, here 81 configs read as non-existent. **Exit:** the migration tool must parse rather than grep, and must assert it touched 128 files. Cross-refs: **C-114**, **C-124**. |

### C-128 — ADR-019's "REQUIRE only refuses" rule is false for its one mandatory key, so a validator author must guess

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Implementing the `REQUIRE` validator, or writing the first real delivery file under ADR-019. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P3 |
| **Status** | Open |
| **Location** | `docs/ADRs/019_delivery_declaration.md` §2 (the testable rule, and the "may be omitted entirely" allowance) vs §4 (Freshness) |
| **Notes** | §2 states the rule that defines the whole two-block design: *"removing a line from `REQUIRE` must never change what is produced, only what is allowed through."* §4 then requires that a `live()` delivery **must** declare `max_age`. Removing `max_age` does not widen what is accepted — it makes the file invalid, so nothing is produced. A second instance sits in the same paragraph: §2 offers *"`REQUIRE` may be omitted entirely when there is nothing to assert"*, but every `live()` delivery must carry `max_age` and every real delivery is live, so the allowance is never actually available. **Why this is Tier 2 rather than a wording nit:** an implementer who resolves the contradiction toward "`REQUIRE` is purely assertive" will not enforce freshness — and the missing freshness bound is precisely the failure that already occurred (**#320**, 145 days of silent non-delivery). The contradiction points the implementation at the bug. **Now corrected in the ADR**; the entry records the class so the next absolute-sounding rule is checked against its own exceptions. Cross-refs: **C-121**, **C-126**, **D-09**. |

### C-129 — The delivery declaration has no home for the key that actually arms the delivery

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Building `deliveries/un_fao.py` (ADR-017 §11 Phase 1), or adding the second consumer under **#333**. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P8 (adequacy) |
| **Status** | Open |
| **Location** | `postprocessors/un_fao/configs/config_meta.py:26-27` (`wire_contract`, `wire_upload_enabled`); `docs/ADRs/019_delivery_declaration.md` §3 (the key set) |
| **Notes** | The real FAO config declares **eight** keys. Five map cleanly onto ADR-019's schema; three had no home — `algorithm`, `wire_contract`, and `wire_upload_enabled`. The last is the **arming switch**: views-postprocessing ADR-013 §11.4 sets `UPLOAD_ENABLED = False` and makes that launcher key its only override. ADR-019 mentioned it **zero times** while the delivery map mentioned it twice. The consequence is exact and self-inflicted: ADR-019 exists because a delivery-deciding line sits buried in a file whose docstring calls itself inert — and the design moved the `ensemble` line out while **leaving the on/off switch behind in that same file**. Worse, `intent = live()｜paused()` and `wire_upload_enabled: True｜False` are the same fact in two places, which is the duplication ADR-019 §8 rejects by name. **Resolved in the ADR by declaring `intent` the repo-side arming switch, from which the launcher key is *derived*** — derivation, not duplication, the same principle as the filename carrying the consumer. `wire_contract` is recorded as an unconditional constant (the legacy leg retired in **#149**) and `algorithm` as framework plumbing that stays put, so all eight keys now have a stated home. **The residual risk this entry tracks:** until `deliveries/` is built, the two switches co-exist and can disagree. Cross-refs: **C-110** (the switch exists only in an uncommitted tree), **C-63**, **C-126**. |

### C-130 — The maturity migration leaves ten sources with no destination value

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Executing the `deployment_status` → `maturity` rename across the fleet (ADR-017 §11 Phase 2). |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P4 |
| **Status** | Open |
| **Location** | the 6 sources declaring `baseline`; `models/cool_cat`, `models/teenage_dirtbag`, `models/test_model`, `ensembles/test_ensemble` (no `config_deployment.py`) |
| **Notes** | ADR-017 correctly holds that `baseline` is a **role**, not a maturity, and that it leaves `config_deployment.py` entirely — but every source still needs *some* maturity, and the migration table originally sent `baseline` to *(nothing)*. Six real sources carry it, and a research assistant migrating the fleet would have had no value to write. A further four source directories carry no `config_deployment.py` at all, so they have no `deployment_status` to migrate *from*. **Both now resolved in ADR-017's migration table** (`baseline` → `candidate`, role preserved where it already lives; the four unconfigured sources named explicitly). Entry retained because the migration is not yet executed and the table is the only thing that makes it mechanical. Cross-refs: **C-127**. |

### C-131 — "Production-tier consumer" reads as a discriminating condition while `tier` has one value

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A reader deriving `is_in_production` from ADR-017 §4e without also reading ADR-019 §3; or the addition of a second `tier` value. |
| **Source** | falsify audit of the four delivery documents (2026-08-04), probe P2 |
| **Status** | Open |
| **Location** | `docs/ADRs/017_source_composition_delivery.md` §4e (the ⟺ definition) and §5 (the tier rule) vs `docs/ADRs/019_delivery_declaration.md` §3 (`tier`) |
| **Notes** | §4e defines *in production ⟺ maturity is `graduate` **and** a delivery ships it to a **production-tier** consumer.* ADR-019 §3 fixes `tier` at exactly one value, `prod`, and says plainly that the gate is therefore *currently unconditional*. §4e did not, so the definition read as two independent conditions when one is presently always true. Not a correctness defect — the definition stays true as written, and becomes discriminating the moment a second tier value exists — but it is the kind of gap that makes a newcomer believe a check exists that does not. **Corrected by a clause in §4e pointing at ADR-019 §3.** The second tier value is itself blocked on ADR-017 §12's open shadow-destination question. Cross-refs: **C-129**. |


## Disagreements

### D-01 — Intentional config duplication vs. DRY principle

| Field | Value |
|---|---|
| **Trigger** | Partition boundary update requires editing 73 files atomically |
| **Source** | expert-code-review (Martin vs. Ousterhout/Hickey) |
| **Status** | **Resolved (closed during review-rr, 2026-07-31)** — the resolution was not just chosen, it was **built** |
| **Notes** | Martin (Clean Code) considers 73 identical files a DRY violation creating coordination nightmares. Ousterhout (Complexity) and Hickey (Simplicity) support the duplication because it eliminates shared-state reasoning and keeps each model self-contained. Resolution: the duplication is load-bearing; build a migration tool rather than centralizing. Related to C-01. **Closed 2026-07-31:** the disagreement is settled in practice and in code. `meta/partitions.json` is the single source of truth and `tools/partitions/bump.py` is the migration tool the resolution called for — 37 unit tests, 3 falsification rounds, invariant validation, atomic writes, a JSONL lockfile — and C-01 is correspondingly de-tiered (1 → 3) in this same pass. The Ousterhout/Hickey position is now a **standing repo convention**: model config files stay self-contained and are NOT centralized; centralizing `config_partitions.py` is explicitly out of bounds. Nothing remains in tension — keeping this Open implied a live decision that was made and executed 3+ months ago. Reopen only if someone proposes centralizing configs again, in which case this entry is the prior. |

---

### D-02 — Hardcoded algorithm-to-package mapping vs. factory pattern

| Field | Value |
|---|---|
| **Trigger** | A new algorithm is added and the test mapping must be manually updated |
| **Source** | expert-code-review (GoF vs. Beck/Hickey) |
| **Status** | Open (deferred — closure trigger added 2026-07-31) |
| **Notes** | Gang of Four would prefer a factory in `views_pipeline_core` that maps algorithm→manager, eliminating the need for `ALGORITHM_TO_PACKAGE` in `test_algorithm_coherence.py`. Beck accepts the mapping as pragmatic (test failure = correct signal). Hickey prefers data (dict) over abstraction (factory). Resolution: correct for this repo's scope; factory is a cross-repo decision for `views_pipeline_core`. **Closure trigger added during review-rr (2026-07-31)** — the resolution deferred to another repo without saying when to look again, which is how a deferral becomes a permanent Open. **Revisit when either:** (a) `views_pipeline_core` ships an algorithm→manager registry or factory (at which point delete `ALGORITHM_TO_PACKAGE` and close as GoF-resolved), **or** (b) `ALGORITHM_TO_PACKAGE` needs a fourth manual edit in one release cycle (at which point the maintenance cost has outgrown Beck's pragmatism and views-models should ask pipeline-core for the registry). Until one of those fires, the hardcoded dict stands and this entry needs no attention. |

---

### D-03 — `config_queryset.py` dependency exception: essential or architectural violation

| Field | Value |
|---|---|
| **Trigger** | Decision to refactor config loading or extend test coverage to querysets |
| **Source** | expert-code-review (Martin vs. Kleppmann vs. Ousterhout) |
| **Status** | Open (resolution is theoretical — closure tied to C-02 on 2026-07-31) |
| **Notes** | Martin considers `config_queryset.py`'s external dependencies an architectural boundary violation — configs should be pure. Kleppmann notes it's where data correctness is defined and can't be simplified away. Ousterhout acknowledges the mental tax but accepts it as irreducible complexity. Resolution: the dependency is essential (querysets require the `viewser` DSL). The gap is in testing — AST-based validation of column structure could create a testable seam without requiring external packages. Related to C-02, C-06. **Honest status, review-rr (2026-07-31):** the *dependency* half is genuinely settled (Kleppmann/Ousterhout prevailed; C-06 is `Accepted` under ADR-002). The *testing* half is *theoretical* — the AST seam this resolution names has been the proposed exit for **C-02 since April 2026** and nobody has started it; C-02 is still Open and is a member of **Cluster A**. **Closure now tied to C-02:** this entry closes when C-02's minimum-viable test lands (`generate()` exists, returns the declared type, datafactory descriptors carry `source`/`zarr_url`/`features`) — or, if that seam is judged not worth building, it closes as *"testing gap accepted, no AST seam"*, which is a legitimate answer but must be stated rather than left implied by inaction. |

---

### D-04 — Static analysis tests vs. behavioral execution tests

| Field | Value |
|---|---|
| **Trigger** | A model passes all pytest structural tests but fails at runtime |
| **Source** | test-review (Beck vs. Nygard) |
| **Status** | **Subsumed by C-106 (review-rr, 2026-07-31)** |
| **Notes** | The test suite is almost entirely static analysis (AST parsing, importlib loading, regex extraction). Beck notes this gives exceptional speed (1.41s for 2374 tests) and clean behavioral contracts. Nygard counters that the gap between "structure is correct" and "system works" is wide and uncovered — no `main.py` is ever executed, no training pipeline is ever triggered. The suite validates the blueprint but never builds the house. Related to C-03, C-15. **Subsumed 2026-07-31:** C-106 ("STRATEGIC ROOT: the test architecture verifies declarations exhaustively but runtime behavior nowhere in CI — the config-vs-behavior gap") states the identical finding, carries the same Beck/Nygard framing, names the cluster of ~10 entries that are its instances, **and has a partially-built exit** (`tests/test_runtime_smoke.py` + `runtime_smoke.yml`, PR #272 — 21 baseline models executed end-to-end at PR time). This is a resolved tension, not a live disagreement: Nygard's position prevailed and work started on it. Tracking it in two places split the evidence. Following the C-108 → Appwrite Seam Contract precedent, ownership moves to C-106; this entry is retained as the historical record of where the finding was first named. Do not fix from here — fix from C-106, **Cluster A**. |

---

### D-05 — Is the 131-files-to-11-environments mismatch a defect, or a resource necessity?

| Field | Value |
|---|---|
| **Trigger** | A proposal to give each model its own environment, or to reduce the number of `requirements.txt` |
| **Source** | expert-code-review (2026-08-02; Ousterhout/Kleppmann vs. Nygard) |
| **Status** | Open |
| **Notes** | **Ousterhout and Kleppmann:** the root defect. 131 declaration points imply 131 configuration points; there are 11, so the interface lies, provenance is unrecoverable, and a reader must know three facts that are not in the file they are reading. **Nygard dissents on cost:** `envs/views-baseline` is 9.0G and only 3 of 11 environments exist on the maintainer's laptop; 131 environments would be 100-200G per machine, on laptops, for a team with no ops engineer. Sharing is the only thing that fits the hardware. **Provisional resolution:** both hold, and the fix is neither more environments nor fewer files — the environment count stays at 11 and what changes is that its contents become a declared, committed artifact (**C-116**, **C-117**). Recorded rather than settled because the resolution has not been built. |

---

### D-06 — Does a repo-wide hygiene test that starts by accepting today's exceptions have value, or is it governance theatre?

| Field | Value |
|---|---|
| **Trigger** | Writing `tests/test_requirements_hygiene.py`, or any repo-wide invariant test over the 131 `requirements.txt` |
| **Source** | expert-code-review (2026-08-02; Hickey vs. Beck/Feathers) |
| **Status** | Open |
| **Notes** | **Hickey:** an allowlist of accepted exceptions is a place to hide, and its length is the metric — a test that begins by blessing the mess has inverted its own purpose. **Beck and Feathers:** the objection is about *size*, and size is a choice of ordering. Fix the one unparseable specifier (#316), then the three missing trailing newlines, then the 27 unbounded ceilings (**C-118**) — each rule is red for a real reason, gets fixed, and goes green with no baseline at all. Only the divergent-spec rule needs a recorded exception, and after that sequence it holds one entry: the r2darts split (**C-115**), with its reason. **Martin adds** the deciding criterion: an exception carrying a written reason is documentation; an exception without one is theatre. **Provisional resolution:** build in Beck's order and the disagreement does not arise; revisit if the exception list ever exceeds two entries. |

---

### D-07 — Is the delivery defect structural, or observational?

| Field | Value |
|---|---|
| **Trigger** | Choosing between a declared composition (structure) and a freshness assertion (observation) as the next change |
| **Source** | expert-code-review (2026-08-04; Nygard/Beck vs Martin/Kleppmann/Ousterhout) |
| **Status** | Open |
| **Notes** | **Nygard and Beck:** the incident that cost 145 days was not a wrong order — it was a delivery step that never ran, and which would have republished March data without objecting. A correctly ordered manifest would not have delivered anything either. So the assertion (**C-121**) addresses the failure that happened and the structure (**C-122**) does not. **Martin, Kleppmann and Ousterhout:** an unrepresented causal dependency in a partner-facing pipeline is a defect whether or not it has fired, and leaving it invites the next silent failure. **Provisional resolution:** both, in that order — the assertion now, the structure behind C-122's named trigger. Recorded because the ordering is the actual decision, and it is easy to reverse it in the name of tidiness. |

---

### D-08 — Does WET-before-DRY forbid a declared composition?

| Field | Value |
|---|---|
| **Trigger** | Proposing a composition manifest, a dependency resolver, or moving composition into views-pipeline-core |
| **Source** | expert-code-review (2026-08-04; Hickey vs Martin/Beck) |
| **Status** | Open |
| **Notes** | **Hickey:** views-models has exactly **one** composition (`monthly_run.sh`). Abstracting at n=1 is precisely the wrong-abstraction risk the rule exists to prevent, and a manifest that acquires conditionals has become a program. **Martin and Beck:** the second instance already exists one repo away — `crafd/` was cloned from `unfao/` per `docs/CLONING.md`, and views-postprocessing #211 is the recorded scar of that clone (*"every partner-scoped guard was scoped to ONE partner"*). **Provisional resolution:** the trigger has fired for views-postprocessing, **not** for views-models. Defer, behind C-122's named trigger. When it fires, copy views-postprocessing's remedy — a declared list asserted against the filesystem — not a framework. Unanimous against a dependency resolver and against moving composition into pipeline-core. |

---

### D-09 — Should the `REQUIRE` block be mandatory, or is it ceremony?

| Field | Value |
|---|---|
| **Trigger** | Writing a delivery file that has nothing to assert |
| **Source** | expert-code-review (2026-08-04; Martin/Ousterhout vs Nygard/Kleppmann) |
| **Status** | Open |
| **Notes** | **Martin and Ousterhout:** make the block optional. A `REQUIRE` holding one line will read as boilerplate, and the first person who deletes an empty one teaches everyone else to delete theirs. A block that is always present stops carrying information. **Nygard and Kleppmann:** make it mandatory — `max_age` and `reconciled` are exactly the assertions whose absence caused real failures, and optional safety is not safety. **Provisional resolution (written into ADR-017 §5):** the *block* is optional; specific *rules* are conditional on the delivery's shape — `reconciled` is required with two or more sources, `max_age` is required when `intent = live()`. Requirement follows from what the delivery is, not from ceremony. Revisit if a delivery file appears with an empty `REQUIRE`. |
