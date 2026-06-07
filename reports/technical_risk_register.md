# Technical Risk Register — views-models

**Last updated:** 2026-06-06  
**Governing ADR:** [ADR-010](../docs/ADRs/010_technical_risk_register.md)  
**Total entries:** 64 (60 concerns + 4 disagreements)  
**Concerns:** Open 22 | Mitigated 12 | Resolved 21 | Accepted 3 | Partially Resolved 1 | New 1  
**Disagreements:** Open 4  

---

## Open Concerns

### C-01 — Partition boundary updates require atomic edits to 73 files

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A decision is made to change calibration, validation, or forecasting partition boundaries |
| **Source** | repo-assimilation |
| **Status** | Mitigated |
| **Notes** | `meta/partitions.json` is the single source of truth. `tools/partitions/bump.py` (replaces deleted `scripts/update_partitions.py`) rewrites all 100 files with invariant validation, temporal plausibility (val test end ≤ Dec previous year), post-write verification, atomic writes, and JSONL lockfile with git state. `test_config_partitions.py` enforces consistency via shared parser from `tools.partitions.fileops`. Override mechanism (`# PARTITION_OVERRIDE:`) permits declared deviations — see C-56 for staleness risk. **2026-06-06:** ADR-011 migration procedure still references the deleted `scripts/update_partitions.py` — must be updated to reference `python -m tools.partitions.bump`. See ADR-011. |

---

### C-02 — No static validation of queryset correctness

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A VIEWS database column is renamed or removed, or a queryset references a non-existent column |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `config_queryset.py` is the most complex config file (up to 734 lines) with zero test coverage. Failures are runtime-only (data fetch phase). Validation would require access to the VIEWS database schema. **2026-04-22 (test-review):** This gap was the root cause of the bright_starship `dict.publish()` crash — `generate()` returns a plain dict for datafactory models but no test validates return type or shape. Minimum viable test: verify `generate()` exists, returns correct type, and that datafactory descriptors contain required keys (`source`, `zarr_url`, `features`). See C-40 (return type contract mismatch). |

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
| **Notes** | `test_algorithm_coherence.py::TestRequirementsCoherence` validates that `requirements.txt` package name (normalized hyphens to underscores) matches the package imported in `main.py`. |

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
| **Trigger** | A constituent model produces NaN, Inf, or wildly off-scale predictions; ensemble silently aggregates or propagates them |
| **Source** | expert-code-review (Nygard, Kleppmann) |
| **Status** | Open |
| **Notes** | `white_mustang` (deployed ensemble) aggregates via median. No NaN/Inf check or range validation occurs before aggregation. If multiple constituent models produce garbage, the ensemble output degrades silently. Downstream consumers (UN FAO API) receive degraded data. |

---

### C-14 — Concurrent model training can silently overwrite artifacts

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | Two training runs for the same model execute simultaneously, writing to the same `artifacts/` directory |
| **Source** | expert-code-review (Kleppmann) |
| **Status** | Open |
| **Notes** | Artifacts have no run ID or timestamp in filenames. The second writer silently overwrites the first. Low probability but destroys reproducibility when it occurs. W&B logs exist but are not cross-referenced with artifact files. |

---

### C-15 — Zero CIC failure mode test coverage

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | Any CIC-documented failure mode occurs in production and the system does not behave as declared |
| **Source** | test-review (Nygard) |
| **Status** | Mitigated |
| **Notes** | `test_failure_modes.py` expanded from 4 to 9 tests (2026-04-06). New tests cover: empty config files, import errors, runtime errors, integration test runner exit codes. Remaining gap: no tests for scaffold builder `FileExistsError`, no tests for ensemble aggregation failure. 9 of 21 CIC failure modes now covered. **2026-05-20 (test expansion):** `test_failure_modes.py` expanded to ~30 tests with new red-team classes: `TestPartitionBoundaryValidation` (steps=0/−1/default across all models), `TestEnsembleConstituentIntegrity` (config loadability, partition alignment, malformed model lists), `TestMalformedQuerysetDescriptor` (missing keys, None return, circular import). Scaffold builder `FileNotFoundError` now tested in `test_scaffold_builders.py`. Estimated 15 of 21 CIC failure modes covered. |

---

### C-16 — Zero direct unit tests for any CIC class

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A refactor of `build_model_scaffold.py`, `create_catalogs.py`, or any other CIC class introduces a regression |
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

### C-22 — No idempotency guarantee in model training artifacts

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A model is re-trained and previous artifacts are silently overwritten without versioning |
| **Source** | expert-code-review (Kleppmann) |
| **Status** | Open |
| **Notes** | Models write artifacts to `artifacts/` and W&B logs to `wandb/`. Re-running overwrites previous artifacts without versioning or deduplication. `force_reset: true` in DARTS hyperparameters acknowledges this but doesn't solve it. Related to C-14 (concurrent overwrite) but also applies to sequential re-runs. |

---

### C-23 — Test suite is overwhelmingly beige; red coverage is low

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A failure mode occurs that convention/structure tests cannot detect |
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
| **Notes** | **v1 test (2026-04-11 morning):** `TestModelDirectoryStructure` added to `tests/test_model_structure.py`. The class uses the existing `model_dir` fixture (`tests/conftest.py:72`, parametrized over `ALL_MODEL_DIRS`) and asserted every model contained four runtime-critical subdirectories: `artifacts/`, `data/raw/`, `data/generated/`, `logs/`. **Regressed (2026-04-11 evening):** the v1 test had two structural gaps that let C-32 recur unnoticed. (1) `REQUIRED_SUBDIRS` omitted `notebooks/` and `reports/` even though `ModelPathManager._initialize_directories` validates both at runtime (`views-pipeline-core/.../model_path.py:442,458`); a model missing either directory would pass the test and crash on first instantiation. (2) The check used `path.is_dir()` on the local filesystem, so any developer who had ever run a model locally would see the test pass regardless of whether the directory was tracked in git — the exact failure mode C-33 was meant to prevent (fresh-clone absence). C-32's `/home/simmaa/` recurrence was a direct consequence: `invisible_string` passed C-33 locally but had no tracked `notebooks/.gitkeep`. **v2 test (commit `cd668ea`):** `REQUIRED_SUBDIRS` extended to the full set `[artifacts, data/raw, data/generated, data/processed, logs, notebooks, reports]` — parity with `ModelPathManager` runtime validation. The assertion replaced `path.is_dir()` with a `git ls-files` probe via a helper `_git_tracks_path()`, so "pass" means "tracked in the git index" — fresh-clone state, not working-tree state. Coverage now 74 models × 7 subdirs = 518 tracked-path assertions; full suite 3243 passing. See also C-32 (now re-mitigated with 37 backfilled .gitkeeps), C-07 (scaffold builder testing), C-16 (CIC class testing gaps). |

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

### C-34 — `--library` filter in `run_integration_tests.sh` silently excludes models lacking `requirements.txt`

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A user runs `bash run_integration_tests.sh --library baseline` and one of the eligible models is missing `requirements.txt` (or the file is unreadable); the model is excluded from the run with no warning |
| **Source** | code-review (2026-04-11) — discovered during C-26 fix in Sprint 2 |
| **Status** | Open |
| **Notes** | `run_integration_tests.sh:128-137` uses `if [ -f "$req_file" ] && grep -q "views-${FILTER_LIBRARY}" "$req_file"`. A missing or unreadable `requirements.txt` causes silent exclusion — the same class of bug C-26 had in the `--level` filter, but in the `--library` filter. C-26's Sprint 2 fix added the `CLASSIFICATION_ERRORS` fail-fast pattern (lines 109-153) for level classification only; the library filter was left untouched because it does not crash and the legitimate "model declares no matching library" case must remain a silent skip. The remaining gap: a model that lacks `requirements.txt` entirely cannot be distinguished from one that declares a different library. **Recommended fix:** when `requirements.txt` does not exist for a model in the candidate set, emit a `WARNING: cannot classify <model> by library: missing requirements.txt` to stderr and exclude it explicitly (don't fail fast — this is milder than C-26 because it doesn't indicate a broken file). After C-08 (requirements coherence test) and C-27 (rude_boy backfill), this gap is mostly future-protection — it would re-emerge if a new model is added without `requirements.txt` and `--library` filtering is used before C-08 catches the omission. See also C-26 (same pattern, resolved 2026-04-11), C-08 (requirements coherence — mitigated), C-27 (rude_boy `requirements.txt` — resolved). |

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
| **Tier** | 2 |
| **Trigger** | A developer runs `python main.py -r calibration` in `views-hydranet-env` (or any env with `views_hydranet` + `views_pipeline_core`) without `datafactory_query` installed, and `calibration_viewser_df.parquet` is not cached |
| **Source** | falsify (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/main.py:33` (`from configs.config_queryset import fetch_data`), `models/bright_starship/configs/config_queryset.py:115` (`from datafactory_query import load_dataset`), `models/shining_codex/main.py:27` (same pattern), `models/shining_codex/configs/config_queryset.py:90` (same pattern) |
| **Notes** | **Falsification audit F-1/F-2 chain.** `views-datafactory` (which provides `datafactory_query`) is declared in `requirements.txt` but not installed in `views-hydranet-env` — the only conda environment that has both `views_hydranet` and `views_pipeline_core`. When `_ensure_data()` encounters a cache miss, it imports `datafactory_query` at line 96 and crashes with `ModuleNotFoundError`. Two of three run_types (`validation`, `forecasting`) have cached parquets from a prior session, masking the missing dependency. `calibration` has no cache — the standard first run (`-r calibration -t -e`) fails immediately. The local `envs/views-hydranet` directory expected by `run.sh` also does not exist; `run.sh` would create it and install deps from `requirements.txt` (which includes the git+https datafactory dep), but that's a ~10 min bootstrap, not "ready to run." **Fix:** `conda run -n views-hydranet-env pip install 'views-datafactory @ git+https://github.com/views-platform/views-datafactory.git@development'`. See also C-06 (config_queryset external deps — accepted for viewser; this is the datafactory equivalent), C-37 (bright_starship partition deviation), C-40 (generate() contract mismatch). **Cross-repo:** views-pipeline-core C-51 (`get_data()` hardcodes viewser), C-52 (drift detection loss), C-53 (`use_saved` overload). |

---

### C-39 — All 70 `run.sh` scripts use `#!/bin/zsh` — will fail on Linux servers and CI

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Any `run.sh` is executed on a Linux server, Docker container, or CI runner where zsh is not installed (i.e., most deployment targets) |
| **Source** | review-diff (2026-04-21) |
| **Status** | Resolved |
| **Location** | `models/*/run.sh`, `ensembles/*/run.sh`, `apis/*/run.sh`, `extractors/*/run.sh`, `postprocessors/*/run.sh`, `models/execute_all.sh` (82 scripts total) |
| **Notes** | **Resolved (2026-04-21).** All 79 `#!/bin/zsh` shebangs changed to `#!/usr/bin/env bash`. `models/execute_all.sh` line 10 changed from `zsh "$script"` to `"$script"` (delegates to shebang). 35 missing trailing newlines and 23 missing executable permissions also fixed. `scripts/audit_shell_health.sh` added to verify: 82 scripts, 490 checks, CLEAN verdict. No zsh-specific syntax was found in any script — all were plain POSIX/bash. |

---

### C-40 — `generate()` return type contract mismatch — dict vs Queryset, no validation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new model migrates to views-datafactory and its `config_queryset.generate()` returns a dict descriptor; `views-pipeline-core` calls `.publish()` on it and crashes |
| **Source** | expert-code-review (2026-04-21) |
| **Status** | Open |
| **Location** | `models/bright_starship/configs/config_queryset.py` (returns dict), `models/shining_codex/configs/config_queryset.py` (returns dict), `views-pipeline-core/views_pipeline_core/data/model_path.py:691-692` (`get_queryset()` returns raw `generate()` output with no type checking) |
| **Notes** | Standard viewser models return a `Queryset` object from `generate()`. bright_starship and shining_codex (datafactory models) return a plain dict with `"source": "views-datafactory"`, `"zarr_url"`, `"features"` keys. `get_queryset()` in views-pipeline-core performs no type checking — it calls `generate()` and returns whatever it gets. Downstream, `_fetch_data_from_viewser()` calls `.publish()` on the result, crashing with `AttributeError: 'dict' object has no attribute 'publish'`. The contract between views-models (config producer) and views-pipeline-core (config consumer) is entirely implicit. **Phase 1 workaround:** `args.saved = True` in bright_starship's `main.py` routes around the viewser path. **Phase 2 fix (views-pipeline-core):** type dispatch in `get_data()` based on descriptor type + `generate()` return type validation in `get_queryset()`. **Cross-repo:** views-pipeline-core C-51 (root cause — `get_data()` hardcodes viewser), C-42 (missing ViewsDataLoader CIC). See also C-06 (config_queryset external deps), C-38 (datafactory_query not installed). |

---

### C-41 — shining_codex has no readiness tests

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A developer clones the repo and runs `python main.py -r calibration` for shining_codex without the `views-r2darts2` environment and `datafactory_query` installed |
| **Source** | tech-debt-cleanup (2026-04-21) |
| **Status** | Open |
| **Location** | `models/shining_codex/` (no `tests/` directory or test files) |
| **Notes** | bright_starship has readiness tests (`test_bright_starship_readiness.py`) that verify environment prerequisites (conda env, `datafactory_query`, `DartsForecastingModelManager` import) and config structural validity. shining_codex, cloned from bright_starship, has no equivalent tests. Without readiness tests, failures will surface only at runtime with opaque error messages (e.g., `ModuleNotFoundError` for `datafactory_query` or `views_r2darts2`). See C-38 (datafactory_query not installed), C-03 (integration tests manual-only). |

---

### C-42 — Synthetic models depend on unreleased `views-pipeline-core` branch

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | The `feature/hydranet_ensamble_africa_me` branch of views-pipeline-core changes its synthetic data API (pattern names, queryset descriptor keys, or `DataFrameEnsembleManager`/`PredictionFrameEnsembleManager` constructor) before merge, breaking synthetic models and ensembles |
| **Source** | pr-review (2026-05-20) |
| **Status** | Open |
| **Location** | `models/vertical_dream/configs/config_queryset.py`, `models/horizontal_dream/configs/config_queryset.py`, `models/diagonal_dream/configs/config_queryset.py`, `ensembles/synthetic_chorus/main.py`, `models/lucid_dream/configs/config_queryset.py`, `models/vivid_dream/configs/config_queryset.py`, `models/waking_dream/configs/config_queryset.py`, `ensembles/synthetic_chant/main.py` |
| **Notes** | PR #56 adds `vertical_dream`, `horizontal_dream`, `diagonal_dream`, and `synthetic_chorus` — all four depend on the `"source": "synthetic"` queryset descriptor and `DataFrameEnsembleManager`, which exist only on the `feature/hydranet_ensamble_africa_me` branch of `views-pipeline-core`. If that branch renames pattern values (e.g., `"vertical_stripe"` → `"v_stripe"`), changes required descriptor keys, or alters the `EnsembleManager` import path, the synthetic models will fail at data-load time with no structural test catching the mismatch — `test_model_structure.py` validates directory layout but not queryset descriptor validity against pipeline-core. This is the same class of cross-repo coupling as C-31 and C-38 but with a sharper trigger: the dependency is on an unreleased, in-flux branch rather than a released package. Risk resolves naturally once the pipeline-core branch merges and the API stabilizes. **2026-05-24 (PR #58):** Three additional PredictionFrame synthetic models (`lucid_dream`, `vivid_dream`, `waking_dream`) and one ensemble (`synthetic_chant`) added. These extend the dependency surface to `PredictionFrameEnsembleManager`, `ConflictologyModel`, and `MixtureBaseline` distributional outputs. All run successfully against `views-pipeline-core v2.3.0` — if that version is released, this risk may be resolved. **2026-05-26 (confirmed):** `envs/views_ensemble` created by ensemble `run.sh` installs `views-pipeline-core` from PyPI, which lacks `PredictionFrameEnsembleManager`. `synthetic_chant` ensemble failed with `ImportError: cannot import name 'PredictionFrameEnsembleManager'` until local editable install replaced the PyPI version. This confirms the trigger: any fresh clone or CI environment that creates `views_ensemble` from `requirements.txt` will fail for PredictionFrame ensembles. See also C-31 (upstream API breaks), C-38 (datafactory_query not installed), C-40 (generate() return type contract mismatch), C-50 (views-baseline version spec mismatch — same class of fresh-clone failure). |

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

### C-44 — Concat aggregation degrades ensemble CRPS when constituent model quality varies

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Building a concat ensemble where constituent models have heterogeneous performance on a specific target (e.g., one model's CRPS is 50%+ worse than others on that target) |
| **Source** | golden_hour calibration run (2026-05-25) |
| **Status** | Open |
| **Location** | `ensembles/golden_hour/configs/config_meta.py` (`aggregation: "concat"`), `views-pipeline-core` PredictionFrameEnsembleManager concat path |
| **Notes** | Observed 53% CRPS degradation on `lr_sb_best` vs best individual model (golden_hour: 0.233 vs purple_alien: 0.152). blue_stranger (0.223) contributed 64 poor-quality samples that diluted the 128 better samples from purple_alien and violet_visitor. Concat treats all posterior samples equally — no mechanism to down-weight poor contributors. For future ensembles, consider weighted aggregation or model selection for targets where constituent quality varies significantly. Models were uncalibrated so this finding may not hold after hyperparameter optimization. See also C-13 (no prediction quality validation before aggregation). |

---

### C-45 — Ensemble `-t` flag causes full retraining cascade when models are pre-trained

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | Running any PredictionFrameEnsembleManager or DataFrameEnsembleManager with `-t` when constituent models already have trained artifacts in their `artifacts/` directories |
| **Source** | golden_hour calibration run (2026-05-25) |
| **Status** | Open |
| **Location** | `views-pipeline-core` EnsembleManager train path (invokes constituent `run.sh` subprocesses) |
| **Notes** | Running `python main.py -r calibration -t -e` on a pre-trained ensemble causes: (1) retrain all constituent models via run.sh subprocess (~2h), (2) create new model artifacts with new timestamps, (3) discover no predictions exist for those new timestamps, (4) re-evaluate all constituent models via run.sh subprocess (~3h), (5) finally perform the actual aggregation (~30 min). This wasted ~6 hours on golden_hour. The correct command when models are already trained: `python main.py -r calibration -e --saved`. The `-t` flag on ensembles should either warn when artifacts already exist, or detect and reuse existing timestamps rather than creating new ones. See also C-22 (no idempotency guarantee in training artifacts). |

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
| **Notes** | The viewser trio (purple_alien, blue_stranger, violet_visitor) trains on `ged_*_best_sum_nokgi` — the summed, no-known-geographical-imprecision variant of UCDP fatality counts. The datafactory trio (bright_starship, bold_comet, blazing_meteor) trains on `ged_*_best` — the base variant. Despite different variable names, both deliver functionally identical values. **Resolved (2026-05-26):** Direct cell-by-cell comparison of cached training parquets (4,876,920 rows × 6 columns) showed 99.99% exact match for all three target variables: lr_sb_best (614 differing rows of 4.9M), lr_ns_best (138), lr_os_best (182). Correlations all >0.999. The `_sum_nokgi` suffix does not indicate a different aggregation — both sources deliver the same fatality sums per PRIO-GRID cell-month. The ~600 differing rows have small absolute differences reflecting timing differences in UCDP data ingestion. This concern is fully disproven as a source of prediction divergence. See `reports/parity_investigation_20260526.md` for full analysis. See also C-02 (queryset validation), C-40 (generate() contract mismatch). |

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
| **Tier** | 2 |
| **Trigger** | A developer clones the repo on a new machine and runs any baseline model's `run.sh`, which creates `envs/views-baseline` and fails at `pip install -r requirements.txt` because `views-baseline>=1.0.0,<2.0.0` has no matching distribution on PyPI |
| **Source** | synthetic ensemble run (2026-05-26) |
| **Status** | Open |
| **Location** | `models/lucid_dream/requirements.txt`, `models/vivid_dream/requirements.txt`, `models/waking_dream/requirements.txt`, `models/vertical_dream/requirements.txt`, `models/horizontal_dream/requirements.txt`, `models/diagonal_dream/requirements.txt`, `models/red_ranger/requirements.txt`, `models/green_ranger/requirements.txt`, `models/blue_ranger/requirements.txt`, `models/black_ranger/requirements.txt`, `models/pink_ranger/requirements.txt`, `models/yellow_ranger/requirements.txt`, `models/white_ranger/requirements.txt`, `models/light_strider/requirements.txt`, `models/heavy_strider/requirements.txt`, `models/average_cmbaseline/requirements.txt`, `models/average_pgmbaseline/requirements.txt`, `models/zero_cmbaseline/requirements.txt`, `models/zero_pgmbaseline/requirements.txt`, `models/locf_cmbaseline/requirements.txt`, `models/locf_pgmbaseline/requirements.txt` (21 models total) |
| **Notes** | All 21 baseline models declare `views-baseline>=1.0.0,<2.0.0` in `requirements.txt`. The `views-baseline` package is not published to PyPI at all — it is only available as a local editable install from `~/Documents/scripts/views_platform/views-baseline` at version `0.1.0`. On existing developer machines with the pre-existing `envs/views-baseline` env, the pip dry-run check succeeds because the package is already installed, and `run.sh` proceeds normally. On a fresh clone (new machine, CI, new contributor), `run.sh` creates the conda env, `pip install` fails with `No matching distribution found for views-baseline`, and the model crashes with `ModuleNotFoundError: No module named 'views_baseline'`. **Observed (2026-05-26):** All 6 synthetic model runs showed `ERROR: No matching distribution found for views-baseline<2.0.0,>=1.0.0` but succeeded because the env already had the local install. **Fix options:** (1) publish `views-baseline` to PyPI at version `>=1.0.0`, (2) change `requirements.txt` to use a git+https URL (matching the `views-datafactory` pattern in HydraNet models), (3) update `run.sh` to install from local path if available (but run.sh must not be modified — see feedback constraint). See also C-38 (same class: `datafactory_query` not installed), C-42 (same class: `views-pipeline-core` from PyPI lacks features), C-08 (requirements coherence). |

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
| **Trigger** | A developer adds heavy_freighter to a production ensemble's `config_meta.models` list without realizing it uses a global grid (360×720 vs regional 180×180), producing incompatible spatial dimensions |
| **Source** | tech-debt-cleanup (2026-06-02) |
| **Status** | Open |
| **Location** | `models/heavy_freighter/configs/config_hyperparameters.py` (`height: 360`, `width: 720` — global grid vs regional 180×180) |
| **Notes** | heavy_freighter uses global grid coverage (360×720) vs the regional Africa-ME grid (180×180) used by all ensemble-eligible models. Its training params (tobit, 200 lessons, 16 samples, scheduled sampling) now match the production models — only the grid differs. It is correctly excluded from golden_hour and stellar_horizon ensembles. The risk is that no directory convention, marker file, or test distinguishes global-grid models from regional models — the only signal is reading the config. Low severity because incompatible spatial dimensions would cause a shape mismatch error at ensemble aggregation time. |

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
| **Status** | Open |
| **Location** | `tools/partitions/fileops.py:extract_values()` and `rewrite_values()` — regex `"calibration":\s*\{(.*?)\}` matches first occurrence |
| **Notes** | The regex matches the first occurrence of `"calibration": {` in the file. If that's in a comment, docstring, or dead code, `extract_values` reads wrong values and `rewrite_values` modifies the wrong location. No current file triggers this, but a single comment addition would cause silent corruption. **Tier 2 justification:** silent data corruption — the tool reports success while leaving the actual partition values unchanged. |

---

### C-58 — `_load_canonical()` has no error handling for missing/corrupt partitions.json

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `meta/partitions.json` is deleted, moved, or edited with invalid JSON; the bump tool prints a raw Python traceback instead of a helpful error message |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Open |
| **Location** | `tools/partitions/bump.py:_load_canonical()` |
| **Notes** | The function is two lines: `open()` + `json.load()` with no try/except. Missing file → `FileNotFoundError`. Corrupt JSON → `JSONDecodeError`. Missing keys → `KeyError` from `PartitionBoundaries.from_json()`. For annual critical infrastructure run by a maintainer, a raw traceback is a robustness failure. |

---

### C-59 — `write_atomic()` does not clean up temp files on `os.replace()` failure

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | `os.replace()` fails during a bump (permission error, disk full) after the temp file has been written; orphaned `.tmp` files remain in config directories |
| **Source** | falsify: bump edge cases (2026-06-06) |
| **Status** | Open |
| **Location** | `tools/partitions/fileops.py:write_atomic()` |
| **Notes** | Creates `NamedTemporaryFile(delete=False)` and calls `os.replace()`. No try/finally to clean up the temp file if replace raises. A failed run touching 100 files could leave up to 100 orphaned `.tmp` files. Low probability in practice (os.replace rarely fails on same-filesystem renames) but easy to fix with try/except around os.replace. |

---

### C-60 — Repo root and scripts/ mix operational tooling, scaffolding, and investigations with no structural separation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor tries to understand the tooling layout and must read 8+ filenames at the root and 14+ files in scripts/ to distinguish operational tools from scaffold builders from investigation scripts |
| **Source** | falsify: tools organization (2026-06-07) |
| **Status** | Open |
| **Location** | Repo root (6 Python scripts, 2 shell scripts), `scripts/` (3 Python, 11 shell, 1 log file), `tools/` (partitions only) |
| **Notes** | Violates CCP (catalog scripts change together but aren't grouped), CRP (4 unrelated responsibilities in one directory), and screaming architecture (flat layout requires reading every filename). Fix: organize into `tools/catalogs/`, `tools/scaffold/`, `tools/partitions/`; move investigation scripts to `investigations/`; move root shell scripts to appropriate locations. See ADR-011, C-01. |

---

## Disagreements

### D-01 — Intentional config duplication vs. DRY principle

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Partition boundary update requires editing 73 files atomically |
| **Source** | expert-code-review (Martin vs. Ousterhout/Hickey) |
| **Status** | Open |
| **Notes** | Martin (Clean Code) considers 73 identical files a DRY violation creating coordination nightmares. Ousterhout (Complexity) and Hickey (Simplicity) support the duplication because it eliminates shared-state reasoning and keeps each model self-contained. Resolution: the duplication is load-bearing; build a migration tool rather than centralizing. Related to C-01. |

---

### D-02 — Hardcoded algorithm-to-package mapping vs. factory pattern

| Field | Value |
|---|---|
| **Tier** | 4 |
| **Trigger** | A new algorithm is added and the test mapping must be manually updated |
| **Source** | expert-code-review (GoF vs. Beck/Hickey) |
| **Status** | Open |
| **Notes** | Gang of Four would prefer a factory in `views_pipeline_core` that maps algorithm→manager, eliminating the need for `ALGORITHM_TO_PACKAGE` in `test_algorithm_coherence.py`. Beck accepts the mapping as pragmatic (test failure = correct signal). Hickey prefers data (dict) over abstraction (factory). Resolution: correct for this repo's scope; factory is a cross-repo decision for `views_pipeline_core`. |

---

### D-03 — `config_queryset.py` dependency exception: essential or architectural violation

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Decision to refactor config loading or extend test coverage to querysets |
| **Source** | expert-code-review (Martin vs. Kleppmann vs. Ousterhout) |
| **Status** | Open |
| **Notes** | Martin considers `config_queryset.py`'s external dependencies an architectural boundary violation — configs should be pure. Kleppmann notes it's where data correctness is defined and can't be simplified away. Ousterhout acknowledges the mental tax but accepts it as irreducible complexity. Resolution: the dependency is essential (querysets require the `viewser` DSL). The gap is in testing — AST-based validation of column structure could create a testable seam without requiring external packages. Related to C-02, C-06. |

---

### D-04 — Static analysis tests vs. behavioral execution tests

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A model passes all pytest structural tests but fails at runtime |
| **Source** | test-review (Beck vs. Nygard) |
| **Status** | Open |
| **Notes** | The test suite is almost entirely static analysis (AST parsing, importlib loading, regex extraction). Beck notes this gives exceptional speed (1.41s for 2374 tests) and clean behavioral contracts. Nygard counters that the gap between "structure is correct" and "system works" is wide and uncovered — no `main.py` is ever executed, no training pipeline is ever triggered. The suite validates the blueprint but never builds the house. Related to C-03, C-15. |
