# Technical Risk Register — views-models

**Last updated:** 2026-04-11  
**Governing ADR:** [ADR-010](../docs/ADRs/010_technical_risk_register.md)  
**Total entries:** 39 (35 concerns + 4 disagreements)  
**Concerns:** Open 10 | Mitigated 9 | Resolved 13 | Accepted 3  
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
| **Notes** | `meta/partitions.json` is now the single source of truth. `scripts/update_partitions.py` rewrites all 73 files from it. `test_config_partitions.py` reads canonical values from the same source and covers models, ensembles, extractors, and postprocessors. Override mechanism (`# PARTITION_OVERRIDE:`) permits declared deviations. Full resolution would require `views_pipeline_core` to support centralized partition loading. See ADR-011. |

---

### C-02 — No static validation of queryset correctness

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A VIEWS database column is renamed or removed, or a queryset references a non-existent column |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | `config_queryset.py` is the most complex config file (up to 734 lines) with zero test coverage. Failures are runtime-only (data fetch phase). Validation would require access to the VIEWS database schema. |

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
| **Notes** | `test_failure_modes.py` expanded from 4 to 9 tests (2026-04-06). New tests cover: empty config files, import errors, runtime errors, integration test runner exit codes. Remaining gap: no tests for scaffold builder `FileExistsError`, no tests for ensemble aggregation failure. 9 of 21 CIC failure modes now covered. |

---

### C-16 — Zero direct unit tests for any CIC class

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | A refactor of `build_model_scaffold.py`, `create_catalogs.py`, or any other CIC class introduces a regression |
| **Source** | test-review (Beck, Feathers) |
| **Status** | Open |
| **Notes** | All 5 CIC-documented classes (`ModelScaffoldBuilder`, `EnsembleScaffoldBuilder`, `PackageScaffoldBuilder`, `CatalogExtractor`, `IntegrationTestRunner`) have zero direct unit tests. Tests validate their *outputs* (model directory structure) but never instantiate or exercise the classes. 33 CIC guarantees total, only 2 directly tested (6%), 6 indirectly tested (18%), 25 untested (76%). |

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
| **Notes** | Red coverage improved from 4 to 9 tests (2026-04-06). New tests cover config loading edge cases and integration test runner failure modes. Distribution still heavily beige (~64%) but red category is no longer negligible. Further improvement requires testing scaffold builder and ensemble aggregation failure modes. |

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
| **Status** | Open |
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
