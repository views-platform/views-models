# Technical Risk Register — views-models

**Last updated:** 2026-04-06  
**Governing ADR:** [ADR-010](../docs/ADRs/010_technical_risk_register.md)  
**Total entries:** 30 (26 concerns + 4 disagreements)  
**Concerns:** Open 17 | Mitigated 4 | Resolved 3 | Accepted 3  
**Disagreements:** Open 4  

---

## Open Concerns

### C-01 — Partition boundary updates require atomic edits to 73 files

| Field | Value |
|---|---|
| **Tier** | 1 |
| **Trigger** | A decision is made to change calibration, validation, or forecasting partition boundaries |
| **Source** | repo-assimilation |
| **Status** | Open |
| **Notes** | ADR-002 mandates self-contained configs (intentional duplication). `test_config_partitions.py` detects drift. The coordination cost is the price of model independence. A migration tool could reduce risk but does not exist. |

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
| **Notes** | `run_integration_tests.sh` is the only mechanism testing actual model training/evaluation. It runs locally and takes hours. The CI pytest workflow (`run_tests.yml`) only runs fast structural tests. A model can be merged broken. |

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
| **Status** | Open |
| **Notes** | `test_darts_reproducibility.py` validates DARTS models against a canonical parameter list, but stepshifter (~20 models) and baseline (~6 models) have no equivalent check. A missing hyperparameter surfaces only at training time. |

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
| **Status** | Open |
| **Notes** | `build_model_scaffold.py` and `build_ensemble_scaffold.py` generate the initial structure for all new models. No test validates that scaffold output satisfies `test_model_structure.py` or `test_config_completeness.py`. |

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
| **Status** | Open |
| **Notes** | `create_catalogs.py` (237 lines), `update_readme.py` (281 lines), `generate_features_catalog.py` (117 lines) have near-zero test coverage. These are change-prone scripts modified when model metadata patterns change. `create_catalogs.py` only has `exec()` absence validated; core logic `extract_models()` is untested. |

---

### C-18 — `build_model_scaffold.py` I/O coupling prevents testability

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | Any attempt to write automated tests for `ModelScaffoldBuilder` or `EnsembleScaffoldBuilder` |
| **Source** | test-review (Beck), expert-code-review (Martin, Ousterhout) |
| **Status** | Open |
| **Notes** | `build_model_scripts()` uses `input()` directly for user prompts and makes network calls to GitHub API. No dependency injection seam exists. The class cannot be instantiated in a test without monkey-patching stdin. This is the root cause of C-07 (scaffold builder untested). |

---

### C-19 — `create_catalogs.py` has no transactional file write safety

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | `create_catalogs.py` crashes between reading and writing `README.md` |
| **Source** | test-review (Feathers), expert-code-review (Martin) |
| **Status** | Open |
| **Notes** | The `__main__` block reads README.md, modifies it in memory, and writes it back without transactional safety. If the script crashes mid-write, README could be corrupted. The CI workflow `update_catalogs.yml` auto-commits the result. Should write to temp file then atomically rename. |

---

### C-20 — No timeout or circuit breaker in data fetch path

| Field | Value |
|---|---|
| **Tier** | 2 |
| **Trigger** | The VIEWS database is slow or unreachable during model training |
| **Source** | expert-code-review (Nygard) |
| **Status** | Open |
| **Notes** | Models fetch data via `viewser.Queryset.publish()` with no timeout, retry limit, or fallback. A database outage hangs every model indefinitely during normal operation. The only timeout is the external `timeout` command in `run_integration_tests.sh` (1800s), which only applies during integration testing. |

---

### C-21 — Partition boundary semantics undocumented

| Field | Value |
|---|---|
| **Tier** | 3 |
| **Trigger** | A new contributor or auditor asks "why 121? why 444?" and finds no answer |
| **Source** | expert-code-review (Kleppmann) |
| **Status** | Mitigated |
| **Notes** | ViewsMonth-to-date mapping added as comment in `models/adolecent_slob/configs/config_partitions.py` and in `docs/ADRs/README.md` ADR-011 candidate description (2026-04-06). Full rationale for split points still pending as a future ADR. |

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
| **Status** | Open |
| **Notes** | The `--level` filter shells out to Python with `2>/dev/null`. If `config_meta.py` is broken, the model is silently excluded from the filtered set. A broken model escapes testing because the test runner can't classify it, and no one is notified. CIC documents this as a known deviation. |

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
