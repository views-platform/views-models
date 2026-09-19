# Class Intent Contract: create_catalogs.extract_models()

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-05
**Related ADRs:** ADR-003, ADR-008, ADR-009

---

## 1. Purpose

> `extract_models()` loads metadata from a model's config files and produces a dictionary suitable for catalog/README generation. It is the boundary function between raw config files and documentation output.

Located in: `tools/catalogs/create_catalogs.py:extract_models()`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate config correctness (that's the test suite's job)
- Does **not** modify config files
- Does **not** run models or load data
- Does **not** handle ensemble-specific table formatting (that's `generate_ensemble_table()`'s job)

---

## 3. Responsibilities and Guarantees

- Loads `config_meta.py` via `importlib.util` and calls `get_meta_config()`
- Loads `config_modelset.py` via `importlib.util` and calls `get_modelset_config()` (ensembles)
- Loads the source's maturity file via `importlib.util`: `config_maturity.py` (`get_maturity_config()`, key `maturity`) when present, else the legacy `config_deployment.py` (`get_deployment_config()`, key `deployment_status`, translated by ADR-017 §3's table). A source carries exactly one of the two (ADR-017 Phase 2)
- Creates GitHub markdown links for querysets, hyperparameters, and model sets
- Extracts implementation date from git history via `subprocess`
- Returns a merged dictionary containing all catalog-relevant fields

---

## 4. Inputs and Assumptions

- Receives a `ModelPathManager` or `EnsemblePathManager` instance
- Assumes config files exist and define the expected functions
- Assumes config functions return dicts (no type enforcement)

---

## 5. Outputs and Side Effects

Returns a dict with keys from the meta config, a `maturity` key in ADR-017's vocabulary (`candidate` / `graduate` / `retired`, translated from the legacy file where needed), plus:
- `model_dir_path`: `Path` to the model/ensemble directory (used for name links in catalog tables)
- `queryset`: markdown link to config_queryset.py, `'N/A'` for baselines, or `'None'` if no queryset exists
- `data_source`: `viewser` / `datafactory` / `synthetic` / `none` / `unknown`, read from `config_queryset.py` by AST via `tools/catalogs/data_source.py` — the one reader the per-model README uses too (#474). `unknown` is reported, never guessed, when a file imports both clients or neither
- `hyperparameters`: markdown link to config_hyperparameters.py
- `implementation_date`: `YYYY-MM-DD` string from git history (falls back to `2026-01-01`)
- `modelset_link`: markdown link to config_modelset.py (ensembles only, when config_modelset.py exists)

No side effects beyond logging and subprocess calls to `git log`.

---

## 6. Failure Modes and Loudness

- If a config file has a syntax error, `importlib` raises `SyntaxError` — currently crashes the entire catalog run
- If `get_meta_config()`, `get_maturity_config()` or `get_deployment_config()` is missing from a file that exists, `AttributeError` is raised
- No per-model error isolation (known deviation — see ADR-008)

---

## 7. Boundaries and Interactions

- Depends on: `importlib.util`, `os`, `pathlib`, `subprocess`, `views_pipeline_core.managers.model.ModelPathManager`
- Called by: `tools/catalogs/create_catalogs.py` main block
- Feeds into: `generate_model_table()`, `generate_ensemble_table()`, `update_readme_with_tables()`

---

## 8. Examples of Correct Usage

```python
model_class = ModelPathManager("counting_stars", validate=True)
model_dict = extract_models(model_class)
# model_dict = {"name": "counting_stars", "algorithm": "XGBRegressor", ...}
```

---

## 9. Examples of Incorrect Usage

```python
# Wrong: calling with a path string instead of a PathManager
model_dict = extract_models("models/counting_stars")  # TypeError

# Wrong: expecting runtime validation of config values
# extract_models does not check if the maturity value is valid; an unknown legacy value translates to ''
```

---

## 10. Test Alignment

- `tests/test_catalogs.py::TestNoExecUsage` — validates this function uses importlib, not exec()
- `tests/test_catalogs.py::TestReplaceTableInSection` — validates downstream markdown generation (requires views_pipeline_core)
- `tests/test_catalogs.py::TestGenerateModelTable` — validates model table generation with correct headers and formatting
- `tests/test_data_source_catalog.py` — every branch of the `data_source` classifier on synthetic files; no model in the fleet is `unknown`; the fleet split pinned (77 viewser / 34 datafactory / 6 synthetic, changed on purpose when a model migrates)
- `tests/test_catalogs.py::TestGenerateEnsembleTable` — validates ensemble table has "Constituent Models" column and shows aggregation
- `tests/test_tooling_scripts.py::TestGenerateModelTable` — characterization tests for model table generator
- `tests/test_tooling_scripts.py::TestGenerateEnsembleTable` — characterization tests for ensemble table generator
- No direct test of `extract_models()` return value (requires views_pipeline_core)

---

## 11. Evolution Notes

- Should add per-model try/except to prevent one broken config from crashing the entire catalog run
- Consider extracting the importlib loading pattern into a shared utility (currently duplicated from `conftest.py:load_config_module()`)

---

## Known Deviations

- No per-model error isolation — one broken config crashes all catalog generation
- The function still lacks consistent per-model error handling

---

## End of Contract

This document defines the **intended meaning** of `tools/catalogs/create_catalogs.extract_models()`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
