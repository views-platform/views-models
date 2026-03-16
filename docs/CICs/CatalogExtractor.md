# Class Intent Contract: create_catalogs.extract_models()

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-03-15
**Related ADRs:** ADR-003, ADR-008, ADR-009

---

## 1. Purpose

> `extract_models()` loads metadata from a model's config files and produces a dictionary suitable for catalog/README generation. It is the boundary function between raw config files and documentation output.

Located in: `create_catalogs.py:extract_models()`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate config correctness (that's the test suite's job)
- Does **not** modify config files
- Does **not** run models or load data
- Does **not** handle ensemble-specific logic (uses same interface for both)

---

## 3. Responsibilities and Guarantees

- Loads `config_meta.py` via `importlib.util` and calls `get_meta_config()`
- Loads `config_deployment.py` via `importlib.util` and calls `get_deployment_config()`
- Creates GitHub markdown links for querysets and hyperparameters
- Returns a merged dictionary containing all catalog-relevant fields

---

## 4. Inputs and Assumptions

- Receives a `ModelPathManager` or `EnsemblePathManager` instance
- Assumes config files exist and define the expected functions
- Assumes config functions return dicts (no type enforcement)

---

## 5. Outputs and Side Effects

Returns a dict with keys from merged meta and deployment configs, plus:
- `queryset`: markdown link (or `'None'`)
- `hyperparameters`: markdown link to config_hyperparameters.py

No side effects beyond logging.

---

## 6. Failure Modes and Loudness

- If a config file has a syntax error, `importlib` raises `SyntaxError` — currently crashes the entire catalog run
- If `get_meta_config()` or `get_deployment_config()` is missing, `AttributeError` is raised
- No per-model error isolation (known deviation — see ADR-008)

---

## 7. Boundaries and Interactions

- Depends on: `importlib.util`, `os`, `pathlib`, `views_pipeline_core.managers.model.ModelPathManager`
- Called by: `create_catalogs.py` main block
- Feeds into: `generate_markdown_table()`, `update_readme_with_tables()`

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
# extract_models does not check if deployment_status is valid
```

---

## 10. Test Alignment

- `tests/test_catalogs.py::TestNoExecUsage` — validates this function uses importlib, not exec()
- `tests/test_catalogs.py::TestReplaceTableInSection` — validates downstream markdown generation (requires views_pipeline_core)
- No direct test of `extract_models()` return value (requires views_pipeline_core)

---

## 11. Evolution Notes

- Should add per-model try/except to prevent one broken config from crashing the entire catalog run
- Consider extracting the importlib loading pattern into a shared utility (currently duplicated from `conftest.py:load_config_module()`)

---

## Known Deviations

- No per-model error isolation — one broken config crashes all catalog generation
- The `tmp_dict` variable was a holdover from the `exec()` pattern and has been removed, but the function still lacks consistent error handling

---

## End of Contract

This document defines the **intended meaning** of `create_catalogs.extract_models()`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
