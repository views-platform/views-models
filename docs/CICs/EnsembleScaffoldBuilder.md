# Class Intent Contract: EnsembleScaffoldBuilder

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-03-15
**Related ADRs:** ADR-001, ADR-002

---

## 1. Purpose

> `EnsembleScaffoldBuilder` creates and validates the directory structure and scripts for a new ensemble model. It inherits from `ModelScaffoldBuilder` and overrides script generation to use ensemble-specific templates.

Located in: `build_ensemble_scaffold.py`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** aggregate model predictions or perform reconciliation
- Does **not** validate that constituent models exist
- Does **not** generate `config_queryset.py` or `config_sweep.py` (ensembles don't use these)

---

## 3. Responsibilities and Guarantees

- Creates an ensemble directory at the path determined by `EnsemblePathManager`
- Inherits directory creation and assessment from `ModelScaffoldBuilder`
- Generates ensemble-specific scripts: `config_deployment.py`, `config_hyperparameters.py`, `config_meta.py`, `main.py`, `run.sh`, `requirements.txt`
- Validates name uniqueness across both models and ensembles

---

## 4. Inputs and Assumptions

- Ensemble name must pass `ModelPathManager.validate_model_name()` (same convention as models)
- Must not collide with existing model or ensemble names
- `views_pipeline_core` must be installed

---

## 5. Outputs and Side Effects

- Creates filesystem directory tree under `ensembles/{ensemble_name}/`
- Creates config and script files from ensemble templates

---

## 6. Failure Modes and Loudness

- Same as `ModelScaffoldBuilder` — inherits `FileExistsError` and `FileNotFoundError` behavior

---

## 7. Boundaries and Interactions

- Inherits from: `ModelScaffoldBuilder` (`build_model_scaffold.py`)
- Depends on: `views_pipeline_core.managers.ensemble.EnsemblePathManager`, `views_pipeline_core.templates.ensemble.*`
- Must not depend on: individual ensemble code, model code

---

## 8. Examples of Correct Usage

```python
builder = EnsembleScaffoldBuilder("happy_ensemble")
builder.build_model_directory()
builder.build_model_scripts()
builder.update_gitkeep_empty_directories()
```

---

## 9. Examples of Incorrect Usage

```python
# Wrong: using a model name that already exists
builder = EnsembleScaffoldBuilder("purple_alien")  # Fails — model already exists
```

---

## 10. Test Alignment

- No direct unit tests — relies on structural convention tests
- The output must conform to patterns validated by `tests/test_model_structure.py` (when extended to ensembles)

---

## 11. Evolution Notes

- Currently uses `self.requirements_path` from parent class, which is set during `build_model_directory()`. This coupling between directory creation and attribute state could be made more explicit.

---

## Known Deviations

- Inherits the `_model` attribute name from `ModelScaffoldBuilder` even though it represents an ensemble — naming is misleading

---

## End of Contract

This document defines the **intended meaning** of `EnsembleScaffoldBuilder`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
