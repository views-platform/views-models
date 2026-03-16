# Logging & Observability Standard

**Status:** Active
**Governing ADRs:** ADR-003 (Authority of Declarations), ADR-005 (Testing), ADR-008 (Observability)

---

## 1. Purpose

This document defines operational standards for logging behavior in views-models.

views-models is a thin orchestration layer — most ML logic lives in external packages.
Logging in this repo covers:
- Config loading and validation
- Model launcher initialization
- Scaffold building
- Catalog generation
- CI/CD pipeline output

---

## 2. Core Principles

### 2.1 Fail Loud and Persist

- Config loading failures must raise exceptions (not log and continue)
- `ModelPathManager` initialization failures must raise `RuntimeError` with context
- Missing config keys are caught by tests, not by runtime logging

### 2.2 Logs Must Support Understanding

Logs must:
- identify which model is being processed
- provide sufficient context to reconstruct errors
- include the operation stage (scaffold building, catalog generation, model run)

---

## 3. Log Levels

### ERROR
- Config file cannot be loaded (syntax error, missing function)
- Model directory creation fails
- Catalog generation encounters a broken config

### WARNING
- A model directory already exists during scaffolding
- A subdirectory already exists (skipped, not recreated)

### INFO
- Model directory created successfully
- Config script generated
- Catalog generation completed

### DEBUG
- Not used in current codebase

---

## 4. Current Patterns

### Scaffold builders (`build_model_scaffold.py`, `build_ensemble_scaffold.py`)
```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
Uses `logging.info()`, `logging.error()` for directory operations.

### Catalog generation (`create_catalogs.py`)
```python
logging.basicConfig(level=logging.ERROR, ...)
logger = logging.getLogger(__name__)
```
Uses `logging.info()` for config discovery.

### Model launchers (`models/*/main.py`)
No logging — wraps `ModelPathManager` in try/except and raises `RuntimeError`.
External package managers handle their own logging.

---

## 5. Known Deviations

- `create_catalogs.py` sets logging to `ERROR` level, which suppresses its own `INFO` messages about config discovery. This is effectively silent unless something breaks.
- Model launchers use `warnings.filterwarnings("ignore")` broadly — this suppresses all Python warnings, including potentially useful ML library warnings.
- No structured logging (JSON/key-value) is used anywhere.
- No alerting is configured for CI failures.

---

## 6. Anti-Patterns (Prohibited)

- Swallowing exceptions without logging
- Logging and continuing after an invariant violation (ADR-003)
- Using `print()` for structural diagnostics in tooling scripts
- Downgrading errors to warnings to "keep catalog generation running"

---

## 7. Evolution

If logging becomes more important (e.g., operational monitoring of monthly runs),
consider:
- Adding a `LoggingManager` wrapper (already available in `views_pipeline_core`)
- Structured logging for catalog generation
- CI notification on catalog generation failures
