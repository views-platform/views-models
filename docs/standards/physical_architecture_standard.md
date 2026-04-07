# Physical Architecture Standard

**Governing ADR:** ADR-002 (Topology and Dependency Rules)  
**Status:** Active (heuristic, not strict)  
**Last reviewed:** 2026-04-05  

---

## Scope

This standard defines the **default heuristic** for file and directory organization in views-models. It is the ruling convention in the absence of a good argument for locality. When locality provides a meaningful benefit (e.g., keeping a model's configs together with its entrypoint), deviation is acceptable and does not require an ADR.

---

## 1. The 1-Class-1-File Heuristic

**Every non-trivial class should live in its own file named after the class in `snake_case`.**

- **Preferred:** `PackageScaffoldBuilder` lives in `build_package_scaffold.py`
- **Acceptable deviation:** Module-level functions (e.g., `extract_models()` in `create_catalogs.py`) that are not classes — the 1-class-1-file rule does not apply to pure functions.
- **Exception:** Trivial data containers or closely related helpers may coexist in the same file.

### Current State in views-models

The repo is primarily composed of:
- **Config files:** Pure functions returning dicts (no classes) — standard does not apply
- **Model entrypoints:** `main.py` files with no class definitions — standard does not apply
- **Scaffold builders:** One class per file (`build_model_scaffold.py`, `build_ensemble_scaffold.py`, `build_package_scaffold.py`) — **compliant**
- **Catalog scripts:** Module-level functions — standard does not apply

The 1-class-1-file heuristic is naturally satisfied because most logic in this repository lives in external packages, not in local classes.

---

## 2. Directory Ontology

Files must be located in directories that match their **functional category**.

| Directory | Category | Contents |
|---|---|---|
| `models/` | Domain entities | Self-contained model directories (configs, entrypoint, artifacts) |
| `ensembles/` | Domain entities | Self-contained ensemble directories |
| `apis/` | Data pipeline | API connectors |
| `extractors/` | Data pipeline | Data extraction modules |
| `postprocessors/` | Data pipeline | Output post-processing |
| `tests/` | Verification | Pytest modules and conftest |
| `docs/` | Governance | ADRs, CICs, protocols, standards |
| `envs/` | Infrastructure | Conda environments (local, not tracked) |
| `.github/workflows/` | Infrastructure | CI/CD pipelines |
| `reports/` | Observability | Risk register, generated reports |
| `logs/` | Observability | Integration test logs |
| Root (`*.py`) | Tooling | Scaffold builders, catalog generators |

### Locality Exception

Model directories (`models/{name}/`) bundle configs, data, artifacts, logs, and entrypoints together. This violates strict ontological separation (configs aren't in a top-level `configs/` directory) but is the correct design — models are independently deployable units, and locality is the stronger organizing principle here. This is explicitly sanctioned by ADR-002.

---

## 3. Symmetrical Hubs

When heterogeneous logic of the same kind must be consolidated, use a symmetrical hub file.

### Current Hubs in views-models

This repo has minimal need for symmetrical hubs because most logic lives in external packages. The closest analogue is:
- `tests/conftest.py` — consolidates all test fixtures and parametrization logic

### When to Create a Hub

If this repository accumulates:
- Multiple custom exceptions → create `utils/exceptions.py`
- Multiple shared helpers → create `utils/helpers.py`
- Multiple shared data transforms → create `utils/transforms.py`

Until then, no hub files are needed.

---

## 4. Import Conventions

- **Explicit imports:** Avoid `from module import *`
- **No circular dependencies:** Follow ADR-002's dependency direction
- **Config files are import-isolated:** Config files (except `config_queryset.py`) must not import from other repo modules

These conventions are already enforced by ADR-002 and tested by `test_cli_pattern.py`.

---

## 5. Enforcement

This standard is a **heuristic, not a gate**. It is not enforced by CI or automated tooling.

- During code review: prefer the standard unless the contributor provides a good argument for locality
- During scaffold generation: the standard is embedded in `build_model_scaffold.py` templates
- During repo assimilation: note deviations as informational, not defects

The standard exists to prevent drift toward disorganization, not to reject legitimate locality-driven decisions.
