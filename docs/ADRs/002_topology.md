# ADR-002: Topology and Dependency Rules

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models is a configuration-and-orchestration layer. All ML logic lives in external packages (`views_pipeline_core`, `views_stepshifter`, `views_r2darts2`, `views_hydranet`, `views_baseline`). The repository's internal dependency structure must remain simple — models should not depend on each other, and shared infrastructure should be minimal.

---

## Decision

### Dependency Direction

```
External Packages (views_pipeline_core, views_stepshifter, etc.)
         ↑
    models/*/main.py  (each model imports ONE manager from ONE package)
         ↑
    models/*/configs/  (config files import from ingester3, viewser)
```

### Self-Contained Config Files

**Each model must have its own self-contained `config_partitions.py`.** The `views_pipeline_core` framework loads config files via `importlib` from each model's directory path. Config files cannot depend on repo-internal packages (like a hypothetical `common/` directory) because the repo root may not be on `sys.path` at runtime.

This means partition logic is duplicated across ~66 models. This duplication is intentional — it ensures each model can run independently. Tests (`tests/test_config_partitions.py`) detect drift between models.

### Allowed Dependencies

| From | May Depend On |
|------|--------------|
| `models/*/main.py` | `views_pipeline_core`, one algorithm package, `pathlib` |
| `models/*/configs/config_partitions.py` | `ingester3` only (for `ViewsMonth`) |
| `models/*/configs/config_queryset.py` | `viewser`, `views_pipeline_core` |
| `models/*/configs/config_*.py` (others) | Nothing (pure dict-returning functions) |
| `ensembles/*/main.py` | `views_pipeline_core` |
| Tooling scripts (root) | `views_pipeline_core`, `importlib`, standard library |
| `tests/` | `conftest.py` helpers, `importlib`, standard library |

### Forbidden Dependencies

- **No cross-model imports** — `models/A/` must never import from `models/B/`
- **No model → tooling imports** — models must not import from root-level scripts
- **No repo-internal imports in config files** — config files must only import from installed packages (`ingester3`, `viewser`), not from repo-local modules
- **No config files with side effects** — config files must be pure functions returning dicts (exception: `config_queryset.py` which builds `Queryset` objects)

---

## Known Deviations

- `config_queryset.py` files import from `viewser` and `views_pipeline_core`, making them impossible to load without these packages installed. This is an accepted deviation — querysets require the VIEWS data layer.

---

## Consequences

### Positive
- Models remain independently deployable
- Tests can run without ML dependencies
- Adding a new model requires no changes to existing models

### Negative
- Each model's `main.py` is a thin launcher with significant boilerplate duplication

---

## References

- ADR-001 (Ontology)
- `tests/test_cli_pattern.py` — enforces CLI import conventions
