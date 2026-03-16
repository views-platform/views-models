# ADR-001: Ontology of the Repository

**Status:** Accepted
**Date:** 2026-03-15
**Deciders:** Simon (project maintainer)
**Informed:** All contributors

---

## Context

views-models contains many different types of entities — model launchers, configuration files, ensembles, tooling scripts, shared infrastructure, and CI workflows. Without a clear taxonomy, contributors may place new code in the wrong location, conflate concerns, or introduce entities that don't fit the existing structure.

---

## Decision

The repository recognizes the following ontological categories:

### Domain Entities
| Category | Location | Description |
|----------|----------|-------------|
| **Models** | `models/*/` | Individual forecasting model launchers (~66). Each is a thin `main.py` + config directory that delegates to an external architecture package. |
| **Ensembles** | `ensembles/*/` | Ensemble aggregation launchers (5). Aggregate predictions from constituent models. |

### Configuration Entities
| Category | Location | Description |
|----------|----------|-------------|
| **Model Configs** | `models/*/configs/` | Six config files per model: `config_meta.py`, `config_deployment.py`, `config_hyperparameters.py`, `config_sweep.py`, `config_queryset.py`, `config_partitions.py` |
| **Ensemble Configs** | `ensembles/*/configs/` | Subset of config files per ensemble |

### Infrastructure Entities
| Category | Location | Description |
|----------|----------|-------------|
| **CI/CD** | `.github/workflows/` | Automated catalog generation |
| **APIs** | `apis/` | External API integrations (e.g., UN FAO) |

### Data Processing Entities
| Category | Location | Description |
|----------|----------|-------------|
| **Extractors** | `extractors/` | Data extraction modules (e.g., UCDP) |
| **Postprocessors** | `postprocessors/` | Output transformation modules |

### Tooling Entities
| Category | Location | Description |
|----------|----------|-------------|
| **Scaffolding** | `build_model_scaffold.py`, `build_ensemble_scaffold.py`, `build_package_scaffold.py` | Interactive CLI tools for creating new model/ensemble directories |
| **Catalog Generation** | `create_catalogs.py`, `update_readme.py`, `generate_features_catalog.py` | Scripts that generate documentation from configs |

### Testing Entities
| Category | Location | Description |
|----------|----------|-------------|
| **Tests** | `tests/` | Config completeness, structural conventions, CLI consistency, partition delegation, catalog safety |

### Stability Levels

| Level | Categories | Change Policy |
|-------|-----------|---------------|
| **Stable** | Config key requirements, partition boundaries | Changes require ADR or team discussion |
| **Conventional** | Config structure, CLI pattern, naming conventions | Changes require updating all models + tests |
| **Volatile** | Individual model hyperparameters, querysets | Changed freely by model owners |

---

## Consequences

### Positive
- Clear guidance on where new code belongs
- Stability levels prevent accidental changes to load-bearing conventions

### Negative
- Adding a new entity type requires updating this ADR

---

## References

- ADR-002 (Topology)
- `tests/test_model_structure.py` — enforces structural conventions
