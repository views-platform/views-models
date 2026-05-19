
# ADR-004: Rules for Evolution and Stability

**Status:** Accepted  
**Date:** 2026-04-05  
**Deciders:** Project maintainers  
**Informed:** All contributors  

---

## Context

The preceding ADRs establish:

- **ADR-001:** the ontology of the repository (what exists)
- **ADR-002:** the topology of the repository (how components may relate)
- **ADR-003:** semantic authority (who owns meaning and how it is declared)

Together, these decisions define the system's structure and semantics at a point in time.

What they do **not** yet define is how the system is allowed to **change over time**:
- which components are expected to be stable
- which components may evolve freely
- what constitutes a breaking change
- when compatibility guarantees apply
- when a new ADR is required

In views-models, these questions are now concrete:

- 68 models and 5 ensembles share identical partition boundaries across 73 files; a boundary change is a coordinated multi-file update (Risk R1).
- External consumers (the VIEWS platform, UN FAO API) depend on `white_mustang` ensemble output; breaking changes have real downstream cost.
- The config key vocabulary (`regression_targets`, `prediction_format`, `rolling_origin_stride`) is enforced by tests; adding or renaming required keys is a breaking change to all 68+ models.
- Contributors regularly express uncertainty about what is safe to change (hyperparameters: freely; partition boundaries: never without coordination).

Multiple trigger conditions from the original deferred ADR-004 template are now met.

---

## Decision

The repository adopts a three-tier stability classification for its components:

### Tier 1 — Stable (change requires ADR or explicit team decision)

| Component | Examples | Rationale |
|---|---|---|
| Partition boundaries | `(121, 444)`, `(445, 492)`, `(493, 540)` | Cross-model comparability depends on identical splits |
| Required config keys | `name`, `algorithm`, `level`, `steps`, `time_steps`, `deployment_status` | Enforced by `test_config_completeness.py`; adding/removing breaks all models |
| Config file set | The 6 config files per model | Enforced by `test_model_structure.py`; scaffold builder generates this set |
| CLI argument contract | `-r`, `-t`, `-e`, `-f`, `--sweep` | All `run.sh` and integration tests depend on this interface |
| Deployment status vocabulary | `shadow`, `deployed`, `baseline`, `deprecated` | Enforced by test; production gating depends on it |

### Tier 2 — Conventional (change requires updating all models + tests)

| Component | Examples | Rationale |
|---|---|---|
| Model naming convention | `adjective_noun` lowercase | Enforced by `test_model_structure.py`; catalog scripts depend on pattern |
| Directory structure | `configs/`, `artifacts/`, `data/`, `main.py`, `run.sh` | Enforced by tests; scaffold builder generates this layout |
| CLI import pattern | `from views_pipeline_core.cli import ForecastingModelArgs` | Enforced by `test_cli_pattern.py` |
| Ensemble dependency declarations | `config_meta["models"]` list | Enforced by `test_ensemble_configs.py` |

### Tier 3 — Volatile (changed freely by model owners)

| Component | Examples | Rationale |
|---|---|---|
| Hyperparameters | All keys in `config_hyperparameters.py` beyond `steps`/`time_steps` | Algorithm-specific; model owner's domain |
| Querysets | Feature selection and transformation chains in `config_queryset.py` | Model owner's domain |
| W&B experiment tracking | Run names, tags, logging frequency | Operational convenience |
| Model-specific README content | Beyond scaffold-generated sections | Documentation convenience |

---

## Rationale

The three-tier model makes the cost of change explicit:

- **Stable** components have high coordination cost and downstream impact. Changes require an ADR or explicit team decision, plus updates to all affected models and tests.
- **Conventional** components have moderate coordination cost. Changes propagate across the model zoo but don't affect external consumers.
- **Volatile** components are model-local. No coordination required.

This classification reflects the existing reality (tests already enforce Stable and Conventional tiers) while making the rules discoverable for contributors.

---

## Consequences

### Positive
- Contributors can immediately determine whether a change is safe to make unilaterally
- The cost of adding new required config keys is made explicit before the change is attempted
- Partition boundary changes are recognized as architectural events, not routine updates

### Negative
- Stable components resist change even when change is desirable — the coordination cost is real
- The 73-file partition duplication (intentional per ADR-002) amplifies the cost of Stable-tier changes
- Model owners may be tempted to treat Conventional components as Volatile; tests are the enforcement mechanism

---

## Implementation Notes

- Stability tiers are enforced primarily by the test suite, not by tooling
- The integration test runner (`run_integration_tests.sh`) provides behavioral verification but is not in CI; Stable-tier changes should include an integration test run
- ADR-001 already defines a stability classification consistent with these tiers; this ADR makes the rules actionable

---

## Open Questions

- Should partition boundary changes require a formal migration tool (updating all 73 files atomically)?
- Should there be a deprecation protocol for removing models (currently only `electric_relaxation` is deprecated)?
- Should Tier 2 changes require a PR review from a specific set of maintainers?

---

## References

- [ADR-001](001_ontology.md) — Ontology stability levels
- [ADR-002](002_topology.md) — Self-contained config files (why duplication is intentional)
- [ADR-003](003_authority.md) — Authority of declarations
- [ADR-005](005_testing.md) — Testing enforces tiers
- [ADR-009](009_boundary_contracts.md) — Boundary contracts define the Stable-tier interface
