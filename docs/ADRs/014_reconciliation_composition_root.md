# ADR-014: Reconciliation Composition Root

**Status:** Accepted
**Date:** 2026-06-26
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-002](002_topology.md) (Topology), [ADR-009](009_boundary_contracts.md) (Boundary Contracts), [ADR-013](013_regression_target_name_agnosticism.md) (config is the single source of truth); views-pipeline-core #194/#195 (Reconciler port), views-frames ADR-014 (geography is injected, never embedded)

---

## Context

views-pipeline-core converted ensemble reconciliation from a hardwired `from views_reporting.reconciliation import ReconciliationModule` into a **Dependency-Inversion seam**: `EnsembleManager(reconciler=None)` accepting a `Reconciler` **Protocol** (`views_pipeline_core.domain.reconciliation`), with a fail-loud `RECONCILER_NOT_INJECTED` if a `pgm_cm_point` run finds no injected reconciler. pipeline-core deliberately does **not** know the concrete reconciler — that is views-postprocessing's frames-native `ReconciliationModule(map_keys, map_vals)`, built with geography that the leaf never embeds (it is injected by the caller).

The **composition root** — the place that constructs the concrete and injects it — is **views-models**. But [ADR-002](002_topology.md) restricts `ensembles/*/main.py` to importing `views_pipeline_core` only, and forbids repo-internal imports in config files. Wiring a reconciler needs to import the concrete (`views_postprocessing`) and source geography (`viewser`/`views-datafactory`). This ADR sanctions exactly that, in a controlled way, and pins where the (irreducible) cross-repo wire is allowed to live.

## Decision

### 1. A single repo-internal composition layer: `reconciliation/`
Reconciliation wiring lives in one repo-internal package, `reconciliation/` (repo root), **one concept per file**:
- `country_mapping.py` — the `CountryMapping` value (`map_keys`, `map_vals`).
- `country_mapping_provider.py` — the `CountryMappingProvider` **port** (abstraction).
- `viewser_country_mapping_provider.py` — the VIEWS-`country_id` concrete (and, later, a datafactory `gaul0_code` concrete in its own file).
- `reconciler_factory.py` — `build_reconciler(...) -> Reconciler`.

This is the composition root's layer. It is **not** a config file and **not** a model; it is the one sanctioned place where concrete dependencies are wired.

### 2. The reconciling `ensembles/*/main.py` may import `reconciliation`
A reconciling ensemble's `main.py` (the entrypoint) **may** `from reconciliation import build_reconciler` and inject `reconciler=` into its manager. Because `run.sh` is immutable production infrastructure (it must not be modified) and the repo root is not guaranteed on `sys.path`, `main.py` bootstraps it explicitly:
```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from reconciliation import build_reconciler
```
This is allowed **only** in the entrypoint `main.py` of reconciling ensembles, **only** for the `reconciliation` package. Config files remain self-contained (ADR-002 §Self-Contained Config Files is unchanged).

### 3. The composition layer's allowed dependencies
The `reconciliation/` layer may depend on:
- `views_pipeline_core` — the `Reconciler` **port** (the abstraction it returns).
- `views_postprocessing` — the concrete `ReconciliationModule` — **confined to `reconciler_factory.py`** (the single file that knows the concrete; CCP/DIP).
- `viewser` (today) / `views-datafactory` (later) — the geography source, **confined to the provider files**, behind the `CountryMappingProvider` port.

The **only new cross-repo coupling** is views-models → `views_postprocessing`, in one file, typed as the port. A composition root must construct its concrete — this wire is irreducible, but it is single, explicit, and named.

### 4. The geography source is pluggable and config-derived
Country ids differ across data sources (viewser VIEWS `country_id` vs datafactory `gaul0_code`); during the viewser→datafactory migration both coexist. The source is therefore a **`CountryMappingProvider` port** with a concrete **selected per ensemble, derived from its data source** (ADR-013: derive from config, don't hardcode). Adding the datafactory provider is a pure one-file extension (OCP); reconciliation never re-wires when an ensemble migrates.

### 5. Dependency direction (ADP / SDP)
`ensembles/*/main.py` → `reconciliation/` → {`views_pipeline_core` port, `views_postprocessing` concrete, geography source}. No cycle. The ensemble (unstable) depends on the composition layer, which depends on stable abstractions (the port). Geography flowing from the **data layer** instead of views-models (zero geography coupling) is a future pipeline-core seam improvement, kept open by this design.

## Consequences

### Positive
- Reconciliation is wired correctly through the DIP port; `pgm_cm_point` runs no longer fail loud.
- The concrete-binding coupling is one file, behind the port — discoverable, testable (inject a fake), swappable.
- The viewser→datafactory migration cannot re-block reconciliation: the source is pluggable.

### Negative
- One new cross-repo dependency (views-models → views-postprocessing) — irreducible for a composition root; minimized to one file.
- `main.py` `sys.path` bootstrap is a small deliberate deviation, forced by run.sh immutability.

## Implementation Notes
- Only reconciling ensembles (`reconciliation: "pgm_cm_point"`) wire a reconciler; all others pass `reconciler=None` (CRP — not forced to depend on it).
- A guard test asserts every `pgm_cm_point` ensemble wires a reconciler (named, not accidental).
- Parity is the acceptance gate: reconciled output must reproduce the current VIEWS-`country_id` numbers (wrong grouping = silent corruption).

## References
- EPIC #172; stories #173–#180; tracking #182.
- [ADR-002](002_topology.md) (amended — composition-layer row), [ADR-013](013_regression_target_name_agnosticism.md), [ADR-006](006_intent_contracts.md) (CIC for the package).
- views-pipeline-core #194/#195 (port + seam); views-postprocessing `ReconciliationModule`; views-frames ADR-014 (injected geography).
