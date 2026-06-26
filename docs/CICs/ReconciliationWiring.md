# Class Intent Contract: `reconciliation` composition layer

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-26
**Related ADRs:** ADR-014 (Reconciliation Composition Root), ADR-002 (Topology), ADR-013 (config is the single source of truth), ADR-006 (Intent Contracts)

---

## 1. Purpose

Wire the reconciler Dependency-Inversion seam at the views-models composition root.
views-pipeline-core defines the `Reconciler` port and fails loud if a `pgm_cm_point`
run finds no injected reconciler; views-postprocessing provides the concrete
`ReconciliationModule(map_keys, map_vals)`. This layer is the single sanctioned
place that builds the geography and constructs the concrete, injected by reconciling
ensemble `main.py` files as `reconciler=`.

## 2. Non-Goals (Explicit Exclusions)

- It does **not** implement reconciliation math (that is views-postprocessing).
- It does **not** embed geography statically (it is built per run from the data source).
- It does **not** decide *whether* an ensemble reconciles (that is `config_meta.reconciliation`).
- It does **not** touch non-reconciling ensembles (they pass `reconciler=None`).

## 3. Responsibilities and Guarantees

- `CountryMapping` — immutable `(time, priogrid_gid) -> country_id` value with shape/dtype invariants.
- `CountryMappingProvider` (port) — `build() -> CountryMapping`; the stable abstraction.
- `ViewserCountryMappingProvider` — the VIEWS-`country_id` concrete; **parity-preserving** (matches the current path: `priogrid_month` `country_id` from `country_month`, `.first()` per grid).
- `build_reconciler(start, end, source="viewser", provider=None) -> Reconciler` — selects the provider, builds the mapping, constructs the concrete; the **only** file importing `views_postprocessing`.
- `build_reconciler_for_run(ensemble_dir) -> Reconciler` — sizes the window from the ensemble's partition config and builds. Called by `main.py`.

## 4. Inputs and Assumptions

- A forecast month window (derived from `config_partitions`; union of test ranges + buffer — a superset is safe).
- viewser available at composition time (the provider fetches the country mapping); pipeline-core seam + views-postprocessing dev-installed.

## 5. Outputs and Side Effects

- Returns an object satisfying the pipeline-core `Reconciler` port (the concrete `ReconciliationModule`). Side effect: one viewser fetch per run (the geography).

## 6. Failure Modes and Loudness

- Unknown geography `source` → `ValueError` (fail loud).
- Malformed mapping shape → `ValueError` at `CountryMapping`.
- viewser fetch failure → propagates (the run fails loud — no silent un-reconciled output).
- A `pgm_cm_point` ensemble with no wiring → pipeline-core's `RECONCILER_NOT_INJECTED` at runtime, and `test_ensemble_configs.py::test_pgm_cm_point_ensemble_wires_a_reconciler` at CI.

## 7. Boundaries and Interactions

- **Only new cross-repo coupling:** views-models → `views_postprocessing`, confined to `reconciler_factory.py`, typed as the pipeline-core port (ADP: no cycle; SDP: depends on the stable abstraction).
- Imported only by reconciling `ensembles/*/main.py` (composition root), via a `sys.path` bootstrap (run.sh is immutable).
- **Future (lower coupling):** geography could flow from the data layer (pipeline-core's PGM dataset already has `_country_id_cache`), dropping the viewser touch to zero — a pipeline-core seam follow-up; the provider port keeps it open.

## 8. Examples of Correct Usage

```python
# In a reconciling ensemble's main.py (composition root):
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconciliation import build_reconciler_for_run
reconciler = build_reconciler_for_run(Path(__file__).resolve().parent)
manager = EnsembleManager(ensemble_path=..., reconciler=reconciler)
```

## 9. Examples of Incorrect Usage

```python
# Wrong: importing the concrete reconciler outside reconciler_factory.py
# from views_postprocessing.reconciliation import ReconciliationModule  # NO — breaks ADR-014

# Wrong: wiring a reconciler into a non-reconciling ensemble (CRP violation)
```

## 10. Test Alignment

- `tests/test_reconciliation_country_mapping.py` (S1), `..._viewser_provider.py` (S2), `..._factory.py` (S3), `..._composition.py` (S4), `..._e2e.py` (S5 — the conservation/parity gate), `test_ensemble_configs.py::test_pgm_cm_point_ensemble_wires_a_reconciler` (S6 guard).

## 11. Evolution Notes

- **Source is derived, not hardcoded (EPIC #192, C-88):** `composition._derive_source` reads the source from the `reconcile_with` CM partner's constituents (`source_detection.detect_ensemble_source`) and **fails loud** if the source has no provider (datafactory before #196) or if PGM↔CM sources disagree — never a silent viewser fallback.
- **Migration-proof:** adding a datafactory-sourced reconciling ensemble = one `_PROVIDERS` entry + one `datafactory_country_mapping_provider.py` (`gaul0_code`, #196); no caller change (OCP). Different country-id system → its own parity validation.
- **Transitional substrate (C-89):** the viewser provider depends on viewser + pandas (being phased out); the reconciliation flow still rides pipeline-core's `_PGDataset`/`_CDataset` adapter. Retired by the datafactory provider (#196) + the `views-frames-reconciler` relocation.
- **Lockstep:** depends on pipeline-core #194/#195/#217 (the seam) + views-postprocessing dev (the concrete); integrated on dev branches (no release needed). Closing this seam unblocks views-reporting#72.
- **white_mustang:** deployed and wired (reconciles with `cruel_summer`) but **not in `monthly_run.sh`** — runs on demand. To schedule it monthly, add `cruel_summer` then `white_mustang` to `monthly_run.sh` (CM before PGM).
