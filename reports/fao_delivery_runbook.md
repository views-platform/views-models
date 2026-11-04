# FAO Delivery Runbook

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-27
**Related:** epic #145 (FAO global delivery) + tracking #148; #143 (rusty_bucket / no-collapse), #127 (land_gaul flip), #149 (no-collapse contract), #77 (ensemble ref); deep detail in `reports/un_fao_delivery_{prerun,postrun}_postmortem.md` and `postprocessors/un_fao/README.md` + `apis/README.md`

> This is the single end-to-end map for delivering VIEWS data to the UN FAO. None of it was
> written in one place before, which is how the recent offline incidents happened. Read the
> two postmortems for the *why*; this runbook is the *how* and the *gates*.

---

## Overview — two independent streams

FAO receives data from **two decoupled streams**, produced here and served by `views-faoapi`:

1. **Historical actuals** — the `un_fao` **postprocessor** (`postprocessors/un_fao/`): datafactory
   UCDP fatality actuals (`lr_ged_sb/ns/os`), enriched with GAUL attribution, uploaded to Appwrite.
2. **Forecast** — the `rusty_bucket` **ensemble** (`ensembles/rusty_bucket/`): pooled posterior
   draws (Track A, `y_pred.npy` of shape `(N, 1024)`), shipped **uncollapsed**.

(One `un_fao` run actually delivers *both* — it reads historical actuals AND downloads the latest
forecast, enriches both, uploads both. See the postprocessor README.)

## Ground rule 1 — the NO-COLLAPSE contract (the one that bites)

`rusty_bucket` exists to ship the **full posterior mixture** as pooled draws. The single,
principled MAP/HDI collapse must happen **once, downstream**, in `views_frames_summarize`
(views-frames#89) — **never** upstream. Each repo on the path has a duty:

| Repo | Duty | Status |
|---|---|---|
| **views-models** | produce pooled draws uncollapsed (`aggregation: concat`, no `_best`/mean) | ✅ rusty_bucket built |
| **views-postprocessing** | carry the draws through delivery without collapsing to a point | ⚠️ **vpp#45 open** — the unfao delivery is still point/DataFrame-based; it would drop the draws |
| **views-faoapi** | serve the draws uncollapsed | downstream |
| **views-frames** | the ONE collapse: robust MAP + nested HDI (views-frames#89) | in progress |

**If any link bakes out a `_best`/mean/median scalar, the mixture is silently flattened and
nobody is alerted.** That is the failure this stream exists to prevent (#149 is the enforcement
contract). **Until vpp#45 lands, the forecast half cannot be delivered uncollapsed.**

## Ground rule 2 — the coverage asymmetry (intended, not a bug)

The two streams have **different geographic coverage on purpose**:

- **Historical** actuals go **global** — `land_gaul` (64,736 cells) once #127 flips the region.
- **Forecast** coverage stays at **the ensemble's region** (whatever `rusty_bucket`'s constituents cover).

These are **decoupled by design** — do not "fix" the asymmetry. Today `un_fao` `REGION =
"africa_me_legacy"` (13,110 cells); the global flip is #127.

## Prerequisites

Run from this repo: `postprocessors/un_fao/main.py`. The runtime needs:

1. **Directory scaffold** — `PostprocessorPathManager` validates `artifacts/`, `notebooks/`,
   `reports/`, `data/{generated,processed,raw}/`, `logs/`, `configs/`. Guarded by
   `tests/test_model_structure.py::TestPostprocessorDirectoryStructure`.
2. **Data factory** — `views-datafactory` installed + `~/.netrc` for the Zarr host `204.168.219.108`.
3. **Appwrite credentials** — **13 `APPWRITE_*` env vars; only 3 are secrets**
   (`ENDPOINT`, `DATASTORE_PROJECT_ID`, `DATASTORE_API_KEY`) and they live in **`views-faoapi/.env`**,
   not here. The 10 others are bucket/collection identifiers. To assemble the full set: load
   `views-faoapi/.env` + the 4 `PROD_FORECASTS_*` identifiers. (Full topology: postprocessor README + prerun postmortem.)
4. **A forecast in the store** — the forecast download is filtered by `{category: forecast, name: <ensemble>}`,
   so the referenced `ensemble` (#77) must have a forecast in `production_forecasts`. `rusty_bucket`
   has none yet → the forecast half can't run until a real-forecast ensemble is referenced or one is delivered.

## Phase 1 — Historical (un_fao) delivery

### Step 1.0 — dry verify the enrichment first (NO upload, NO forecast)
There is **no dry-run / skip-upload flag** in the manager. So to validate the run safely, exercise
the **historical-only enrichment** path directly (the pattern used to verify vpp#24):
instantiate `UNFAOPostProcessorManager`, call `_read_historical_data()` then `_append_metadata()`,
inspect the GAUL columns. Read-only, no Appwrite, no upload. (See the postrun postmortem §2 for the snippet.)

**Expected:** enriched frame with `country_iso_a3`, `admin0/1/2_gaul*`, coords; a small number of
GAUL-uncovered cells (remote islands — e.g. Marion Island) will be null and "fail validation"
under `africa_me_legacy`. **That caveat self-resolves under `land_gaul`** (it excludes uncovered cells).

### Step 1.1 — the real delivery
```
conda run -n views_pipeline python -m postprocessors.un_fao.main --run_type calibration
```
This performs the **real Appwrite upload** to the FAO bucket. Low-stakes for `africa_me_legacy`
(an existing region). Confirm it completes without error.

## Phase 2 — Forecast (rusty_bucket) delivery
**Blocked on vpp#45** (the delivery path must carry pooled draws) and on `rusty_bucket` having a
real forecast in the store. Until both land, the forecast half is not deliverable uncollapsed.
When unblocked: run the ensemble forecast, confirm `y_pred.npy` is `(N, 1024)` at every hop to FAO
(the #149 contract), with the single collapse only in views-frames#89.

## Phase 3 — The land_gaul global flip (#127) — DRY RUN FIRST
The go-global act. Discipline (vpp register C-32 — the historical frame grows to ~28M rows):

1. **One-line change** in `config_queryset.py`: `REGION = "africa_me_legacy"` → `"land_gaul"`.
2. **Full-volume DRY RUN on the run machine, NO Appwrite upload** — fetch → enrich → validate →
   write parquet locally only. **Record wall-clock + peak memory** against the Stage-0 baseline.
3. Only then the **real run** (with upload).

Preconditions before flipping (do NOT flip until all are met): views-datafactory#159 region merged+released (✅),
`datafactory_query` updated on the machine, vpp#24 (enrichment swap) verified green (✅ enrichment verified),
the Stage-0 baseline schema at hand.

## Known caveats (carried from the smoke test)
- **No dry-run flag** in the un_fao manager — Phase 1.0 is the workaround.
- **~5 `africa_me_legacy` cells** (Marion Island + offshore) have no GAUL match → fail completeness
  validation; resolves under `land_gaul`.
- **`PROD_FORECASTS_COLLECTION_ID`** documented value (`forecasts_metadata`) was wrong in the live
  Appwrite during the smoke test — confirm the real collection ID before a forecast delivery.

## "No undocumented step remains" checklist (#147 acceptance)
- [x] prereqs (scaffold, datafactory, creds, forecast-in-store)
- [x] the no-collapse contract + responsible repos
- [x] the coverage asymmetry
- [x] the dry-run-before-real-run discipline
- [x] how to run each stream + the safe enrichment-only verify
