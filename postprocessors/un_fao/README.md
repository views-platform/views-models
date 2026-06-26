# un_fao postprocessor

Delivers VIEWS data to the **UN FAO** (served downstream by `views-faoapi`). One
run delivers **two things**, not one:

1. **Historical actuals** — fetched from the VIEWS **data factory** (Zarr store).
2. **The latest forecast** — downloaded from the Appwrite `production_forecasts` bucket.

Both are enriched with country/admin (GAUL) attribution and **uploaded to the FAO
Appwrite bucket**. (Earlier docs framed this as historical-only — it is not.)

Created: 2025-10-17.

## Where the pieces live (read this first — it is non-obvious)

`un_fao` spans **four repos in three roles** plus a shared substrate. Confusing
them is the #1 way to get lost here:

| Role | Lives in | What it is |
|---|---|---|
| **Producer** (config + entrypoint) | `postprocessors/un_fao/` (here) | configs + a thin `main.py` that calls the vpp manager. **Zero secret code.** |
| **Manager** (the logic) | `views-postprocessing/.../unfao/managers/unfao.py` | reads the Appwrite env vars, downloads the forecast, reads the datafactory actuals, enriches both (`GaulLookupEnricher`), uploads both |
| **Package** (the API) | `views-faoapi` (repo) | the FastAPI service that *serves* the delivered data; holds the Appwrite **secrets** (`.env`) |
| **Launcher** (deploy) | `apis/un_fao/` (this repo) | installs + runs `views-faoapi` (see `apis/README.md`) |
| **Substrate** | `views-pipeline-core` | the shared Appwrite/datastore client (`modules/datastore`, `configs/prediction_store.py`) |

## What it produces

Three UCDP fatality targets at PRIO-GRID month level:

| descriptor feature | renamed to | UCDP source |
|---|---|---|
| `ged_sb_best` | `lr_ged_sb` | state-based |
| `ged_ns_best` | `lr_ged_ns` | non-state |
| `ged_os_best` | `lr_ged_os` | one-sided |

Data source is the **data factory** (not viewser) — see `configs/config_queryset.py`.
Current coverage region: **`africa_me_legacy`** (~13,110 cells). Going global to
`land_gaul` (64,736 cells) is tracked in #127 — and it also clears the handful of
GAUL-uncovered cells (remote islands) that fail completeness validation under
`africa_me_legacy` (see the postmortems).

## Credential topology

The run needs **13 `APPWRITE_*` env vars**, of which only **3 are real secrets**:

- **Secrets (3):** `APPWRITE_ENDPOINT`, `APPWRITE_DATASTORE_PROJECT_ID`,
  `APPWRITE_DATASTORE_API_KEY` — they live in **`views-faoapi/.env`** (the serving
  repo). views-models stores **no secrets**.
- **Identifiers (10):** `APPWRITE_PROD_FORECASTS_*` (the forecast-download bucket),
  `APPWRITE_UNFAO_*` (the upload target), `APPWRITE_METADATA_DATABASE_*` — bucket/
  collection names, not secrets.

The manager reads them via ambient `os.getenv` (plus an optional
`load_dotenv(<ensemble>/.env)` overlay — which finds nothing today, so the
`ensemble` ref does **not** supply credentials; a known red herring). To run it,
provide all 13 in the process env (e.g. load `views-faoapi/.env` + set the 4
`PROD_FORECASTS_*`). The cleanest long-term home is one canonical source referenced
by all consumers, not copied — see the postmortems.

## Prerequisites

**Directory structure.** The postprocessor is run through `PostprocessorPathManager`,
which validates the full scaffold (`artifacts/`, `notebooks/`, `reports/`,
`data/{generated,processed,raw}/`, `logs/`, `configs/`) — a missing dir crashes
before any work. These are guarded by
`tests/test_model_structure.py::TestPostprocessorDirectoryStructure`.

**Data factory.** The runtime env **must** have:

1. **`views-datafactory` installed** (the `datafactory_query` module):
   ```
   pip install 'views-datafactory @ git+https://github.com/views-platform/views-datafactory.git@development'
   ```
   If missing, `configs/config_queryset.py` fails loud at config load (not an opaque
   `ModuleNotFoundError`).
2. **`~/.netrc` for the Zarr host `204.168.219.108`** (see the `bright_starship`
   model README for the format). Without it the Zarr fetch fails at runtime.

Verify these in the postprocessor's **own** runtime, not only a model env (#95).

## The ensemble reference

`configs/config_meta.py` carries an `ensemble` field. It does **not** select an
aggregation. The manager's forecast download is **filtered by this ensemble name**
(`{category: forecast, name: <ensemble>}`), so it must reference an ensemble that
**has** a forecast in the `production_forecasts` bucket. It also (optionally) locates
`.env` credentials and tags uploads. Must be a real ensemble (#77).

## Running

```
conda run -n views_pipeline python -m postprocessors.un_fao.main
```

This runs the full pipeline including the **Appwrite upload** to the FAO bucket —
there is **no dry-run / skip-upload flag**, so do not run it casually. For a local,
no-upload check of just the **enrichment** (no forecast, no Appwrite), call the
manager's `_read_historical_data()` + `_append_metadata()` directly (the pattern used
to verify vpp#24 — see `reports/un_fao_delivery_postrun_postmortem.md`). For a
data-equivalence check, exercise `configs/config_queryset.py::generate()` + the
datafactory fetch (see `tests/test_un_fao_datafactory_equivalence.py`).

## Further reading

- `reports/un_fao_delivery_prerun_postmortem.md` — the machinery map + credential topology.
- `reports/un_fao_delivery_postrun_postmortem.md` — the verified-with-caveat #24 result + the forecast-path gaps.
- `apis/README.md` — the launcher/package distinction.
