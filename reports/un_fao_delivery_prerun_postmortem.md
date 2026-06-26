# Pre-Run Postmortem: un_fao `africa_me_legacy` smoke test (vpp#24 enrichment-swap verification)

**Date:** 2026-06-26
**Author:** Simon / Claude (prompted by Simon)
**Status:** Pre-run — credential + machinery investigation complete; awaiting execution (`go`)
**Scope:** views-models `postprocessors/un_fao/` delivery path, with established facts spanning views-postprocessing, views-faoapi, and views-pipeline-core
**Related:** views-postprocessing#24 (enrichment swap), views-models#127 (land_gaul flip), #77 (ensemble ref), the FAO delivery epic #145; pairs with a forthcoming **post-run** postmortem

---

## 1. Executive Summary

We set out to do one small thing — run the `un_fao` postprocessor once over `africa_me_legacy` as a smoke test — and it took a long, winding investigation to get to the point of being *able* to press go. Nothing was technically hard; the cost was entirely **discovery**: the pieces of the FAO delivery live in four repos under non-obvious names, our mental model of "where things are" was wrong in roughly six places, and the run's prerequisites (credentials especially) were scattered and mislabeled. This document records what we did, every place reality differed from our assumptions ("the flips"), what the machinery actually is, and the exact run plan — so the post-run postmortem + these two together become the basis for real documentation.

**Why this run exists.** views-postprocessing#24 swapped the un_fao enrichment from a geopandas runtime mapper to a precomputed `GaulLookupEnricher` (ADR-011, merged to vpp `development` via PR #25). Its planned safety check — diff new-enricher output vs old-mapper output — is **impossible** because the old mapper was deleted (PR #42 / C-39). So the agreed verification is **option A: one real `africa_me_legacy` delivery as a smoke test**. Green → close vpp#24 → unblock views-models#127 (the global `land_gaul` flip). The run is triggered **from this repo** (`postprocessors/un_fao/main.py`).

**Key findings (the short list — details in §3–§5):**
1. The Appwrite credentials are **not** in views-models; they live in `views-faoapi/.env`.
2. The 4 `APPWRITE_PROD_FORECASTS_*` vars we thought were "missing secrets" are **public bucket identifiers**, documented in plain text in vpp's README.
3. `un_fao` is **not** historical-only — one run delivers historical actuals **and** a downloaded forecast.
4. There are **two** `un_fao` directories in views-models; the `apis/un_fao/` one is **not dead** — it's the launcher for the `views-faoapi` service, with a real build-out stranded on an unmerged branch.
5. The `load_dotenv(<ensemble>/.env)` we feared coupled the historical delivery to `rusty_bucket` is a **red herring** — the manager reads ambient `os.getenv`; the ensemble `.env` is an optional overlay.
6. vpp#24 shows `open`, but its **code is merged** — the issue is open only because it awaits this smoke test.

---

## 2. What we did (the path)

1. Asked "what's the next concrete FAO move" → identified the vpp#24 smoke test as the one thing ready to execute.
2. Confirmed (against vpp `development`) that the enrichment swap is merged and the old mapper is deleted.
3. Tried to determine the run's prerequisites → discovered the credential requirement and went hunting for the keys.
4. Found `views-faoapi/.env` (11 keys) but it lacked the `PROD_FORECASTS_*` set → assumed the maintainer had to "produce" secret credentials.
5. Ran an `/expert-code-review` on the credential topology (the "secrets in two places" unease), then a sharpened recommendation (single canonical source now; secrets manager later).
6. Got pulled into the `apis/un_fao` question (is it deletable?) → discovered it's a live-but-stranded service launcher, **not** a dead stub.
7. Re-anchored to the goal, hunted the `PROD_FORECASTS_*` values properly → found they are **documented, non-secret bucket IDs**.
8. Verified the full 13-variable set resolves (13/13) → **credential blocker gone**; ready to run.

The "two seconds of progress" feeling is accurate: the actual work product is this map, not yet a delivery.

## 3. What the machinery actually is (where things live)

The FAO `un_fao` concern is spread across **four repos** in **three roles** plus a shared substrate:

| Role | Lives in | What it is |
|---|---|---|
| **Producer** (config + entrypoint) | `views-models/postprocessors/un_fao/` | `config_meta.py` (targets `lr_ged_sb/ns/os`, `ensemble: rusty_bucket`), `config_queryset.py` (`REGION = "africa_me_legacy"`, datafactory source + `FEATURE_RENAME`), and a thin `main.py` that calls vpp's manager. **Zero secret-reading code.** |
| **Manager** (the logic) | `views-postprocessing/.../unfao/managers/unfao.py` | Reads ~13 `APPWRITE_*` env vars; `_read_forecast_data` downloads the latest `category=forecast` file; `_read_historical_data` reads datafactory actuals (Zarr via `~/.netrc`); `_append_metadata` enriches via `GaulLookupEnricher` (**the #24 change**); `_save` uploads historical + forecast files to the UNFAO bucket. |
| **Package** (the API) | `views-faoapi` (repo) | The FastAPI service that *serves* the delivered data to FAO. Holds `.env` with the Appwrite secrets. |
| **Launcher** (deploy entrypoint) | `views-models/apis/un_fao/` | A views-models-convention wrapper whose `run.sh` `pip install`s `views-faoapi` and runs it (multi-worker uvicorn). Full build-out stranded on `sweep_week_dylan`; only a stub on `development`. |
| **Substrate** | `views-pipeline-core` | Owns the shared Appwrite/datastore abstraction (`modules/datastore/datastore.py`, `modules/appwrite/file.py`, `configs/prediction_store.py`). Each consumer still reads creds itself from env. |

**Crucial mechanics for the run:**
- **One run, two deliveries.** `un_fao` is *combined*: it downloads a forecast **and** reads historical actuals, enriches both, uploads both. This is why a "historical" smoke test drags in forecast/Appwrite plumbing.
- **Forecast download is not ensemble-filtered.** `_read_forecast_data` calls `download_latest_file(filters={"category":"forecast"})` — it grabs the latest forecast in the `production_forecasts` bucket regardless of which ensemble produced it. The `ensemble` ref (`rusty_bucket`) only drives an optional `load_dotenv(<ensemble>/.env)` for credentials (which finds nothing → falls back to ambient `os.getenv`) and runs `validate=False`.
- **No dry-run.** `_save()` always uploads; there is no skip-upload flag. The smoke test is therefore a *real* (low-stakes, existing-region) delivery.

## 4. The flips (expected vs. actual)

| # | We assumed | Reality |
|---|---|---|
| 1 | Appwrite creds live in views-models (maybe `postprocessors/un_fao` or `apis/un_fao`) | They live in **`views-faoapi/.env`**; views-models is deliberately secret-free |
| 2 | `PROD_FORECASTS_*` are missing secret credentials the maintainer must produce | They are **public bucket identifiers** (`production_forecasts`, `Production Forecasts`, `forecasts_metadata`, `Forecasts Metadata`), documented in vpp's README |
| 3 | `un_fao` delivers FAO **historical** data | It delivers **historical + forecast** in one run |
| 4 | `apis/un_fao/` is a dead duplicate scaffold, safe to delete | It's the **launcher for `views-faoapi`** (FastAPI, multi-worker), with a 297-line spec'd build-out **stranded on `sweep_week_dylan`** (Jan; maintained by Dylan into May) |
| 5 | The historical delivery is coupled to `rusty_bucket`'s `.env` (via `load_dotenv`) | Red herring — the manager reads ambient `os.getenv`; the ensemble `.env` is an optional, currently-empty overlay |
| 6 | vpp#24 is `open` ⇒ not done | Its **code is merged** (PR #25); it's open only pending this smoke test |
| 7 | Only the 3 datastore vars are real secrets | Correct, and confirmed: `ENDPOINT`, `DATASTORE_PROJECT_ID`, `DATASTORE_API_KEY` are the real secrets; the other 10 are identifiers |

## 5. Credential topology (what we learned)

The run needs **13** `APPWRITE_*` variables, of which only **3 are secrets**:

- **Secrets (3)** — `APPWRITE_ENDPOINT`, `APPWRITE_DATASTORE_PROJECT_ID`, `APPWRITE_DATASTORE_API_KEY`. In `views-faoapi/.env`.
- **Identifiers (10)** — `APPWRITE_PROD_FORECASTS_{BUCKET_ID,BUCKET_NAME,COLLECTION_ID,COLLECTION_NAME}` (documented values), `APPWRITE_UNFAO_{BUCKET_ID,BUCKET_NAME,COLLECTION_ID,COLLECTION_NAME}` (in faoapi/.env), `APPWRITE_METADATA_DATABASE_{ID,NAME}` (in faoapi/.env).

**Verified:** combining `views-faoapi/.env` + the 4 documented `PROD_FORECASTS_*` resolves **13/13**.

**The topology smell (separate from this run — see the expert review):** the secrets risk being copied into a second `.env` to run the producer; the `load_dotenv(<ensemble>/.env)` couples credential location to an unrelated forecast concept; `views-faoapi/.env` is an *incomplete* replica (lacks `PROD_FORECASTS_*`). Recommendation (one path, sequenced): **one canonical secret source referenced not copied, now**; a **secrets manager later** (trigger: a prod machine / rotation need). `apis/un_fao` becoming a live API would make it a *third* Appwrite consumer — reinforcing single-source.

## 6. The plan (exact run procedure)

**Command** (assembles env in-process — no secret copied to disk, no values printed):
```bash
cd views-models && conda run -n views_pipeline python -c "
from dotenv import load_dotenv; import os, runpy
load_dotenv('../views-faoapi/.env')
os.environ.update({
  'APPWRITE_PROD_FORECASTS_BUCKET_ID':'production_forecasts',
  'APPWRITE_PROD_FORECASTS_BUCKET_NAME':'Production Forecasts',
  'APPWRITE_PROD_FORECASTS_COLLECTION_ID':'forecasts_metadata',
  'APPWRITE_PROD_FORECASTS_COLLECTION_NAME':'Forecasts Metadata',
})
runpy.run_module('postprocessors.un_fao.main', run_name='__main__')
"
```

**Preconditions:**
- `~/.netrc` has the Zarr host `204.168.219.108` — **confirmed present**.
- 13/13 Appwrite vars resolve — **confirmed**.
- `REGION = "africa_me_legacy"` (the smoke-test region) — current config.

**Definition of green:** the run fetches actuals + the latest forecast, enriches both via `GaulLookupEnricher`, and uploads the delivery without error. On green → close **vpp#24** → **#127** unblocked.

## 7. Risks / unknowns the run will surface (record outcomes in the post-run doc)

1. **faoapi API-key permissions** — does the key have *read* on `production_forecasts` and *write* on the UNFAO bucket? (faoapi is the *serving* repo; its key may be scoped differently.)
2. **Forecast availability** — is there a `category=forecast` file in `production_forecasts` to download? (If empty, the download step fails — unrelated to #24's enrichment.)
3. **`wandb.login()`** in `main.py:23` — needs a wandb session/key on the machine.
4. **Outward write** — `_save` performs a real delivery to the UNFAO bucket (low-stakes: africa_me_legacy is an existing region).
5. **Bucket-ID correctness** — the documented `PROD_FORECASTS_*` values are assumed canonical; a "bucket not found" error means they need correcting to the live Appwrite IDs.

## 8. Discoveries to fold into documentation + open decisions

- **un_fao architecture note** — the producer/manager/package/launcher map (§3) belongs in `postprocessors/un_fao/README.md` (and/or a platform doc), because it is non-obvious and cost us hours.
- **Credential single-source** — adopt one canonical Appwrite secret source; add a fail-loud env check to the un_fao entrypoint; remove the ensemble-dir `load_dotenv`. (Driver: a new Appwrite bucket is imminent — single-source is needed *now*, a manager is not.)
- **`apis/un_fao` revive-vs-defer** — Dylan's FAO-serving-API build-out is stranded on `sweep_week_dylan`; decide to revive/merge or leave the stub. **Do not delete it.** (Coordinate with Dylan.)
- **Manager gaps** — no dry-run/skip-upload and no startup credential validation; both are cheap, high-value additions in vpp.
- **Stale sibling** — `apis/seldon_api/` is a dead stub (Dylan removed it on his branch); align development with that.

---

*Next step: on `go`, execute §6 and capture the actual outcome (and which §7 unknowns bit) in the paired **post-run postmortem**.*
