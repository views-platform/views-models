# Post-Run Postmortem: un_fao `africa_me_legacy` smoke test (vpp#24 enrichment-swap verification)

**Date:** 2026-06-26
**Author:** Simon / Claude (prompted by Simon)
**Status:** Run complete — #24 enrichment **verified**; full delivery blocked on forecast-path config gaps (not #24); one benign coverage caveat
**Scope:** views-models `postprocessors/un_fao/`, executed against live views-datafactory + Appwrite
**Related:** pairs with `un_fao_delivery_prerun_postmortem.md`; views-postprocessing#24, views-models#127, #77

---

## 1. Executive Summary

We executed the `un_fao` `africa_me_legacy` smoke test. It took **three attempts** to get a verdict, each one informative:

1. **Full run** → failed immediately on a missing directory (`PostprocessorPathManager` validation).
2. **Full run (re-try, dirs created)** → got past auth into Appwrite, then failed at the **forecast download** — for two reasons unrelated to #24.
3. **Historical-only enrichment check** (the clean #24 test) → **passed**: the `GaulLookupEnricher` correctly enriched 5.7M `africa_me_legacy` actuals with GAUL metadata.

**Bottom line:** the #24 enrichment swap (geopandas runtime mapper → precomputed `GaulLookupEnricher`) **works correctly**. The full *delivery* is separately blocked by two forecast-path config gaps and one benign coverage caveat — none of which is the enrichment itself.

## 2. What we ran, and what each attempt showed

### Attempt 1 — full run, fresh
`PostprocessorPathManager(Path(.../un_fao/main.py))` raised `FileNotFoundError: .../postprocessors/un_fao/artifacts does not exist`. **Finding:** the un_fao postprocessor directory was missing **7 required scaffold dirs** — it had only `configs/` + `logs/`; the manager requires `artifacts`, `notebooks`, `reports`, `data/{generated,processed,raw}`. (un_fao is the only postprocessor, so nothing was there to copy from.) **Fixed:** created the 7 dirs with `.gitkeep` (uncommitted — a real structural fix to land).

### Attempt 2 — full run, dirs present
Auth to Appwrite **succeeded** (faoapi creds valid). It reached `_read_forecast_data` and failed there:
```
Search ... filters: {'category': 'forecast', 'name': 'rusty_bucket'}
ERROR - Collection with the requested ID 'forecasts_metadata' could not be found
FileNotFoundError: No forecast file found in the prediction store (category='forecast')
```
**Findings — corrected against merged vpp `development` (`unfao/managers/unfao.py::_read_forecast_data`, post-#64):**
- **Forecast selection is category-only (newest-wins); ensemble identity is enforced *after* selection, not in the query.** The log showed a `{'category':'forecast','name':'rusty_bucket'}` filter, and an earlier draft of this doc concluded the download "IS ensemble-name-filtered." **That conclusion was wrong.** Merged vpp queries `get_latest_file_id(filters={"category":"forecast"})` **alone**, then checks the resolved file's `name`/`loa` via `identity.assert_forecast_identity(...)` — the S3/C-25 identity guard, which fails loud on mismatch (the code comment: *"filtered by category alone (newest-wins), so resolve the file, then verify its identity before delivering it"*). The `name` in the observed filter came from a **non-clean vpp working copy**, not shipped `development` — **confirmed (2026-06-27)**: vpp `development`'s `_read_forecast_data` filters on `category` **alone** (no `name` key exists in the query), so the observed `{'category':'forecast','name':'rusty_bucket'}` log *cannot* have come from `development`; the run used a modified checkout. So the pre-run's "not ensemble-filtered" was *closer to right* (selection is category-only) but **incomplete** — it omitted the post-selection identity guard. **Why it matters:** the wrong "name-filtered" framing would teach a reader that ensemble identity is handled by the query, making the S3/C-25 guard look redundant and inviting someone to remove it. The merged design is the opposite: category-only selection, identity enforced by the guard.
- **`APPWRITE_PROD_FORECASTS_COLLECTION_ID='forecasts_metadata'` (the documented value) is wrong** — that collection does not exist in the live Appwrite, and the run actually died **here**, before the selection/identity path was meaningfully exercised. The vpp README value is **not canonical** for these IDs.

The run never reached the enrichment, so it told us nothing about #24 — only about the forecast path.

### Attempt 3 — historical-only enrichment check (the actual #24 test) ✅
Called the manager directly: `_read_historical_data()` then `_append_metadata()`, skipping the forecast path entirely (no Appwrite, no upload). Result (exit 0):
- `GaulLookupEnricher` loaded `64742 cells` from `gaul_lookup.parquet` (`version=land_gaul@f74d3b2b`).
- Datafactory fetched `africa_me_legacy` actuals via `~/.netrc`: **5,729,070 rows** (13,110 cells × 437 months), targets `lr_ged_sb/ns/os`.
- Enrichment joined the **9-column GAUL metadata contract** (`gaul_schema.METADATA_COLS`, shared verbatim with views-faoapi's `FAO_PGMDataset._METADATA_COLS`): `pg_xcoord`, `pg_ycoord`, `country_iso_a3`, `admin1_gaul0_{code,name}`, `admin1_gaul1_{code,name}`, `admin2_gaul2_{code,name}` — note gaul0 **and** gaul1 both sit under the `admin1_` prefix, gaul2 under `admin2_` (not `admin0_gaul0`). Sample correct (e.g. ZAF → Western Cape → South Africa → Overberg).

**#24's enrichment mechanism is verified working.**

## 3. The coverage caveat (benign, self-resolving under #127)

The enricher warned: **2,185 rows = exactly 5 cells** have no GAUL-lookup match (*"will fail validation"*). Mapped to coordinates (PRIO-GRID gid → lat/lon):

| gid | lat | lon | what it is |
|---|---|---|---|
| 62356 | −46.75 | +37.75 | **Marion Island / Prince Edward Is.** (sub-Antarctic) |
| 94776 | −24.25 | +47.75 | offshore SE of Madagascar |
| 99027 | −21.25 | +13.25 | offshore Namibian coast (Atlantic) |
| 107733 | −15.25 | +46.25 | Mozambique Channel (NW Madagascar) |
| 107742 | −15.25 | +50.75 | Indian Ocean, E of Madagascar |

These are remote islands / offshore ocean cells — exactly the "land cells GAUL doesn't cover" that #127 documents (82 globally). **This is not an enrichment bug.** It is an artifact of the *old* `africa_me_legacy` region, which has no GAUL-exclusion built in. **It self-resolves under #127**: the `land_gaul` region definition excludes uncovered cells upstream, so a `land_gaul` delivery never sees them. A full `africa_me_legacy` delivery, however, would fail the manager's existing `_validate` null-gate (the 9 GAUL metadata columns must be non-null) on these 5 cells until they are excluded — this is the existing validation, not a new coverage contract.

## 4. Status after the run

- **vpp#24 (enrichment swap):** the **enrichment mechanism** is verified (the `_append_metadata` GAUL join). Scope of that verdict: attempt 3 stopped after `_append_metadata`, so the downstream `_validate`/coverage gating and the **forecast-frame** enrichment (same enricher applied to the prediction frame) were **not** exercised — the historical-frame join is what ran. Recommend **close with the coverage caveat + this scope noted** — the validation behaviour on uncovered cells is the expected GAUL exclusion, cleanest under `land_gaul`.
- **views-models#127 (land_gaul flip):** still gated on vpp#24's close, and now shown to be the *cleaner* region for un_fao (it removes the 5-cell caveat by construction).
- **Full un_fao delivery:** blocked on the forecast path — (a) no forecast in the store for the referenced ensemble (`rusty_bucket` produces none; ties to #77 / the forecast track), and (b) the wrong `PROD_FORECASTS_COLLECTION_ID`.
- **Credentials/datafactory/enricher end-to-end on the historical path:** all working.

## 5. Corrections to the pre-run postmortem

- Pre-run §3 "forecast download is not ensemble-filtered" → **substantially correct** (merged vpp selects by `category` alone), but **incomplete** — it omitted the post-selection identity guard (`assert_forecast_identity`, S3/C-25). An earlier draft of *this* post-run wrongly "corrected" it to name-filtering; that was an artifact of a non-clean vpp working copy (§2) and is itself now corrected against merged `development`.
- Pre-run §7 "bucket-ID correctness unknown" → **confirmed wrong**: `forecasts_metadata` collection ID does not exist.
- New, not anticipated pre-run: the **missing-scaffold-dirs** prerequisite.

## 6. Action items

- **views-models:** commit the 7 un_fao postprocessor scaffold dirs (structural fix); flip to `land_gaul` (#127) once vpp#24 closes — it also clears the 5-cell caveat.
- **views-postprocessing:** correct/locate the real `PROD_FORECASTS_COLLECTION_ID` (README value is wrong); add fail-loud credential + collection-existence validation at startup; consider a dry-run/skip-upload flag (there is none today); confirm the un_fao validation excludes GAUL-uncovered cells (the 5).
- **forecast track:** un_fao's forecast delivery needs a forecast in the store for its referenced ensemble — `rusty_bucket` provides none. Either point the `ensemble` ref at a real-forecast ensemble for delivery, or the forecast track (#143/#146/#77/vpp#45) must produce one.
- **secrets:** the single-canonical-source recommendation (from the expert review) stands and is reinforced.

## 7. The two-doc value

Together with the pre-run postmortem, this captures the full machinery map, the credential topology, the six pre-run flips + three run-time corrections, and the verified-with-caveat #24 result — ready to lift into `postprocessors/un_fao/README.md` and a platform FAO-delivery runbook (#147).
