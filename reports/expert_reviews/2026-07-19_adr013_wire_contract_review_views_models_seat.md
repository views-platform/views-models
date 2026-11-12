# Review: views-postprocessing ADR-013 "The Sampled-Forecast Wire Contract (v1.5)"

**Reviewing seat:** views-models (maintainer-commissioned)
**Date:** 2026-07-19
**Document reviewed:** `views-postprocessing/docs/ADRs/013_sampled_forecast_wire_contract.md`
at vpp `development` HEAD `76c4a20` (post adversarial-sweep text)
**Commission:** dual-purpose — (1) does the ADR correctly understand the world;
(2) does the views-models seat correctly understand the world. All confusions,
discrepancies, and hand-waviness to be investigated, not noted.

**Verdict up front: the ADR is substantially correct and unusually honest —
nearly every discrepancy hunted was already self-flagged in its own errata.
The larger corrections fall on the reviewing seat's world-model: three
operationally significant beliefs held by this seat were wrong or materially
incomplete (§3). Two open items the ADR flags are owed by the maintainer (§2.3).**

---

## 1. ADR claims verified TRUE against the repos (receipts, 2026-07-19)

| ADR claim | Verification |
|---|---|
| §10 golden fixture published; root hash `b1f3878df9ef74b25dce53a070e1711db39dfdf1c6ca3e1f5a716875ceb32f44` | `tests/fixtures/wire_contract/` exists (5 artifacts + README + SHA256SUMS); `sha256sum SHA256SUMS` reproduces the hash byte-for-byte |
| Hop-A publish leg shipped (pipeline-core #269 via PR #276, `66328be`) | Commit present on pipeline-core `development`; plus a newer conformance commit the ADR postdates (`1e14689`, PR #278 — producer emits the canonical fixture bytes) |
| Hop-A legacy guard merged (vpp PR #99) | `views_postprocessing/unfao/managers/unfao.py:33` — `LEGACY_FORECAST_FILTERS = {"category": "forecast", "type": "ensemble"}`, applied at `:121` |
| Hop-B legacy guard merged (faoapi PR #200, `type="model"` pin) | faoapi `development` `d42043d`; C-161 (guard must reach production before first contract upload) tracked; deploy epic #184 in visible motion (release-prep PR #201, runbook fixes at HEAD) |
| §6 `draws.py` exists | `views_postprocessing/delivery/draws.py`, sibling of coverage/identity/observed_range/provenance invariants |
| Hop-A source adapter built | vpp PR #101 (`unfao/track_a_source.py`) |
| §4.1a: forecast docs upload under the *ensemble's* name; only historical complies; faoapi name-filters unconditionally | Current code: forecast doc `name=self.ensemble_path_manager.model_name` (`unfao/managers/unfao.py:325`), historical `name=self._model_path.model_name` (`:314`), both `type="model"` — exactly the invisibility mechanism §4.1a describes |
| vpp #91 (Hop-B sink adapter) open — contract wire not yet end-to-end | Confirmed open; ADR never claims otherwise |
| §0.3/§3.5 retention owner OPEN; §6 "region-pinned" S_min undefined | Confirmed as flagged — see §2.3 below |

## 2. Discrepancies found IN the ADR (all minor)

**2.1 Appendix B paths/lines are stale.** It cites `delivery/unfao.py:110/:123/:303/:314`;
the module has moved to `unfao/managers/unfao.py` and the lines shifted
(≈`:121`/`:314`/`:325`). The Appendix disclaims this ("evidence at verified
HEADs, not obligations") — acceptable, but a one-line refresh note would spare
a future reader the wild-goose chase this review went on.

**2.2 A §10.1 success story the text does not tell.** The fixture bytes were
once silently dropped by a blanket `*.zip`/`*.parquet` gitignore (fixed vpp
PR #102, `7f1914c`). The pinned-hash test caught it — the strongest existing
evidence *for* the §10.1 vendoring mechanism, and worth a line in the
post-adoption record.

**2.3 Two self-flagged OPEN items, both owed by the maintainer:**
(a) the `production_forecasts` **retention owner** (§0.3/§3.5 — ≈28.6 GB per
full-S run, accumulating, no TTL mechanism anywhere); (b) the **§6
"region-pinned" production S_min definition**. Neither blocks the walking
skeleton; both bite before the first full-S production run.

**2.4 Trivia (no action):** "ADR-013" is itself number-overloaded across the
platform (views-models ADR-013 = target-name agnosticism) — the per-repo ADR
numbering that produced three ADR-046s (Erratum E2) guarantees recurrence.

## 3. Corrections to the REVIEWING SEAT's world-model (the review's second purpose)

**3.1 "FAO forecast delivery stalled 131 days" — wrong axis; the truth is
worse: the FAO forecast product has NEVER been served.** The views-models
liveness instrument (`tools/liveness/unfao_delivery.py`) watches *storage
files* and truthfully saw a `forecast_dataset` parquet of 2026-03-10. But
faoapi selects via *metadata documents* through an unconditional `name`
filter, and every forecast document ever uploaded carried the ensemble's name
— confirmed live 2026-07-15 (faoapi `02432ca`: six stranded
`orange_ensemble`-named forecast docs; "the invisibility has been firing all
along and is the actual reason forecast serving is empty"). Even when the
liveness check read DELIVERING, FAO could not GET forecasts. **Files-in-bucket
and visible-to-consumer are different axes; the liveness suite measures only
the former** (limitation recorded against views-models C-102).

**3.2 The seat's register C-97 narrative is partially stale.** It cites the
pre-guard `unfao.py:106` newest-wins selection with no `type` filter; the
type guard now exists (vpp PR #99). C-97's core (recency-as-identity, no
name/month addressing) still stands. Register update owed.

**3.3 The "delivery epic" definition of done was under-specified.** "Files
land in `unfao_bucket`" is necessary but insufficient — a forecast document
under the wrong `name` lands and stays invisible. The true DoD is
views-models#230 step 5: **faoapi returns non-empty forecast *with posterior
draws***.

**3.4 Blocker status of views-models#230, live-verified 2026-07-19:**
- **A — env: STILL BROKEN.** The current manager reads
  `APPWRITE_PROD_FORECASTS_{BUCKET,COLLECTION}_{ID,NAME}` (both vocabularies);
  `views-faoapi/.env` contains **none of the four** (UNFAO_* + metadata +
  datastore only). Correct values are known from the 2026-07-19 liveness
  forensics (db `file_metadata`, collection `production_forecasts`).
- **B — `rusty_bucket` has never produced a forecast: CONFIRMED** (its
  `artifacts/` and `logs/` are empty).
- **C — point-shaped forecast path (vpp#45): OPEN.** #230 recommends
  minimal-C (pass sample columns through the nested-parquet encoding faoapi
  already ingests). Corollary either way: the forecast doc must upload under
  the name faoapi resolves, or it lands invisible like all its predecessors.

**3.5 Stale statement in views-models#230 itself:** "no forecast file has
ever landed in unfao_bucket" — false as written (March file; orange-era
docs); the accurate statement is "no forecast has ever been *servable*."
The ADR's post-adoption record has the corrected picture; #230's body
predates it. Issue-body correction owed on the views-models side.

## 4. Confusions investigated to ground

- **`orange_ensemble`** — not a ghost: a retired ensemble present in
  views-models *git history* (e.g. `e3b1d318`, `7e82282a`), deleted since.
  The six stranded docs date that era.
- **faoapi "11 files, all historical" (2026-06-29) vs liveness seeing a March
  forecast file** — not a contradiction: faoapi enumerated what it can see
  through the name filter; liveness enumerates raw storage. Two instruments,
  two axes, both truthful.
- **`*_ID` vs `*_NAME` env vocabulary** — the postmortems say
  `COLLECTION_ID`; the current manager reads **both** families. A blocker-A
  fix must set all four PROD_FORECASTS variables.

## 5. Consequence for the delivery push (2026-07-19)

Corrected chain: **`rusty_bucket` run (views-models) → `production_forecasts`
shelf → un_fao postprocessor (launched from views-models
`postprocessors/un_fao`, code in vpp) → `unfao_bucket` under a
faoapi-visible name, carrying draws → faoapi serves (deploy epic #184)**.
views-models' duties: the four env vars (values known), run `rusty_bucket`,
trigger the postprocessor, verify serving. Gates outside views-models:
vpp#45 (minimal-C), the name-visibility fix (vpp), and C-161 only if
contract-typed artifacts are uploaded (legacy-shaped minimal-C does not trip
it). ADR §11.4 ordering holds: guards → producers → run 0; both guards are
merged on development, the Hop-B guard's *production* deploy rides #184.
