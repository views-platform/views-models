# Pre-registration v3 — does an FAO delivery arrive, intact and ours?

**Written 2026-09-29 after `opponent` — the only reviewing seat that was not in the room —
established that v2 lost the four probes v1 called its strongest.** Supersedes
`prereg_fao_delivery_v2.md` for delivery verification. Its model-health half moved to
`prereg_model_health_v1.md`; the two are siblings and neither is a superset.

v1 (`prereg_rusty_bucket_landing_20260929.md`) and v2 are retained **unedited**. A
pre-registration that can be rewritten after the fact is not one.

---

## How to audit this document in ten seconds

**Read down the hops.** Every probe is pinned to the hop it exercises. **A hop with no probe is a
hole** — you do not have to notice an omission, you have to see an empty row.

That mechanism is the whole point of v3. v1 was indexed by the chain and its coverage could be
audited this way. v2 re-indexed on a taxonomy of failure classes — good analysis, bad table of
contents, because **a taxonomy has no empty cells**: a class can look full while a hop is empty.
Four probes fell out and none of five reviewers noticed.

---

## The claim

> A named FAO delivery has arrived, is being served, and the values FAO receives are the values we
> produced.

Three parts, and v2 could only test the first. **Arrived** is structural. **Served** is a
provenance question. **Ours** is a value question, and it is the one that has never been tested.

---

## The chain

```
L1  pod        produce and pool                                          P1
L2  Hop-A      publish -> production_forecasts    (§3.2 manifest per (run,target))   P2 P3 E3
L3  transform  un_fao postprocessor, land -> land_gaul                    P6 B1 B4 B5
L4  Hop-B      unfao_bucket                       (§4.2 manifest per run) E1 E2
L5  serve      faoapi ingest -> /data endpoints                        P5 P4 P7 P11 B6
ACROSS         served values == produced values                           P8  <- the spine
CONTROL        nothing outside the declared surface moved                 P9
```

**Two manifest kinds.** Conflating them cost three sessions an afternoon. §3.2 is the Hop-A
producer manifest, one per *(run, target)*. §4.2 is the Hop-B run manifest, one per *run*, with a
`targets` list.

**Which endpoint is authoritative: `forecast_serving_state`** (`reason`, `fallback_available`).
`/health`'s `status` and `forecast_freshness` read the newest record in the store, **not what is
served** — they reported `healthy, age 0.0 days` while the run was refused.
`/provenance/forecast` answers a *legacy* resolution path. **v1 believed `/provenance`.**

---

## Reporting rules

1. **PASS / FAIL / UNRUN.** Reasoned about but not executed is UNRUN, never a pass.
2. **Never report a bare count of passes.** Report per hop, so the reader sees the shape of what
   was and was not tested. "5 of 10 passed" treats a file count and a value check as exchangeable,
   and it is the artefact that leaves the room and becomes "the pre-registered suite passed".
3. **A FAIL is a FAIL.** Not a caveat.
4. **A badly designed probe is recorded as a defect in this document**, re-run corrected — not
   dropped, not counted.
5. **R / D labels, against a stated scope.** **R** = tests a failure mode already observed
   *anywhere on this platform*. **D** = not yet observed anywhere on this platform. v2 claimed
   "majority-D" while labelling as D things that were measured and published (the bloom, M54, the
   seed effect). Honest count here: **the delivery suite is majority R, and says so.**

---

## L1 — pod: produce and pool

**P1 — the pool is 128 draws, not 16 and not 48.** `R` · *v1 P1, retained*
`8 models × 16 = 128` per regression target, matching `expected_models` and
`expected_samples_per_model`.
*Falsifier:* any count below 128, or a run completing with fewer than 8 contributors. A pool that
quietly drops a constituent still produces a plausible forecast.
*Executable:* yes, pod-side.

---

## L2 — Hop-A: publish to the producer store

**P2 — exactly three targets are published, by name.** `R` · *v1 P2, retained*
Manifests for exactly `lr_ged_sb`, `lr_ged_ns`, `lr_ged_os`.
*Falsifier:* **6** — views-pipeline-core#536's filter did not take, and classification channels
reached the partner store. **0** — the filter computed empty and the run delivered nothing while
reporting success. Any other count.
*Status note:* PASSED on 2026-09-29 against the real store, first exercise ever.

**P3 — no torn commit.** `R` · *v1 P3, retained*
Every manifested shard resolves; no shard exists whose manifest is absent.
*Falsifier:* shards without a manifest, or a manifest referencing a missing shard.
*Status note:* FAILED on 2026-09-29 (network timeout, 35/36 on one target), then PASSED on retry.

**E3 — a re-publish is verified, not assumed.** `R`
Pooling is deterministic (measured: 25/25 anchors byte-identical across a re-pool) and filenames
carry the run id, so **a re-publish writes new names over identical bytes and every artefact can
hit content-hash dedup at once** — N successes, nothing servable. Re-publishing is this platform's
documented remedy for a torn run (views-postprocessing C-105/C-22), so this is the realistic path.
*Falsifier:* after a re-publish, any manifested name fails to resolve.
*Executable:* yes, and **this must be run at least once before the production run**, because the
remedy is the trigger.

---

## L3 — transform: the postprocessor and the curation

**P6 — the curation happened.** `R` · *v1 P6, retained*
`region = land_gaul`, `expected_cell_count = 64742`, `unmapped_count = 0`.
*Status note:* PASSED on 2026-09-29. This was v1's pre-registered "most likely silent failure" and
it did not happen.

**B1 — set equality of cells, not cardinality.** `D`
`set(delivered) == set(load_region_pgids("land_gaul"))`.
*Falsifier:* any difference either way.
**Why this exists:** P6 checks a *number*. `land_gaul ⊂ land` is verified (64,742 of 64,818, 76
dropped, zero additions), so the count is not vacuous — **but it cannot see 76 substituted cells.**
v1 predicted the coverage risk, tested cardinality, and recorded a PASS. That is the probe shaped
to confirm rather than to falsify.

**B4 — no unmapped sentinel reaches FAO.** `D`
`-1` is the unmapped sentinel (163,622 of 259,200 cells are ocean). Assert independently rather
than trusting the manifest's own `unmapped_count`.
**`unmapped_count: 0` is a null-check, not a correctness check** — see "Known wrong" below.

**B5 — both legs share one `lookup_version`.** `D`
The manifest carries **one**, not one per leg. Nothing compares the historical and forecast legs.
*Falsifier:* they differ, or the field is absent from either.

---

## L4 — Hop-B: the partner bucket

**E1 — every artefact the manifest names resolves BY NAME.** `R`
Not "did the upload return an id" — on 2026-09-29 it returned a **real id for the wrong file**.
*Falsifier:* any name the manifest references does not resolve.
*Passes-while-broken if:* we check ids, or counts. **Both were true in v1, and this is what
failed.**

**E2 — object count at L4 equals L2's, per type.** `R`
*Status note:* run-0 has 110 objects; the 2026-09-29 run had **109**. That one-object gap was the
entire incident.

---

## L5 — serve: faoapi ingest and the data endpoints

**P5 — the served forecast is OURS.** `R` · *v1 P5, restored with the endpoint corrected*
*Falsifier:* `forecast_serving_state` reports a refusal; or the served run id is not the one
recorded pod-side before the publish; or `created_at` is not today's.
**Record the pod-side run id BEFORE the publish**, or this degrades into "a forecast exists", which
is not the claim.
**Endpoint correction:** v1 read `/provenance/forecast`. That is a legacy path and it returned
"No forecast prediction files found" while `/health` said `healthy, age 0.0`. Read
`forecast_serving_state`.

**P4 — no silent stem collision.** `D` · *v1 P4, restored*
No two published shards resolve to the same consumer series. Spot-check values under a served
`sb_*` column against the pod's **regression** cube, and confirm they do **not** match the
**classification** cube for the same cells.
*Falsifier:* values under a fatality name match the probability cube.
**Now better founded than in v1:** confirmed that faoapi resolves by tokenising, that
`pred_lr_ged_sb` and `pred_cls_ged_sb` both reduce to the stem `sb`, and that **nothing raises** —
`series_of` rejects zero or ambiguous matches *within one name*, and neither of its callers checks
for a repeated stem. **This is the failure that puts probabilities in front of the UN under a
fatality label with no error anywhere.**

**P7 — the horizon is right.** `R` · *v1 P7, retained*
36 months beginning at 561.
*Falsifier:* 37 (views-models#512's off-by-one reaching the wire), or a range starting before 561.

**P11 — three targets are actually served, not one or two.** `D`

**views-faoapi has asked for this probe because its side will not catch it.** Nothing there asserts
a target count: `value_columns` is checked for **emptiness only** — in the service, in the daily
monitor *and* in the smoke test. A manifest declaring one target is internally consistent, so every
integrity assert passes and **FAO receives one violence series, labelled normally.**

*Check:* `len(value_columns) == 3` against `/health`'s `published.forecast.value_columns`, and
**27 value columns** on the grid parquet's schema (3 series x 9 quantities).
*Falsifier:* any count other than 3 and 27.

This is the one probe in this suite that is **strictly better than what the consumer has**, and it
exists only because views-faoapi said plainly what it could not catch.

**B6 — month boundaries mean the same thing at both ends.** `D`
`month_id = (year-1980)*12 + month`, epoch pinned as a module constant; 561 is 2026-09 everywhere.
Assert min, max and distinct count map to the intended calendar months at **both** ends.
*Note:* the converter returns 0 for Dec-1979 and negatives before it rather than raising, so a bad
date arrives as a plausible small integer.

---

## ACROSS — the seam. The spine of this document.

**P8 — the values served ARE the values we produced.** `D` · *v1 P8, restored and now executable*

**The only probe that cannot be satisfied by coincidence, and the one nobody owned.**

**Check.** Before the publish, record 20 random cell-months plus the 5 highest-valued, to full
precision, in a companion file. After the delivery, query the API for those same
`(month_id, priogrid_id)` pairs and compare.

**v1 could not execute this.** It flagged that FAO is served MAP and HDI summaries rather than raw
draws, and never resolved which statistic to compare — so the probe was a wish.

**Resolved.** The chain, traced and then confirmed by views-faoapi:

    grid_dataset.py:696  _compute_single_map
      -> :684            _tower_collapse
      -> forecast/summarize/estimator.py:54,:103   vfs.tower_point(frame)

where `vfs` is **`views_frames_summarize`**. So this probe imports **`views_frames_summarize.tower_point`**
— not a `views_frames` top-level alias, which does not exist — and calls the identical function on
the identical version (faoapi pins `views-frames>=1.10.2,<2`, installed 1.10.2). That removes the
"compared the wrong thing" failure mode *by construction* rather than by care.

`estimator.py:4` defines it as *"median of the narrowest canonical HDI floor"* — the definition to
reach for if a difference ever needs explaining.

*Falsifier:* values differ beyond floating-point tolerance.
*Carry this caveat, now specific:* the degeneracy observed on this path is at `sample_size == 1`,
where interval bounds collapse onto the point estimate and the served shape is indistinguishable
from a genuine HDI (views-faoapi register **C-265**). A 128-draw pool is nowhere near that, so it
should not arise — **but if P8 ever shows zero-width intervals, check that before suspecting an
estimator mismatch.**

**Why this hop has its own section.** Every reviewing session was a repo owner and contributed
probes about its own repo. **The seam spans two repos and belongs to neither**, so it was in
nobody's scope and the one probe covering it vanished unremarked. That is not one seat's blind
spot; it is **the union of five seats' priors**, which is more dangerous because it looks like
coverage.

---

## CONTROL — nothing outside the declared surface moved

**P9 — negative control.** `R` · *v1 P9, rewritten rather than restored*

**v1's premise was wrong** and this is recorded as a defect in v1, not a pass: it asserted the
historical leg must be unchanged, assuming only the forecast publish writes. The `un_fao`
postprocessor **legitimately** rewrites historical. v1 would have failed itself for correct
behaviour.

**v3's control:** nothing outside the **declared delivery surface** moved — other buckets, other
collections, and specifically **run-0's metadata document**, which `update_file_metadata` touched
during the 2026-09-29 incident and which **nobody has yet been able to read**.

*Falsifier:* any write outside the declared surface.
*Executable:* **partially.** Storage side yes — run-0's file still reads `createdAt == updatedAt`
to the millisecond, name unchanged. **The metadata document is not reachable from outside:**
faoapi exposes four `/files/…` routes, all storage, and `/provenance` reports only the served run.
It needs the operator console. **Marked UNRUN at registration rather than discovered UNRUN
afterwards.**

**Why this is not theoretical:** the Appwrite key carries 20 scopes — full read, write and delete
across every database, table, bucket and file — and both platform keys expire 2026-11-17.

---

## Known wrong, and certified anyway

So that "clean" is never read as "correct".

- **Nine delivered cells carry the wrong administrative unit.** views-datafactory C-353: the
  area-majority join ranks polygons by area in **raw EPSG:4326 — square degrees, not area** — so
  above 55° a cell is a narrow trapezium and the ranking can pick the wrong winner. Measured:
  **9 of 3,581 delivered border cells**; 8 differ at `gaul2`, 1 also at `gaul1`, none at `gaul0`.
  All nine are inside our 64,742, and **`unmapped_count: 0` is true for every one of them.** faoapi
  serves `/gaul2/` endpoints, so the labels are retrievable. Deferred (#471) because the fix
  republishes the reference layer FAO is currently served.
- **`c_id` is frozen across the forecast horizon.** `extrapolate_time` repeats the last observed
  frame and increments only the time column. `priogrid_gid`, `row`, `col` are time-invariant so
  copying is correct; **`c_id` is not.** All 36 forecast months carry one month's country map.
  Possibly intended — nothing states it, nothing checks it. *No probe here can see it: every
  target shares the same frozen `c_id`, so they agree and all pass.*
- **`lookup_version` is a digest of datafactory's ingestion, not of a GAUL edition.** It moves when
  the harvest changes for any reason. **Consequence for views-postprocessing#313: "version the
  sidecar by GAUL edition" is not currently possible** — that field is not carried into the
  artifact.

---

## What this suite cannot do

- **It cannot detect a failure class nobody has articulated.** Findings outside these probes are
  not lesser findings: v1's two most valuable results — that pooling is deterministic, and that
  uploads can report success having written nothing — were both **accidents**.
- **Durability.** A forecast that lands and is then evicted, expired or superseded passes every
  probe here at the moment of measurement. *(v1 said this; v2 dropped it.)*
- **The production run differs.** A green suite at 40 lessons says the plumbing carries data
  correctly. The 300-lesson run differs in duration and memory profile, and those are exactly where
  this platform has failed before. *(v1 said this; v2 dropped it.)*
- **Whether FAO can use it.** A question for FAO. *(v1 said this; v2 dropped it.)*
- **Upstream defects.** C-353's nine cells are invisible to every probe here.
- **Skill.** A delivery can be perfect and the forecasts poor. That is
  `prereg_model_health_v1.md`, and it is a different claim.
- **The seam has no owner.** This document exists partly to give it one. That is a statement about
  the organisation, not about the code, and no probe closes it.
- **`opponent` is now inside the loop.** v2's limitations section named "someone who was not in the
  room" as its remedy. That review has happened, and this document is its result — so it is **a
  review that occurred, not a property this document has.** The next revision needs a seat that has
  not read this one.

*A limitation that arrives with a scheduled fix has been converted into a plan and no longer
constrains how the suite should be read now, which is the only job this section has. None of the
above is offered with a remedy attached.*

---

## Probe provenance — every v1 probe accounted for

v2's failure was four silent deletions sitting beside four explicit retentions in the same section.

| v1 | v3 | |
|---|---|---|
| P1 | retained unchanged | L1 |
| P2 | retained unchanged | L2 |
| P3 | retained unchanged | L2 |
| P7 | retained unchanged | L5 |
| P4 | **restored**, better founded | L5 |
| P5 | **restored**, endpoint corrected | L5 |
| P6 | retained **and** strengthened by B1 | L3 |
| P8 | **restored**, now executable via `tower_point` | ACROSS |
| P9 | **rewritten** — v1's premise was wrong | CONTROL |
| P10 | **moved** to `prereg_model_health_v1.md` as C6 | — |

---

## Acknowledgements

`opponent` found the structural defect, the four dropped probes, the org-chart diagnosis, and that
I had misremembered v2's size by a factor of 2.2. `views-faoapi` established the refusal mechanism
and the authoritative endpoint. `views-datafactory` supplied C-353, the subset proof, the epoch,
and the cross-end criticism. `views-hydranet` supplied the model-health suite. `views-postprocessing`
corrected three of my claims. `library` supplied the reporting rules and the falsifier-first method.

**v1 was written alone. Every serious defect in v1 and v2 was found by someone else.**
