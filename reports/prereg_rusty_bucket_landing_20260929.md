# Pre-registration — proving (or falsifying) that `rusty_bucket` landed at FAO

**Written 2026-09-29 15:30 CEST, BEFORE the pooling, the publish, or the delivery had run.**
Nothing in this document was written with a result in hand. That is the point: every threshold
below is committed in advance so that a disappointing outcome cannot be re-read as a success.

**Status when written:** 3 of 8 models forecast. Pooling has never run. `rusty_bucket` has never
been published. The chain below has never executed end to end in the history of this platform.

---

## The claim under test

> The 40-lesson test forecast produced by `rusty_bucket` on 2026-09-29 has landed in the UN FAO
> delivery and is being served to FAO.

**This is a claim about IDENTITY, not QUALITY.** The models are deliberately undertrained. No
probe below asks whether the numbers are good, and none can. A suite that confirmed "forecasts
are present and plausible" would pass just as happily on someone else's run, on a cached
artefact, or on a partial commit. Every probe is therefore designed to distinguish **our** run
from **any** run.

---

## The baseline, recorded before the fact

Read live at 2026-09-29 15:28 CEST. **This is the anchor. Without it, nothing below is provable.**

```
GET /version
  {"version":"1.7.1","deployed_tag":"v1.7.1","served_contract_version":"1.5"}

GET /provenance/forecast
  {"detail":"No forecast prediction files found in the bucket: unfao_bucket"}

GET /provenance/historical
  file_id      6a7d5d960028cdbb1c04
  filename     historical_dataset_20260813_080043.parquet
  created_at   2026-08-13T06:01:22.142+00:00
  targets      lr_ged_sb, lr_ged_ns, lr_ged_os
  region       land_gaul   expected_cell_count 64742   actual 64742   unmapped 0
  file_hash    19a990b52cf07849587c928ac82d553671a7c7fc8d0872d02652ed0406c76e14
```

**Two findings from the baseline that change the test design.**

1. **There is no stale forecast to displace — there is no forecast at all.** The bucket is empty
   of them. So "a forecast exists" is already a strong signal, and any forecast appearing during
   this window is overwhelmingly likely to be ours. This makes P5 cheap and decisive, and it means
   a *negative* result is unambiguous too.

2. **The served region is `land_gaul` at 64,742 cells. We produce `land` at 64,818.** A curation
   step sits between the publish and what FAO serves — 76 cells are dropped. **Publishing is
   therefore NOT landing.** Any probe that stops at the prediction store proves nothing about FAO.
   This is the single most likely place for the chain to break silently, because both numbers look
   like "global land".

---

## The chain, and where each probe sits

```
  L1  pod        rusty_bucket pools 8 models  ->  128 draws            P1
  L2  store      publish to production_forecasts bucket                P2 P3 P4
  L3  delivery   un_fao postprocessor curates land -> land_gaul        P6
  L4  api        FAO serves it                                         P5 P7
  L5  values     what FAO serves IS what we produced                   P8
      control    nothing else moved                                    P9 P10
```

A probe may only be reported as passed if it was **executed**. A probe reasoned about but not run
is recorded `UNRUN`, never as a pass. This rule exists because it has been violated on this
platform before.

---

## Probes

Ordered cheapest-and-most-decisive first. **Each states what result would falsify the claim** —
if a probe cannot fail, it is not a probe and must be struck from this list rather than reported.

### P1 — the pool is 128 draws, not 16 and not 48

**Check.** Pooled artefacts on the pod carry `8 x 16 = 128` draws per cell-month per regression
target, matching `expected_models: 8` and `expected_samples_per_model: 16`.

**Risky prediction:** exactly **128**.

**Falsifies if:** 16 (only one constituent found), any multiple below 128 (constituents silently
skipped), or the run completes with fewer than 8 models contributing. A pool that quietly drops a
constituent still produces a plausible forecast — this is the probe that catches it.

### P2 — exactly THREE targets are published, by name

**Check.** The store holds manifests for exactly `lr_ged_sb`, `lr_ged_ns`, `lr_ged_os`.

**Risky prediction:** **3**. Not 6, not 0.

**Falsifies if:** **6** — the views-pipeline-core#536 fix did not take, and classification channels
went to the partner store. **0** — the publishable set computed empty and the run delivered
nothing while reporting success (the failure mode pipeline-core added a guard for). Any other
count — something is wrong that nobody predicted.

This probe *is* the #536 fix under test. It has never been exercised against a real store.

### P3 — no torn commit

**Check.** For each manifest, every shard it lists resolves to a file that exists in the bucket;
and no shard exists whose manifest is absent.

**Risky prediction:** shard count == `targets x months`, and manifests present for all of them.

**Falsifies if:** shards exist without a manifest (the publish died mid-run — consumers are
supposed to ignore unmanifested shards, so this would be invisible to FAO but is still a torn
run), or a manifest references a shard that is not there (worse: a commit marker for an
incomplete delivery).

### P4 — no silent stem collision

**Check.** No two published shards resolve to the same consumer series. Then spot-check values
from a published `lr_ged_sb` shard against the pod's **regression** cube — and confirm they do
**not** match the pod's **classification** cube for the same cells.

**Risky prediction:** three distinct series `sb`, `ns`, `os`; and the served `sb` values match the
regression source, not the classification source.

**Falsifies if:** values under a fatality name match the probability cube instead. views-faoapi
resolves names by tokenising, so `pred_lr_ged_sb` and `pred_cls_ged_sb` both reduce to `sb` **and
nothing raises**. This is the failure that would put probabilities in front of the UN under a
fatality label, with no error anywhere. It cannot be detected by counting files.

### P5 — the served forecast is OURS

**Check.** `GET /provenance/forecast`.

**Risky prediction:** it no longer returns `"No forecast prediction files found"`. It returns a
record whose `created_at` is **2026-09-29** and whose run identity carries the `rusty_bucket`
forecasting timestamp generated on the pod today.

**Falsifies if:** still empty (nothing landed), or a record whose date or run id is not today's
(something else landed, or a cached artefact is being served and we are about to congratulate
ourselves for someone else's file).

**Record the pod-side run id BEFORE the publish**, or this probe degrades into "a forecast exists",
which is not the claim.

### P6 — the curation happened

**Check.** The served forecast's coverage.

**Risky prediction:** `region = land_gaul`, `expected_cell_count = 64742`, `unmapped_count = 0` —
matching the historical record exactly.

**Falsifies if:** 64,818 (our raw `land` coverage was served uncurated — 76 cells FAO's lookup
cannot place), or `unmapped_count > 0` (the lookup failed for some cells and the delivery shipped
anyway), or any other count.

This is the probe for the step the baseline revealed. It is the most likely silent failure in the
whole chain, because 64,818 and 64,742 are both "global land" to the naked eye.

### P7 — the horizon is right

**Check.** Months served.

**Risky prediction:** **36** months beginning at **561** — matching what `purple_alien` produced
(verified 561–596 earlier today).

**Falsifies if:** 37 (the off-by-one already filed as views-models#512 has reached the wire), 12,
or a range that starts before 561 (history leaked into the forecast).

### P8 — the values served ARE the values we produced

**The strongest probe, and the only one that cannot be satisfied by coincidence.**

**Check.** Before the publish, select **20 cell-months at random plus the 5 highest-valued cells**
from the pod's pooled output, and record their values to full precision in this file's companion
log. After the delivery, query the API for those same `(month_id, priogrid_id)` pairs and compare.

**Risky prediction:** exact agreement on the derived statistic.

**Procedural caution, agreed in advance so a mismatch is not misread:** the API serves MAP and HDI
summaries (`sb_map`, `sb_hdi90_lower`, …), **not raw draws**. The comparison must therefore be
against the same statistic computed from our 128 draws, using faoapi's own definition — which must
be read from `views-faoapi` **before** comparing, not guessed. A mismatch caused by comparing the
wrong statistic is an error in the probe, not a falsification of the claim, and must be recorded
as such rather than as a failure.

**Falsifies if:** the values differ beyond floating-point tolerance once the statistic is
correctly matched. That would mean something between the pod and FAO altered the numbers —
a transform, a stale cache, or a different run.

### P9 — negative control: nothing else moved

**Check.** `GET /provenance/historical`.

**Risky prediction:** **unchanged** — `file_hash` still
`19a990b52cf07849587c928ac82d553671a7c7fc8d0872d02652ed0406c76e14`, file still
`historical_dataset_20260813_080043.parquet`.

**Falsifies if:** it changed. The forecast publish has no business touching historical data. Given
the API key carries write-and-delete across all buckets and files, this is not a theoretical
concern, and it is the only probe here that would catch collateral damage.

### P10 — adequacy: it must look undertrained

**Check.** The served forecast's magnitudes against what a 300-lesson run produces.

**Risky prediction:** recognisably undertrained — `purple_alien` at 40 lessons gave ~45,410
state-based fatalities over 36 months globally.

**Falsifies if:** the numbers look like a well-trained run. That would mean we published something
other than what we just produced — an older artefact, or a different model set. A result that
looks *too good* is a failure here, not a success.

---

## What this suite CANNOT prove

Stated so that nobody, including me, over-reads a green run:

- **Nothing about forecast quality.** The models are undertrained by design. Every probe tests
  identity, coverage, shape and provenance. None tests skill.
- **Nothing about the 300-lesson run.** A green suite says the plumbing carries data correctly at
  40 lessons. It does not say the production run will succeed — that run differs in duration and
  memory profile, and those are exactly where this platform has failed before.
- **Nothing about whether FAO can USE it.** P5–P8 prove the data is served and correct. Whether it
  is usable is a question for FAO, and the mail sent today explicitly tells them not to try.
- **Nothing about durability.** A forecast that lands and is then evicted, expires, or is
  superseded would still pass every probe here at the moment of measurement.

## Reporting rules, fixed in advance

1. A probe is **PASS**, **FAIL**, or **UNRUN**. There is no "probably fine".
2. Any probe not executed is reported `UNRUN`, even if its outcome seems obvious.
3. A FAIL is reported as a FAIL. It is not downgraded to a caveat, and the run is not described as
   a success with an asterisk.
4. If a probe turns out to be badly designed once run (P8's statistic-matching is the likely
   candidate), that is recorded as a **defect in this document**, and the probe is re-run
   corrected — not silently dropped, and not counted as a pass.
5. The overall verdict is **LANDED** only if P1–P9 all pass. P10 failing means we published the
   wrong artefact, which is also not LANDED.

---

*Companion log for the recorded pre-publish values: `prereg_rusty_bucket_landing_20260929_values.json`,
to be written before the publish and never after.*
