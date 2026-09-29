# Pre-registration v2 — proving (or falsifying) an FAO delivery

**Written 2026-09-29, after the first end-to-end run failed and after adversarial review by four
other sessions.** Supersedes `prereg_rusty_bucket_landing_20260929.md`, which is retained
unedited as the record of what v1 was and how it did.

---

## 0. What this suite is, and what it cannot be

**It tests the failure modes we can currently articulate. It does not and cannot test for unknown
unknowns. A finding outside the registered probes is not a lesser finding** — v1's two most
valuable results (that pooling is deterministic, and that uploads can report success having
written nothing) were both accidents, not probes.

This is stated first because v1 lacked it, and because a pre-registration that makes unexpected
findings feel illegitimate is working against its own purpose.

**The form's limits, named so they cannot flatter us.** Pre-registration is borrowed from clinical
trials, where it prevents fishing over continuous outcomes. Our failure mode is different: not
post-hoc rationalisation but **failure of imagination**. This platform has both, so the form still
earns its place — but it addresses only the first, and its credibility should not be borrowed for
the second.

---

## 1. How v1 did, honestly

Not "5 of 10 passed". That framing treats probes as exchangeable units when a file count and a
value check are nothing alike, and it is a number that leaves the room and becomes "the
pre-registered suite passed majority".

| what was tested | probes | result |
|---|---|---|
| **structure** — counts, spans, alignment, names | 5 | all passed |
| **value correctness** — are the numbers right | **0** | **none existed** |
| **cross-end agreement** — do both ends mean the same thing | **0** | **none existed** |
| **delivery mechanism** — did the upload write what it said | **0** | **none existed, and this is what failed** |
| served to FAO | 1 | FAILED |
| blocked behind the failure | 3 | UNRUN |
| defective premise | 1 | assumed only one component writes |

**The honest summary: v1 tested whether the delivery was well-formed. It could not have told us
whether it was correct.**

**The birth test I failed.** Of v1's ten probes, I expected roughly **one** to fail when writing
them. Writing probes you expect to pass is this method's equivalent of p-hacking. v2's probes are
written falsifier-first — *what observation would let this pass while the system is broken?* — and
a probe whose falsifier is hard to state is too weak to keep.

---

## 2. Reporting rules

1. **PASS / FAIL / UNRUN.** A probe reasoned about but not executed is UNRUN, never a pass.
2. **Never report a bare count of passes.** Report by class (§4), so the reader sees the *shape* of
   what was and was not tested.
3. **A FAIL is a FAIL.** Not a caveat, not an asterisk.
4. **A probe that turns out badly designed is recorded as a defect in this document** and re-run
   corrected — not dropped, not counted.
5. **Every probe is labelled R (regression — tests something that has already failed) or
   D (discovery — tests a failure mode not yet observed).** v2 is deliberately majority-D; a suite
   that is all-R only ever catches yesterday.

---

## 3. The chain, correctly modelled

v1's central structural error: it treated "published" and "landed" as one step. There are five.

```
 L1  pod        8 models forecast -> rusty_bucket pools -> 128 draws
 L2  Hop-A      publish to production_forecasts   (§3.2 manifest PER (run,target))
 L3  transform  un_fao postprocessor: curate land -> land_gaul, re-emit
 L4  Hop-B      unfao_bucket                      (§4.2 manifest PER run)
 L5  serve      views-faoapi ingest -> /data endpoints
```

**Two different manifest kinds.** Conflating them cost three sessions an afternoon.

**Which endpoint is authoritative:** `forecast_serving_state` (`reason`, `fallback_available`).
`/health`'s `status` and `forecast_freshness` read the newest record in the store, **not what is
served** — they reported `healthy, age 0.0 days` while the run was refused.
`/provenance/forecast` answers a legacy resolution path. v1 believed `/provenance`.

---

## 4. The probes

### Class E — did the delivery mechanism actually do what it reported? (R)

*The class that did not exist in v1, and the one that failed.*

**E1 — every artefact the manifest names resolves by name.** (R)
Not "did the upload return an id" — it returned a real id for the wrong file. Resolve each of the
110 filenames.
*Falsifier:* any name the manifest references does not resolve. *Passes-while-broken if:* we check
ids rather than names, or check a count. **Both were true in v1.**

**E2 — object count at L4 equals L2's, per type.** (R)
Run-0 has 110; our failed run had 109.

**E3 — a re-publish is verified, not assumed.** (D)
Because pooling is deterministic and filenames carry the run id, a re-publish writes new names over
identical bytes and **all 110 can hit content-hash dedup at once** — 110 successes, nothing
servable. Re-publishing is this platform's own documented remedy for a torn run (C-105/C-22), so
this is the realistic path, not the exotic one.
*Falsifier:* after a re-publish, any name fails to resolve.

### Class B — do both ends mean the same thing? (mostly D)

*datafactory's core point: every v1 probe tested a property of the delivery in isolation. A shifted
convention, a stale lookup, a re-derived identifier all produce internally perfect artefacts.*

**B1 — set equality of cells, not cardinality.** (R)
`set(delivered) == set(load_region_pgids("land_gaul"))`.
*Falsifier:* any difference either way. *Passes-while-broken if:* we compare `len()` — **v1 did
exactly that, and I recorded it as a PASS having predicted this very risk.** `land_gaul ⊂ land` is
verified (64,742 of 64,818, 76 dropped, zero additions), so the count is not vacuous — but it
cannot see 76 *substituted* cells.

**B2 — geometry round-trip on every delivered id.** (D)
`latlon_to_pgid(pgid_to_latlon(ids)) == ids`. Vectorised, instant.
*Falsifier:* any id fails. Catches reindexing or convention change end to end.

**B3 — named-place anchors.** (D)
Five cells whose location can be stated in words (e.g. 148759 = Mekelle, Tigray, 13.25N 39.25E).
Assert coordinates *and* admin labels. A convention flip survives every aggregate check and dies
here.

**B4 — no unmapped sentinel reaches FAO.** (D)
`-1` is the unmapped sentinel (163,622 of 259,200 cells are ocean). Assert independently rather
than trusting the manifest's `unmapped_count` to have computed it.
**`unmapped_count: 0` is a null-check, not a correctness check** — see §6.

**B5 — both legs share one `lookup_version`.** (D)
The manifest carries **one** `lookup_version`, not one per leg. Nothing compares the historical and
forecast legs. Assert explicitly.

**B6 — month boundaries at both ends.** (R)
`month_id = (year-1980)*12 + month`, epoch pinned; 561 is 2026-09 everywhere. Assert min, max and
distinct count map to the intended calendar months.
*Note:* the converter returns 0 for Dec-1979 and negatives before, rather than raising — a bad date
arrives as a plausible small integer.

### Class C — are the numbers right? (all D)

*hydranet's contribution. v1 had nothing here at all.*

**C1 — total mass by horizon, h=1..36, per model and pooled.** (D)
**The highest-value number we were not computing.** A healthy gated rollout is flat or mildly
decaying. **Monotone growth is the bloom** (Epic #193, ADR-070) — gate/body decoupling in the
36-step autoregressive rollout, measured at ~230× unfixed and ~12× fixed. All 8 configs set
`rollout_feedback: 'sample'`, which is the fix, **so we should be in the fixed regime — measure it,
do not assume it.**
*Falsifier:* h36 total many times h1.

**C2 — zero-fraction by horizon.** (D)
Both directions are failures: rising toward 1.0 is zero-collapse; falling is the bloom. Flat is
healthy.

**C3 — zero-fraction decomposed by model; exactly 3 must be all-or-nothing.** (D)
**The gate is not one mechanism.** 5 models are `soft_gate` (per-draw Bernoulli); 3 are
`threshold_gate` with *different* τ — `violet_visitor` 0.20, `blazing_meteor` 0.16, `bold_comet`
0.14 (verified in the configs). Threshold models emit blocks of 16 that are all-zero or
all-nonzero.
*Falsifier:* the count of all-or-nothing models is not exactly 3 — a config did not load as
believed.
**Consequence to record, not test:** the pooled zero-fraction is a mixture of two mechanisms and is
**not** interpretable as "the ensemble's P(event) = nonzero/128". No single occurrence threshold
exists across the eight.

**C4 — draw-block ordering, checked by behaviour not metadata.** (D)
Draws 0–15 are model A, 16–31 model B, and so on. Confirm the all-or-nothing blocks land on the
three threshold models. A free axis-order check that uses model behaviour rather than trusting
metadata.

**C5 — spatial correlation against the last observed month.** (D)
The h=1 forecast's nonzero cells should overlap the truth month heavily. **This repo measured (M54)
that rolling the recurrent state by 90 cells moves the forecast by 90 cells with r≈0.90 while skill
collapses 48×.** A translated forecast is cell-count-perfect and completely wrong; no structural
probe can see it.

**C6 — order of magnitude against truth, with a directional threshold.** (D)
v1's P10 said "must look undertrained" with no threshold, which is not a test. The threshold:
**the ensemble should undershoot observed totals by roughly 5–20×.** The body is documented as
timid (`size_ratio` measured as low as 0.02).
*Falsifier — and note the direction:* **an ensemble that matches observed totals is the alarming
result, not the good one.**

**C7 — gate↔body target pairing.** (D)
`to_cube_samples` slices `gate[..., :n_reg]` and applies gate channel *j* to regression target *j*,
positionally. **No validator asserts `classification_targets[j]` is the gate for
`regression_targets[j]`.** A reordered config gates sb's body with ns's gate: right shape, right
cells, right totals, wrong pairing, no error. Assert the two lists are the same targets in the same
order, read from the config the run actually used.

**C8 — per-model max draw: record, do not gate.** (D)
`M_max` **false-flags** — a single enormous draw once made a clean field look like "blooms
everywhere". Record it; if one model's max is orders above the others, look at that model. Do not
fail on it. *This is the correction to v1's handling of its own anchor cell (§6).*

### Class D — is the ensemble what it claims to be? (D)

**D1 — draw independence across models sharing a seed.** (D)
`torch_seed`: pink_pirate 42, violet_visitor 42, blue_stranger 43, bright_starship 43,
purple_alien 44, bold_comet 45, blazing_meteor 46, heavy_freighter 47. The cube generator is seeded
from `torch_seed` **and nothing else** — no model identity — so two models with the same seed draw
from byte-identical streams.

**Measured on calibration draws, 2026-09-29 (origin_0, `lr_sb_best`):**

| pair | identical 16-draw mask | per-draw agreement | excess over independence |
|---|---|---|---|
| seed 43 both (`blue_stranger`, `bright_starship`) | **3.42%** | 72.21% (indep. 69.51%) | **+2.70 pp** |
| control (44 / 47) | 0.93% | 69.67% (indep. 69.68%) | **−0.01 pp** |

The control is exactly independent. The shared-seed pair shows 3.7× the identical-mask rate and a
real excess. **The mechanism is confirmed; the magnitude is modest** — consistent with the NB
rejection sampler diverging the streams after the first target's draws. So 128 pooled draws are not
128 independent draws, though they are much closer than the mechanism alone would suggest.
*Falsifier for a run:* excess over independence materially above the ~3 pp measured here.
**Why the determinism check cannot substitute:** re-pooling and getting byte-identical output proves
*determinism*, not *non-duplication*. That is precisely the gap.

### Class A — structure (R, and known weak)

v1's P1, P2, P3, P7 retained unchanged: 128 draws; exactly 3 targets published by name and 3
withheld; 36 shards per target with a manifest each; months 561–596.

**Kept because they are cheap and catch gross regressions. Labelled weak because passing them is
nearly uninformative** — they all passed on a delivery that was unservable.

---

## 5. Cross-end, not self-consistent

The single sharpest criticism received: **every v1 probe tested a property of one artefact.**
`expected_cell_count`, `unmapped_count`, row counts are all self-consistency checks. The failures
that matter produce artefacts that are internally perfect.

Class B exists entirely to compare the two ends. Where a probe can be written as either a
self-check or a cross-check, **write the cross-check.**

---

## 6. Known-wrong things we will certify anyway

Stated so that "clean" is never read as "correct".

- **Nine delivered cells carry the wrong administrative unit, right now.** views-datafactory C-353:
  the area-majority join ranks polygons by area in raw EPSG:4326 — square degrees, not area — so
  above 55° the ranking can pick the wrong winner. Measured: **9 of 3,581 delivered border cells**;
  8 differ at `gaul2`, 1 also at `gaul1`, none at `gaul0`. All nine are inside our 64,742, and
  `unmapped_count: 0` is true for every one. faoapi serves `/gaul2/` endpoints, so the labels are
  retrievable. Fix deferred (#471) because applying it republishes the reference layer FAO is
  currently served.
- **`c_id` is frozen across the forecast horizon.** `extrapolate_time` repeats the last observed
  frame and increments only the time column. `priogrid_gid`, `row`, `col` are time-invariant so
  copying is correct; **`c_id` is not.** All 36 forecast months carry the last observed month's
  country map. Anything aggregating grid→country by `c_id` uses one month's map for three years.
  This may even be intended — but nothing states it and nothing checks it. *v1 could not have seen
  this: it checked identifier alignment across targets, and all targets share the same frozen
  `c_id`, so they agree and all pass.*
- **`lookup_version` is a digest of datafactory's ingestion, not of a GAUL edition.** It moves when
  the harvest changes for any reason, including a re-harvest producing identical geography.
  **Consequence for views-postprocessing#313: "version the sidecar by GAUL edition" is not
  currently possible** — that field is not carried into the artifact.
- **v1's anchor cell was misread.** 121/128 zeros with one draw of 29,536 was recorded as
  "undertrained behaviour". It is the shape of gate/body decoupling: all 7 nonzero draws came from
  soft-gate models, implying a gate near 0.09, while the body emitted `log1p ≈ 10.29` against a
  training range topping out near 11.6 — a near-maximum value in a cell the gate believes is empty.
  **One cell cannot settle it** (M_max false-flags), which is why the diagnostic is C1, not this.

---

## 7. What this suite still cannot do

- It cannot detect a failure class nobody has articulated. §0.
- It cannot see upstream defects: C-353's nine mislabelled cells are invisible to every probe here.
- It says nothing about forecast **skill**. A delivery can be perfect and the forecasts poor.
- It is written by the seat that ran the failed delivery, so it inherits that seat's priors.
  **The standing recommendation is that someone who was not in the room writes probes for the next
  revision** — the `opponent` session exists for exactly this.

---

## 8. Acknowledgements, because the corrections were load-bearing

`views-faoapi` established the refusal mechanism and that coverage curation was correct.
`views-postprocessing` corrected three of my claims, including that xarray is capped rather than
unpinned, and that my proposed findability fix would not have worked. `views-datafactory` supplied
C-353, the subset proof, the epoch, and the cross-end criticism that reshaped this document.
`views-hydranet` supplied all of Class C and the seed mechanism in Class D. The `library` seat
supplied §0, §1, §2 and the falsifier-first method.

**v1 was written alone in a morning. Its most serious defects were all found by someone else.**
