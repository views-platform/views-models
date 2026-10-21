# S7 / #572 — M1–M4 readout

**Date:** 2026-10-21 · **Cost:** $0 (local CPU over retrieved weights) · **Seat:** views-models

This seat measures. It does not author the prereg outcome — `2026-10-18_weight_decay_arms_prereg.md`
belongs to the views-hydranet seat, and the numbers below are handed over, not written in.

## Method

Seven arms swept with the **committed** harness (`2026-10-12_underfire_diagnosis/p2_gate_calibration.py`)
via `run_p2.py`, which patches `vehicle.EXPECTED_SIDECAR["output_distribution"]` `'nb' -> 'truncated_nb'`
under an assert and changes nothing else. Posterior-mean gate over D=4 locked-dropout draws,
`BASE_ORIGIN=456`, `N_ORIGINS=13`, `LEADIN_START=421`, CPU, P2 bin edges.

Run in an isolated directory so nothing in views-hydranet was written. **A previous run of this
harness overwrote that seat's `2026-10-12_underfire_diagnosis/p2_result.npz`** (it writes next to
its own script); the file was regenerated from the Oct-12 vehicle and verified to carry
sha `4acbf4d3…`, verdict D2, pos_weight 2.0.

**Common bin support, 11 bins** — a bin enters a fit only where every arm has `0 < p < 1`. Four of
those have `n >= 1000` in all six. `silver014` joins without changing the intersection: its gate
cube is bit-identical to `silver`'s, so its usable bins are the same bins. Fitting each arm on its own bins makes a `b` difference partly a
difference in which bins were fitted; the numbers below do not.

## M4 — reproducibility: **bit-exact**

`silver_kestrel` is `amber_kestrel`'s config at the same seed (42). It did not land "far from
1.1875" — it landed **on top of amber**:

| | |
|---|---|
| weight tensors compared | **190** |
| bit-identical | **190** |
| gate cube `(4,13,64818)` | `np.array_equal` **True**, `max|diff| = 0.0` |

Different pods, different days, views-hydranet **0.1.2 vs 0.1.3**, different file `sha256`
(metadata only). Two consequences, both measured rather than argued:

- **`0.1.2 -> 0.1.3` is numerically inert** for this config.
- **`0.1.3 -> 0.1.4` is numerically inert too** — added 2026-10-21 from `silver014`, seed 42 on
  0.1.4: again 190/190 bit-identical tensors and `np.array_equal` True on the gate cube. **Three
  library versions, one model.**
- **Pod hardware does not affect the result**, so it is excluded as an explanation for anything else.
- ⛔ **Stochastic run-to-run variation is zero.** At fixed config+seed+version this pipeline is
  deterministic. "Run-to-run" σ̂ therefore measures *seed* sensitivity, and nothing else.

## M2 — does `weight_decay: 0.0` train?

**Yes.** `indigo_kestrel` reached `Lesson 300/300` and produced a loadable artifact
(`b0d8829c…`, sidecar present). **Surviving runs: 1 of 1 launched; 1 of 2 planned arms run**
(`russet_kestrel` has never been launched). Stated before any slope below, per the DoD.

## M3 — `a`, `b` and calibration error, common support

`b` (an honest gate gives 1.000):

| arm | seed | ver | wd | pos_w | n>=1000 | all | sqrt-n |
|---|---|---|---|---|---|---|---|
| amber | 42 | 0.1.2 | 0.1 | 1.0 | 1.1529 | 1.1615 | 1.1385 |
| copper | 42 | 0.1.3 | 0.1 | 2.0 | 1.4190 | 1.2507 | 1.4559 |
| silver | 42 | 0.1.3 | 0.1 | 1.0 | 1.1529 | 1.1615 | 1.1385 |
| silver014 | 42 | **0.1.4** | 0.1 | 1.0 | 1.1529 | 1.1615 | 1.1385 |
| bronze | 43 | **0.1.4** | 0.1 | 1.0 | 1.0939 | 0.9880 | 1.0632 |
| cobalt | 44 | **0.1.4** | 0.1 | 1.0 | 1.0906 | 1.0157 | 1.0634 |
| indigo | 42 | **0.1.4** | **0.0** | 1.0 | 1.2578 | 1.1744 | 1.2357 |

`a` (an honest gate gives 0.000):

| arm | n>=1000 | all | sqrt-n |
|---|---|---|---|
| amber | +0.1002 | +0.1300 | +0.0222 |
| copper | +1.6451 | +0.9016 | +1.7768 |
| silver | +0.1002 | +0.1300 | +0.0222 |
| silver014 | +0.1002 | +0.1300 | +0.0222 |
| bronze | −0.1837 | −0.6975 | −0.4057 |
| cobalt | −0.3734 | −0.7159 | −0.5551 |
| indigo | +0.0403 | −0.3374 | −0.1010 |

Signed error beside absolute, as required — and it carries the clearest signal in the readout:

| arm | median \|gate−p\| | median signed | sign pattern across the 11 bins |
|---|---|---|---|
| amber | 0.0209 | −0.0062 | **crosses zero** (+ low, − high) |
| silver | 0.0209 | −0.0062 | **crosses zero** (+ low, − high) |
| silver014 | 0.0209 | −0.0062 | **crosses zero** — identical to `silver` in every bin |
| copper | 0.1438 | −0.1438 | negative throughout |
| bronze | 0.1352 | **+0.1352** | **positive throughout** |
| cobalt | 0.1218 | **+0.1218** | **positive throughout** |
| indigo | 0.0598 | +0.0598 | positive, then ~0 at the top bin |

⛔ **CORRECTED 2026-10-21 by pod five.** This section originally read *"the two `<= 0.1.3` arms
cross zero, the three `0.1.4` arms do not"* and read the split as a **version** effect. That was
wrong. `silver014` — seed 42 on **0.1.4** — reproduces `amber`/`silver`'s zero-crossing exactly,
bin for bin. The split is by **seed**: 42 crosses zero, 43 and 44 do not. It remains a qualitative
difference in shape rather than a shift in magnitude, and still larger than anything in `b` — but
it is a seed effect.

## M1 — σ̂_b, and why it is reported as two numbers

⚠️ **The common-support arm set is a second unpinned parameter.** The prereg mandates common bin
support but says only "every arm", which can mean every arm in the experiment or every arm in the
variance cell. The two readings move σ̂ by up to 19%. Both are reported; neither is selected.

**Common support over all six arms (11 bins; 4 with n≥1000):**

| estimator | as launched {42,43,44} *mixed version* | same-version pair {43,44} |
|---|---|---|
| common n>=1000 | 0.0351 (n=3, df=2, range 0.0623) | **0.0023** (raw \|Δ\| 0.0033 ÷ √2) |
| common all | **0.0932** (range 0.1735) | **0.0196** (raw \|Δ\| 0.0277 ÷ √2) |
| common sqrt-n | 0.0434 (range 0.0753) | **0.0002** (raw \|Δ\| 0.0003 ÷ √2) |

**Common support over the three cell members only (12 bins; 5 with n≥1000)** — the clean cell,
seeds 42/43/44 all on 0.1.4, `silver014` in place of `silver`:

| estimator | b per seed (42, 43, 44) | σ̂_b | vs W-5 0.0601 | vs 0.0356 |
|---|---|---|---|---|
| n>=1000 | 1.1462, 1.0764, 1.0879 | 0.0374 | under | **over** |
| all_bins | 1.1771, 0.9733, 1.0003 | **0.1107** | **OVER** | **over** |
| sqrt-n | 1.1389, 1.0575, 1.0599 | 0.0463 | under | **over** |

⛔ Because `silver014` ≡ `silver` bit for bit, **the clean cell and the as-launched cell are the
same measurement.** Removing the version confound changed nothing.

⛔ **CORRECTED — the feasibility threshold I first used is not the live one.** This section
originally read *"σ̂_b exceeds the 0.0356 feasibility bound under every estimator … unlike the W-5
verdict this is robust"*. **Prereg Amendment 10 (`:1018`) explicitly retires 0.0356**: it was
derived for n=1 vs n=2, the approved set is n=3 vs n=2, and the plan's critical value is *"1.81×
too harsh"*. The live boundaries are **0.0645 pooled** (`:1015`) and **0.0477 conservative**
(`:1016`).

| threshold | source | 0.0374 | 0.0463 | 0.1107 | verdict |
|---|---|---|---|---|---|
| **0.0645** feasibility, pooled | A10 `:1015` | under | under | **over** | **straddles → NULL** |
| **0.0477** feasibility, conservative | A10 `:1016` | under | under | **over** | **straddles → NULL** |
| **0.0601** W-5 separability | A10 `:1033` | under | under | **over** | **straddles → NULL** |
| ~~0.0356~~ | retired by A10 | over | over | over | *not a live reading* |

⛔ **All three live readings are NULLs, and all three are null for the same reason:** two
estimators sit under every boundary and one sits above all of them. The design is not shown
unreadable — it is shown *unresolved*.

⚠️ **The lone dissenting estimator is `all_bins`.** It gives every bin equal weight, so a bin with
**65** observations counts as much as one with **808,517**. `n>=1000` excludes those sparse bins and
√n-weighting down-weights them; both land at 0.0374–0.0463, under every live boundary. That is a
statement about the estimator's leverage, not a selection — **the prereg pins no estimator
(`:676`, `:682`), so this seat reports the family and does not choose.** Choosing is the
hydranet seat's call.

**DoD item 5, read against Amendment 10's own replicate table (`:493`–`:502`):** at n=2 per cell
the design needs σ̂ < 0.0436; at **n=3** it needs σ̂ < 0.0827; at n=4, σ̂ < 0.1084. Two of three
estimators clear even the n=2 requirement; `all_bins` clears none of them. ⛔ **So the pre-registered
response to an unresolved σ̂ applies as written: more replicates per cell, not more doses** —
Amendment 10 §4 pre-sizes that at **n=3 on two cells, 6 pods**, and calls it "the single
highest-leverage change available to this design".

⚠️ **df is 2, not the 3 the acceptance criteria ask for.** Three runs give 2 df; the planned 3 df
assumed pooling with the `indigo`/`russet` pair, and `russet` was never launched.

⛔ **W-5 is a NULL.** The as-launched family spans **0.0351 … 0.0932** and 0.0601 sits inside it,
so whether W-5 fires is a choice of estimator. The prereg (`:676`, `:682`) names `σ̂_b` and its
thresholds but pins neither the weighting nor the bin filter. I am not selecting the variant that
reads cleaner.

**The same-version pair is under both thresholds under every estimator** (0.0002–0.0196 vs 0.0601
and 0.0356) — but on **1 df**, which is as thin as an estimate gets.

## The confound — raised, then measured, then dismissed

⛔ **This section originally concluded, by elimination, that the σ̂ excess "sits on the 0.1.3/0.1.4
boundary" and that the as-launched σ̂ "is not a seed variance". Pod five measured it directly and
both statements are false.** They are kept here, struck, because they were published to #572.

The elimination argument was: stochastic noise is zero, hardware is excluded, `bronze` vs `cobalt`
isolates seed at fixed version at 0.0002–0.0196, the as-launched σ̂ is 4–200× larger, so the excess
must be the version. Structurally sound; empirically wrong. The missing term was that `bronze` vs
`cobalt` measures the 43↔44 distance, which happens to be small, and says nothing about how far
**42** sits from either.

**The direct measurement** — `silver014`, seed 42 on 0.1.4, same config, same git sha `2e73de3`,
same seeds, nothing else moved:

| compared | result |
|---|---|
| weight tensors | **190/190 bit-identical** to `silver` (0.1.3) |
| gate cube | `np.array_equal` **True** — 4×13×64818 = 3,370,536 values |
| signed error | identical in all 12 bins |
| `a`, `b` | identical to 4 dp under all three estimators |
| only differing field in the npz | `sha`, which records which `.pt` produced it |

So `0.1.3 → 0.1.4` is **numerically inert** for this config. With `amber` (0.1.2) ≡ `silver`
already established, **three library versions produce the same model.** The as-launched σ̂ was an
honest seed variance all along, and seed 42 is simply the distant draw.

⚠️ **F13 still cannot see the library version** — `ARM_IDENTICAL_FILES` and `CORE_GENOME` both
omit it, so a replicate set spanning a version boundary still passes F13 silently. That hole
(views-models C-167) is real and unclosed. What pod five shows is that *this particular* boundary
had zero effect, not that the hole is harmless.

⛔ **F13 cannot detect this.** It asserts the replicates are byte-identical across
`ARM_IDENTICAL_FILES` and identical in `CORE_GENOME` except the seed keys. **The library version
is in neither**, so a replicate set spanning a version boundary passes F13 and deflates or
inflates σ̂ silently. Same shape as the seed-collision hole Amendment 11 closed.

## No causal sentence

⛔ **CORRECTED 2026-10-21 by pod five.** This originally reported `indigo` (`weight_decay 0.0`)
against the **mean of `bronze` and `cobalt`** — seeds 43 and 44 — giving **Δb = +0.166**, and
called it 4.7σ against the as-launched σ̂. `indigo` is seed **42**, so that comparison carried the
whole 42↔{43,44} seed distance inside it. `silver014` is the matched control: seed 42, version
0.1.4, `weight_decay 0.1`, everything else identical.

| estimator | wd 0.1 (`silver014`) | wd 0.0 (`indigo`) | Δb, seed-matched | Δb, old seed-confounded |
|---|---|---|---|---|
| n>=1000 | 1.1462 | 1.2348 | **+0.0886** | +0.1527 |
| all_bins | 1.1771 | 1.1617 | **−0.0154** | +0.1748 |
| sqrt-n | 1.1389 | 1.2338 | **+0.0949** | +0.1751 |

⛔ **With the seed confound removed the Δb family straddles zero** (−0.0154 … +0.0949) and sits
inside σ̂_b (0.0374 … 0.1107). Roughly half of the original +0.166 was the seed. **On `b`, the
zero-decay arm is a NULL** — and like W-5, that null is a property of the estimator family, not of
one chosen variant. The level `a` moves consistently negative (−0.14 … −0.55) and is the one
component that does not change sign.

**F12 stands: no sentence here attributes a cause, and F12 is checked by the other seat, not
self-certified.**

## What pod five closed — RESOLVED 2026-10-21

`silver_kestrel`'s config at seed 42 on **0.1.4** answers three open questions at once:

1. **σ̂ at df=2 with no version confound** — seeds 42/43/44, all 0.1.4.
2. **The version question directly** — same seed, same config, 0.1.3 vs 0.1.4. Determinism means
   any difference is the version, with no noise term to argue about.
3. **The matched-seed control for `indigo`** — removes the seed confound from Δb above.

**Pre-registered prediction, recorded before the run:** if the version boundary is real, seed-42
on 0.1.4 shows the *positive-throughout* signed-error pattern of `bronze`/`cobalt`. If it
reproduces `amber`/`silver`'s zero-crossing, the version hypothesis is dead and `silver` was
simply a distant seed.

⛔ **OUTCOME: the second branch. The version hypothesis is dead by its own pre-registered
criterion.** `silver014` reproduced the zero-crossing exactly — 190/190 identical weight tensors,
identical gate cube. Taking the three questions in order:

1. **σ̂ at df=2 with no version confound** — 0.0374 / 0.1107 / 0.0463. Identical to the
   as-launched figures, because the confound carried no signal. **W-5 remains a NULL**; the
   **0.0356 feasibility bound is exceeded robustly.**
2. **The version question** — answered: `0.1.3 → 0.1.4` is numerically inert for this config.
3. **The matched-seed control for `indigo`** — delivered, and it **halved the zero-decay Δb and
   made it cross zero**: +0.0886 / −0.0154 / +0.0949 against the previously published +0.166.
   See "No causal sentence". **F12 stands: stated, not interpreted.**

`silver` was a distant seed. The run-to-run variance the prereg set out to measure was being
measured correctly the whole time.

## Artifacts

`results/p2_result_{amber,copper,silver,silver014,bronze,cobalt,indigo}.npz`, sha256-verified
after copy (`silver014` = `17b8adaf…`) · `version_single_variable.py` (the single-variable version
test and both σ̂ cells) · `sweep_silver_0_1_4.log` ·
`sigma_run_to_run.py` (deliberately **not** named `sigma`: `e1_sigma.py` is the across-draw logit
sd against floors 0.85/1.20, ~20× a different scale from this across-run slope sd) ·
`readout_m1_m4.py` · `sweep_all.sh` + `sweep_all.log`
