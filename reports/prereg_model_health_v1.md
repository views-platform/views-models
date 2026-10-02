# Pre-registration — is the forecast itself sound?

**Written 2026-09-29.** Sibling to `prereg_fao_delivery_v3.md`. Neither is a superset of the other.

These probes were first drafted inside `prereg_fao_delivery_v2.md` as "Class C and D". `opponent`
established that they do not belong there, and it is right: **every probe here is computed
pod-side from the raw 128-draw cube, and FAO is served MAP and HDI summaries. None of them can
cross the seam.** Keeping them in a delivery document created the impression that "are the numbers
right?" was covered for the delivery, when what was covered is "are the numbers healthy at the
source" — a different claim, with different falsifiers, for a different reader.

Substantively this is views-hydranet's work. It is recorded here because views-models runs the
models and owns the decision to ship what they produce.

---

## The claim

> The pooled forecast is a sound object: the rollout has not diverged, the eight models composed
> the way their configs say, the draws are what they claim to be, and the magnitudes are the ones
> an undertrained ensemble should produce.

**This says nothing about skill.** Whether the forecasts are *good* is evaluation's question, and
it is answered by metrics against outcomes, not by any probe here. A model can be perfectly healthy
and forecast poorly.

---

## R / D, against a stated scope

**R** = tests a failure mode already observed *anywhere on this platform*.
**D** = not yet observed anywhere on this platform.

Applied honestly, **this suite is majority R** — the bloom, the translation failure, the timid
body and `M_max` false-flagging are all measured, published and carry issue numbers. v2 labelled
them D and claimed to be "deliberately majority-D", which inverted the truth. A smaller honest
count is worth more than a larger one that cannot be defended.

---

## Reporting rules

PASS / FAIL / UNRUN. A probe reasoned about but not executed is UNRUN. Never report a bare count
of passes — report per question, so the reader sees what was and was not tested. A FAIL is a FAIL.
A badly designed probe is recorded as a defect in this document and re-run corrected.

---

## The rollout

**C1 — total mass by horizon, h = 1…36, per model and pooled.** `R`

**The highest-value number this platform was not computing.** A healthy gated rollout is flat or
mildly decaying across the horizon. **Monotone growth is the bloom** — gate/body decoupling in the
36-step autoregressive rollout (Epic #193, ADR-070), measured at roughly **230× unfixed** and
**12× fixed**.

All eight configs set `rollout_feedback: 'sample'`, which is the fix. **So we should be in the
fixed regime — measure it, do not assume it.**

*Falsifier:* the h=36 total is many times the h=1 total.
*Executable:* yes, pod-side, from the pooled cube.

**C2 — zero-fraction by horizon.** `R`

Both directions are failures. Rising toward 1.0 is the zero-collapse this repo has fought for
months; falling is the bloom. **Flat is healthy.**

---

## The composition

**C3 — zero-fraction decomposed by model; exactly three must be all-or-nothing.** `D`

**The gate is not one mechanism.** Verified in the configs:

| gate | models | τ |
|---|---|---|
| `soft_gate` — per-draw Bernoulli | blue_stranger, bright_starship, heavy_freighter, pink_pirate, purple_alien | — |
| `threshold_gate` — deterministic mask | violet_visitor | **0.20** |
| | blazing_meteor | **0.16** |
| | bold_comet | **0.14** |

`compose_samples` draws `torch.bernoulli(gate)` **independently per draw** for soft gates, and
applies `(gate >= τ)` to the whole cell at once for threshold gates. So the three threshold models
contribute **blocks of 16 that are all-zero or all-nonzero**.

*Falsifier:* the count of all-or-nothing models is not exactly three — a config did not load the
way we believe.

**Consequence to record, not test:** the pooled zero-fraction is a **mixture of two mechanisms**
and is **not** interpretable as "the ensemble's P(event) = nonzero/128". τ differs across the three,
so *"the ensemble's implied occurrence threshold"* is not a number that exists. A cell just below
0.14 / 0.16 / 0.20 loses 48 draws to exact zero regardless of how large its body is.

**C4 — draw-block ordering, checked by behaviour rather than metadata.** `D`

Draws 0–15 are model A, 16–31 model B, and so on. Confirm the all-or-nothing blocks land on the
three threshold models. **A free axis-order check on the draw axis that uses model behaviour
instead of trusting metadata.**

**C7 — the gate↔body target pairing.** `D`

`to_cube_samples` slices `gate[..., :n_reg]` and applies gate channel *j* to regression target *j*,
**positionally**. **No validator asserts that `classification_targets[j]` is the gate for
`regression_targets[j]`** — `config_initializer` checks only a count on `pos_weight`.

It holds today by convention. A config that reorders one list gets sb's body gated by ns's gate:
right shape, right cells, right range, right totals, **wrong pairing, nothing errors.**

*Check:* assert the two lists are the same targets in the same order, per model, read from the
config the run actually used.
*Ownership:* the validator belongs in views-hydranet; this probe is the interim guard.

---

## The magnitudes

**C5 — spatial correlation against the last observed month.** `R`

The h=1 forecast's nonzero cells should overlap the truth month heavily.

**Why this matters more than it sounds:** this repo measured (M54) that **rolling the recurrent
state by 90 cells moves the forecast by 90 cells with r ≈ 0.90 while skill collapses 48×.** A
translated forecast is cell-count-perfect, total-perfect, and completely wrong. **No structural
probe anywhere can see a translation.**

**C6 — order of magnitude against truth, with a direction.** `R`

v2's predecessor said "it must look undertrained", which is not a test because it has no threshold.

**The threshold: the ensemble should undershoot observed totals by roughly 5–20×.** The body is
documented as timid; `size_ratio` has been measured as low as **0.02**.

*Falsifier, and note the direction:* **an ensemble that matches observed totals is the alarming
result, not the good one.** At 40 lessons, matching means something composed wrong.

*Caveat:* the reference magnitude for observed global state-based fatalities must be taken from our
own truth data, not from recollection.

**C8 — per-model max draw: record, do not gate.** `R`

**`M_max` false-flags.** A single enormous draw once made a clean field look like "blooms
everywhere" when there was no bloom. Record it; if one model's maximum is orders above the others,
look at that model. **Do not fail on it.**

*This is the correction to how v1 handled its own anchor cell — see below.*

---

## The ensemble's identity

**D1 — draw independence across models sharing a seed.** `R`

`torch_seed`, verified in the configs: pink_pirate **42**, violet_visitor **42**, blue_stranger
**43**, bright_starship **43**, purple_alien 44, bold_comet 45, blazing_meteor 46, heavy_freighter
47.

The cube generator is seeded from `torch_seed` **and nothing else** — no model identity — and
`to_cube_samples` derives its per-step seed as `base + pass_index*1_000_003 + tt*10_007`. So two
models with the same seed draw from **byte-identical random streams**.

The pair that matters: **blue_stranger and bright_starship — same seed, both soft-gated, both
D×K = 4×4.** Their Bernoulli masks are `1{u < gate}` computed from the same `u`.

**Measured 2026-09-29** on calibration draws (`origin_0`, `lr_sb_best`):

| pair | identical 16-draw mask | per-draw agreement | excess over independence |
|---|---|---|---|
| seed 43 both | **3.42%** | 72.21% (independent: 69.51%) | **+2.70 pp** |
| control, seeds 44 / 47 | 0.93% | 69.67% (independent: 69.68%) | **−0.01 pp** |

The control is exactly independent. The shared-seed pair shows 3.7× the identical-mask rate and a
real excess. **The mechanism is confirmed; the magnitude is modest** — consistent with the NB
rejection sampler diverging the streams after the first target's draws.

**So 128 pooled draws are not 128 independent draws**, though far closer than the mechanism alone
would suggest.

*Falsifier for a future run:* excess over independence materially above the ~3 pp measured here.

**Three limits of this measurement, stated because they are easy to omit:**

1. **Calibration data, not forecasting.** The rejection sampler is the mechanism credited for
   divergence, and rejection rates are distribution-dependent — so ~3 pp does not transfer to
   forecasting draws by argument, only by assumption.
2. **n = 1 control.** One control pair, one shared-seed pair, no error bar. "Materially above
   ~3 pp" inherits an undefined *materially* from a single measurement.
3. **Determinism cannot substitute for this.** Re-pooling to byte-identical output proves
   *determinism*, not *non-duplication*. That is exactly the gap, and it is why the determinism
   result — however satisfying — does not speak to this at all.

**D1 does not stand alone in its class.** Seed-sharing is one way the eight configs fail to be
eight independent models; config duplication and shared initialisation are others, untested here.

**And this compounds with C3, which the v2 taxonomy separated.** C3 says the pooled zero-fraction
is a mixture and not a probability. D1 says the effective draw count is inflated. **Both bear on
any credible interval FAO is served**, and neither is visible from the other's section.

---

## The anchor cell v1 misread — a worked example of C8's rule

v1 recorded a cell with 121 of 128 draws at zero, one draw at 29,536, mean 283.8, as "undertrained
behaviour" and moved on.

**It is not undertraining. It has the shape of gate/body decoupling.** 121 zeros means ≤7 nonzero;
the threshold models emit in blocks of 16, so none of those 7 came from them — all came from the
five soft models, implying a gate around **0.09**. Meanwhile `log1p(29536) = 10.29` against a
training input range topping out near **11.6**. The body emitted a near-maximum value in a cell
the gate believes is almost certainly empty.

**But one cell cannot settle it**, and that is the point of C8: `M_max` false-flags, and v1 handed
itself an `M_max` and read it as reassurance. The diagnostic is **C1**, not this.

---

## What this suite cannot do

- **Skill.** Not tested here, not testable here.
- **Anything after the pod.** Every probe reads the raw cube. The delivery is
  `prereg_fao_delivery_v3.md`, and a green run here says nothing about whether FAO received it.
- **The production run.** A green 40-lesson run says the composition is sound at 40 lessons. The
  300-lesson run differs in duration and memory profile, and those are where this platform has
  failed before.
- **Failure modes nobody has articulated.** Findings outside these probes are not lesser findings.
- **Upstream data defects.** views-datafactory C-353's nine mislabelled cells are invisible here.

---

## Acknowledgements

Every probe in this document originates with the **views-hydranet** session, which was asked what
could be wrong with the numbers that a shape check would pass, and answered with six findings, the
measurement that should replace `M_max`, and the seed mechanism. The seed effect was measured here
rather than relayed. `opponent` established that these probes needed their own document.
