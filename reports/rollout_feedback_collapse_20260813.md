# The horizon collapse is the rollout feedback, and the blooming is its mirror

**Date:** 2026-08-13
**Test bed:** `bold_comet`, calibration, origin_6, 13,110 cells, 36-month horizon
**Method:** re-evaluate the SAME trained artifact (`calibration_model_20260812_215145.pt`)
with the SAME fetched data, varying one inference-time knob. `rollout_feedback` is read at
inference (`views_hydranet/utils/hydranet_inference.py:99`), so each test costs ~10 minutes
rather than a retrain.

---

## Verdict

**The trained models are fine. The failure is entirely in the rollout feedback, and both
deployable settings are broken in opposite directions.**

Gate — `by_sb_best`, mean P(conflict) per horizon month:

| feedback | m1 | m3 | m6 | m12 | m24 | m36 |
|---|---|---|---|---|---|---|
| `sample` *(current roster)* | 0.0301 | 0.0186 | 0.0084 | 0.0026 | 0.0010 | **0.0010** |
| `teacher_forced` *(probe)* | 0.0301 | 0.0270 | 0.0259 | 0.0288 | 0.0358 | **0.0364** |
| `mean` | 0.0301 | 0.0860 | 0.1659 | 0.3419 | 0.6471 | **0.8335** |

Body — `lr_sb_best`, mean magnitude:

| feedback | m1 | m3 | m6 | m12 | m24 | m36 |
|---|---|---|---|---|---|---|
| `sample` | 0.152 | 0.057 | 0.030 | 0.004 | 0.000 | **0.000** |
| `teacher_forced` | 0.152 | 0.121 | 0.124 | 0.126 | 0.251 | **0.168** |
| `mean` | 0.152 | 0.427 | 1.382 | 2.538 | 6.519 | **8.127** |

**The control reproduced the 2026-08-12 run exactly** (0.0301 / 0.0186 / 0.0084 / 0.0026 /
0.0010), so the comparison is valid and not run-to-run noise.

## What each result means

**`teacher_forced` holds flat.** Feed the model the truth each month and the gate neither
decays nor grows: 0.030 → 0.036 over three years. **The learned model is stable.** Nothing
about the months of modelling work is lost. This is not a deployable setting — it needs
future actuals — but as a probe it isolates the loop completely, and the loop is the whole
story.

**`sample` starves.** The model feeds back its own draws. On a target that is ~99.8% zeros,
a sampled feedback is almost always exactly zero, so the model reads "nothing happened",
lowers its gate, and feeds back an even emptier signal. Monotonic, self-reinforcing, no
floor. 30× decay by month 36.

**`mean` blooms.** 28× on the gate, 53× on the magnitude, approaching P(conflict)=0.83
everywhere. **This is the historical blooming failure, reproduced on demand.**

## Why this matters more than a bug report

The blooming problem was fought and beaten over months. The evidence here says it was beaten
**by switching the feedback mode from an expectation-like signal to a sampled one** — which
traded an explosion for a collapse. The stabilisation was real; it just moved the failure to
the far horizon, where nothing was measuring.

Two things follow:

1. **Neither available mode is correct.** `mean` over-feeds, `sample` under-feeds. The fix
   is a third behaviour, not a choice between these two.
2. **The far horizon was unmonitored.** Both failures are invisible at short horizons — all
   three settings agree exactly at m1 (0.0301). Anything that only checked the first few
   steps would have passed all three.

## Why CRPS did not catch it

From step 6 onward, CRPS is **identical to three decimals across all eight roster models**
(0.116 / 0.112 / 0.135 / 0.133 / 0.875). Once every model predicts ~zero on a zero-inflated
target, CRPS measures the actuals, not the model. Pooled, CRPS spans 0.9% across the roster
while MCR spans 10×.

**Ranking these models on CRPS ranks noise.** This is register C-84's concern in a new
instance, and it is why the MCR guardrail exists.

## What was ruled out

- **Not the training.** Same artifact across all three tests.
- **Not the data.** Same fetched parquet, `-sa` on every run.
- **Not run-to-run variance.** The control reproduced to four decimals.
- **Not model-specific.** All eight roster models show the same monotonic gate decay; the
  magnitudes differ (9× on `purple_alien`, 30× on `bold_comet`) but the shape is identical.

## Round 2 result: the draw count is irrelevant — ANSWERED 2026-08-13 11:44

`E_sample64x1` — `n_posterior_samples: 64`, `n_head_samples: 1`, i.e. the pre-roster
produced-count of 64, sampled feedback, everything else unchanged. 2h15m (64 draws is
15.7× the sampling work per origin).

| case | produced draws | m1 | m3 | m6 | m12 | m24 | m36 |
|---|---|---|---|---|---|---|---|
| control `sample` 4×4 | 16 | 0.0301 | 0.0186 | 0.0084 | 0.0026 | 0.0010 | 0.0010 |
| **E** `sample` 64×1 | **64** | 0.0299 | 0.0174 | 0.0082 | 0.0025 | 0.0010 | 0.0010 |

**Four times the draws, the same collapse, to four decimals.** Two conclusions:

1. **The roster lock's `n_posterior_samples: 64 → 4` cut is not the cause.** It changed the
   posterior width, not the horizon behaviour.
2. **"Sample more" is not the fix.** The starvation is the feedback *mode*, not sparsity of
   the draws. A sampled feedback on a ~99.8%-zero target reads as zero often enough to
   collapse the gate no matter how many draws are taken.

Worth carrying to the sample-count discussion (C-90 / C-99): more draws did not improve
magnitude here at all. Whatever the case for 128 or 512 per constituent, it cannot be made
on the basis of horizon calibration.

### `loss_reg` is also ruled out, for free

`teacher_forced` ran with `loss_reg: 'mse'` — the same post-roster loss — and held flat.
So the `shrinkage → mse` change in `f0b4436f` is **not** the cause of the horizon decay.
It may still affect magnitude calibration at short horizons, but it does not need a
retrain to rule out for this question. That saves ~90 minutes and removes the second
suspect.

### `F_sample16x1` — a third parameterisation, same answer (12:19)

16 draws taken entirely in the posterior head rather than split 4×4, i.e. the same produced
count as the control with a different D/K shape:

| case | draws | D×K | m3 | m6 | m12 | m36 |
|---|---|---|---|---|---|---|
| control | 16 | 4×4 | 0.0186 | 0.0084 | 0.0026 | 0.0010 |
| **F** | 16 | **16×1** | 0.0179 | 0.0084 | 0.0025 | 0.0010 |
| **E** | **64** | 64×1 | 0.0174 | 0.0082 | 0.0025 | 0.0010 |

**Three independent parameterisations of `sample` — 4×4, 16×1, 64×1 — collapse identically
to three decimals.** Neither the draw count nor the D/K split moves it. Combined with
`teacher_forced` holding flat under the same loss and the same artifact, the conclusion is
as tight as this method can make it.

**What remains: the feedback mode, and only the feedback mode.**

## The complete ablation — all six cases, closed 2026-08-13 14:49

`bold_comet`, one trained artifact, one dataset, one knob varied. Gate P(conflict):

| case | draws | m1 | m3 | m6 | m12 | m24 | m36 |
|---|---|---|---|---|---|---|---|
| control `sample` 4×4 | 16 | 0.0301 | 0.0186 | 0.0084 | 0.0026 | 0.0010 | **0.0010** |
| F `sample` 16×1 | 16 | 0.0300 | 0.0179 | 0.0084 | 0.0025 | 0.0010 | **0.0010** |
| E `sample` 64×1 | 64 | 0.0299 | 0.0174 | 0.0082 | 0.0025 | 0.0010 | **0.0010** |
| A `teacher_forced` | 16 | 0.0301 | 0.0270 | 0.0259 | 0.0288 | 0.0358 | **0.0364** |
| B `mean` 4×4 | 16 | 0.0301 | 0.0860 | 0.1659 | 0.3419 | 0.6471 | **0.8335** |
| G `mean` 64×1 | 64 | 0.0299 | 0.0858 | 0.1655 | 0.3395 | 0.6472 | **0.8345** |

**The draw count is irrelevant in BOTH directions.** Three `sample` parameterisations collapse
identically; two `mean` parameterisations bloom identically (0.8335 vs 0.8345 at m36). The
feedback mode determines the outcome; the sampling budget does not touch it.

Config verified restored afterwards — `sample` / 4 / 4, byte-identical to committed.

**But see the root-cause section below: this ablation characterises the rollout's response,
not the underlying defect.** All six cases share the same activation deficit at m1, which is
where the real problem lives.

## Open, and being tested

Whether `sample` starves *only* because the roster lock cut the draw count. Pre-lock the
config was `n_posterior_samples: 64` with no head samples (64 produced); the lock made it
4 × 4 = 16. With 64 draws there are 4× more chances to land a non-zero in the feedback.

Round 2 (`E_sample64x1`, `F_sample16x1`, `G_mean64x1`) tests exactly that. Note round 1's
64-sample cases were **void**: they left `n_head_samples: 4`, giving 64 × 4 = 256 produced
draws, and the RAM guard correctly refused at 6.67 GB needed vs 6.70 GB available.

Untested and next if round 2 is negative: `loss_reg` went `shrinkage` → `mse` in the same
roster lock (`f0b4436f`, 2026-08-10). That is training-time and needs a real retrain.

## ROOT CAUSE — the feedback mode is the messenger, not the cause (2026-08-13, deep trace)

### 1. The feedback IS a sample. The design decision is correctly implemented.

Traced end to end, because it was challenged and needed proving rather than asserting:

- `hydranet_inference.py:467,545` — with `rollout_feedback == "sample"`, the fed-back tensor
  is `self._sample_feedback(...)`, never `t1_pred` (the emitted mean).
- `_sample_feedback` (`:293`) draws `k=1` from the NB family — a genuine count draw.
- `compose_samples` (`distributions/composition.py:55`) applies the gate as
  **`torch.bernoulli(gate_k)`** — a real Bernoulli realisation, then a 0/1 multiply.
- The expectation path is a *separate* function, `compose_mean` (`:62`, `gate * mean`), used
  only for the emitted prediction.

**Sample and mean are correctly kept apart, and the feedback takes the sample branch.**
Any suggestion to switch the feedback to `mean` is wrong on the design's own terms and is
withdrawn.

### 2. The real defect: the gate is under-persistent by ~4.6×

Real conflict is strongly persistent. Measured on `bold_comet`'s own training parquet
(5,034,240 rows, `lr_sb_best`):

| | active fraction | **P(on\|on)** | P(on\|off) |
|---|---|---|---|
| **REAL DATA** | 0.0046 | **0.4181** | 0.0027 |
| control `sample` 4×4 | 0.0007 | **0.0901** | 0.0005 |
| `sample` 64×1 | 0.0005 | **0.0395** | 0.0004 |
| `teacher_forced` | 0.0019 | **0.1152** | 0.0016 |

A real conflict cell stays lit with p=0.42. The model's stays lit with p=0.09 — it
extinguishes at 0.91/month where reality says 0.58. Compounded over a 36-month rollout,
that is total extinction.

**`teacher_forced` reaches only 0.115 — with real inputs.** So the deficit is in the
trained model, not the inference path. `teacher_forced` looks stable because the oracle
re-injects active cells every month; the model is not sustaining them.

This explains all three modes with one mechanism:

- **`sample`** — marginal is right, but 0.09 persistence cannot sustain a population → extinction.
- **`mean`** — puts a small positive in *every* cell, so ignition fires everywhere and swamps
  the weak persistence → bloom.
- **`teacher_forced`** — truth re-lights the map each step, masking the deficit entirely.

### 3. Why the model never learned persistence — already documented

`reports/archived/2026-06-05_rollout_training_dossier/00_README.md`:

> `HydraBNUNet06_LSTM4` is trained one-step-ahead but **run 36 steps free-running** at
> inference. The prediction→input feedback loop ... receives **zero gradient** during
> training (`training_engine.py:200`, `prev_pred = t1_pred.detach()`).

One-step-ahead training never teaches multi-step persistence. The dossier is explicit that
the fix must live in **the training algorithm**, *"not an inference-time hard-prior hack"* —
which is exactly what choosing between `sample` and `mean` is.

### 4. The fix exists, is designed, and its revisit trigger has fired

`DISPOSITION.md`, 2026-06-10:

> **PARKED — documented fallback, not rejected.** **Revisit if:** the ZINB head fails the
> explosion-check or eval. Rollout training (B1 pushforward / B2 GTF) remains the principled
> answer **if the autoregressive runaway turns out to be recurrence-deep rather than
> dissolved by the ZINB softplus link.** Issues #77/#78 parked. C-139, epic #97.

The distributional head did not dissolve it — it inverted the sign. And `teacher_forced`
demonstrates the deficit is recurrence-deep. **The stated condition is met; Axis B should be
unparked.**

### 5. Why "T=0-neutral" let this through

The `sample` mitigation is documented (`hydranet_inference.py:95`) as *"T=0-neutral so the
scored T=0 product is byte-unchanged."* Its acceptance criterion was that it change nothing
at month 1 — and all three feedback modes are identical at month 1 (0.0301). The mitigation
was validated on the one horizon where it provably cannot differ.

## Recommended, once round 2 lands

1. **Do not go global on the current roster.** Whatever the draw-count answer, the horizon
   behaviour must be fixed first — global costs hours per model and would only confirm this.
2. **Take this to views-hydranet as a design question**, not a config tweak. A feedback mode
   that neither starves nor blooms is the actual requirement — scheduled sampling, a
   calibrated/dithered feedback, or feeding the gate probability rather than a realisation.
3. **Add a far-horizon guard.** A test asserting the gate at m36 stays within a band of the
   gate at m1 would have caught both failures on the day each was introduced.

## Reproduce

```
scratchpad/ablate.sh    # round 1: control, teacher_forced, mean
scratchpad/ablate2.sh   # round 2: 64x1 sample, 16x1 sample, 64x1 mean
```
Both restore `config_hyperparameters.py` on exit, including on crash.
