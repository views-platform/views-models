# The eight-model roster, scrutinised — calibration run 2026-08-13

**Status:** all 8 trained and evaluated, 160 lessons each, fresh data, fresh artifacts.
**Verdict: do not promote. Seven of the eight collapse to near-zero magnitude within six months.**

Run: `calibration`, 13,110 cells (Africa + Middle East), 36 steps, 13 rolling origins,
16 produced draws per model (D=4 × K=4, ADR-015 §6). Frame parity confirmed: every model
produced 78 frames of shape `(471960, 16) float32`.

---

## 1. The headline: magnitude decays to nothing

`MCR = mean(y_pred) / mean(y_true)`; **1.00 is calibrated**
(`views_evaluation/.../native_metric_calculators.py:141`). Target `lr_sb_best`:

| model | s1 | s2 | s3 | s6 | s12 | s18 | s24 | s36 |
|---|---|---|---|---|---|---|---|---|
| violet_visitor | 0.25 | 0.18 | 0.08 | 0.04 | 0.01 | 0.00 | 0.00 | 0.00 |
| bright_starship | 0.70 | 0.98 | 0.53 | 0.16 | 0.01 | 0.00 | 0.00 | 0.00 |
| bold_comet | 1.45 | 0.60 | 0.45 | 0.16 | 0.04 | 0.01 | 0.00 | 0.00 |
| blazing_meteor | 0.58 | 0.31 | 0.06 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 |
| heavy_freighter | 2.69 | 1.63 | 1.00 | 0.37 | 0.18 | 0.05 | 0.02 | 0.00 |
| pink_pirate | 1.18 | 0.34 | 0.18 | 0.08 | 0.03 | 0.02 | 0.02 | 0.00 |
| blue_stranger | 1.08 | 0.74 | 0.55 | 0.34 | 0.11 | 0.03 | 0.01 | 0.00 |
| **purple_alien** | **1.27** | **0.85** | **0.81** | **0.71** | **0.37** | **0.25** | **0.14** | 0.01 |

Most models are roughly calibrated at step 1 and predict **effectively zero conflict from
about month 6 onward**. The FAO forecast window is 36 months. On this evidence, months
6–36 would carry almost no signal.

## 2. purple_alien is the outlier, and it is the good one

At s12 it holds **0.37** where the next best is 0.18 and five are at 0.00–0.03. At s24 it
is **0.14** against a field of 0.00–0.02. It is roughly an order of magnitude better at
retaining magnitude across the horizon.

Worth knowing exactly what is different about its configuration before reading anything
into this — it is one run, one partition, and it is also the model whose first evaluation
was killed by the RAM guard and re-run separately. **The re-run used the same artifact
(`calibration_model_20260813_062540.pt`) and the same fetched data, so the numbers are
comparable** — but the difference is large enough to deserve a second look rather than a
celebration.

## 3. CRPS cannot see any of this — and that is the trap

Same target, same runs:

| model | s1 | s2 | s3 | s6 | s12 | s18 | s24 | s36 |
|---|---|---|---|---|---|---|---|---|
| violet_visitor | 0.130 | 0.128 | 0.125 | 0.116 | 0.112 | 0.135 | 0.133 | 0.875 |
| bright_starship | 0.144 | 0.138 | 0.130 | 0.117 | 0.112 | 0.135 | 0.133 | 0.875 |
| bold_comet | 0.148 | 0.133 | 0.129 | 0.117 | 0.112 | 0.135 | 0.133 | 0.875 |
| blazing_meteor | 0.143 | 0.135 | 0.128 | 0.117 | 0.112 | 0.135 | 0.133 | 0.875 |
| heavy_freighter | 0.162 | 0.144 | 0.133 | 0.116 | 0.112 | 0.135 | 0.133 | 0.875 |
| pink_pirate | 0.144 | 0.132 | 0.126 | 0.116 | 0.112 | 0.135 | 0.133 | 0.875 |
| blue_stranger | 0.143 | 0.134 | 0.127 | 0.116 | 0.112 | 0.135 | 0.133 | 0.875 |
| purple_alien | 0.142 | 0.132 | 0.126 | 0.116 | 0.111 | 0.134 | 0.133 | 0.875 |

**From step 6 onward, all eight are identical to three decimals.** Not similar —
identical. Once every model predicts ~zero on a zero-inflated target, CRPS stops
measuring the model and starts measuring the actuals. It has no discriminating power over
most of the horizon.

Pooled means tell the same story in miniature: CRPS spans 0.0820–0.0827 (a 0.9% spread)
while MCR spans 0.02–0.20 (10×). **Ranking these models on CRPS would rank noise.**

This is register **C-84**'s concern in a new instance: a headline metric that rewards
timid under-prediction, read without the magnitude guardrail beside it.

## 4. Classification is uniform and unhelpful too

`Brier_cls_sample`, pooled over the three `by_*` targets: 0.0052–0.0057 across all eight.
purple_alien and violet_visitor tie best at 0.0052. On a target this sparse, Brier is
dominated by the zeros; it is not separating these models either.

Note `AP` (classification point) is declared on `rusty_bucket` but is not in the
per-model eval outputs, so the occurrence channel is not yet independently assessed here.

## 5. The step-36 discontinuity

CRPS jumps to **0.875 for every model at s36**, from ~0.133 at s24 — a 6.6× step change,
identical across all eight. That is a property of the evaluation window, not of the
models. Worth understanding before anyone quotes a 36-month number.

---

## What this means

**The roster is not ready to promote to global or to the server.** The point of the eight
was to replace the conflictology placeholders in `rusty_bucket`; a pool whose members
predict near-zero past month 6 would be worse than the placeholder it replaces, and the
pooled ensemble cannot fix a systematic magnitude collapse shared by all members.

**Do not rank on CRPS.** It is flat. Any comparison of these eight must lead with MCR.

## Suggested next steps, cheapest first

1. **Understand purple_alien's advantage** — diff its config against the other seven. If
   the difference is real and portable, it may be the fix rather than an outlier.
2. **Ask whether the decay is expected.** A gated/hurdle NB whose gate saturates closed
   over the horizon would produce exactly this shape. That is a modelling question for
   views-hydranet, not a pipeline bug.
3. **Explain the s36 discontinuity** before it reaches anyone external.
4. **Only then** consider global training. Global is ~5× the data and several hours per
   model; spending it on a configuration that collapses at month 6 buys an expensive
   confirmation of what these numbers already say.

## Caveats on this report

- One partition (`calibration`), one region (13,110 cells, Africa + ME). Global behaviour
  is unmeasured and could differ.
- Means are over 36 steps and, where stated, pooled over three targets — the pooling hides
  that `ns` (non-state) is by far the worst channel: MCR 0.01–0.02 for every model.
- No ensemble has been run. `rusty_bucket` at calibration would need ~18.9 GB
  (pipeline-core#463) and has not been attempted.
