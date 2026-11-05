# Do the smoothed arms create support between the bulk and the tail?

*First measurement, 2026-11-04. Ready to be filed as a views-baseline issue — I was blocked from
creating it, so it lives here until someone posts it.*

## The question

Take a cell whose history is roughly

```
[1, 2, 3, 1, 1, 2, 3, 4, 5, 25]
```

A bulk around 1–5, and a rare extreme at 25. The values *between* the bulk and the tail — 6
through 24 — are real magnitudes that genuinely occur in conflict data. They are simply unobserved
in this cell's short window.

`ConflictologyModel` (arm A) resamples observed values, so its support is exactly `{1,2,3,4,5,25}`.
It can never draw 11. The whole point of arms B (`InterpolatedQuantileConflictology`),
C (`KDEConflictology`) and E (`EmpiricalKDEMixtureConflictology`) is to place mass in that interior
band — and, equally, inside naked patches *within* the bulk or *within* the tail, since no cell has
much data.

**So: do B/C/E actually create support between the bulk and the tail, and at what rate?**

This question is the scientific reason the epic exists, and until now it was recorded nowhere —
not in views-baseline#104, not in #109, not in an ADR.

## First measurement

`short_linen` (arm B), calibration run, `origin_12` / `lr_sb_best`, 2,333,448 cell-horizons
× **64 draws** (reduced from the design 256 to fit the pod's memory and disk).
Reproduce with `gap_diag.py` in this directory.

| statistic | value |
|---|---|
| rows with all-zero draws | 2,301,422 (**98.63%**) |
| rows with max draw ≥ 10 | 13,684 (**0.586%**) — the only rows where a bulk/tail gap can exist |
| draws that are exact integers | 100.00% |
| distinct draw values per row | median 11 of 64 (p10 2, p90 29) |
| occupied deciles of the row's own range | median 7 of 10 |
| largest interior empty run of deciles | median 2, p90 8 |
| rows with a hole ≥ 3 deciles | 37.8% |

**The mechanism demonstrably works.** The row with the largest tail draws

```
0, 2, 3, 5, 7, 30, 37, 49, 53, 57, 134, 144, 222, 249, … 1045, 1332, 1511, 1526
```

— 49 distinct values from 64 draws. The values at 30–57 sit strictly between the bulk and the tail
and are not resampled atoms; a window of ~12 observations cannot yield 49 distinct values.

**The rate is not yet established, and the 37.8% above must not be read as one.** At 64 draws the
median row carries only 11 distinct values, so an empty decile is as likely to mean "no draw landed
here" as "there is no support here". Reporting it as a null would be the `absence_is_not_a_pass`
error: a measurement that cannot see its subject must refuse, not pass.

## What makes it answerable

1. **The A/B comparison at matched n.** The absolute hole rate is inflated by n=64, but comparing
   arm A against arm B *at the same 64 draws* is valid, because the sampling penalty is identical.
   If A's hole distribution matches B's, B buys nothing. `plain_burlap` (arm A) is the control and
   is being evaluated now.
2. **A re-measurement at n=256** on the rows that matter. Only 13,684 rows of 2.3M are eligible, so
   the interesting slice is small enough to re-draw densely without a full-scale run.
3. **Arms C and E** (`*_velvet`, `*_tweed`) extend the same comparison across bandwidth and
   mixture weight.

## Why it matters beyond the baselines

This is the property a conventional count model cannot provide, and it is the stated reason these
baselines exist at all. If none of B/C/E creates interior support at a materially better rate than
A, the epic's premise is not supported and the finding belongs in views-baseline#109. If they do,
the rate is the number that should appear in the Phase-1 findings — not a claim that smoothing
"fills gaps".

Note also the integer-valued output: interpolation is happening (49 distinct values proves it) but
the result is discretised to integers somewhere in the path. Harmless for the question above, since
the interior band 6..24 is representable as integers, but it should be understood rather than
assumed.

Related: views-baseline#104 (epic), #105 (ADR-023 + evaluation plan), #109 (Phase-1 experiment and
findings), #120 (the streaming fix that made land-scale evaluation possible at all).

---

## Answer (2026-11-04, paired A vs B at matched n=64)

The sampling confound above is **controlled** by comparing arm A and arm B on the *same cells* with
the *same* 64-draw budget. Reproduce with `ab_support_compare.py` in this directory.

Per-arm, `origin_12` / `lr_sb_best`:

| statistic | A `plain_burlap` | B `short_linen` |
|---|---|---|
| all-zero rows | 97.93% | 98.63% |
| eligible rows (max draw >= 10) | 22,351 | 13,684 |
| distinct draws per row (median) | 6 of 64 | 11 of 64 |
| occupied deciles of own range (median) | 4 of 10 | 7 of 10 |
| largest interior empty band (median) | 5 deciles | 2 deciles |
| rows with no interior hole | 0.3% | 12.5% |
| rows with a hole >= 3 deciles | 81.3% | 37.8% |

Paired on the 13,012 cell-horizons eligible in **both** arms:

| statistic | value |
|---|---|
| distinct draws per row | A median 8, B median 11; B > A in 64.4% of rows |
| largest interior hole | A median 4, B median 2 |
| B's hole strictly smaller than A's | **62.3%** of rows |
| equal | 21.9% |
| B's hole strictly **larger** than A's | 15.8% |
| rows where B draws >=1 interior value absent from A's support | **78.1%** |
| median count of such values per row | 7 (p90 20) |

### Verdict

**The premise holds.** In 78.1% of the cells where a bulk/tail gap can exist, B places draws at
magnitudes that are inside A's range but absent from A's support — the "6 through 24" of the worked
example. A cannot produce them by construction. The median largest empty band halves, from 4 deciles
to 2.

**Two qualifications, both material to the design decision:**

1. **It is not uniformly better.** In 15.8% of paired rows B's largest interior hole is *wider* than
   A's. Interpolation redistributes mass; it does not only add support.
2. **B pays for interior support with conservatism elsewhere.** B zeroes out ~16,000 more
   cell-horizons than A (98.63% vs 97.93% all-zero) and has a smaller eligible set (13,684 vs
   22,351). Part of the apparent gap-filling improvement is B declining to put any mass in cells
   that A keeps alive. **The trade to adjudicate is this one, not whether smoothing fills gaps —
   it does.**

Still open: arms C (`*_velvet`, KDE) and E (`*_tweed`, empirical-KDE mixture), and a re-measurement
at the design n=256 on the ~13k eligible rows, which is small enough to re-draw densely without a
full-scale run.

---

## The sweep (2026-11-04) — and a correction to the single-arm answer above

The paired A-vs-B answer above used `short_linen` alone. **Swept across B's window and across to
arm C, `short_linen` turns out to be the conservative outlier, and its figures are not
representative of the family.** Reproduce with `multi_arm_support_compare.py`.

`origin_12` / `lr_sb_best`, all arms at n=64:

| arm | all-zero | eligible (max >= 10) | distinct/row | median hole | hole >= 3 |
|---|---|---|---|---|---|
| A  empirical resample | 97.93% | 22,351 | 6 | 5 | 81.3% |
| B  interp-quantile, **short** window | **98.63%** | **13,684** | 11 | 2 | 37.8% |
| B  interp-quantile, mid window | 97.93% | 21,649 | 8 | 3 | 58.8% |
| C  KDE, SJ bandwidth | 97.93% | 22,885 | 9 | 4 | 67.9% |

Paired against A on cells eligible in both:

| arm | n paired | rows with novel interior values | median # | hole < A | hole > A |
|---|---|---|---|---|---|
| B short | 13,012 | 78.1% | 7 | 62.3% | 15.8% |
| B mid | 20,280 | **86.3%** | 5 | 58.4% | 11.7% |
| C KDE SJ | 20,954 | **83.8%** | 4 | 47.1% | 12.8% |

### Corrected verdict

**Retracted:** "the median largest empty band halves, 4 deciles to 2." That holds for
`short_linen` and nothing else. `short_linen` is the only arm whose all-zero rate *exceeds* the
control's (98.63% vs 97.93%) and its eligible set is 40% smaller, so part of its apparent
gap-filling is simply declining to put mass in cells the control keeps alive. Across the swept
family the median largest hole goes 5 -> 2 / 3 / 4, and **arm C is the weakest at closing holes
(47.1% better than A, 12.8% worse) despite having the largest eligible set of any arm.**

**Survives the sweep, and is the answer to the question:** every smoothed arm draws at least one
interior value absent from the control's entire support in **78%-86%** of paired cells, median 4-7
such values per cell. This is robust to the window parameter and to replacing interpolation with a
KDE outright. The interior band between bulk and tail does get populated; *how much* of it is
strongly parameter-dependent and is the thing to tune.

**Methodological note.** Had the finding been reported from `short_linen` alone it would have
overstated hole reduction and understated the conservatism cost — the
`feedback_sweep_before_summarising` shape exactly. The free parameter must be swept before any
scalar from this diagnostic is quoted.

Still open: arm E (`*_tweed`, empirical-KDE mixture), the remaining C bandwidths
(`plush_velvet` silverman, `flat_velvet` fixed 1.0), `long_linen` (long window), and a
re-measurement at the design n=256 on the ~20k eligible rows.

---

## All four mechanisms (2026-11-04, n=64, origin_12 / lr_sb_best)

| arm | all-zero | eligible | distinct | median hole | hole >= 3 |
|---|---|---|---|---|---|
| A  empirical resample | 97.93% | 22,351 | 6 | 5 | 81.3% |
| B  interp-quantile, short window | 98.63% | 13,684 | 11 | 2 | 37.8% |
| B  interp-quantile, mid window | 97.93% | 21,649 | 8 | 3 | 58.8% |
| B  interp-quantile, long window | 97.68% | 24,509 | 7 | 4 | 68.7% |
| C  KDE, SJ bandwidth | 97.93% | 22,885 | 9 | 4 | 67.9% |
| E  empirical-KDE mixture, a=0.5 | 97.94% | 22,706 | 8 | 4 | 72.6% |

Paired against A on cells eligible in both:

| arm | n paired | novel interior | median # | hole < A | hole > A |
|---|---|---|---|---|---|
| B short | 13,012 | 78.1% | 7 | 62.3% | 15.8% |
| B mid | 20,280 | 86.3% | 5 | 58.4% | 11.7% |
| B long | 19,081 | **89.9%** | 4 | 51.9% | 19.8% |
| C KDE SJ | 20,954 | 83.8% | 4 | 47.1% | 12.8% |
| E mixture a=0.5 | 20,949 | 79.2% | 3 | **37.9%** | 13.6% |

### Two structures

**1. The window is a trade-off, not a dial.** Across B's three windows, lengthening the window
engages more cells (all-zero 98.63% -> 97.68%, eligible 13.7k -> 24.5k) and finds novel interior
values in more of them (78% -> 90%), but the support *within* each cell thins: distinct draws
11 -> 7, median hole 2 -> 4, and the share of cells where B does **worse** than the control nearly
doubles to 19.8%. Coverage breadth and support density pull against each other. Any single-window
claim about arm B is a claim about one point on this trade.

**2. Arm E is weakest by construction.** `EmpiricalKDEMixtureConflictology` at a=0.5 draws half its
samples from the plain empirical resampler, so half of every cell's support *is* A's atoms. Lowest
novel count (median 3) and worst hole reduction (37.9%) follow mechanically.

### Risky prediction, recorded before the data

If the account in (2) is right, arm E's gap-filling must rise **monotonically with alpha**:

```
light_tweed (a=0.25)  <  even_tweed (a=0.50)  <  heavy_tweed (a=0.75)
```

on both `novel interior` and `hole < A`. `light_tweed` and `heavy_tweed` were not yet measured when
this was written. **If the ordering fails, the mechanical explanation above is wrong** and the
mixture is doing something other than linearly blending the two supports.

---

## ~~FINAL READING~~ (2026-11-05) — SUPERSEDED, see the correction at the end of this file

> ⛔ **The tail-magnitude claim in this section is RETRACTED.** The stratified table below is
> grouped by the OBSERVED outcome, which produces its 4-5x / 1.4x / 0.3x pattern even for a
> perfectly calibrated forecast (regression to the mean; Murphy-Winkler: condition on the
> FORECAST, not the observation). No magnitude defect is established by it. The arm *ordering*
> within the `truth > 10` group survives, because every arm shares the same grouping. The
> P(fire) reliability result in this section is unaffected — it bins by prediction, which is
> the correct conditioning.

This section supersedes the scoring conclusions above. The support-diagnostic sections are unaffected
(they use draws only, never truth). Scripts: `battery.py`, `cond_size.py`,
`strat_and_reliability.py`.

### What was retracted, and why

1. **"CRPS ranks the arms, B-short wins."** CRPS spread across ten arms is 4.0%. A metric's dynamic
   range must be checked before ranking on it (views-hydranet evaluation report §2.2).
2. **"Every arm under-forecasts the typical conflict cell by 4-6x."** My `size_ratio` used the mean
   of ALL draws, so it was `P(fire) x E[size|fires]` — occurrence leaking into a magnitude statistic.
   Caught by the views-baseline session.
3. **"Conditional magnitude is well calibrated (cond_med = 1.0 on nine of ten arms)."** That pooled
   median was **cancellation across strata**, not calibration. Also caught by views-baseline, who
   predicted integer ties; the real cause turned out to be larger than ties.

Each retraction has the same cause: **reporting a statistic without checking what it is composed
of.** A metric having dynamic range is not evidence that it measures one thing — the mixed
`size_ratio` had the largest spread of anything measured (45.5%) and was the least meaningful.

### Conditional size by truth magnitude — the informative cut

lr_sb_best / origin_12 / S=64. 9,134 conflict cells: `truth==1` 1,677 (18.4%), `2-10` 4,488 (49.1%),
`>10` 2,969 (32.5%). Values are the median over the stratum of (median positive draw / truth).

| arm | truth=1 | truth 2-10 | truth >10 |
|---|---|---|---|
| A  resample | 4.00 | 1.38 | 0.31 |
| B  interp short | — | — | — |
| B  interp mid | 4.00 | 1.38 | 0.32 |
| B  interp long | 4.00 | 1.33 | **0.26** |
| C  KDE sj | 4.50 | 1.50 | 0.33 |
| C  KDE silverman | 5.00 | 1.62 | **0.37** |
| C  KDE fixed 1.0 | 4.00 | 1.40 | 0.31 |
| E  mix a=0.25 | 4.00 | 1.43 | 0.32 |
| E  mix a=0.50 | 4.00 | 1.50 | 0.33 |
| E  mix a=0.75 | 4.00 | 1.50 | 0.33 |

**The shrinkage signature.** Every arm over-predicts small cells 4-5x, over-predicts the middle
~1.4x, and under-predicts the large cells by 2.7-3.8x. The pooled median of 1.0 was these cancelling.
The tail stratum is 32.5% of conflict cells and carries the counts that matter.

### P(fire) reliability — the occurrence forecast is too extreme

ECE 0.00138-0.00191 across arms (`short_linen` worst). Small in absolute terms, systematic in shape.
Arm A, predicted vs realised:

| predicted bin | n | predicted | realised | ratio |
|---|---|---|---|---|
| 0.00-0.02 | 6,650 | 0.0156 | 0.0382 | 2.44 |
| 0.02-0.05 | 12,525 | 0.0382 | 0.0490 | 1.28 |
| 0.05-0.10 | 9,022 | 0.0750 | 0.0829 | 1.11 |
| 0.10-0.20 | 7,461 | 0.1419 | 0.1441 | 1.02 |
| 0.20-0.35 | 5,451 | 0.2619 | 0.2282 | 0.87 |
| 0.35-0.50 | 2,323 | 0.4174 | 0.3612 | 0.87 |
| 0.50-0.70 | 2,213 | 0.5839 | 0.4844 | 0.83 |
| 0.70-1.01 | 2,564 | 0.8561 | 0.5811 | **0.68** |

Under-confident at the bottom, over-confident at the top. Where the model says 86% it is right 58% of
the time. The error is concentrated where the model is confident.

### The answer to the original question

**The support between bulk and tail does get filled; it does measurably improve tail magnitude; and
it closes roughly a seventh of the gap that matters.**

In the `truth > 10` stratum the gap-filling ranking tracks magnitude fidelity: C-Silverman, the best
gap-filler of all ten (56.3% of cells improved over control), has the best tail ratio (0.37);
`long_linen`, a poor gap-filler, has the worst (0.26). Smoothing helps exactly where its mechanism
predicts — the upper range — and nowhere else. The effect is real but small: 0.26 -> 0.37 across ten
arms, ~1.4x, against a tail deficit of ~3x.

So smoothing is **not** the lever for these baselines, but it is not inert either, and the reason it
looked inert earlier was that the pooled statistics averaged its only real effect away.

### What is now the candidate work, on this evidence

1. **Tail magnitude.** A ~3x under-prediction on 32.5% of conflict cells, uniform across all ten
   arms. This is the defect with the largest size and the clearest interpretation.
2. **Occurrence over-confidence at the top of the probability range** (0.86 predicted vs 0.58
   realised). Nothing in the battery saw this until the reliability curve was computed.
3. **views-baseline#112, spatial pooling** — attacks the upstream cause (per-cell history too thin),
   and the tail is exactly where a single cell's own history is least informative.
4. **Not** more smoothing variants. Three mechanisms x seven settings span 1.4x on the tail stratum
   and 1.04x on discrimination.

### Scope

One target (`lr_sb_best`), one origin (`origin_12`), S=64 against a declared 256, calibration
partition only. `short_linen`'s stratified row was lost to output truncation on the rerun and is the
one gap. Extending to all three targets and 13 origins is a ten-minute job against the local
archives.

---

## CORRECTION AND FINAL RESULT (2026-11-05) — condition on the forecast, and the horizon mechanism

### The fourth retraction

The stratified table in the superseded section groups conflict cells by the **observed outcome**.
That produces its 4-5x / 1.4x / 0.3x pattern even from a perfectly calibrated forecast: the cells
that land at `truth > 10` are, on average, those whose outcome came out high relative to their own
predictive distribution, so an honest forecast "under-predicts" them. This is regression to the mean
(Murphy-Winkler: condition on the **forecast**, not the observation). **No magnitude defect is
established by it.** Only the arm *ordering* within a fixed grouping survives.

Four retractions tonight, one error wearing four costumes — each time I checked the previous
objection and then took the repaired statistic at face value:

| claim | what I failed to check |
|---|---|
| CRPS ranks the arms | its **dynamic range** (4.0% across ten arms) |
| arms under-forecast 4-6x | its **composition** (`mean of ALL draws` = P(fire) x size) |
| conditional magnitude is calibrated | its **aggregation** (pooled across opposite-signed strata, returns exactly 1.0000) |
| ~3x tail magnitude deficit | its **conditioning** (grouped by outcome, not forecast) |

The P(fire) reliability curve is the only result that survived every round, and it survived precisely
because it bins by prediction — the conditioning the fourth error got wrong.

### Size calibration conditioned on the forecast

Binned by **predicted** median positive draw; ratio = mean truth / mean predicted; cells with >=1
positive draw. Script: `cond_fcst.py`.

| predicted draw | A n | pred | truth | ratio | C-silv ratio | B-short ratio |
|---|---|---|---|---|---|---|
| 0.0-1.5 | 9,444 | 1.00 | 0.46 | 0.46 | 0.30 | 1.62 |
| 1.5-2.5 | 7,075 | 1.97 | 1.81 | 0.92 | 0.79 | 0.79 |
| 2.5-5.5 | 12,635 | 3.84 | 2.14 | 0.56 | 0.48 | 0.71 |
| 5.5-10.5 | 8,582 | 7.60 | 3.56 | 0.47 | 0.46 | 0.62 |
| 10.5-25.5 | 6,835 | 16.17 | 5.00 | 0.31 | 0.32 | 0.40 |
| 25.5-100.5 | 3,089 | 45.70 | 7.30 | 0.16 | 0.14 | 0.23 |
| >100.5 | 549 | 197.67 | 8.75 | **0.04** | 0.05 | 0.01 |

Direction **reverses**: the arms **over**-predict, progressively worse the larger the forecast. Where
the control predicts ~198, reality averages 8.75.

⚠️ Caveat stated up front: this bins cells with >=1 positive draw, not cells with `truth > 0`, and the
median truth is 0.0 in **every** bin. It is a statement about the forecast's conditional mean — what
a consumer experiences — but it blends occurrence with size and does not isolate "size given
occurrence". Given the horizon result below, much of it is probably the same decay. Separating them
needs this table per horizon band, or a PIT histogram on the positive part.

### The horizon mechanism — CONFIRMED

views-baseline's hypothesis, stated before the measurement: the window is fixed at `train_end` (#82),
so a cell's P(fire) is constant across all 36 horizons while reality mean-reverts over three years.
That predicts the over-confidence should **grow with horizon**.

P(fire) reliability, top bin (predicted 0.70-1.01):

| arm | h=1 | h=18 | h=36 |
|---|---|---|---|
| A resample | 0.75 | 0.70 | **0.33** |
| C KDE silverman | 0.78 | 0.71 | **0.36** |
| B interp short | 0.74 | 0.69 | **0.27** |

And the 0.50-0.70 bin: A 1.08 -> 0.92 -> 0.45; C 1.06 -> 0.86 -> 0.31; B 0.93 -> 0.75 -> 0.34.
Six independent monotone sequences.

**The detail that nails it:** the predicted value is flat across horizon (A: 0.8548 / 0.8522 /
0.8510) while the realised rate falls 0.639 -> 0.600 -> 0.280. The forecast's confidence does not
decay; reality does.

⚠️ n is 70-82 cells per horizon in the top bin, so each ratio is noisy. The monotonicity is what is
solid. A publishable version should pool h1-12 / h13-24 / h25-36.

### What is established, and what is not

**Established:** P(fire) over-confidence grows monotonically with horizon while the prediction stays
flat, on every arm tested. Conditioned on the forecast, magnitude is over-predicted, worse at larger
predicted values. Arm orderings within a fixed grouping. Everything draws-only (the support
diagnostic and its sweeps).

**Not established:** any tail magnitude *deficit* (retracted). That the size over-prediction is
size-specific rather than another face of the horizon decay. Whether the h=1 residual over-confidence
(~0.75) is real or small-n noise.

### The verdict on the baselines

They are **too aggressive at long horizon**, in both probability and magnitude, and the single
plausible cause is that a window fixed at `train_end` cannot know that hot cells cool over three
years. One mechanism explains both findings. That makes the actionable work **temporal** —
views-baseline#82 (window placement) and #111 (temporal decay) — not #112, and my earlier
reordering that put #111 down is withdrawn.

### The verdict on smoothing

Unchanged, now correctly grounded. The smoothed arms differ from the empirical control by 1.04x on
discrimination and inherit the same horizon decay with the same monotone shape (C-Silverman
0.78/0.71/0.36 against A's 0.75/0.70/0.33). **Smoothing does not touch the mechanism that actually
degrades these forecasts.** The support-filling is real — 71.5-89.9% novel interior coverage,
robust across three mechanisms — and it is orthogonal to the defect. That was the conclusion two
retractions ago, reached for the wrong reason; it now has the right one.

---

## THE TARGET ITSELF (2026-11-05) — one event is 38% of the metric

### What was found

Checking an edge-of-data confound raised by the views-baseline session, the per-calendar-month
prevalence turned out normal at the window's end (month 504: 210 cells > 0, 0.3240%, inside the
0.30-0.51% range throughout) — but **total** state-based events at month 504 were 115,635 against
2,200-10,600 for every other month. One cell held 113,395 of it.

`priogrid_id` 148759, `month_id` 504, `c_id` 124. That cell has 9 nonzero months in 384 and a
previous maximum of 1,934 — so this is 59x its own record, and 21x the largest value anywhere else
in the 48-month test window. `lr_ns_best` and `lr_os_best` at that cell-month are both exactly zero.

### What it is — answered by the views-datafactory session

**A real UCDP value on the wrong support.** One event, id 463131, `best` = 113,368 (the cell's
113,395 is that plus three ordinary events of 24, 2, 1):

| field | value |
|---|---|
| `date_start` / `date_end` | 2021-01-01 / 2021-12-31 — the entire year |
| `date_prec` | 5 — UCDP's own flag for "year or coarser" |
| `where_prec` | 5 — "larger than admin-1" |
| `where_desc` | "Tigray, Afar, Amhara" — three Ethiopian regions |
| conflict | Ethiopia: Government — TPLF (the Tigray war) |

A true annual death toll for the Tigray war across three regions, assigned wholesale to one
0.5-degree cell in one month by `source_distribution_map={"annual": "date_end_only"}` plus
`spatial_distribution_strategy="passthrough"` — both deliberate VIEWSER GedLoader parity. UCDP coded
its imprecision honestly; the viewpoint layer discarded that information.

Filed as **views-datafactory#533**, raised by views-hydranet two days ago on the same cell, with a
catalog at `reports/investigation_summary_event_concentration/findings.md`. Found independently from
two directions: hydranet via h36 truth mass (62.2%), this session via a 48-month window (35.8%).

### Three of my four inferences were wrong

| my inference | verdict |
|---|---|
| a `month_last` boundary accumulation | **wrong** — `date_end` is 2021-12-31 and month 504 *is* December 2021. Coincidence. Truncate at 503 and it vanishes; extend to 560 and it stays at 504. |
| ns/os both zero implies an artefact | **backwards** — one event has exactly one `type_of_violence`, so zeros elsewhere are what a *real* event looks like |
| implausible as a cell-month toll | **right**, and the whole point |

### Scope beyond this cell

21,766 events (5.8%) carrying **21.95% of all fatality mass** are assigned to cell-months their own
precision fields disown. Ten cell-months hold 12.4% of every fatality UCDP records 1989-2026. Cell
148759 alone holds 7.79% across 8 months. **In 2020-2024 — the calibration window — 48.8% of all
fatality mass is concentrated this way.**

A larger one sits outside this window: event 463137, 121,848 fatalities, Aug-Oct 2022, same cell,
lands at month 514 — **94.7% of all state-based fatalities recorded anywhere on earth that month.**
Any calibration partition ending after 504 hits it.

### Sensitivity of the metric — the prediction was right, the conclusion was wrong

Script: `outlier_sens.py`.

| arm | CRPS as-is | excl. the 1 cell | excl. month 504 | outlier share of summed CRPS |
|---|---|---|---|---|
| B interp short | 0.12518 | 0.07659 | 0.07712 | 38.8% |
| B interp mid | 0.12621 | 0.07762 | 0.07816 | 38.5% |
| C KDE fixed1.0 | 0.12628 | 0.07769 | 0.07825 | 38.5% |
| A resample | 0.12670 | 0.07811 | 0.07866 | 38.4% |
| E mix a=0.25 | 0.12676 | 0.07816 | 0.07872 | 38.3% |
| E mix a=0.50 | 0.12686 | 0.07826 | 0.07883 | 38.3% |
| C KDE sj | 0.12687 | 0.07828 | 0.07876 | 38.3% |
| E mix a=0.75 | 0.12692 | 0.07832 | 0.07887 | 38.3% |
| C KDE silverman | 0.12802 | 0.07942 | 0.07995 | 38.0% |
| B interp long | 0.13017 | 0.08158 | 0.08220 | 37.3% |

**One cell is 37.3-38.8% of total summed CRPS.** But the ordering is *byte-identical* with the cell
excluded, and excluding all of month 504 produces one adjacent swap (`E a=0.5` / `C KDE sj`, 0.09%
apart — noise). **So the ranking is robust to it**, because the cell hits every arm within 1.5
percentage points of the same share and behaves as a large additive constant that cancels.

**What it did instead was hide the signal.** Spread: 4.0% as-is, **6.5%** excluding the cell, 6.6%
excluding month 504. The outlier was inflating every arm's level identically and thereby
*compressing* CRPS's dynamic range by 1.6x. One of the two reasons CRPS looked non-discriminating
tonight is this single mis-located event.

This does not rescue CRPS: 6.5% is still thin, and `short_linen` still wins it while placing last on
AP. The honest statement about the instrument is that **CRPS's level here is ~38% an artefact of one
event's support, and its apparent blindness was partly that artefact.**

### The remedy is not the obvious one

Per views-datafactory, and reached independently by views-hydranet: **do not winsorise or threshold
on magnitude.** UCDP underreporting is non-linear in event size — ~30% for large events against
>100% for single-fatality ones (Vesco et al.) — and counts are substantially more accurate than
locations (Weidmann 2015). **The number is the reliable half; the coordinate is the unreliable half.**
Thresholding discards the reliable one.

What is recommended instead: report both with and without as a declared sensitivity (done above), and
move to a weighted or scale-invariant proper score — twCRPS / CL / CSL (Lerch et al. 2017, the
forecaster's dilemma) or SCRPS (Bolin & Wallin 2023, aimed exactly at "the score is in count units so
magnitude buys leverage"). Both are in the library.

ADR-049's distributed profile was measured: it takes the cell from 113,395 to 13,866, but the largest
state-based cell that month is still 41,645 — ~36% of the month. Distribution reduces the spike 2.7x
rather than removing it. `production_parity` remains what production serves.

### Fifth error, and the one that should have come first

Four earlier retractions were about the statistics. This one is about the data: **I computed
aggregates over a target without ever looking at its distribution.** A single value holding 38% of a
metric would have been visible in one `max()` call before any of the rest. The order of operations
was wrong from the start.

---

## ~~THE SCORING RESOLUTION~~ (2026-11-05) — SUPERSEDED; see the fine-grid section at the end

> ⛔ **The mechanism sentence in this section is RETRACTED.** "CRPS puts its mass where there is
> no information and its information where there is no mass" compared the HEIGHTS of a Brier
> density at eight unequally-spaced thresholds. CRPS integrates BS(t)·dt, so the columns are
> densities, not contributions. Caught by views-hydranet; it had already been quoted verbatim
> onto views-datafactory#533 and has been struck there. The measured replacement — rank
> **crossing**, not dilution — is at the end of this file. The SCRPS and restricted-AP numbers
> in this section stand.

Two tests specified by the views-datafactory session, each with a prediction recorded first.
Scripts: `scrps_thresh.py`, `scrps_paired.py`.

### Threshold decomposition (Gneiting & Ranjan 2011) — my prediction was WRONG

CRPS is the integral of Brier scores over thresholds. Brier x 1000, lr_sb_best / origin_12 / S=64:

| arm | t<=0 | t<=1 | t<=2 | t<=5 | t<=10 | t<=25 | t<=100 | t<=1000 |
|---|---|---|---|---|---|---|---|---|
| A resample | 3.0561 | 2.5606 | 2.2326 | 1.6418 | 1.2083 | 0.6744 | 0.1444 | 0.0031 |
| B interp short | 3.1030 | 2.6148 | 2.2778 | 1.6294 | 1.1989 | **0.6536** | **0.1344** | 0.0034 |
| B interp mid | 3.0556 | 2.5656 | 2.2371 | 1.6485 | 1.2124 | 0.6732 | 0.1429 | 0.0030 |
| B interp long | 3.0628 | 2.5724 | 2.2448 | 1.6576 | 1.2295 | 0.6821 | 0.1532 | 0.0032 |
| C KDE sj | 3.0613 | 2.5707 | 2.2359 | 1.6531 | 1.2161 | 0.6744 | 0.1453 | 0.0031 |
| C KDE silverman | 3.0581 | 2.5808 | 2.2500 | 1.6713 | 1.2327 | 0.6830 | 0.1489 | 0.0031 |
| C KDE fixed1.0 | 3.0581 | 2.5715 | 2.2334 | 1.6400 | 1.2050 | 0.6694 | 0.1428 | 0.0031 |
| E mix a=0.25 | 3.0623 | 2.5634 | 2.2321 | 1.6412 | 1.2051 | 0.6706 | 0.1447 | 0.0031 |
| E mix a=0.50 | 3.0626 | 2.5658 | 2.2345 | 1.6451 | 1.2091 | 0.6721 | 0.1452 | 0.0031 |
| E mix a=0.75 | 3.0600 | 2.5657 | 2.2340 | 1.6460 | 1.2105 | 0.6723 | 0.1457 | 0.0031 |
| **spread** | **1.6%** | 2.1% | 2.1% | 2.6% | 2.8% | 4.5% | **14.0%** | 10.7% |

I predicted the dynamic range would live at the zero/nonzero boundary, making CRPS "a dressed-up
occurrence score". **Falsified.** Spread is 1.6% at `t<=0` and 14.0% at `t<=100` — the information is
in the tail.

**THE MECHANISM, exactly:** on a 99.6%-zero count field, **CRPS puts its mass where there is no
information and its information where there is no mass.** Twenty-fold more Brier mass at `t<=0`
(3.06) than at `t<=100` (0.14), and nine times *less* discrimination there (1.6% vs 14.0%).
Integrating the two together is what produces the 4% spread.

This is a sharper account than "proper but non-discriminating" (views-hydranet report §2.2): it says
*why*, and it predicts the remedy — any reweighting that moves attention up the threshold range
should recover dynamic range. That is what twCRPS does by construction.

It also explains `short_linen`: it has the **best** tail Brier (0.6536, 0.1344) and the **worst**
boundary Brier (3.1030). It wins CRPS on the thresholds that carry the discrimination.

### SCRPS (Bolin & Wallin 2023) — both predictions held

`SCRPS(F,y) = E|X-y| / E|X-X'| + 0.5 log E|X-X'|`, minimised. Ensemble forms via sorted draws, with
`E|X-X'|` the Gini mean difference `(2/m^2) sum_i (2i-m-1) x_(i)`.

**My prediction (recorded first): SCRPS is undefined on ~98% of cells.** Held — a degenerate
all-zero forecast has zero predictive spread, so the division and the log are both undefined.
Defined on **2.07%** of cell-horizons.

**views-datafactory's prediction (recorded first): SCRPS widens the spread.** Held, and on a first
pass the arms were scored on *different* cell sets (32,026 to 54,086), with the arm penalised
hardest scored on the fewest — not a paired comparison. Re-run on the intersection where **every**
arm is non-degenerate, 25,601 cells (1.097%):

| arm | SCRPS | CRPS (same cells) |
|---|---|---|
| **A resample** | **2.44855** | 5.6754 |
| C KDE sj | 2.45003 | 5.6890 |
| B interp mid | 2.52003 | 5.6313 |
| E mix a=0.75 | 2.52463 | 5.6939 |
| C KDE silverman | 2.55029 | 5.7912 |
| C KDE fixed1.0 | 2.56603 | 5.6385 |
| E mix a=0.25 | 2.64263 | 5.6791 |
| E mix a=0.50 | 2.70604 | 5.6889 |
| B interp short | 3.05044 | **5.5531** |
| B interp long | 3.24850 | 6.0019 |

**SCRPS spread 32.7% against CRPS's 8.1% on identical cells — 4x the dynamic range.**

**And it resolves the tension that ran through the whole campaign:** on the same cells, CRPS ranks
`short_linen` **first** and SCRPS ranks it **ninth of ten**. SCRPS agrees with AP; CRPS is the
outlier. The timid arm profits under CRPS and is penalised under a score normalised by predictive
spread — precisely what SCRPS was designed to do.

**The control wins.** `plain_burlap` is first on SCRPS, with `C KDE sj` 0.06% behind (tied). The
simplest arm, no smoothing, tops the best-discriminating proper score available.

### Limits

- The intersection is 1.097% of the field and is selected by requiring *every* arm non-degenerate,
  which biases toward the most active cells. This is a comparison on the active subset, not the map.
- **SCRPS is undefined on 97.9% of cell-horizons.** For a production scoring path that is
  disqualifying. twCRPS is the candidate that is defined everywhere while still moving attention up
  the threshold range.
- One target, one origin, S=64, calibration partition.

### What the campaign concludes about scoring

1. CRPS alone must not select among these arms: 4.0% spread (8.1% on the active subset), and it
   ranks first the arm that AP and SCRPS both rank near last.
2. The reason is structural and now measured: mass at low thresholds, information at high thresholds.
3. The remedy is a threshold-reweighted or spread-normalised proper score, not exclusion and not
   winsorising.
4. A single mis-located event consumes 1.6x of CRPS's dynamic range without biasing its ranking.

---

## THE UNIFICATION (2026-11-05) — the data defect and the scoring defect are one defect

Framing contributed by the views-datafactory session on reading the threshold decomposition, and it
is the right way to hold the whole investigation:

**The data-side defect** — a year's conflict aggregate piled onto one cell-month (UCDP 463131, the
Tigray war, `date_prec` 5 / `where_prec` 5, views-datafactory#533) — and **the scoring-side defect**
— CRPS concentrating its mass at thresholds that carry no discrimination — **are the same defect seen
from opposite ends.** The threshold decomposition is the instrument that sees both:

- moving the aggregate off the cell reduces the extreme-threshold mass;
- threshold weighting stops the boundary mass drowning the signal.

**Neither alone is sufficient.** Distribution reduces the concentration 2.7x and leaves the largest
cell at ~36% of its month; threshold reweighting recovers dynamic range but still scores a value
sitting on the wrong support. They are one workstream, not two.

### Instrument choice, settled

**twCRPS-primary**, with SCRPS as the diagnostic that proved the mechanism exists. The deciding fact
is the constraint, not the discrimination: SCRPS is **undefined on 97.9% of cell-horizons**, and a
score that cannot be computed on 98% of the field is not a production score however well it
discriminates on the remainder. twCRPS is defined everywhere and shifts threshold weight *by
construction* rather than incidentally via spread normalisation — so it is both the better instrument
and the better-understood one.

### Paired support is a REQUIREMENT, not a caveat

My unpaired SCRPS run scored arms on 32,026-54,086 cells each, with the arm penalised hardest scored
on the fewest. For an allocation comparison the trap is worse: the allocation itself changes which
cells are degenerate, so an unpaired comparison **measures the support change and attributes it to
the remedy** — a confound invisible from inside the result.

Any spread comparison must be computed on the intersection of cells scorable under *both* conditions.
Baseline figures for calibration: CRPS spread 4.0% as-is, 6.5% excluding the outlier cell, 8.1% on
the 25,601-cell active intersection, SCRPS 32.7% on that same intersection.

### Pre-registered prediction, recorded before any data exists

If a `kgi_distributed` pull is ever scored (unmerged, unapproved, Simon's call — not requested here):
**the distributed allocation recovers part of the 1.6x dynamic-range loss and not all of it**, because
the largest state-based cell at month 504 is still 41,645 (~36% of the month) after distribution.

⚠️ Measure first: `kgi_distributed` produces 2,414,552 viewpoint rows against `production_parity`'s
372,303 — a **6.5x increase**, since each imprecise event becomes many. Effect on a pgm pull's size
and on per-arm degeneracy frequencies is unknown, and the second one determines the paired support
the comparison depends on.

### Scoreboard on pre-registration

Four predictions recorded before their data across this session, between two sessions: three held
(alpha-monotonicity on arm E's lower half; SCRPS widens the spread; SCRPS undefined on ~98%) and one
was falsified (the dynamic range lives at the zero/nonzero boundary). **The falsified one taught the
most** — being wrong about it is what produced the mass/information inversion, which is stranger and
more useful than the prediction would have been.

---

## THE MECHANISM, MEASURED ON A FINE GRID (2026-11-05) — it is CANCELLATION, not dilution

Sixth retraction, and the only one that reached a public artefact before being caught. Script:
`thresh_fine.py`. 261 thresholds (every integer 0–200 plus 60 geometric points to 2000), trapezoid.

### What was wrong, twice over

**My claim:** "CRPS puts its mass where there is no information and its information where there is no
mass", from a table of raw BS(t) at thresholds 0,1,2,5,10,25,100,1000. CRPS = ∫BS(t)dt, so those
columns are **densities**. I compared heights and called them mass. The "twenty-fold more mass at
t≤0" is a height ratio of 21.2x; the contributions run 42.5x the other way.

**The correction to my claim was also wrong in magnitude.** A left-rectangle over the same eight
points (computed independently by both views-hydranet and views-datafactory) put 59.3% of the
integral in [100, 1000). The fine grid says **25.4%** — the estimate carried the t≤100 height across
900 units of width and overshot by ~2.3x.

### The actual distribution

| region | contribution | share | integrated abs. spread | share | spread per unit mass |
|---|---|---|---|---|---|
| [0, 1] | 0.0028 | 3.7% | 0.0001 | 0.8% | 0.22x |
| [1, 10] | 0.0152 | 19.9% | 0.0003 | 5.1% | 0.26x |
| [10, 25] | 0.0134 | 17.5% | 0.0005 | 7.5% | 0.43x |
| [25, 100] | 0.0232 | **30.4%** | 0.0016 | 24.0% | 0.79x |
| [100, 1000] | 0.0194 | 25.4% | 0.0041 | **62.0%** | **2.44x** |
| [1000, 2000] | 0.0022 | 2.9% | 0.0000 | 0.5% | 0.17x |

The mass **peaks at [25, 100]**. A real mass/information mismatch exists — the tail carries 2.44x its
proportional share of discrimination, [0, 10] carries ~0.25x — but it is roughly 10x, not 20-vs-9,
and it should be stated as the ratio column and never again as a sentence with a symmetry in it.

### The mechanism: the arms CROSS OVER

Spearman of each threshold's per-arm BS ordering against the ordering at t=0:

| t≤ | 0 | 1 | 2 | 5 | 10 | 25 | 50 | 100 | 200 | 492 | 992 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rho | 1.000 | 0.479 | 0.273 | **−0.236** | **−0.236** | **−0.382** | −0.006 | 0.103 | −0.018 | 0.600 | 0.685 |
| best | B mid | A | E .25 | B short | B short | B short | B short | B short | B short | B mid | B mid |

**The orderings go negatively correlated.** Three different arms hold "best" at different thresholds.
Integrated absolute spread is 8.63% against a realised pooled spread of 4.0% — the gap is the
orderings annihilating in the integral.

⛔ **The consequence, which is worse than dilution:** upweighting the tail does not *recover* range,
it **picks a side of the crossing**, and which side is then a free parameter of the score. **A single
scalar cannot represent this comparison.** This replaces the retracted "any reweighting should
recover dynamic range" — and it is a stronger finding, because it says the problem is not that CRPS
is badly weighted but that one number cannot carry the answer.

### SCRPS's verdict is partly, not wholly, a selection effect

views-hydranet's separator: AP restricted to the same 25,601 cells where every arm is non-degenerate.

| arm | restricted AP | | arm | restricted AP |
|---|---|---|---|---|
| B interp mid | 0.48987 | | E mix a=0.25 | 0.48772 |
| A resample | 0.48966 | | E mix a=0.50 | 0.48764 |
| C KDE silverman | 0.48946 | | C KDE sj | 0.48740 |
| C KDE fixed1.0 | 0.48946 | | B interp short | 0.48721 |
| E mix a=0.75 | 0.48801 | | B interp long | 0.48196 |

**Spearman(SCRPS, restricted AP) = 0.685.** Not ~1.0, so SCRPS adds something beyond occurrence —
but substantial, so part of the verdict reversal *is* selection. Both rank `short_linen` 9th, which
is corroboration on that arm, not a general validation of SCRPS. I had treated the SCRPS/AP agreement
as confirmation while having just published the hypothesis that predicts it.

### An unplanned self-validation

The integral gives mean CRPS **0.07632** against **0.1268** measured directly per cell. It reconciles
exactly: the grid stops at 2000 while one cell's truth is 113,395, contributing
1 × (113,395 − 2000) / 2,333,448 = **0.0477** beyond the grid. 0.0763 + 0.0477 = 0.124 ≈ 0.1268.

And that beyond-grid contribution is **37.6% of the total** — independently re-deriving the
37.3–38.8% outlier share measured earlier by per-cell exclusion. Two unrelated computations agreeing
on the same number, and I did not design the check.

**Method note for any repeat:** extend the grid to the max TRUTH, not the max draw. Mine stopped at
2000 because draws top out at 1,526; the reconciliation only worked because the shortfall was
noticed. On a window containing a concentrated cell, that truncation silently understates the
integral by exactly the outlier's share.

---

## SIGNAL AGAINST NOISE (2026-11-05) — nothing was blind, and that is the real result

Seventh retraction, and the one that resolves the campaign. Reframing contributed by the
views-hydranet session: **a metric's dynamic range is only meaningful against its own noise on the
same population.** Script: `signal_noise.py`.

### The measurement

All ten arms declare `seed: 42`, so no seed replicates exist. Substitute floor: **split-half draw
noise** — every metric computed on draws 0–31 and again on 32–63, same model, same data, same cells,
different draw subsample.

| metric | between-arm (signal) | within-arm split-half (noise) | **S/N** |
|---|---|---|---|
| CRPS | 3.99% | 0.16% | **25.13x** |
| AP | 4.37% | 0.28% | **15.40x** |
| act_ratio | 5.49% | 0.18% | **31.04x** |

Halves are S=32 against a headline S=64, so noise is overstated ~sqrt(2) and S/N is conservative.

**Why split-half is the right floor for these models and would not be for a neural net:**
`ConflictologyModel` fits by building per-cell history tables (`hist_per_entity`, `window_months`) —
deterministic given the pull. The seed enters only at `predict()`. So refit variance is essentially
zero and seed variance reduces to draw variance, which is what split-half measures. ⚠️ Reasoned from
the artifact's attributes, **not proven by re-fitting at a second seed** — that needs a pod. If fit
were seed-dependent, the floor is too low and these S/N figures are upper bounds.

### What is retracted

⛔ **"AP has no dynamic range on this arm set (4.4%)"** — it is 15.4x its own noise. And ⛔ **every
description of CRPS tonight as "blunt", "non-discriminating" or "blind" on this arm set** — its 4.0%
is 25x its noise floor.

The arms differ by a **small** amount, **reliably**. I treated a small effect as an absent one, which
is the same error as reading a wide interval as a null. This was the framing underneath the whole
campaign and it was wrong from the first measurement.

### What it strengthens — the campaign's actual result

CRPS reliably ranks `short_linen` **first**. AP reliably ranks it **last**. Both at 15–25x above
their own noise.

⇒ **The two metrics reliably DISAGREE.** The rank crossing is not noise finding a pattern in ten
arms — it is two instruments measuring genuinely different properties, each with high precision.

This makes "a single scalar cannot represent this comparison" harder to escape rather than easier. If
the orderings were noisy, more draws would fix it. They are not noisy. **They genuinely conflict.**

### And "is this metric discriminating?" has no model-independent answer

views-hydranet's seed table gives AP a 1.10x span (~10% relative) across four seeds on their
vehicle. My between-arm AP spread is 4.37%. On a vehicle with that refit variance, my ten arms would
be **entirely swamped** — 4.4% signal against a 10% floor is S/N 0.44, a null.

⇒ Same metric, same task, opposite verdict, purely from the model family's stochasticity. The reason
my arms are separable is not that my metrics are better; it is that climatology baselines have no
refit noise to swamp them. **So the axis rule needs a second clause: a metric discriminates only
along the axis the arms differ on, AND only against that model family's own noise.**

### Closing tally

Seven retractions, six caught by peer sessions (views-baseline x3, views-hydranet x2, and
views-hydranet supplying the tool for the seventh). Every one was the same error in different dress —
asserting a statistic without interrogating its construction:

| claim | what went unchecked |
|---|---|
| CRPS ranks the arms | its dynamic range |
| arms under-forecast 4–6x | its composition (occurrence x size) |
| conditional magnitude is calibrated | its aggregation (pooled over opposite-signed strata) |
| ~3x tail deficit | its conditioning (outcome, not forecast) |
| the metric is sound | the target's distribution (one cell = 38% of CRPS) |
| mass where there is no information | densities read as contributions |
| AP is blind here | the metric's own noise floor |

---

## LEAD vs CALENDAR, AND THE LATE-2021 CLIFF (2026-11-05)

Run for views-baseline ADR-024 / PR #128. Scripts: `lead_vs_calendar.py`, `eth_step.py`.
Arm A (`plain_burlap`), all 13 origins, top bin = predicted P(fire) >= 0.70.

### The check as asked was not identifiable

One trained artifact serves all 13 origins, so the resampling pool is fixed at `train_end = 456`
(this answers views-baseline#82 for these models structurally, not by measurement). Therefore
**lead = target_month − 456 is collinear with calendar month** and the two cannot be separated.

What *is* separable is horizon-from-origin: month 480 appears at h=24 in origin_0, h=18 in origin_6,
h=12 in origin_12.

### Horizon-from-origin has NO effect

| cal months | h1–12 | h13–24 | h25–36 |
|---|---|---|---|
| 469–480 | 0.75 (n=5,611) | 0.75 (n=5,529) | — |
| 481–492 | — | 0.71 (n=5,495) | 0.71 (n=5,523) |
| 493–504 | — | — | 0.66 (n=5,631) |

Identical across each row, moving only down the column. At n~5,500 per cell this is not an
underpowered null. **The variation is entirely calendar/lead.**

### ⛔ It is a STEP, not a decay — and my earlier three-point table was sampling error

Per-month ratio, months 457–500: 0.86 0.70 0.72 0.88 0.85 0.77 0.73 0.76 0.78 0.69 0.74 0.69 0.75
0.68 0.74 0.68 0.77 0.85 0.83 0.77 0.77 0.77 0.67 0.71 0.74 0.72 0.67 0.78 0.68 0.70 0.75 0.64 0.72
0.71 0.72 0.70 0.69 0.74 0.69 0.75 0.73 0.70 0.67 0.61 — **no trend, noise around ~0.73.** Then:

| month | 500 | 501 | 502 | 503 | 504 |
|---|---|---|---|---|---|
| ratio | 0.61 | **0.29** | 0.40 | 0.36 | 0.33 |
| n | 366 | 287 | 215 | 148 | 75 |

⛔ **My earlier "over-confidence grows monotonically with horizon" (0.75 → 0.70 → 0.33 at
h1/h18/h36, origin_12) sampled months 469 / 486 / 504 — two points on the flat region and one on the
cliff.** The smooth curve was an artefact of where I sampled. Same class of error as reading eight
threshold points as a density.

### The cliff is GLOBAL, not the #533 aggregation — hypothesis refuted

I hypothesised the cliff was Ethiopia-specific: a year of Tigray events collapsed onto cell 148759
would leave neighbouring Ethiopian cells reading zero, and those are the high-P(fire) cells.

**Both halves wrong.** Ethiopia (`c_id` 124) contributes **zero cells to the top bin in any month**,
so it cannot have caused the statistic. And Ethiopia's own record is not quiet — months 499–504 carry
11, 9, 8, 14, 16, 17 active cells, so no stripping occurred.

| | 457–500 | 501–504 | change |
|---|---|---|---|
| rest of world | 0.73 (n=32,607) | **0.34** (n=725) | **−0.39** |
| Ethiopia | insufficient n | insufficient n | — |

The cliff is real: at n=287 for month 501, a realised rate of 0.29 is ~12 SE below the expected 0.63.

⇒ **In the last four months of the pull, globally, conflict occurred in different cells than
predicted, while overall prevalence stayed normal (0.3101 / 0.3224 / 0.3302 / 0.3240%).** The
persistently-violent cells went quiet and others became active.

**Cause: OPEN.** One candidate, explicitly not asserted: views-datafactory#533 records 21,766 events
(5.8%) carrying 21.95% of fatality mass landing on cell-months their own precision fields disown. If
coarse-precision events cluster in *recent* months — as they would if precision improves with
revision — recent events are assigned to wrong cells, giving normal prevalence with empty
high-predicted cells. That is testable via `where_prec`/`date_prec` over time, which is
views-datafactory's data. **Not tested here; stated as a candidate only.**

### Consequence for ADR-024: none

views-baseline excluded all of 2021 from their multi-cutoff refit before this ran, so their age
result never depended on the cliff. Their refit (pools at 396–456, age separated from calendar year,
and a drop-one-cutoff robustness check) locates a real age effect that this view was structurally
blind to:

- **quiet cells** (`P(fire) = 0` exactly, forecast-conditioned): realised rate rises with age —
  sb +0.89 per 1,000 by age 7 against a base of ~0.4, stable at +0.88/+0.94 when the earliest or
  latest cutoff is dropped;
- **hot cells**: `os` decays (−0.33, sign robust but magnitude moving 75% on a dropped cutoff), `ns`
  similar, and **`sb` about nil** — which is exactly the flat ~0.73 measured here.

⛔ **My "the defect is temporal" claim was measured on sb HOT cells — the one stratum of nine with no
age effect — and generalised to the whole.** Third instance of the axis rule tonight: measured where
the effect isn't, then spoke about everything.

### The synthesis both results support

| | behaviour | age dependence |
|---|---|---|
| sb quiet (`P = 0`) | realised rate ~0.4 → ~1.5 per 1,000 by age 8 | **grows ~3.7x** |
| sb hot (`P >= 0.70`) | says 86%, right 58% | **flat at every age** |

**The predictive distribution is too narrow at BOTH ends, and only the zero end degrades with age.**
Too confident of action where it predicts action; too confident of nothing where it predicts nothing;
and the nothing-side error grows as the window recedes, because a single cell's own quiet history
stops being evidence of quiet. That is the case for nested rings — not "the forecast fails to decay",
which is what I wrongly supplied. The constant sb hot-cell offset is a separate calibration matter.
