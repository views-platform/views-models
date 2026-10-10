<!-- The README.md that was uploaded with this delivery on 2026-09-29, before
     the delivery note existed. Superseded by the note and by the generated
     README.txt; kept because it is the record of what the recipient was
     actually given first. Not edited. -->

# VIEWS HydraNet calibration frames — three point estimators

Produced 2026-09-29 from the 2026-09-28 global-PGM calibration run.
Contact: Simon (simmaa@prio.org). Provenance: views-models#505, ADR-023 Amendment 1.

## What is here

Eight HydraNet models, each collapsed three different ways from the **same** posterior draws.
The models are **not** re-run between frames — one set of predictions, three estimators.

```
<model>/
  I_mean/                 predictions_calibration_<ts>_<NN>.parquet   (13 files)
  II_q95/                 "
  III_conditional_mean/   "
```

| frame | estimator | what it is |
|---|---|---|
| **I** | `arithmetic_mean` | mean over the 16 posterior draws. **This is the delivered estimator** — what every model declares, and byte-identical to the parquets delivered separately. |
| **II** | `q95` | 95th percentile across the 16 draws. |
| **III** | `conditional_mean` | `E[y\|y>0]` — mean over the *positive* draws only. Equivalently `E[y] / P(y>0)`. |

## Columns

`month_id`, `priogrid_id`, `pred_lr_sb_best`, `pred_lr_ns_best`, `pred_lr_os_best`

Raw fatality counts (state-based, non-state, one-sided). Not logged, not scaled.
`priogrid_id` is PRIO-GRID at 0.5°: `row = (id-1)//720`, `col = (id-1)%720`,
`lat = -90 + (row+0.5)*0.5`, `lon = -180 + (col+0.5)*0.5`.

## Read this before using the files

**The 13 files per frame are 13 forecast origins, and their windows overlap.** Each file is one
origin: 64,818 cells × 36 months ahead. Across the 13 the same `(month_id, priogrid_id)` appears
many times, from different origins. **Do not concatenate them and treat the result as a panel.**
Pick an origin, or handle the origin dimension explicitly.

## Known properties — please do not read these as bugs

1. **Frame I under-predicts total fatalities by roughly 5×** against observed. This is a known
   property of the estimator, not of this export. Frames II and III run ~4.6× frame I, which
   happens to land near the observed total — that is what makes them interesting, and is the
   reason all three are here.
2. **All three frames are non-zero on exactly the same cells** (1,469,004 cell-months across the
   eight models). They differ in magnitude only, never in *where* conflict is.
3. **q95 is coarse.** With 16 draws it sits between the 15th and 16th order statistics, so a
   single draw moves it. Reported, not corrected for.
4. **The forecast is flat after ~5 steps.** Steps 5–36 are close to constant; the model carries
   little dynamic signal past five months. Reading month 30 as a forecast is a misuse.
5. **Frames II and III are experimental.** No model declares them, and they are not a calibrated
   forecast. They are alternative point estimators, intended as a probe.

## Check it yourself

`audit_selector_frames.ipynb` sits in this folder and audits **this folder**. Start Jupyter here
and Run All — it needs only `pandas`, `pyarrow`, `numpy` and `matplotlib`, and reads the model
directories beside it. It opens every file, checks every value, draws the maps, and ends in an
explicit PASS/FAIL cell that raises rather than prints if anything is wrong.

The saved outputs in it are from the run that checked this exact copy of the data, so you can
read it without running anything.

## Verification

Every file was checked in full — 312 files, 728,035,776 rows, not a sample:
columns and dtypes exact; no NaN, inf or negative values; no repeated
`(month_id, priogrid_id)` within any file; every file a complete 64,818 × 36 block.
Maps were inspected: conflict concentrates in the Levant, Afghanistan/Pakistan, the Sahel,
the Horn, Nigeria and eastern DRC, with nothing over water.
