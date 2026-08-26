# Parity Investigation: purple_alien (viewser) vs bright_starship (datafactory)

**Date:** 2026-05-26
**Investigator:** Claude (prompted by Simon)
**Status:** OPEN — root cause not yet identified

## Executive Summary

The prediction outputs from purple_alien and bright_starship diverge dramatically
(r=0.64 for sb, r≈0 for ns/os), but the **raw training data is 99.99% identical**.
The divergence is NOT caused by different data sources — the datafactory zarr store
and viewser database deliver equivalent values for all three target variables.
The root cause lies somewhere downstream in the training/evaluation pipeline.

## 1. Prediction-Level Divergence (the symptom)

Calibration predictions compared across all 13 origins, 471,960 rows each
(13,110 cells × 36 steps × 64 posterior samples):

| Target | Avg Correlation | Grade | Scale (viewser/factory) |
|--------|----------------|-------|------------------------|
| lr_sb_best | 0.636 | POOR | 3.1x (origin 0) |
| lr_ns_best | 0.003 | DIVERGENT | 1,848x (origin 0) |
| lr_os_best | 0.120 | DIVERGENT | 334x (origin 0) |

For lr_ns_best and lr_os_best, bright_starship predicts near-zero everywhere
(mean ≈ 0.00001) while purple_alien predicts meaningful values (mean ≈ 0.02).

## 2. Raw Training Data Comparison (the surprise)

Direct cell-by-cell comparison of cached training parquets
(`calibration_viewser_df.parquet` vs `calibration_datafactory_df.parquet`):

**Both datasets: 4,876,920 rows × 6 columns, identical MultiIndex (month_id, priogrid_gid).**

### 2a. Target Variables — Near-Identical

| Column | Exact Match | Correlation | Scale Ratio | Differing Rows |
|--------|-------------|-------------|-------------|----------------|
| lr_sb_best | 99.99% (4,876,306/4,876,920) | 0.9998 | 1.008x | 614 |
| lr_ns_best | 100.00% (4,876,782/4,876,920) | 0.9966 | 0.999x | 138 |
| lr_os_best | 100.00% (4,876,738/4,876,920) | 0.9999 | 1.000x | 182 |

The viewser `ged_*_best_sum_nokgi` and datafactory `ged_*_best` produce
**functionally identical values** after renaming to `lr_*_best`. The `_sum_nokgi`
suffix does not indicate a different aggregation — both sources deliver
the same fatality sums per PRIO-GRID cell-month.

The ~600-900 differing rows have small absolute differences (mostly single-digit)
with occasional larger discrepancies (max diff 760 for sb_best). These likely
reflect timing differences in when UCDP data was ingested into each store.

### 2b. Spatial Features — Identical

| Column | Match |
|--------|-------|
| col | 100.00% |
| row | 100.00% |

Both sources provide identical PRIO-GRID coordinates. The earlier concern
(C-49) that datafactory models lacked spatial features was incorrect —
the data loading pipeline adds col/row to both.

### 2c. Country Identity — Completely Different (but likely irrelevant)

| Column | Match |
|--------|-------|
| c_id | 0.00% |

Viewser uses VIEWS-internal `country_id` (e.g., 192); datafactory uses
FAO `gaul0_code` (e.g., 159, or -1 for unassigned cells). However,
`c_id` is in `identity_cols`, NOT in `features`:

```python
'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],   # model inputs
'input_channels': 3,                                        # only 3 channels
'identity_cols': ['month_id', 'priogrid_gid', 'c_id', 'row', 'col'],  # metadata
```

HydraNet uses only the 3 target variables as input channels. `c_id` should
be metadata only. **BUT** — if any part of the training pipeline (curriculum
sampling, stratified evaluation, geographic masking) uses `c_id` values,
the -1 entries in the datafactory version could cause unexpected behavior.

## 3. Variables Available in the Datafactory Zarr Store

The store provides 53 variables. For UCDP fatalities:

| Variable | Description | Values (month 480, Africa+ME) |
|----------|-------------|-------------------------------|
| `ged_sb_best` | State-based fatalities (sum) | nonzero=730, mean(nz)=32.3, max=1528 |
| `ged_sb_count` | State-based events (count) | nonzero=788, mean(nz)=5.1, max=129 |
| `ged_ns_best` | Non-state fatalities (sum) | nonzero=325, mean(nz)=24.3, max=593 |
| `ged_ns_count` | Non-state events (count) | nonzero=356, mean(nz)=4.3, max=83 |
| `ged_os_best` | One-sided fatalities (sum) | nonzero=302, mean(nz)=13.4, max=344 |
| `ged_os_count` | One-sided events (count) | nonzero=320, mean(nz)=1.8, max=9 |

The `_best` variables are fatality sums (ratio best/count ≈ 8.6x), confirming
they are NOT event counts. The datafactory is serving the correct variable.

## 4. Risk Register Updates

### C-48 — REVISED

Original finding: "Viewser vs datafactory variable variant mismatch confounds
parity comparison." This is **disproven** by the raw data comparison. The
variables produce 99.99% identical values. The risk should be downgraded or
closed, and replaced with a new entry for the actual root cause (once identified).

### C-49 — PARTIALLY DISPROVEN

- col/row: identical (not missing in datafactory) → DISPROVEN
- c_id encoding: confirmed different, but likely metadata-only → REDUCED SEVERITY
- NA handling: not yet investigated → OPEN

## 5. Training Timeline Forensics

### Concurrent processes (C-14 risk)

Two training processes ran simultaneously on bright_starship:

| PID | Command | Started | Finished | Model Saved |
|-----|---------|---------|----------|-------------|
| 439229 | `-r forecasting -t` | ~22:43 | 01:03 | `forecasting_model_20260526_010355.pt` (sha256: 6aa2...) |
| 456752 | `-r calibration -t -e` | 23:34 | 03:19 | `calibration_model_20260526_013733.pt` (sha256: d8d8...) |

Overlap period: 23:34 to 01:03 (~90 minutes of concurrent GPU training).

The weight files have different prefixes and different SHA-256 hashes, so they
are distinct files. The calibration evaluation ran from 01:37 to 03:19 and
produced predictions at `predictions_calibration_20260526_013733/` — the
timestamp matches the calibration model, not the forecasting model.

**H1 (weight file corruption) appears unlikely** — the file naming convention
separates run types. However, concurrent GPU training could have caused:
- Memory pressure / OOM fallbacks affecting gradient computation
- Non-deterministic CUDA operations interleaving between processes
- Data loader interference (shared disk I/O)

### Log rotation during run

The Python logging framework rotated at midnight (2026-05-26 00:00):
- `views_pipeline_INFO.log.2026-05-25`: Contains PID 456752 entries (pre-midnight)
- `views_pipeline_INFO.log`: Contains only PID 439229 entries (post-midnight)

PID 456752's training entries span both files.

## 6. Prediction Scale Analysis

A closer look at the prediction magnitudes reveals something unusual:

| Target | purple_alien max | bright_starship max | Ratio |
|--------|-----------------|--------------------| ------|
| lr_sb_best | 848.97 | 461.87 | 1.8x |
| lr_ns_best | 51.62 | 0.02 | 2,581x |
| lr_os_best | 14.08 | 1.19 | 11.8x |

For lr_ns_best, bright_starship's MAXIMUM prediction across all 471,960 cells
is 0.02 — essentially noise. The model has learned to predict near-zero for
this target. Yet the training data for ns_best is 100% identical between the
two models (only 138 of 4.9M rows differ).

This pattern — one target (sb) partially correlated, two targets (ns, os)
collapsed to near-zero — suggests the model may be in a **degenerate solution**
where the multi-task loss landscape allows the model to "give up" on sparse
targets and allocate capacity to the densest target (sb_best, 0.43% nonzero
vs ns_best 0.13% and os_best 0.22%).

## 7. Hypotheses for Prediction Divergence (ranked)

### H1: Concurrent GPU training interference (LIKELY)
Two PyTorch processes sharing a GPU for ~90 minutes. GPU memory contention
could force fallback to smaller effective batch sizes, alter gradient
accumulation, or cause silent numerical differences that push the optimizer
into a different basin. The sparse targets (ns, os) are most vulnerable
because their gradient signal is dominated by zero-valued cells.

### H2: Degenerate multi-task solution (COMPOUNDING)
With 3 regression + 3 classification heads sharing a single backbone, the
model can minimize total loss by "sacrificing" sparser targets. A slight
perturbation (from concurrent GPU training or the ~600 differing cells)
could tip the optimizer into a solution that effectively zeroes out ns and os.

### H3: c_id downstream usage
If curriculum sampling or cell selection uses `c_id` values, the -1 entries
in the datafactory version (vs proper country IDs in viewser) could alter
which cells are selected for training. Unlikely to cause this magnitude of
divergence, but worth checking.

### H4: Data preprocessing subtlety
The viewser `.transform.missing.replace_na()` might apply forward-fill or
interpolation rather than simple zero-fill. The cached parquets show identical
values, but the transform is applied BEFORE caching. Worth verifying.

### H5: Pure GPU non-determinism (UNLIKELY as sole cause)
Same seed, same data, same architecture. GPU non-determinism alone typically
produces r > 0.95, not r ≈ 0. Could contribute but cannot explain the
magnitude of divergence.

## 8. Recommended Next Steps

1. **Re-run bright_starship calibration** with NO concurrent processes.
   If predictions match purple_alien (r > 0.9), H1 is confirmed.
2. If divergence persists, **check per-target training loss curves**.
   If ns/os loss plateaus early, H2 (degenerate solution) is likely.
3. **Grep HydraNet source** for `c_id` usage beyond metadata to test H3.
4. Compare NaN patterns in raw data before/after viewser transform for H4.
