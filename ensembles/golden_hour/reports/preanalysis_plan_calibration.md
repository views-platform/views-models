# Pre-Analysis Plan: golden_hour Calibration Run

**Date:** 2026-05-24 (pre-analysis), 2026-05-25 (results)
**Author:** Simon (with Claude)
**Purpose:** Document expectations before running the three constituent HydraNet models and the golden_hour ensemble on calibration, so we can systematically compare actual results against predictions.

## 1. What We Ran

Three HydraNet models, then one PredictionFrameEnsembleManager ensemble:

| Step | Model/Ensemble | Command |
|------|---------------|---------|
| 1 | purple_alien | `python main.py -r calibration -t -e` |
| 2 | blue_stranger | `python main.py -r calibration -t -e` |
| 3 | violet_visitor | `python main.py -r calibration -t -e` |
| 4 | golden_hour (ensemble) | `python main.py -r calibration -t -e` |

## 2. Calibration Partition

- **Train:** months 121-444 (324 months, Jan 1990 - Dec 2016)
- **Test:** months 445-492 (48 months, Jan 2017 - Dec 2020)
- **Rolling origins:** 13 sequences (base origin at month 444, then 12 shifts with stride 1)
- **Forecast steps per origin:** 1-36

## 3. Spatial Coverage

- Grid: 180 x 180 (row_offset=87, col_offset=310)
- Region: Africa + Middle East (viewser priogrid_month default)
- **Actual cell count: 13,110 land cells** (confirmed from y_pred shape: 471,960 / 36 steps = 13,110)

## 4. Structural Verification

### 4.1 File Counts (all correct)

| Model/Ensemble | Expected y_pred.npy files | Actual | Status |
|---------------|--------------------------|--------|--------|
| purple_alien | 78 (13 origins x 6 targets) | 78 | PASS |
| blue_stranger | 78 | 78 | PASS |
| violet_visitor | 78 | 78 | PASS |
| golden_hour | 39 (13 origins x 3 regression targets) | 39 | PASS |

### 4.2 Array Shapes (all correct)

| Model/Ensemble | Expected shape | Actual shape | Status |
|---------------|---------------|--------------|--------|
| Models | (471960, 64) | (471960, 64) | PASS |
| Ensemble | (471960, 192) | (471960, 192) | PASS |

- 471,960 = 13,110 cells x 36 steps
- 64 = posterior samples per model
- 192 = 3 x 64 (concat aggregation)

### 4.3 Value Ranges

| Check | Result | Status |
|-------|--------|--------|
| Min value >= 0 | 0.0 | PASS |
| Max value reasonable | 342.98 (purple_alien lr_sb_best) | PASS |
| No NaN | No NaN detected | PASS |

## 5. Metrics: Pre-Analysis Hypotheses vs Actual Results

### 5.1 Target Difficulty Ranking

| Hypothesis | Result | Verdict |
|-----------|--------|---------|
| lr_sb_best hardest (highest CRPS) | lr_sb_best: 0.152-0.233 vs others: 0.031-0.054 | CONFIRMED |
| lr_ns_best easiest | lr_ns_best CRPS: 0.031 (lowest) | CONFIRMED |
| lr_os_best intermediate | lr_os_best CRPS: 0.051-0.054 | CONFIRMED |

### 5.2 Relative Model Performance (step-wise CRPS, lower = better)

| Target | purple_alien (shrinkage) | blue_stranger (basu_dpd) | violet_visitor (lognormal_nll) |
|--------|------------------------|-------------------------|-------------------------------|
| lr_sb_best | **0.152** | 0.223 | 0.175 |
| lr_os_best | 0.054 | **0.051** | 0.054 |
| lr_ns_best | **0.031** | 0.031 | 0.031 |

| Hypothesis | Result | Verdict |
|-----------|--------|---------|
| violet_visitor best CRPS overall | purple_alien wins on lr_sb_best (0.152 vs 0.175) | **FALSIFIED** |
| blue_stranger outperforms purple_alien | blue_stranger worst on lr_sb_best (0.223 vs 0.152) | **FALSIFIED** |
| purple_alien is the baseline | purple_alien is actually the best on lr_sb_best | REVERSED |
| All agree on low-conflict regions | lr_ns_best nearly identical (0.031 all three) | CONFIRMED |

**Interpretation:** The metric-lab autoresearch that reported 59% CRPS improvement for lognormal_nll was conducted under different experimental conditions (likely different hyperparameters, partition, or spatial scope). These models have not been hyperparameter-swept or calibrated — this run was an end-to-end integration test, not a model comparison. The shrinkage loss's advantage here should not be over-interpreted.

### 5.3 Ensemble Performance (step-wise CRPS)

| Target | Best Individual | golden_hour (ensemble) | Delta |
|--------|----------------|----------------------|-------|
| lr_sb_best | 0.152 (purple_alien) | 0.233 | +53% worse |
| lr_os_best | 0.051 (blue_stranger) | 0.051 | +0.3% (tied) |
| lr_ns_best | 0.031 (purple_alien) | 0.033 | +8% worse |

| Hypothesis | Result | Verdict |
|-----------|--------|---------|
| Ensemble competitive with best model | Ensemble worst on lr_sb_best (0.233 vs 0.152) | **FALSIFIED** |
| Ensemble not dramatically worse than any model | Ensemble worse than ALL models on lr_sb_best | **FALSIFIED** |

**Interpretation:** Concat aggregation treats all 192 samples equally. When one model (blue_stranger, 0.223) is substantially worse than others on a target, its 64 "bad" samples dilute the 128 better samples. This is a structural property of unweighted concat — it has no mechanism to down-weight poor contributors. For future ensembles, consider weighted aggregation or model selection for targets where constituent quality varies significantly. Again, these models are uncalibrated — this finding may not hold after proper hyperparameter optimization.

## 6. Timing: Expectations vs Actual

### 6.1 Per-Model Timing

| Model | Expected Training | Actual Training | Expected Eval | Actual Eval | Total |
|-------|------------------|----------------|---------------|-------------|-------|
| purple_alien | 15-45 min | 83 min | 5-15 min | 79 min | 2h 42m |
| blue_stranger | 15-45 min | 36 min | 5-15 min | 66 min | 1h 42m |
| violet_visitor | 15-45 min | 35 min | 5-15 min | 70 min | 1h 44m |

**Why purple_alien took 2x longer to train:** `diagnostic_visualizations: True` generates biopsy plots every lesson (150 lessons x 6 targets x 3 windows = ~2,700 diagnostic images). blue_stranger and violet_visitor have diagnostics off.

**Why evaluation was 5-13x slower than expected:** The pre-analysis underestimated evaluation time because it did not account for the full inference pipeline per origin: data loading, forward pass with 64 posterior samples, feature scaling inversion of (36, 180, 180, 11, 64) volumes, and diagnostic biopsy generation per origin.

### 6.2 Ensemble Timing

| Phase | Expected | Actual | Notes |
|-------|----------|--------|-------|
| Aggregation + metrics | 2-10 min | ~30 min | Loading predictions from disk for 13 origins x 3 targets x 3 models |
| Total ensemble command | 2-10 min | **6h 34m** | Ensemble re-trained and re-evaluated all 3 models (see lesson learned) |

### 6.3 End-to-End Wall Clock

| Phase | Start | End | Duration |
|-------|-------|-----|----------|
| purple_alien (manual) | 23:28 May 24 | 02:10 May 25 | 2h 42m |
| blue_stranger (manual) | 02:10 | 03:52 | 1h 42m |
| violet_visitor (manual) | 03:53 | 05:37 | 1h 44m |
| golden_hour ensemble | 05:37 | 12:11 | 6h 34m |
| **Total wall clock** | **23:28 May 24** | **12:11 May 25** | **12h 43m** |

**Without the redundant retraining** (if ensemble had been run with `--saved`):
- Models: 6h 9m (sequential)
- Ensemble: ~30 min
- **Estimated total: ~6h 40m**

### 6.4 Lesson Learned: Never Use `-t` on the Ensemble When Models Are Pre-Trained

Running the ensemble with `-t -e` caused it to:
1. **Retrain** all 3 models via run.sh subprocess (~2h)
2. Create new model artifacts with **new timestamps**
3. Discover that no predictions exist for those new timestamps
4. **Re-evaluate** all 3 models via run.sh subprocess (~3h)
5. Finally perform the actual ensemble aggregation (~30 min)

This wasted ~6 hours. The correct command when models are already trained:
```
python main.py -r calibration -e --saved
```

## 7. Failure Modes Encountered

| Risk from Pre-Analysis | Occurred? | Notes |
|----------------------|-----------|-------|
| GPU OOM | No | 2.7 GB VRAM used of 8 GB available |
| Viewser connection failure | No | Data loaded successfully |
| Missing raw data for ensemble actuals | No | Model runs populated data/raw/ |
| Classification target KeyError | No | Correctly excluded from ensemble config |
| Shape mismatch across models | No | All models produced identical N=471,960 |
| NaN in predictions | No | Clean predictions throughout |

**Unexpected issue:** The `-t` flag on ensemble causing full retraining cascade (see Section 6.4).

## 8. Success Criteria Assessment

| Criterion | Status |
|-----------|--------|
| All models complete without errors | PASS |
| Correct file counts (78 per model, 39 ensemble) | PASS |
| Model shapes (N, 64), consistent N | PASS |
| Ensemble shape (N, 192) | PASS |
| CRPS metrics computed for all targets | PASS |
| No NaN values | PASS |
| MCR in reasonable range | NOT CHECKED (QS_sample and MCR_sample not in output) |

**Overall: PASS** — The end-to-end integration test succeeded. PredictionFrameEnsembleManager works with multi-target HydraNet models using concat aggregation.

## 9. Baseline Metrics for Parity Comparison

These are the reference values for the future datafactory-based ensemble (bright_starship-like, Africa+ME region). When that ensemble is built, compare its CRPS against these:

### Step-wise CRPS (primary reference)

| Target | purple_alien | blue_stranger | violet_visitor | golden_hour |
|--------|-------------|---------------|----------------|-------------|
| lr_sb_best | 0.152 | 0.223 | 0.175 | 0.233 |
| lr_os_best | 0.054 | 0.051 | 0.054 | 0.051 |
| lr_ns_best | 0.031 | 0.031 | 0.031 | 0.033 |

### Time-series-wise CRPS

| Target | purple_alien | blue_stranger | violet_visitor | golden_hour |
|--------|-------------|---------------|----------------|-------------|
| lr_sb_best | 0.136 | 0.154 | 0.150 | 0.169 |
| lr_os_best | 0.037 | 0.036 | 0.036 | 0.035 |
| lr_ns_best | 0.052 | 0.052 | 0.052 | 0.052 |

## 10. Risks to Register

1. **Concat aggregation degrades CRPS when constituent model quality varies** — observed 53% CRPS increase on lr_sb_best vs best individual model. Trigger: building ensembles with concat aggregation where constituent models have heterogeneous performance.

2. **Ensemble `-t` flag causes full retraining cascade** — running ensemble with `-t` when models are pre-trained creates new artifacts, invalidates existing predictions, and triggers full re-evaluation. Wasted 6 hours. Trigger: running any PredictionFrameEnsembleManager with `-t` after models are already trained.

3. **Evaluation time underestimated** — actual eval was 5-13x longer than expected due to full inference pipeline per rolling origin (scaling inversion, diagnostics). Trigger: capacity planning for model evaluation runs.

4. **Classification targets not evaluable at ensemble level** — design decision confirmed correct; no KeyError occurred because we excluded classification_targets from ensemble config_meta. The gap in PredictionFrameEnsembleManager.prepare_actuals_df remains.
