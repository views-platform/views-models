# Parity Validation Runbook (v2 — post-curriculum-learner fix)

> ⚠️ **DO NOT use this runbook to check determinism / bit-reproducibility (added 2026-06-15).**
> Its weight fingerprint is a **`sha256sum` of the `.pt` file, which is UNRELIABLE** — `torch.save` writes a zip
> embedding non-deterministic mtimes, so the *same* `state_dict` saved twice yields *different* file shas
> (`fe07ce3d` ≠ `1fc215d0`, proven). It therefore cannot distinguish identical from different weights. Likewise
> `investigations/compare_parity.py` measures *similarity* (pearson `r`, "EXCELLENT" at `r>0.99`) and collapses
> posterior samples to their mean — similarity ≠ identity. For determinism checks use the reliable
> **weight-tensor hash** method and procedure in
> **`views-hydranet/reports/reproducibility_runbook.md`** (+ `scripts/compare_run_determinism.py`). This runbook's
> *ground rules* (one model on GPU at a time, frozen data, log every run) remain sound; its fingerprint does not.

**Created:** 2026-05-26
**Supersedes:** parity_runbook v1 (same file, pre-purge)
**Experiment log:** [`reports/parity_experiment_log.md`](parity_experiment_log.md)

---

## Ground Rules

1. **ONE model on the GPU at a time.** No exceptions. Verify before every run.
2. **Log every run** in the sidecar experiment log. No run is valid without a log entry.
3. **Do not proceed past a decision gate** until results are recorded and evaluated.
4. **All comparisons use PredictionFrame** (numpy): `y_pred.npy` + `identifiers.npz`.

---

## Models Under Test

| Pair | Viewser | Datafactory | Loss | Expected Parity |
|------|---------|-------------|------|-----------------|
| A | purple_alien | bright_starship | shrinkage | r > 0.9 all targets |
| B | blue_stranger | bold_comet | basu_dpd | r > 0.9 all targets |
| C | violet_visitor | blazing_meteor | lognormal_nll | r > 0.9 all targets |
| Ens | golden_hour | stellar_horizon | (concat) | r > 0.9 all targets |

Targets: `lr_sb_best`, `lr_ns_best`, `lr_os_best` (+ classification heads `by_*`).
Calibration: 13 origins. Forecasting: 1 origin. Validation: TBD origins.

---

## Phase 0: Prerequisites

- [ ] **P0.1** Curriculum learner fix landed in views-hydranet
  ```bash
  cd ~/Documents/scripts/views_platform/views-hydranet && git log --oneline -5
  ```
  Record commit hash in experiment log.

- [ ] **P0.2** Environment updated
  ```bash
  conda run -n views-hydranet-env pip show views-hydranet | grep -E "^(Version|Location)"
  ```

- [ ] **P0.3** GPU free
  ```bash
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  ```
  Must show "no running processes" or header only.

- [ ] **P0.4** Disk space
  ```bash
  df -h /home/simon --output=avail | tail -1
  ```
  Must show >80 GB.

- [ ] **P0.5** All model directories are clean (no stale artifacts)
  ```bash
  for m in purple_alien bright_starship blue_stranger bold_comet violet_visitor blazing_meteor; do
    echo "$m: $(find models/$m/artifacts models/$m/data -type f ! -name '.gitkeep' | wc -l) files"
  done
  for e in golden_hour stellar_horizon; do
    echo "$e: $(find ensembles/$e/artifacts ensembles/$e/data -type f ! -name '.gitkeep' | wc -l) files"
  done
  ```
  All counts must be 0.

---

## Phase 1: Calibration — Viewser Trio

Each step: verify GPU is solo → run → verify outputs → log.

### Step 1.1: purple_alien (shrinkage, viewser)

- [ ] **1.1a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **1.1b** Run
  ```bash
  conda run -n views-hydranet-env python models/purple_alien/main.py -r calibration -t -e
  ```
- [ ] **1.1c** Verify artifacts
  ```bash
  ls -lh models/purple_alien/artifacts/calibration_model_*.pt
  sha256sum models/purple_alien/artifacts/calibration_model_*.pt
  ```
- [ ] **1.1d** Verify predictions (13 origins × 6 targets = 78 y_pred.npy files)
  ```bash
  find models/purple_alien/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **1.1e** Check error log
  ```bash
  tail -5 models/purple_alien/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **1.1f** Log in experiment log: timestamp, PID, weight SHA-256, file count, duration.

### Step 1.2: blue_stranger (basu_dpd, viewser)

- [ ] **1.2a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **1.2b** Run
  ```bash
  conda run -n views-hydranet-env python models/blue_stranger/main.py -r calibration -t -e
  ```
- [ ] **1.2c** Verify artifacts
  ```bash
  ls -lh models/blue_stranger/artifacts/calibration_model_*.pt
  sha256sum models/blue_stranger/artifacts/calibration_model_*.pt
  ```
- [ ] **1.2d** Verify predictions
  ```bash
  find models/blue_stranger/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **1.2e** Check error log
  ```bash
  tail -5 models/blue_stranger/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **1.2f** Log in experiment log.

### Step 1.3: violet_visitor (lognormal_nll, viewser)

- [ ] **1.3a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **1.3b** Run
  ```bash
  conda run -n views-hydranet-env python models/violet_visitor/main.py -r calibration -t -e
  ```
- [ ] **1.3c** Verify artifacts
  ```bash
  ls -lh models/violet_visitor/artifacts/calibration_model_*.pt
  sha256sum models/violet_visitor/artifacts/calibration_model_*.pt
  ```
- [ ] **1.3d** Verify predictions
  ```bash
  find models/violet_visitor/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **1.3e** Check error log
  ```bash
  tail -5 models/violet_visitor/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **1.3f** Log in experiment log.

---

## Phase 2: Calibration — Datafactory Trio

### Step 2.1: bright_starship (shrinkage, datafactory)

- [ ] **2.1a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **2.1b** Run
  ```bash
  conda run -n views-hydranet-env python models/bright_starship/main.py -r calibration -t -e
  ```
- [ ] **2.1c** Verify artifacts
  ```bash
  ls -lh models/bright_starship/artifacts/calibration_model_*.pt
  sha256sum models/bright_starship/artifacts/calibration_model_*.pt
  ```
- [ ] **2.1d** Verify predictions
  ```bash
  find models/bright_starship/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **2.1e** Check error log
  ```bash
  tail -5 models/bright_starship/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **2.1f** Log in experiment log.

### Step 2.2: bold_comet (basu_dpd, datafactory)

- [ ] **2.2a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **2.2b** Run
  ```bash
  conda run -n views-hydranet-env python models/bold_comet/main.py -r calibration -t -e
  ```
- [ ] **2.2c** Verify artifacts
  ```bash
  ls -lh models/bold_comet/artifacts/calibration_model_*.pt
  sha256sum models/bold_comet/artifacts/calibration_model_*.pt
  ```
- [ ] **2.2d** Verify predictions
  ```bash
  find models/bold_comet/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **2.2e** Check error log
  ```bash
  tail -5 models/bold_comet/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **2.2f** Log in experiment log.

### Step 2.3: blazing_meteor (lognormal_nll, datafactory)

- [ ] **2.3a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **2.3b** Run
  ```bash
  conda run -n views-hydranet-env python models/blazing_meteor/main.py -r calibration -t -e
  ```
- [ ] **2.3c** Verify artifacts
  ```bash
  ls -lh models/blazing_meteor/artifacts/calibration_model_*.pt
  sha256sum models/blazing_meteor/artifacts/calibration_model_*.pt
  ```
- [ ] **2.3d** Verify predictions
  ```bash
  find models/blazing_meteor/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 78.
- [ ] **2.3e** Check error log
  ```bash
  tail -5 models/blazing_meteor/logs/views_pipeline_ERROR.log 2>/dev/null || echo "no errors"
  ```
- [ ] **2.3f** Log in experiment log.

---

## Phase 3: Calibration Parity — Individual Models

- [ ] **3.1** Disk check (6 models × ~9 GB = ~54 GB consumed so far)
  ```bash
  df -h /home/simon --output=avail | tail -1
  ```

- [ ] **3.2** Run parity comparison
  ```bash
  conda run -n views-hydranet-env python scripts/compare_parity.py --run calibration 2>&1 | tee reports/parity_calibration_models_$(date +%Y%m%d_%H%M%S).txt
  ```

- [ ] **3.3** Record results in experiment log (copy the summary table).

### Decision Gate 1

| Pair | lr_sb | lr_ns | lr_os | Verdict |
|------|-------|-------|-------|---------|
| purple_alien ↔ bright_starship | | | | |
| blue_stranger ↔ bold_comet | | | | |
| violet_visitor ↔ blazing_meteor | | | | |

**Pass criteria:** All 9 cells show grade FAIR or better (r > 0.8).
**Ideal:** All 9 cells show GOOD or better (r > 0.95).

- r > 0.9 all targets → **PROCEED** to Phase 4.
- Any pair POOR (r 0.5–0.8) → **PROCEED WITH CAUTION**, note in log, continue to see if pattern holds.
- Any pair DIVERGENT (r < 0.5) → **STOP**. Do not proceed. Investigate root cause.

---

## Phase 4: Calibration — Ensembles

- [ ] **4.1a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **4.1b** golden_hour calibration eval
  ```bash
  conda run -n views-hydranet-env python ensembles/golden_hour/main.py -r calibration -e --saved
  ```
- [ ] **4.1c** Verify predictions
  ```bash
  find ensembles/golden_hour/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 39 (13 origins × 3 regression targets; ensembles produce `lr_*` only).

- [ ] **4.2a** GPU check
  ```bash
  nvidia-smi --query-compute-apps=pid,name --format=csv
  ```
- [ ] **4.2b** stellar_horizon calibration eval
  ```bash
  conda run -n views-hydranet-env python ensembles/stellar_horizon/main.py -r calibration -e --saved
  ```
- [ ] **4.2c** Verify predictions
  ```bash
  find ensembles/stellar_horizon/data/generated/predictions_calibration_* -name "y_pred.npy" | wc -l
  ```
  Must be 39.

- [ ] **4.2d** Log both runs in experiment log.

---

## Phase 5: Calibration Parity — Ensembles

- [ ] **5.1** Run ensemble parity comparison
  ```bash
  conda run -n views-hydranet-env python scripts/compare_parity.py --run calibration --ensemble 2>&1 | tee reports/parity_calibration_ensemble_$(date +%Y%m%d_%H%M%S).txt
  ```

- [ ] **5.2** Record results in experiment log.

### Decision Gate 2

| Pair | lr_sb | lr_ns | lr_os | Verdict |
|------|-------|-------|-------|---------|
| golden_hour ↔ stellar_horizon | | | | |

**Pass criteria:** Same as Gate 1. Ensemble parity should be at least as good as the weakest constituent pair.

- Consistent with Gate 1 → **PROCEED** to forecasting.
- Worse than constituent models → **STOP**. Investigate PredictionFrameEnsembleManager.

---

## Phase 6: Forecasting — All Models (sequential, one at a time)

### Step 6.1: purple_alien

- [ ] **6.1a** GPU check
- [ ] **6.1b** `conda run -n views-hydranet-env python models/purple_alien/main.py -r forecasting -t -e`
- [ ] **6.1c** Verify: `find models/purple_alien/data/generated/predictions_forecasting_* -name "y_pred.npy" | wc -l` — must be 6 (1 origin × 6 targets)
- [ ] **6.1d** Check error log. Log in experiment log.

### Step 6.2: bright_starship

- [ ] **6.2a** GPU check
- [ ] **6.2b** `conda run -n views-hydranet-env python models/bright_starship/main.py -r forecasting -t -e`
- [ ] **6.2c** Verify: prediction count = 6
- [ ] **6.2d** Check error log. Log.

### Step 6.3: blue_stranger

- [ ] **6.3a** GPU check
- [ ] **6.3b** `conda run -n views-hydranet-env python models/blue_stranger/main.py -r forecasting -t -e`
- [ ] **6.3c** Verify: prediction count = 6
- [ ] **6.3d** Check error log. Log.

### Step 6.4: bold_comet

- [ ] **6.4a** GPU check
- [ ] **6.4b** `conda run -n views-hydranet-env python models/bold_comet/main.py -r forecasting -t -e`
- [ ] **6.4c** Verify: prediction count = 6
- [ ] **6.4d** Check error log. Log.

### Step 6.5: violet_visitor

- [ ] **6.5a** GPU check
- [ ] **6.5b** `conda run -n views-hydranet-env python models/violet_visitor/main.py -r forecasting -t -e`
- [ ] **6.5c** Verify: prediction count = 6
- [ ] **6.5d** Check error log. Log.

### Step 6.6: blazing_meteor

- [ ] **6.6a** GPU check
- [ ] **6.6b** `conda run -n views-hydranet-env python models/blazing_meteor/main.py -r forecasting -t -e`
- [ ] **6.6c** Verify: prediction count = 6
- [ ] **6.6d** Check error log. Log.

### Step 6.7: golden_hour (ensemble)

- [ ] **6.7a** GPU check
- [ ] **6.7b** `conda run -n views-hydranet-env python ensembles/golden_hour/main.py -r forecasting -e --saved`
- [ ] **6.7c** Verify: prediction count = 3 (1 origin × 3 regression targets)
- [ ] **6.7d** Check error log. Log.

### Step 6.8: stellar_horizon (ensemble)

- [ ] **6.8a** GPU check
- [ ] **6.8b** `conda run -n views-hydranet-env python ensembles/stellar_horizon/main.py -r forecasting -e --saved`
- [ ] **6.8c** Verify: prediction count = 3
- [ ] **6.8d** Check error log. Log.

---

## Phase 7: Forecasting Parity

- [ ] **7.1** Run full forecasting parity
  ```bash
  conda run -n views-hydranet-env python scripts/compare_parity.py --run forecasting --all 2>&1 | tee reports/parity_forecasting_$(date +%Y%m%d_%H%M%S).txt
  ```

- [ ] **7.2** Record results in experiment log.

### Decision Gate 3

| Pair | lr_sb | lr_ns | lr_os | Verdict |
|------|-------|-------|-------|---------|
| purple_alien ↔ bright_starship | | | | |
| blue_stranger ↔ bold_comet | | | | |
| violet_visitor ↔ blazing_meteor | | | | |
| golden_hour ↔ stellar_horizon | | | | |

**Pass criteria:** Consistent with calibration results (Gate 1 + Gate 2).

- Consistent → **PROCEED** to validation.
- Degraded relative to calibration → Note in log, investigate before proceeding.

---

## Phase 8: Validation — All Models (sequential, one at a time)

### Step 8.1: purple_alien

- [ ] **8.1a** GPU check
- [ ] **8.1b** `conda run -n views-hydranet-env python models/purple_alien/main.py -r validation -t -e`
- [ ] **8.1c** Verify prediction count (origins × 6 targets)
- [ ] **8.1d** Check error log. Log.

### Step 8.2: bright_starship

- [ ] **8.2a–d** Same pattern. `models/bright_starship/main.py -r validation -t -e`

### Step 8.3: blue_stranger

- [ ] **8.3a–d** Same pattern. `models/blue_stranger/main.py -r validation -t -e`

### Step 8.4: bold_comet

- [ ] **8.4a–d** Same pattern. `models/bold_comet/main.py -r validation -t -e`

### Step 8.5: violet_visitor

- [ ] **8.5a–d** Same pattern. `models/violet_visitor/main.py -r validation -t -e`

### Step 8.6: blazing_meteor

- [ ] **8.6a–d** Same pattern. `models/blazing_meteor/main.py -r validation -t -e`

### Step 8.7: golden_hour (ensemble)

- [ ] **8.7a–d** Same pattern. `ensembles/golden_hour/main.py -r validation -e --saved`

### Step 8.8: stellar_horizon (ensemble)

- [ ] **8.8a–d** Same pattern. `ensembles/stellar_horizon/main.py -r validation -e --saved`

---

## Phase 9: Validation Parity

- [ ] **9.1** Run full validation parity
  ```bash
  conda run -n views-hydranet-env python scripts/compare_parity.py --run validation --all 2>&1 | tee reports/parity_validation_$(date +%Y%m%d_%H%M%S).txt
  ```

- [ ] **9.2** Record results in experiment log.

### Decision Gate 4 (Final)

| Pair | lr_sb | lr_ns | lr_os | Verdict |
|------|-------|-------|-------|---------|
| purple_alien ↔ bright_starship | | | | |
| blue_stranger ↔ bold_comet | | | | |
| violet_visitor ↔ blazing_meteor | | | | |
| golden_hour ↔ stellar_horizon | | | | |

---

## Completion Summary

When all 4 gates are passed, fill in the final matrix:

| Pair | Calibration | Forecasting | Validation | Overall |
|------|-------------|-------------|------------|---------|
| A: purple_alien ↔ bright_starship | | | | |
| B: blue_stranger ↔ bold_comet | | | | |
| C: violet_visitor ↔ blazing_meteor | | | | |
| Ens: golden_hour ↔ stellar_horizon | | | | |

**Parity validated** when all 12 cells show FAIR or better.
**Parity confirmed** when all 12 cells show GOOD or better.

---

## Time Estimates

| Phase | Models | Runs | Est. Time |
|-------|--------|------|-----------|
| 1 | 3 viewser | calibration | ~7.5 hrs |
| 2 | 3 datafactory | calibration | ~7.5 hrs |
| 3 | — | parity check | ~5 min |
| 4 | 2 ensembles | calibration eval | ~1 hr |
| 5 | — | parity check | ~5 min |
| 6 | 6 models + 2 ensembles | forecasting | ~4 hrs |
| 7 | — | parity check | ~5 min |
| 8 | 6 models + 2 ensembles | validation | ~4 hrs |
| 9 | — | parity check | ~5 min |
| **Total** | | | **~24 hrs GPU** |

---

## Failure Recovery

- **Disk full:** Check `df -h`. Delete `data/raw/*.parquet` caches from completed models (they are re-fetched on next run). Each is ~10 MB but there may be larger intermediates.
- **OOM / CUDA error:** Check `nvidia-smi`. Kill stale processes. Restart the failed run only.
- **Evaluation crash, training OK:** Re-run with `-e --saved` (skips training, reloads weights).
- **Data fetch failure:** Check `~/.netrc` credentials (datafactory models) or viewser connectivity (viewser models).
