# Parity Experiment Log

**Runbook:** [`reports/parity_runbook.md`](parity_runbook.md)
**Started:** _(fill when Phase 0 is complete)_

---

## Environment Snapshot (fill once at start)

| Field | Value |
|-------|-------|
| views-hydranet commit | |
| views-pipeline-core commit | |
| views-hydranet-env Python | |
| PyTorch version | |
| CUDA version | |
| GPU model | |
| GPU memory | |
| Disk free at start | |

```bash
# Commands to fill this table:
cd ~/Documents/scripts/views_platform/views-hydranet && git log --oneline -1
cd ~/Documents/scripts/views_platform/views-pipeline-core && git log --oneline -1
conda run -n views-hydranet-env python --version
conda run -n views-hydranet-env python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /home/simon --output=avail | tail -1
```

---

## Training Runs

One row per model × run_type. Fill as each run completes.

### Calibration

| Step | Model | Source | Loss | Start | End | Duration | PID | GPU Solo | Weight File | SHA-256 (first 8) | Pred Dir | y_pred Count | Errors | Notes |
|------|-------|--------|------|-------|-----|----------|-----|----------|-------------|-------------------|----------|--------------|--------|-------|
| 1.1 | purple_alien | viewser | shrinkage | | | | | Y/N | | | | /78 | | |
| 1.2 | blue_stranger | viewser | basu_dpd | | | | | Y/N | | | | /78 | | |
| 1.3 | violet_visitor | viewser | lognormal_nll | | | | | Y/N | | | | /78 | | |
| 2.1 | bright_starship | datafactory | shrinkage | | | | | Y/N | | | | /78 | | |
| 2.2 | bold_comet | datafactory | basu_dpd | | | | | Y/N | | | | /78 | | |
| 2.3 | blazing_meteor | datafactory | lognormal_nll | | | | | Y/N | | | | /78 | | |

### Calibration — Ensembles

| Step | Ensemble | Source | Start | End | Duration | PID | GPU Solo | Pred Dir | y_pred Count | Errors | Notes |
|------|----------|--------|-------|-----|----------|-----|----------|----------|--------------|--------|-------|
| 4.1 | golden_hour | viewser | | | | | Y/N | | /39 | | |
| 4.2 | stellar_horizon | datafactory | | | | | Y/N | | /39 | | |

### Forecasting

| Step | Model | Source | Loss | Start | End | Duration | PID | GPU Solo | Weight File | Pred Dir | y_pred Count | Errors | Notes |
|------|-------|--------|------|-------|-----|----------|-----|----------|-------------|----------|--------------|--------|-------|
| 6.1 | purple_alien | viewser | shrinkage | | | | | Y/N | | | /6 | | |
| 6.2 | bright_starship | datafactory | shrinkage | | | | | Y/N | | | /6 | | |
| 6.3 | blue_stranger | viewser | basu_dpd | | | | | Y/N | | | /6 | | |
| 6.4 | bold_comet | datafactory | basu_dpd | | | | | Y/N | | | /6 | | |
| 6.5 | violet_visitor | viewser | lognormal_nll | | | | | Y/N | | | /6 | | |
| 6.6 | blazing_meteor | datafactory | lognormal_nll | | | | | Y/N | | | /6 | | |
| 6.7 | golden_hour (ens) | viewser | — | | | | | Y/N | | | /3 | | |
| 6.8 | stellar_horizon (ens) | datafactory | — | | | | | Y/N | | | /3 | | |

### Validation

| Step | Model | Source | Loss | Start | End | Duration | PID | GPU Solo | Weight File | Pred Dir | y_pred Count | Errors | Notes |
|------|-------|--------|------|-------|-----|----------|-----|----------|-------------|----------|--------------|--------|-------|
| 8.1 | purple_alien | viewser | shrinkage | | | | | Y/N | | | | | |
| 8.2 | bright_starship | datafactory | shrinkage | | | | | Y/N | | | | | |
| 8.3 | blue_stranger | viewser | basu_dpd | | | | | Y/N | | | | | |
| 8.4 | bold_comet | datafactory | basu_dpd | | | | | Y/N | | | | | |
| 8.5 | violet_visitor | viewser | lognormal_nll | | | | | Y/N | | | | | |
| 8.6 | blazing_meteor | datafactory | lognormal_nll | | | | | Y/N | | | | | |
| 8.7 | golden_hour (ens) | viewser | — | | | | | Y/N | | | | | |
| 8.8 | stellar_horizon (ens) | datafactory | — | | | | | Y/N | | | | | |

---

## Parity Results

### Gate 1 — Calibration Models (Phase 3)

**Timestamp:** ___
**Report file:** `reports/parity_calibration_models_*.txt`

| Pair | lr_sb r | lr_ns r | lr_os r | sb Grade | ns Grade | os Grade | Worst |
|------|---------|---------|---------|----------|----------|----------|-------|
| purple_alien ↔ bright_starship | | | | | | | |
| blue_stranger ↔ bold_comet | | | | | | | |
| violet_visitor ↔ blazing_meteor | | | | | | | |

**Gate 1 verdict:** _(PASS / CAUTION / FAIL)_
**Notes:**

---

### Gate 2 — Calibration Ensemble (Phase 5)

**Timestamp:** ___
**Report file:** `reports/parity_calibration_ensemble_*.txt`

| Pair | lr_sb r | lr_ns r | lr_os r | sb Grade | ns Grade | os Grade | Worst |
|------|---------|---------|---------|----------|----------|----------|-------|
| golden_hour ↔ stellar_horizon | | | | | | | |

**Gate 2 verdict:** _(PASS / CAUTION / FAIL)_
**Notes:**

---

### Gate 3 — Forecasting (Phase 7)

**Timestamp:** ___
**Report file:** `reports/parity_forecasting_*.txt`

| Pair | lr_sb r | lr_ns r | lr_os r | sb Grade | ns Grade | os Grade | Worst |
|------|---------|---------|---------|----------|----------|----------|-------|
| purple_alien ↔ bright_starship | | | | | | | |
| blue_stranger ↔ bold_comet | | | | | | | |
| violet_visitor ↔ blazing_meteor | | | | | | | |
| golden_hour ↔ stellar_horizon | | | | | | | |

**Gate 3 verdict:** _(PASS / CAUTION / FAIL)_
**Notes:**

---

### Gate 4 — Validation (Phase 9)

**Timestamp:** ___
**Report file:** `reports/parity_validation_*.txt`

| Pair | lr_sb r | lr_ns r | lr_os r | sb Grade | ns Grade | os Grade | Worst |
|------|---------|---------|---------|----------|----------|----------|-------|
| purple_alien ↔ bright_starship | | | | | | | |
| blue_stranger ↔ bold_comet | | | | | | | |
| violet_visitor ↔ blazing_meteor | | | | | | | |
| golden_hour ↔ stellar_horizon | | | | | | | |

**Gate 4 verdict:** _(PASS / CAUTION / FAIL)_
**Notes:**

---

## Final Summary

| Pair | Calibration | Forecasting | Validation | Overall |
|------|-------------|-------------|------------|---------|
| A: purple_alien ↔ bright_starship | | | | |
| B: blue_stranger ↔ bold_comet | | | | |
| C: violet_visitor ↔ blazing_meteor | | | | |
| Ens: golden_hour ↔ stellar_horizon | | | | |

**Conclusion:**

---

## Incident Log

Record anything unexpected here: crashes, reruns, GPU conflicts, disk issues, anomalous results.

| Date | Phase/Step | What Happened | Resolution |
|------|-----------|---------------|------------|
| | | | |

---

## Disk Usage Checkpoints

Record after each phase to catch creep early.

| Checkpoint | Free Space | Consumed Since Last |
|------------|------------|---------------------|
| Phase 0 (start) | | — |
| After Phase 2 (6 calibrations done) | | |
| After Phase 4 (ensembles done) | | |
| After Phase 6 (forecasting done) | | |
| After Phase 8 (validation done) | | |
