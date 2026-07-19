# Bashful Dwarf
## Overview

Parametric distributional baseline (epic #33 / ADR-022 in views-baseline). A white_ranger
(ConflictologyModel) clone with the algorithm swapped to a parametric family, for a like-for-like
CRPS comparison against white_ranger. Only the model varies; targets, partitions, `window_months`,
`n_samples`, and `seed` are identical to white_ranger. Full evaluation:
`views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md`.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ParametricHurdleConflictology |
| **Family / Transform** | gamma / log1p |
| **Level of Analysis** | pgm |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status** | deprecated |

## ⚠️ DEPRECATED — do not run

`gamma/log1p` is **pathological** (same `log1p` `expm1` tail blow-up): **validation avg CRPS ≈ 0.29, ~1.6× worse** than conflictology, with `lr_sb` far worse. `deployment_status='deprecated'` blocks it from running. Kept as evidence that `log1p` must never be a default (ADR-022).

## Usage

```
WANDB_MODE=offline python main.py -r calibration -t -e -sa
```
