# Happy Dwarf
## Overview

Parametric distributional baseline (epic #33 / ADR-022 in views-baseline). A white_ranger
(ConflictologyModel) clone with the algorithm swapped to a parametric family, for a like-for-like
CRPS comparison against white_ranger. Only the model varies; targets, partitions, `window_months`,
`n_samples`, and `seed` are identical to white_ranger. Full evaluation:
`views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md`.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ParametricHurdleConflictology |
| **Family / Transform** | lognormal / log1p |
| **Level of Analysis** | pgm |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status** | deprecated |

## ⚠️ DEPRECATED — do not run

`lognormal/log1p` is **pathological**. The `log1p` transform's per-sample `expm1` detransform inflates the heavy right tail into astronomical draws: **validation avg CRPS ≈ 6.14, ~34× worse** than conflictology (0.18). `deployment_status='deprecated'` blocks it from running (CoreConfigSniffer). Kept only as evidence that `log1p` must never be a default (ADR-022).

## Usage

```
WANDB_MODE=offline python main.py -r calibration -t -e -sa
```
