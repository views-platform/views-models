# Stumpy Dwarf
## Overview

Parametric distributional baseline (epic #33 / ADR-022 in views-baseline). A white_ranger
(ConflictologyModel) clone with the algorithm swapped to a parametric family, for a like-for-like
CRPS comparison against white_ranger. Only the model varies; targets, partitions, `window_months`,
`n_samples`, and `seed` are identical to white_ranger. Full evaluation:
`views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md`.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ParametricConflictology |
| **Family / Transform** | zinb / none |
| **Level of Analysis** | pgm |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status** | baseline |

## Status — working baseline (best of the set)

`zinb/none` (zero-inflated negative binomial). **Best parametric baseline tested**: robustly beats plain `nb` on both partitions, and beats conflictology on the low-volume targets `lr_ns`/`lr_os` (validation avg CRPS 0.1773, the lowest). On the high-volume `lr_sb` it is within seed noise. The recommended parametric baseline.

## Usage

```
WANDB_MODE=offline python main.py -r calibration -t -e -sa
```
