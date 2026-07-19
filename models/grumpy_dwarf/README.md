# Grumpy Dwarf
## Overview

Parametric distributional baseline (epic #33 / ADR-022 in views-baseline). A white_ranger
(ConflictologyModel) clone with the algorithm swapped to a parametric family, for a like-for-like
CRPS comparison against white_ranger. Only the model varies; targets, partitions, `window_months`,
`n_samples`, and `seed` are identical to white_ranger. Full evaluation:
`views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md`.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ParametricHurdleConflictology |
| **Family / Transform** | lognormal / none |
| **Level of Analysis** | pgm |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status** | baseline |

## Status — NOT RECOMMENDED (runs, but outperformed)

`lognormal/none`. Won the *calibration* partition (avg CRPS 0.1071, best) but **reversed out-of-sample** (validation avg CRPS 0.1811 — worst of the untransformed families, above conflictology's 0.1799): a textbook in-sample overfit / winner's curse. Also tail-poor (twCRPS worse than conflictology). It runs and forecasts validly, so it is kept as `baseline`, but is superseded by `zinb`/`nb`/`gamma`. Retained as a negative-result reference.

## Usage

```
WANDB_MODE=offline python main.py -r calibration -t -e -sa
```
