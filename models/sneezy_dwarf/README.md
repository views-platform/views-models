# Sneezy Dwarf
## Overview

Parametric distributional baseline (epic #33 / ADR-022 in views-baseline). A white_ranger
(ConflictologyModel) clone with the algorithm swapped to a parametric family, for a like-for-like
CRPS comparison against white_ranger. Only the model varies; targets, partitions, `window_months`,
`n_samples`, and `seed` are identical to white_ranger. Full evaluation:
`views-baseline/reports/dwarf_forecast_evaluation/FINDINGS.md`.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ParametricHurdleConflictology |
| **Family / Transform** | gumbel / none |
| **Level of Analysis** | pgm |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status** | baseline |

## Status — NOT RECOMMENDED (runs, but outperformed)

`gumbel/none`. Consistently ~4–5% worse than conflictology across both partitions (validation avg CRPS 0.1865) and in the tail (twCRPS). It runs and forecasts validly (kept as `baseline`) but is superseded by `zinb`/`nb`/`gamma`. Retained as a reference.

## Usage

```
WANDB_MODE=offline python main.py -r calibration -t -e -sa
```
