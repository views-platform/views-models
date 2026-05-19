# Blue Stranger
## Overview

Clone of `purple_alien` with Basu Density Power Divergence (DPD) regression loss.

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                       |
| **Level of Analysis** | pgm                         |
| **Parent Model**    | purple_alien                   |
| **Key Difference**  | `loss_reg='c'` (Basu DPD, alpha=0.3, sigma=3.0) instead of `loss_reg='b'` (ShrinkageLoss) |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best, by_sb_best, by_ns_best, by_os_best |
| **Deployment Status** | shadow                       |

## Rationale

Basu et al. (1998) Density Power Divergence adds a "suspension system" to regression gradients.
Standard MSE/Shrinkage losses treat a 1-death error and a 1000-death error as differing by orders
of magnitude in gradient. Basu DPD with alpha=0.5 compresses this ratio, allowing the model to
learn from extreme conflict events without being destabilized by tail gradient shocks.

See: `views-metric-lab/reports/research_notes/research_program_loss_physics.md`

## Changes from purple_alien

| Parameter | purple_alien | blue_stranger |
|-----------|-------------|--------------|
| `loss_reg` | `'b'` (ShrinkageLoss) | `'c'` (BasuDPDLoss) |
| `loss_reg_a` | 258 | removed |
| `loss_reg_c` | 0.001 | removed |
| `loss_reg_alpha` | n/a | 0.3 |
| `loss_reg_sigma` | n/a | 3.0 |

All other hyperparameters, architecture, curriculum, and data configuration are identical.

## Usage

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```
