# Violet Visitor
## Overview

Clone of `purple_alien` with LogNormal NLL regression loss (fixed sigma=0.9).

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                       |
| **Level of Analysis** | pgm                         |
| **Parent Model**    | purple_alien                   |
| **Key Difference**  | `loss_reg='d'` (LogNormal NLL, sigma=0.9) instead of `loss_reg='b'` (ShrinkageLoss) |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best, by_sb_best, by_ns_best, by_os_best |
| **Deployment Status** | shadow                       |

## Rationale

Winner of the views-metric-lab autoresearch (2026-04-09): 35 experiments, 59% CRPS
improvement over baseline, 8/9 FAO guardrails passed. LogNormal NLL with fixed
sigma=0.9 outperformed Basu DPD, Huber, L1, and Focal losses on sparse UCDP data.

See: `views-metric-lab/reports/experiments/autoresearch_basu_apr09_report.md`

## Changes from purple_alien

| Parameter | purple_alien | violet_visitor |
|-----------|-------------|----------------|
| `loss_reg` | `'b'` (ShrinkageLoss) | `'d'` (LogNormalFixedSigmaLoss) |
| `loss_reg_a` | 258 | removed |
| `loss_reg_c` | 0.001 | removed |
| `loss_reg_sigma` | n/a | 0.9 |

## Usage

```
python main.py -r calibration -t -e
```
