# Synthetic Choir

## Overview

Parity test ensemble for `DataFrameEnsembleManager` (composition-based, ADR-051). This ensemble is identical to `synthetic_chorus` except it uses `DataFrameEnsembleManager` instead of `EnsembleManager` (inheritance-based). If both produce MSE = 4.34444, empirical equivalence is proven and Phase 3 legacy migration is unblocked.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Models**            | vertical_dream, horizontal_dream, diagonal_dream |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Aggregation**       | mean                             |
| **Metrics**           | MSE                              |
| **Manager**           | DataFrameEnsembleManager (composition-based) |
| **Deployment Status** | shadow                           |

## What

A mean-aggregation ensemble over three LocfModel synthetic models, using the new composition-based `DataFrameEnsembleManager` from ADR-051. The only difference from `synthetic_chorus` is the manager class in `main.py`.

## Why

`EnsembleManager` is inheritance-based (`EnsembleManager` extends `ForecastingModelManager`). `DataFrameEnsembleManager` is a composition-based replacement built as a proving ground for composition-over-inheritance (ADR-051, PR #79). Before the legacy manager can be retired, we need empirical proof that both produce identical outputs on the same inputs. This ensemble provides that proof: same constituent models, same configs, same partitions, same expected MSE.

## How

The ensemble is functionally identical to `synthetic_chorus`. The only code difference is in `main.py`:

```python
# synthetic_chorus (inheritance-based)
from views_pipeline_core.managers.ensemble import EnsembleManager
manager = EnsembleManager(...)

# synthetic_choir (composition-based)
from views_pipeline_core.managers.ensemble import DataFrameEnsembleManager
manager = DataFrameEnsembleManager(...)
```

### Expected results

| Run type    | Expected MSE |
|-------------|-------------|
| Calibration | 4.34444     |
| Validation  | 4.34444     |
| Forecasting | N/A (no evaluation in forecast mode) |

See [synthetic_chorus/README.md](../synthetic_chorus/README.md) for the full analytical MSE derivation. See [ADR-051](https://github.com/views-platform/views-pipeline-core/blob/main/docs/ADRs/051_composition_ensemble_manager.md) in views-pipeline-core for the architectural decision.

### Partition boundaries

All partitions use fixed month-id ranges (no dynamic computation):

| Partition    | Train       | Test        |
|-------------|-------------|-------------|
| Calibration | (121, 444)  | (445, 492)  |
| Validation  | (121, 492)  | (493, 540)  |
| Forecasting | (121, 540)  | (541, 577)  |

## Repository Structure

```
synthetic_choir
├── README.md
├── main.py
├── run.sh
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   └── config_partitions.py
├── data
│   ├── generated
│   └── processed
├── reports
└── logs
```
