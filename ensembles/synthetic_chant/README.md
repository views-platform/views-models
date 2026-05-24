# Synthetic Chant

## Overview

Test ensemble for `PredictionFrameEnsembleManager` (composition-based, numpy-native). Aggregates PredictionFrame output from three distributional synthetic models using sample concatenation.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Models**            | lucid_dream, vivid_dream, waking_dream |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Aggregation**       | concat                           |
| **Metrics**           | CRPS                             |
| **Manager**           | PredictionFrameEnsembleManager   |
| **Deployment Status** | shadow                           |

## What

A concat-aggregation ensemble over three distributional synthetic models, using `PredictionFrameEnsembleManager`. Each constituent produces 64 posterior samples; concat joins them horizontally into `(N, 192)`.

## Why

The repo has two synthetic ensembles testing two of the three ensemble managers:

- `synthetic_chorus` -- `EnsembleManager` (legacy, inheritance-based)
- `synthetic_choir` -- `DataFrameEnsembleManager` (composition-based, DataFrame)

This ensemble completes the coverage by testing `PredictionFrameEnsembleManager`, which works with numpy-native PredictionFrame objects. The constituent models use distributional baselines (ConflictologyModel, MixtureBaseline) that produce posterior samples instead of point predictions.

## How

### Aggregation

`concat` aggregation horizontally joins the posterior sample arrays:

```
lucid_dream:  (N, 64)  ]
vivid_dream:  (N, 64)  ] --> concat --> (N, 192)
waking_dream: (N, 64)  ]
```

### Partition boundaries

| Partition    | Train       | Test        |
|-------------|-------------|-------------|
| Calibration | (121, 444)  | (445, 492)  |
| Validation  | (121, 492)  | (493, 540)  |
| Forecasting | (121, 540)  | (541, 577)  |

## Repository Structure

```
synthetic_chant
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
