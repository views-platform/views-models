# Diagonal Dream

## Overview

Synthetic test model for end-to-end pipeline integration testing. This model exists to verify that the full pipeline (data loading, training, evaluation, forecasting) works correctly without requiring external data services.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Model Algorithm**   | LocfModel (Last Observation Carried Forward) |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Data Source**       | Synthetic (no external dependencies) |
| **Pattern**           | `diagonal_gradient`              |
| **Deployment Status** | shadow                           |

## What

A LocfModel trained on synthetically generated data with a `diagonal_gradient` spatial pattern. The model predicts `synth_target` at the priogrid-month (pgm) level across 1000 grid cells.

## Why

Production models depend on VIEWSER/datafactory for data, making it impossible to test the pipeline in isolation. This model uses `{"source": "synthetic"}` in its queryset config, causing the pipeline to generate deterministic test data internally. If this model runs successfully through calibration, validation, and forecasting, the pipeline machinery is working.

## How

### Synthetic data generation

The pipeline generates data by assigning each grid cell a value based on its spatial position:

```
D(i) = (row(i) + col(i)) mod 20
```

where `row(i) = (priogrid_gid - 1) / 720` and `col(i) = (priogrid_gid - 1) mod 720`. This produces values in [0, 19] along the diagonal. The pattern is **time-invariant** -- every month gets identical values.

### Expected results

Because the data is time-invariant, LocfModel (which carries forward the last observed value) predicts perfectly:

- **MSE = 0** for all run types (calibration, validation, forecasting)

This is by design -- the synthetic pattern is constant over time, so "carry forward" is an exact predictor.

### Partition boundaries

All partitions use fixed month-id ranges (no dynamic computation):

| Partition    | Train       | Test        |
|-------------|-------------|-------------|
| Calibration | (121, 444)  | (445, 492)  |
| Validation  | (121, 492)  | (493, 540)  |
| Forecasting | (121, 540)  | (541, 577)  |

These are safe to hardcode because the synthetic generator produces data for whatever range is requested.

## Repository Structure

```
diagonal_dream
├── README.md
├── main.py
├── run.sh
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   ├── config_partitions.py
│   ├── config_queryset.py
│   └── config_sweep.py
├── artifacts
├── data
│   ├── generated
│   ├── processed
│   └── raw
├── notebooks
├── reports
└── logs
```
