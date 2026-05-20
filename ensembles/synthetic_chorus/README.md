# Synthetic Chorus

## Overview

Synthetic test ensemble for end-to-end pipeline integration testing. Aggregates predictions from three synthetic models to verify that the full ensemble pipeline (model artifact loading, prediction aggregation, evaluation, forecasting) works correctly without requiring external data services.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Models**            | vertical_dream, horizontal_dream, diagonal_dream |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Aggregation**       | mean                             |
| **Metrics**           | MSE                              |
| **Deployment Status** | shadow                           |

## What

A mean-aggregation ensemble over three LocfModel synthetic models, each trained on a different spatial pattern. The ensemble averages their predictions and evaluates the result against the ground truth from the first constituent model.

## Why

Production ensembles depend on their constituent models having run successfully and produced prediction artifacts. This ensemble tests the aggregation and evaluation machinery using deterministic synthetic data where the expected MSE can be derived analytically. If the observed MSE matches the derivation, the ensemble pipeline is correct.

## How

### Constituent model patterns

Each model generates time-invariant synthetic data. With 1000 grid cells (priogrid_gid 1-1000) and a grid width of 720 columns:

| Model            | Pattern              | Formula                              | Range   |
|-----------------|----------------------|--------------------------------------|---------|
| vertical_dream   | `vertical_stripe`   | V(i) = col(i) mod 10                | [0, 9]  |
| horizontal_dream | `horizontal_stripe` | H(i) = row(i) mod 10                | [0, 1]* |
| diagonal_dream   | `diagonal_gradient` | D(i) = (row(i) + col(i)) mod 20     | [0, 19] |

*With 1000 entities, row values are 0 (gids 1-720) and 1 (gids 721-1000).

Since all patterns are time-invariant, each LocfModel predicts its own pattern perfectly (MSE=0 individually).

### Ensemble prediction

The ensemble computes the mean of the three model predictions for each grid cell:

```
pred(i) = (V(i) + H(i) + D(i)) / 3
```

### Ground truth

The ensemble evaluation compares the averaged prediction against the actual `synth_target` values from the first model's prediction files. The model list in `config_meta.py` is `["vertical_dream", "horizontal_dream", "diagonal_dream"]`, so the ground truth comes from vertical_dream, i.e. V(i). This is order-dependent -- changing the model list order would change which model supplies the ground truth and therefore change the expected MSE.

### Expected MSE derivation

The per-entity error is:

```
err(i) = pred(i) - V(i) = (V(i) + H(i) + D(i)) / 3 - V(i) = (H(i) + D(i) - 2*V(i)) / 3
```

MSE is the mean of squared errors across all 1000 entities. The population splits into two rows:

**Row 0** (gids 1-720, row=0, H=0):
```
err = (0 + col%20 - 2*(col%10)) / 3 = (col%20 - 2*(col%10)) / 3
```

**Row 1** (gids 721-1000, row=1, H=1):
```
err = (1 + (1+col)%20 - 2*(col%10)) / 3
```

Computing numerically:

| Subset  | Entities | Mean squared error |
|---------|----------|--------------------|
| Row 0   | 720      | 3.72222             |
| Row 1   | 280      | 5.94444             |
| **Weighted** | **1000** | **4.34444**     |

```
MSE = (720 * 3.72222 + 280 * 5.94444) / 1000 = 4.34444
```

### Expected results

| Run type    | Expected MSE |
|-------------|-------------|
| Calibration | 4.34444     |
| Validation  | 4.34444     |
| Forecasting | N/A (no evaluation in forecast mode) |

The MSE is identical across calibration and validation because the synthetic data is time-invariant -- the same spatial pattern repeats in every month.

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
synthetic_chorus
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
