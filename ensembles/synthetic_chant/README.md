# Synthetic Chant 
## Overview

This folder contains code for the Synthetic Chant model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | lucid_dream, vivid_dream, waking_dream                  |
| **Level of Analysis** | pgm            |
| **Targets**         | synth_target |
| **Aggregation**       |  concat   |
| **Metrics**       |  No information provided    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Synthetic Chant
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── logs
├── artifacts
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   ├── config_modelset.py
│   ├── config_partitions.py
├── data
│   ├── generated
│   ├── processed
├── reports
```

## Setup Instructions

Clone the [views-pipeline-core](https://github.com/views-platform/views-pipeline-core) and the [views-models](https://github.com/views-platform/views-models) repository.


## Usage
Modify configurations in configs/.

If you already have an existing environment, run the `main.py` file. If you don't have an existing environment, run the `run.sh` file. 

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```

<!-- manual -->
## What

A concat-aggregation ensemble over three distributional synthetic models, using `PredictionFrameEnsembleManager` (composition-based, numpy-native). Each constituent produces 64 posterior samples; concat joins them horizontally into `(N, 192)`.

## Why

The repo has two synthetic ensembles testing two of the three ensemble managers:

- `synthetic_chorus` -- `EnsembleManager` (legacy, inheritance-based)
- `synthetic_choir` -- `DataFrameEnsembleManager` (composition-based, DataFrame)

This ensemble completes the coverage by testing `PredictionFrameEnsembleManager`, which works with numpy-native PredictionFrame objects. The constituent models use distributional baselines (ConflictologyModel, MixtureBaseline) that produce posterior samples instead of point predictions.

## Evaluation semantics

The three constituent models are trained on **different synthetic patterns** (lucid_dream: `vertical_stripe`, vivid_dream: `horizontal_stripe`, waking_dream: `diagonal_gradient`). The ensemble evaluates all predictions against the first model's actuals (lucid_dream / `vertical_stripe`), following the standard `models[0]` actuals-selection rule in `EvaluationStage`. This means ensemble CRPS (~1.04) is much higher than any constituent's individual CRPS (0.000, 0.002, 0.043) because 128 of 192 samples come from models trained on different target distributions. The metric reflects cross-pattern disagreement, not prediction quality degradation. This ensemble tests infrastructure correctness (aggregation, serialisation, evaluation pipeline), not meaningful forecast skill.

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
<!-- /manual -->
