# Vivid Dream

## Overview

Synthetic test model for end-to-end PredictionFrame pipeline testing. Uses MixtureBaseline (distributional baseline) to produce numpy-native PredictionFrame output with posterior samples.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Model Algorithm**   | MixtureBaseline                  |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Data Source**       | Synthetic (no external dependencies) |
| **Pattern**           | `horizontal_stripe`              |
| **Prediction Format** | prediction_frame (numpy)         |
| **Samples**           | 64                               |
| **lambda_mix**        | 0.05                             |
| **Deployment Status** | shadow                           |

## What

A MixtureBaseline model trained on synthetically generated `horizontal_stripe` data with `lambda_mix=0.05`. Produces PredictionFrame output (Track A: `y_pred.npy` + `identifiers.npz`) with 64 posterior samples per observation.

## Why

Constituent model for the `synthetic_chant` ensemble, which tests `PredictionFrameEnsembleManager`. Uses a different synthetic pattern and algorithm variant from `lucid_dream` to ensure the ensemble aggregates across heterogeneous distributional models.

## Repository Structure

```
vivid_dream
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
