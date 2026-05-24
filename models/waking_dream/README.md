# Waking Dream

## Overview

Synthetic test model for end-to-end PredictionFrame pipeline testing. Uses MixtureBaseline (distributional baseline) to produce numpy-native PredictionFrame output with posterior samples.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Model Algorithm**   | MixtureBaseline                  |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Data Source**       | Synthetic (no external dependencies) |
| **Pattern**           | `diagonal_gradient`              |
| **Prediction Format** | prediction_frame (numpy)         |
| **Samples**           | 64                               |
| **lambda_mix**        | 0.10                             |
| **Deployment Status** | shadow                           |

## What

A MixtureBaseline model trained on synthetically generated `diagonal_gradient` data with `lambda_mix=0.10`. Produces PredictionFrame output (Track A: `y_pred.npy` + `identifiers.npz`) with 64 posterior samples per observation.

## Why

Constituent model for the `synthetic_chant` ensemble, which tests `PredictionFrameEnsembleManager`. Uses a different lambda_mix value and synthetic pattern from `vivid_dream` to test ensemble behavior with varying mixture weights.

## Repository Structure

```
waking_dream
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
