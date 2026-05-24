# Lucid Dream

## Overview

Synthetic test model for end-to-end PredictionFrame pipeline testing. Uses ConflictologyModel (distributional baseline) to produce numpy-native PredictionFrame output with posterior samples.

| Information           | Details                          |
|-----------------------|----------------------------------|
| **Model Algorithm**   | ConflictologyModel               |
| **Level of Analysis** | pgm                              |
| **Targets**           | synth_target                     |
| **Data Source**       | Synthetic (no external dependencies) |
| **Pattern**           | `vertical_stripe`                |
| **Prediction Format** | prediction_frame (numpy)         |
| **Samples**           | 64                               |
| **Deployment Status** | shadow                           |

## What

A ConflictologyModel trained on synthetically generated `vertical_stripe` data. Produces PredictionFrame output (Track A: `y_pred.npy` + `identifiers.npz`) with 64 posterior samples per observation.

## Why

The existing synthetic models (vertical_dream, horizontal_dream, diagonal_dream) produce DataFrame output via LocfModel. This model exercises the PredictionFrame code path — distributional baselines that output numpy arrays of shape `(N, S)` where S is the number of posterior samples. It serves as a constituent for the `synthetic_chant` ensemble, which tests `PredictionFrameEnsembleManager`.

## Repository Structure

```
lucid_dream
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
