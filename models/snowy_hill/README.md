# Ice Fall
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | SkDistributionalHurdleModel (Classifier: RandomForestClassifier, Regressor: RandomForestQuantileRegressor)                  |
| **Level of Analysis** | cm            |
| **Targets**         | lr_ged_sb, lr_ged_os, lr_ged_ns |
| **Features**       |  fatalities003_joint_broad   |
| **Feature Description**       |  Predicting fatalities, cm level Queryset with baseline and broad list of features from all sources    |
| **Metrics**       |  CRPS, y_hat_bar, twCRPS, QIS, MIS, MCR_sample    |
| **Config Maturity**       |  candidate    |

## Repository Structure

```
Ice Fall
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── logs
├── artifacts
├── configs
│   ├── config_hyperparameters.py
│   ├── config_maturity.py
│   ├── config_meta.py
│   ├── config_partitions.py
│   ├── config_queryset.py
│   ├── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   ├── raw
├── reports
├── notebooks
```

## Setup Instructions

Clone the [views-models](https://github.com/views-platform/views-models) and [views-skstepshifter](https://github.com/views-platform/views-skstepshifter) repositories.


## Usage
Modify configurations in configs/.

If you already have an existing environment, run the `main.py` file. If you don't have an existing environment, run the `run.sh` file.

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```
