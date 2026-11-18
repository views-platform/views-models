# Rocky Cliff
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | SkDistributionalHurdleModel (Classifier: XGBClassifier, Regressor: NGBoostRegressor)                  |
| **Level of Analysis** | pgm           |
| **Targets**         | lr_ged_sb, lr_ged_os, lr_ged_ns |
| **Features**       |  views-datafactory (africa_me_gaul), 67 features   |
| **Feature Description**       |  Predicting fatalities, pgm level FeatureFrame with all data factory features except ACLED, GAUL codes and maincrop    |
| **Metrics**       |  CRPS, y_hat_bar, twCRPS, QIS, MIS, MCR_sample    |
| **Config Maturity**       |  candidate    |

## Repository Structure

```
Rocky Cliff
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
