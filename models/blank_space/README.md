# Blank Space 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HurdleModel (Classifier: LGBMClassifier, Regressor: LGBMRegressor)                  |
| **Level of Analysis** | pgm            |
| **Targets**         | ln_ged_sb_dep |
| **Features**       |  blank_space   |
| **Feature Description**       |  Fatalities natural and social geography, pgm level Predicting ln(fatalities) using natural and social geography features    |
| **Metrics**       |  RMSLE, CRPS, MSE, MSLE, y_hat_bar    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Blank Space
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
│   ├── config_partitions.py
│   ├── config_queryset.py
│   ├── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   ├── raw
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


