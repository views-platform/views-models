# Car Radio 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | XGBRegressor                  |
| **Level of Analysis** | cm            |
| **Targets**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_topics   |
| **Feature Description**       |  Predicting ln(fatalities), cm level Queryset with baseline and Mueller & Rauh topic model features    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Car Radio
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

Clone the [views-pipeline-core](https://github.com/views-platform/views-pipeline-core) and the [views-models](https://github.com/views-platform/views-models) repository.


## Usage
Modify configurations in configs/.

If you already have an existing environment, run the `main.py` file. If you don't have an existing environment, run the `run.sh` file. 

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```


