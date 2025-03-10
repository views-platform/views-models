# Fast Car 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HurdleModel (Classifier: XGBClassifier, Regressor: XGBRegressor)                  |
| **Level of Analysis** | cm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_vdem_short   |
| **Feature Description**       |  Predicting ln(fatalities), cm level Queryset with baseline and short list of vdem features    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
├── README.md
├── artifacts
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   ├── config_queryset.py
│   └── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   └── raw
├── main.py
├── notebooks
├── reports
├── requirements.txt
└── run.sh
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


