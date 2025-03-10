# Dark Paradise 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HurdleModel (Classifier: LGBMClassifier, Regressor: LGBMRegressor)                  |
| **Level of Analysis** | pgm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_pgm_conflictlong   |
| **Feature Description**       |  fatalities longer conflict history, pgm level Predicting ln(ged_best_sb) using conflict predictors, longer version    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Dark Paradise
├── 
│   ├── README.md
│   ├── main.py
│   ├── artifacts
│   ├── configs
│   │   ├── config_deployment.py
│   │   ├── config_hyperparameters.py
│   │   ├── config_meta.py
│   │   ├── config_queryset.py
│   │   ├── config_sweep.py
│   ├── data
│   │   ├── generated
│   │   ├── processed
│   │   ├── raw
│   ├── logs
│   ├── reports
├── requirements.txt
├── run.sh
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


