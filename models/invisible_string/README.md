# Invisible String 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | LGBMRegressor                  |
| **Level of Analysis** | pgm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_pgm_broad   |
| **Feature Description**       |  fatalities broad model, pgm level Predicting ln(ged_best_sb), broad model    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Invisible String
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


