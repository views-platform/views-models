# {{Ensemble_NAME}} 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | bad_blood, blank_space, caring_fish, chunky_cat, dark_paradise, invisible_string, lavender_haze, midnight_rain, old_money, orange_pasta, wildest_dream, yellow_pikachu                  |
| **Level of Analysis** | pgm            |
| **Target**         | ln_ged_sb_dep |
| **Aggregation**       |  mean   |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Skinny Love
├── 
│   ├── README.md
│   ├── main.py
│   ├── artifacts
│   ├── configs
│   │   ├── config_deployment.py
│   │   ├── config_hyperparameters.py
│   │   ├── config_meta.py
│   ├── data
│   │   ├── generated
│   │   ├── processed
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

## Model Created on: 2025-02-19 10:59:44.116719
