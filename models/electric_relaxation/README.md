# Electric Relaxation 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | RandomForestRegressor                  |
| **Level of Analysis** | cm            |
| **Target**         | ged_sb_dep |
| **Features**       |  escwa001_cflong   |
| **Feature Description**       |  Views-escwa conflict history, cm level    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  deprecated    |

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
│   ├── ESCWA_model.ipynb
│   └── ESCWA_script_outputs.ipynb
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


