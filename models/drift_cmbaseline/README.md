# Drift Cmbaseline 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | DriftModel                  |
| **Level of Analysis** | cm            |
| **Targets**         | lr_ged_sb, lr_ged_ns, lr_ged_os |
| **Features**       |  drift_baseline   |
| **Feature Description**       |  Linear trend extrapolation using slope from training window    |
| **Metrics**       |  RMSLE, CRPS, MSE, MSLE, y_hat_bar, MTD, BCD, Pearson, LevelRatio    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Drift Cmbaseline
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


