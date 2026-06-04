# Bright Starship 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best, by_sb_best, by_ns_best, by_os_best |
| **Features**       |  ged_sb_best, ged_ns_best, ged_os_best, gaul0_code   |
| **Feature Description**       |  Synthetic data (unknown)    |
| **Metrics**       |  No information provided    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Bright Starship
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


