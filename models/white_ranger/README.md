# White Ranger 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | ConflictologyModel                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Features**       |  white_ranger   |
| **Feature Description**       |  No description provided    |
| **Metrics**       |  No information provided    |
| **Deployment Status**       |  baseline    |

## Repository Structure

```
White Ranger
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


