# White Mustang 
## Overview

This folder contains code for the White Mustang model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | lavender_haze, blank_space                  |
| **Level of Analysis** | pgm            |
| **Targets**         | ln_ged_sb_dep |
| **Aggregation**       |  mean   |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  deployed    |

## Repository Structure

```
White Mustang
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
├── data
│   ├── generated
│   ├── processed
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


