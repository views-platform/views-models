# {{ENSEMBLE_NAME}} 
## Overview

This folder contains code for the {{ENSEMBLE_NAME}} model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | {{MODELS}}                  |
| **Level of Analysis** | {{LEVEL_OF_ANALYSIS}}            |
| **Targets**         | {{TARGET}} |
| **Aggregation**       |  {{AGGREGATION}}   |
| **Metrics**       |  {{METRICS}}    |
| **Deployment Status**       |  {{DEPLOYMENT}}    |

## Repository Structure

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

{{CREATED_SECTION}}
