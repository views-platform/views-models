# Synthetic Choir 
## Overview

This folder contains code for the Synthetic Choir model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | vertical_dream, horizontal_dream, diagonal_dream                  |
| **Level of Analysis** | pgm            |
| **Targets**         | synth_target |
| **Aggregation**       |  mean   |
| **Metrics**       |  No information provided    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Synthetic Choir
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


