# Big Chungus 
## Overview

This folder contains code for the Big Chungus model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | little_talks, mister_blueshy                  |
| **Level of Analysis** | pgm            |
| **Targets**         | ln_ged_sb, ln_ged_ns, ln_ged_os |
| **Aggregation**       |  concat   |
| **Metrics**       |  y_hat_bar, twCRPS, QIS, MIS, MCR_sample, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Big Chungus
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


