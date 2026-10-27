# Chrome Box 
## Overview

This folder contains code for the Chrome Box model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | ashen_sparrow, brisk_sparrow, candid_sparrow, dusky_sparrow, eager_sparrow, faint_sparrow, gentle_sparrow, humble_sparrow                  |
| **Level of Analysis** | pgm            |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best |
| **Aggregation**       |  concat   |
| **Metrics**       |  No information provided    |
| **Maturity**       |  candidate    |

## Repository Structure

```
Chrome Box
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── logs
├── artifacts
├── configs
│   ├── config_hyperparameters.py
│   ├── config_maturity.py
│   ├── config_meta.py
│   ├── config_modelset.py
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


