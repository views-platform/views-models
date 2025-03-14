# Pink Ponyclub 
## Overview

This folder contains code for the Pink Ponyclub model, an ensemble machine learning model designed for predicting fatalities. 


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | bittersweet_symphony, brown_cheese, car_radio, counting_stars, demon_days, fast_car, fluorescent_adolescent, good_riddance, green_squirrel, heavy_rotation, high_hopes, little_lies, national_anthem, ominous_ox, plastic_beach, popular_monster, teen_spirit, twin_flame, yellow_submarine                  |
| **Level of Analysis** | cm            |
| **Target**         | ln_ged_sb_dep |
| **Aggregation**       |  mean   |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Pink Ponyclub
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


