# Chunky Bunny
## Overview

This folder contains code for the Chunky Bunny model, an ensemble machine learning model designed for predicting fatalities.

`chunky_bunny` is a **clone of the original `big_chungus` ensemble** (the 2026-06-04 calibration run), recreated with its **full 23 constituents** — 13 plain stepshifters + 6 Hurdle stepshifters + 4 deep-learning models. It mirrors `big_chungus` exactly, but its stepshifter constituents now carry the restored `target_transform` fix (plain models → `log1p`), so its results should be sane rather than divergent. (It differs from `pink_ponyclub`, which carries only the 19 stepshifter constituents and omits the 4 deep-learning models.)


| Information         | Details                        |
|---------------------|--------------------------------|
| **Models** | bittersweet_symphony, brown_cheese, car_radio, counting_stars, demon_days, elastic_heart, fast_car, fluorescent_adolescent, good_riddance, green_squirrel, heavy_rotation, high_hopes, little_lies, national_anthem, new_rules, ominous_ox, plastic_beach, popular_monster, revolving_door, smol_cat, teen_spirit, twin_flame, yellow_submarine                  |
| **Level of Analysis** | cm            |
| **Targets**         | lr_ged_sb |
| **Aggregation**       |  mean   |
| **Metrics**       |  MSLE, MSE, MCR_point, y_hat_bar    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
Chunky Bunny
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
python main.py -r calibration -t -e -re

or

./run.sh -r calibration -t -e -re
```
