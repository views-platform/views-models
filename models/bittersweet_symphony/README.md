# Blank Space 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HurdleModel                  |
| **Level of Analysis** | pgm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_pgm_natsoc   |
| **Feature Description**       |  Fatalities natural and social geography, pgm level

                                    Predicting ln(fatalities) using natural and social geography features

                                        |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure
```

adjective_noun/ # should follow the naming convention adjective_noun
|
|-- artifacts/ #   
|   |-- run_type_model_date.pkl # model/ensemble artifacts
|
|-- configs/ # ...
|   |-- config_deployment.py # configuration for deploying the model into different environments
|   |-- config_hyperparameters.py # hyperparameters for the model
|   |-- config_meta # metadata for the model (model architecture, name, target variable, and level of analysis)
|   |-- config_sweep # sweeping parameters for weights & biases
|
|-- data/ # all input, processed, output data
|    |-- generated/ # Data generated - i.e. forecast/ evaluation
|    |-- processed/ # Data processed
|    |-- raw/ # Data directly from VIEiWSER
|
|-- logs/ # all VIEWS pipeline logs
|
|-- notebooks/ # should only contain experimental notebooks (see [ODR #003](https://github.com/views-platform/docs/blob/main/ODRs/general_003_no_jupyter_notebooks_in_production.md))
|
|-- reports/ # dissemination material - internal and external 
|   |-- figures/ # figures for papers, reports, newsletters, and slides 
|   |-- papers/ # working papers, white papers, articles ect.
|   |-- plots/ # plots for papers, reports, newsletters, and slides
|   |-- slides/ # slides, presentation and similar
|   |-- timelapse/ # plots to create timelapse and the timelapse
|
|-- wandb/ # folder that stores Weights & Biases runs
|
|-- main.py
|
|-- README.md
|
|-- requirements.txt
|
|-- run.sh # sets up the environment and executes the main.py file inside the environment


```

## Setup Instructions

Clone the [views-pipeline-core](https://github.com/views-platform/views-pipeline-core) and the [views-models](https://github.com/views-platform/views-models) repository.


## Usage
Modify configurations in configs/.

If you already have an existing environment, run the `main.py` file. If you don't have an existing environment, run the `run.sh` file. 

```
python main.py -r calibration -t -e

or

./run.sh r calibration -t -e
```

## Model Created on: 2024-11-05 10:55:38.515494
