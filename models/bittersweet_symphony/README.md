# Bittersweet Symphony 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | XGBRegressor                  |
| **Level of Analysis** | cm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_all_features   |
| **Feature Description**       |  Predicting ln(fatalities), cm level, queryset with all features    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
├── README.md
├── artifacts
│   ├── calibration_model_20250305_124458.pkl
│   └── validation_model_20250306_104346.pkl
├── configs
│   ├── __pycache__
│   │   ├── config_deployment.cpython-311.pyc
│   │   ├── config_hyperparameters.cpython-311.pyc
│   │   ├── config_meta.cpython-311.pyc
│   │   ├── config_queryset.cpython-311.pyc
│   │   └── config_sweep.cpython-311.pyc
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   ├── config_queryset.py
│   └── config_sweep.py
├── data
│   ├── generated
│   │   ├── calibration_log.txt
│   │   ├── eval_month_calibration_20250305_124458.parquet
│   │   ├── eval_month_validation_20250306_104346.parquet
│   │   ├── eval_step_calibration_20250305_124458.parquet
│   │   ├── eval_step_validation_20250306_104346.parquet
│   │   ├── eval_ts_calibration_20250305_124458.parquet
│   │   ├── eval_ts_validation_20250306_104346.parquet
│   │   ├── predictions_calibration_20250305_124458_00.parquet
│   │   ├── predictions_calibration_20250305_124458_01.parquet
│   │   ├── predictions_calibration_20250305_124458_02.parquet
│   │   ├── predictions_calibration_20250305_124458_03.parquet
│   │   ├── predictions_calibration_20250305_124458_04.parquet
│   │   ├── predictions_calibration_20250305_124458_05.parquet
│   │   ├── predictions_calibration_20250305_124458_06.parquet
│   │   ├── predictions_calibration_20250305_124458_07.parquet
│   │   ├── predictions_calibration_20250305_124458_08.parquet
│   │   ├── predictions_calibration_20250305_124458_09.parquet
│   │   ├── predictions_calibration_20250305_124458_10.parquet
│   │   ├── predictions_calibration_20250305_124458_11.parquet
│   │   ├── predictions_validation_20250306_104346_00.parquet
│   │   ├── predictions_validation_20250306_104346_01.parquet
│   │   ├── predictions_validation_20250306_104346_02.parquet
│   │   ├── predictions_validation_20250306_104346_03.parquet
│   │   ├── predictions_validation_20250306_104346_04.parquet
│   │   ├── predictions_validation_20250306_104346_05.parquet
│   │   ├── predictions_validation_20250306_104346_06.parquet
│   │   ├── predictions_validation_20250306_104346_07.parquet
│   │   ├── predictions_validation_20250306_104346_08.parquet
│   │   ├── predictions_validation_20250306_104346_09.parquet
│   │   ├── predictions_validation_20250306_104346_10.parquet
│   │   ├── predictions_validation_20250306_104346_11.parquet
│   │   └── validation_log.txt
│   ├── processed
│   └── raw
│       ├── calibration_data_fetch_log.txt
│       ├── calibration_viewser_df.parquet
│       ├── validation_data_fetch_log.txt
│       └── validation_viewser_df.parquet
├── logs
│   ├── views_pipeline_CRITICAL.log
│   ├── views_pipeline_DEBUG.log
│   ├── views_pipeline_DEBUG.log.2025-03-05
│   ├── views_pipeline_ERROR.log
│   ├── views_pipeline_INFO.log
│   ├── views_pipeline_INFO.log.2025-03-05
│   ├── views_pipeline_WARNING.log
│   └── views_pipeline_WARNING.log.2025-03-05
├── main.py
├── notebooks
├── reports
├── requirements.txt
├── run.sh
└── wandb
    ├── debug-internal.log
    ├── debug.log
    ├── latest-run
    │   ├── files
    │   │   ├── generated
    │   │   │   ├── eval_month_validation_20250306_104346.parquet
    │   │   │   ├── eval_step_validation_20250306_104346.parquet
    │   │   │   ├── eval_ts_validation_20250306_104346.parquet
    │   │   │   ├── predictions_validation_20250306_104346_00.parquet
    │   │   │   ├── predictions_validation_20250306_104346_01.parquet
    │   │   │   ├── predictions_validation_20250306_104346_02.parquet
    │   │   │   ├── predictions_validation_20250306_104346_03.parquet
    │   │   │   ├── predictions_validation_20250306_104346_04.parquet
    │   │   │   ├── predictions_validation_20250306_104346_05.parquet
    │   │   │   ├── predictions_validation_20250306_104346_06.parquet
    │   │   │   ├── predictions_validation_20250306_104346_07.parquet
    │   │   │   ├── predictions_validation_20250306_104346_08.parquet
    │   │   │   ├── predictions_validation_20250306_104346_09.parquet
    │   │   │   ├── predictions_validation_20250306_104346_10.parquet
    │   │   │   └── predictions_validation_20250306_104346_11.parquet
    │   │   ├── media
    │   │   │   └── table
    │   │   │       ├── evaluation_metrics_month_101_4356a7afafaf57e23ffb.table.json
    │   │   │       ├── evaluation_metrics_step_101_ac9ad62374cc37f2f54d.table.json
    │   │   │       ├── evaluation_metrics_ts_101_5247f14722b8ec946e2e.table.json
    │   │   │       ├── predictions_102_6f0555e01a8150d08114.table.json
    │   │   │       ├── predictions_103_4322ff6e4c0b7723c8b3.table.json
    │   │   │       ├── predictions_104_d813db1ff5e43fae661a.table.json
    │   │   │       ├── predictions_105_544380b76313f7091d50.table.json
    │   │   │       ├── predictions_106_7d7d4846b0a4d91e376b.table.json
    │   │   │       ├── predictions_107_ab277efa7a7a5f08af1a.table.json
    │   │   │       ├── predictions_108_d9195d5b9e41a0d126b4.table.json
    │   │   │       ├── predictions_109_3206154e1f17befb3a58.table.json
    │   │   │       ├── predictions_110_24991ccff4fffe41e1e4.table.json
    │   │   │       ├── predictions_111_61773ee1ee8a08ef50ef.table.json
    │   │   │       ├── predictions_112_a861716cbd11a431a8cb.table.json
    │   │   │       └── predictions_113_fe9ab345d72a37a97da6.table.json
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-dbqgajah.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250305_123845-2hgovzm7
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-2hgovzm7.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250305_123953-uqhsnz2l
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-uqhsnz2l.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250305_124316-zzt9pmxg
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-zzt9pmxg.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250305_124341-f0h9ev3h
    │   ├── files
    │   │   ├── generated
    │   │   │   ├── eval_month_calibration_20250305_124458.parquet
    │   │   │   ├── eval_step_calibration_20250305_124458.parquet
    │   │   │   ├── eval_ts_calibration_20250305_124458.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_00.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_01.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_02.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_03.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_04.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_05.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_06.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_07.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_08.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_09.parquet
    │   │   │   ├── predictions_calibration_20250305_124458_10.parquet
    │   │   │   └── predictions_calibration_20250305_124458_11.parquet
    │   │   ├── media
    │   │   │   └── table
    │   │   │       ├── evaluation_metrics_month_101_6256b95079bfd458b8c9.table.json
    │   │   │       ├── evaluation_metrics_step_101_b8b7bf881b4dd5875900.table.json
    │   │   │       ├── evaluation_metrics_ts_101_d421ab65735476dd4b7a.table.json
    │   │   │       ├── predictions_102_e40e4ed52bf6dbb7d473.table.json
    │   │   │       ├── predictions_103_568163d38684ce837039.table.json
    │   │   │       ├── predictions_104_a15ce32b8ddfe64d1cf6.table.json
    │   │   │       ├── predictions_105_cf13512647fcc80061aa.table.json
    │   │   │       ├── predictions_106_0e95e19bdae14cd9dd57.table.json
    │   │   │       ├── predictions_107_0de1f85a8442a1af572d.table.json
    │   │   │       ├── predictions_108_7efabeee6cc1e700616a.table.json
    │   │   │       ├── predictions_109_0a7e9a6fe830c639b1cf.table.json
    │   │   │       ├── predictions_110_99f957541617dfcc913e.table.json
    │   │   │       ├── predictions_111_47578e50ba7f2e76715b.table.json
    │   │   │       ├── predictions_112_ab87125d11ca834c2360.table.json
    │   │   │       └── predictions_113_9ec9848558a8cab442d9.table.json
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-f0h9ev3h.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250306_104205-iuvw9d4q
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-iuvw9d4q.wandb
    │   └── tmp
    │       └── code
    └── offline-run-20250306_104229-dbqgajah
        ├── files
        │   ├── generated
        │   │   ├── eval_month_validation_20250306_104346.parquet
        │   │   ├── eval_step_validation_20250306_104346.parquet
        │   │   ├── eval_ts_validation_20250306_104346.parquet
        │   │   ├── predictions_validation_20250306_104346_00.parquet
        │   │   ├── predictions_validation_20250306_104346_01.parquet
        │   │   ├── predictions_validation_20250306_104346_02.parquet
        │   │   ├── predictions_validation_20250306_104346_03.parquet
        │   │   ├── predictions_validation_20250306_104346_04.parquet
        │   │   ├── predictions_validation_20250306_104346_05.parquet
        │   │   ├── predictions_validation_20250306_104346_06.parquet
        │   │   ├── predictions_validation_20250306_104346_07.parquet
        │   │   ├── predictions_validation_20250306_104346_08.parquet
        │   │   ├── predictions_validation_20250306_104346_09.parquet
        │   │   ├── predictions_validation_20250306_104346_10.parquet
        │   │   └── predictions_validation_20250306_104346_11.parquet
        │   ├── media
        │   │   └── table
        │   │       ├── evaluation_metrics_month_101_4356a7afafaf57e23ffb.table.json
        │   │       ├── evaluation_metrics_step_101_ac9ad62374cc37f2f54d.table.json
        │   │       ├── evaluation_metrics_ts_101_5247f14722b8ec946e2e.table.json
        │   │       ├── predictions_102_6f0555e01a8150d08114.table.json
        │   │       ├── predictions_103_4322ff6e4c0b7723c8b3.table.json
        │   │       ├── predictions_104_d813db1ff5e43fae661a.table.json
        │   │       ├── predictions_105_544380b76313f7091d50.table.json
        │   │       ├── predictions_106_7d7d4846b0a4d91e376b.table.json
        │   │       ├── predictions_107_ab277efa7a7a5f08af1a.table.json
        │   │       ├── predictions_108_d9195d5b9e41a0d126b4.table.json
        │   │       ├── predictions_109_3206154e1f17befb3a58.table.json
        │   │       ├── predictions_110_24991ccff4fffe41e1e4.table.json
        │   │       ├── predictions_111_61773ee1ee8a08ef50ef.table.json
        │   │       ├── predictions_112_a861716cbd11a431a8cb.table.json
        │   │       └── predictions_113_fe9ab345d72a37a97da6.table.json
        │   ├── output.log
        │   ├── requirements.txt
        │   └── wandb-metadata.json
        ├── logs
        │   ├── debug-core.log
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-dbqgajah.wandb
        └── tmp
            └── code
```
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
