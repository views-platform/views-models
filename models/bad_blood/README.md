# Bad Blood 
## Overview


| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | LGBMRegressor                  |
| **Level of Analysis** | pgm            |
| **Target**         | ln_ged_sb_dep |
| **Features**       |  fatalities003_pgm_natsoc   |
| **Feature Description**       |  Fatalities natural and social geography, pgm level Predicting ln(fatalities) using natural and social geography features    |
| **Metrics**       |  RMSLE, CRPS    |
| **Deployment Status**       |  shadow    |

## Repository Structure

```
├── README.md
├── artifacts
├── configs
│   ├── config_deployment.py
│   ├── config_hyperparameters.py
│   ├── config_meta.py
│   ├── config_queryset.py
│   └── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   └── raw
│       ├── forecasting_data_fetch_log.txt
│       └── forecasting_viewser_df.parquet
├── logs
│   ├── views_pipeline_CRITICAL.log
│   ├── views_pipeline_DEBUG.log
│   ├── views_pipeline_ERROR.log
│   ├── views_pipeline_INFO.log
│   └── views_pipeline_WARNING.log
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
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-3zmnuaf1.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250310_102213-ny4hutat
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ny4hutat.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250310_102251-r2cqxsvx
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-r2cqxsvx.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250310_102714-qenfi8ef
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-qenfi8ef.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250310_102754-q2tuxed0
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-q2tuxed0.wandb
    │   └── tmp
    │       └── code
    ├── offline-run-20250310_103246-ijwsgpx6
    │   ├── files
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   └── wandb-metadata.json
    │   ├── logs
    │   │   ├── debug-core.log
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-ijwsgpx6.wandb
    │   └── tmp
    │       └── code
    └── offline-run-20250310_103326-3zmnuaf1
        ├── files
        │   ├── output.log
        │   ├── requirements.txt
        │   └── wandb-metadata.json
        ├── logs
        │   ├── debug-core.log
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-3zmnuaf1.wandb
        └── tmp
            └── code
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


