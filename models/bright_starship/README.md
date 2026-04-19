# Bright Starship
## Overview

Datafactory-powered variant of purple_alien. Same HydraNet architecture, same hyperparameters, same training loop. The only difference is the data source: instead of pulling from viewser, input data is pre-generated from views-datafactory via `generate_consumer_data.py`.

This model exists to prove the datafactory consumer path works end-to-end with a real training script (v1.2 milestone M11).

| Information         | Details                        |
|---------------------|--------------------------------|
| **Model Algorithm** | HydraNet                  |
| **Level of Analysis** | pgm            |
| **Data Source** | views-datafactory (not viewser) |
| **Targets**         | lr_sb_best, lr_ns_best, lr_os_best, by_sb_best, by_ns_best, by_os_best |
| **Features**       | lr_sb_best, lr_ns_best, lr_os_best |
| **Deployment Status**       |  shadow    |

## Prerequisites

### 1. Install views-datafactory

```bash
pip install "views-datafactory @ git+https://github.com/views-platform/views-datafactory.git@development"
```

### 2. Configure Hetzner credentials

The model fetches data from the VIEWS data factory zarr store on Hetzner (`204.168.219.108`). Authentication uses HTTP Basic Auth via `~/.netrc`.

Add this entry to `~/.netrc` (create the file if it doesn't exist):

```
machine 204.168.219.108
    login <your-username>
    password <your-password>
```

Then restrict permissions: `chmod 600 ~/.netrc`

Contact the VIEWS team for credentials.

## Data Flow

On first run, `main.py` checks for cached parquets in `data/raw/`. If the cache is missing, it fetches data from the Hetzner zarr store via `datafactory_query.load_dataset()`, renames columns to VIEWSER convention, and saves the parquet. Subsequent runs use the cache directly.

To force a re-fetch, delete the cached parquet:

```bash
rm data/raw/calibration_viewser_df.parquet
```

## Usage

```bash
# Train on calibration partition
./run.sh -r calibration -t

# Evaluate on calibration partition
./run.sh -r calibration -e

# Train and evaluate
./run.sh -r calibration -t -e
```

## Repository Structure

```
bright_starship
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
│   ├── config_queryset.py
│   ├── config_sweep.py
├── data
│   ├── generated
│   ├── processed
│   ├── raw
├── reports
├── notebooks
```
