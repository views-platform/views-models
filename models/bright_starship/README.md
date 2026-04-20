# Bright Starship
## Overview

Datafactory-powered variant of purple_alien. Same HydraNet architecture, same hyperparameters, same training loop. The only difference is the data source: instead of pulling from viewser, bright_starship fetches from the VIEWS data factory zarr store on Hetzner at runtime.

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

## Differences from purple_alien

### Country identity (`c_id`)

purple_alien's `c_id` uses Gleditsch & Ward / ETH C-Shapes codes — a **time-varying** country assignment where a grid cell's country can change across months (e.g., Sudan/South Sudan split in 2011). This conflates spatial identity with temporal political signal.

bright_starship's `c_id` uses FAO GAUL codes — a **time-invariant** assignment where each grid cell maps to exactly one country code across all months. This separates identity from signal: `c_id` is metadata for grouping and tracing, not a feature the model should learn from.

If temporal boundary information is needed as a predictive feature (e.g., sovereignty transitions), it should be constructed as an explicit, named feature — not carried implicitly through the identity column. See [ADR-025](https://github.com/views-platform/views-datafactory/blob/development/docs/ADRs/025_country_identity_gaul.md).

### Event values

Event columns (`lr_sb_best`, `lr_ns_best`, `lr_os_best`) show ~0.05-0.14% cell-level differences due to UCDP annual data versions — the factory uses v25.1, viewser uses an older version. This is a data freshness difference, not a pipeline bug. See `reports/consumer_parity_investigation.md` in views-datafactory.

### Data audit

Run `scripts/audit_data_parity.py` to compare bright_starship's Hetzner fetch against purple_alien's viewser data:

```bash
cd views-datafactory
uv run python ../views-models/models/bright_starship/scripts/audit_data_parity.py
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
