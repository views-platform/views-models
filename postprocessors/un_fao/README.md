# UN FAO Postprocessor

**Model Name:** `un_fao`  
**Type:** Postprocessor  
**Level of Analysis:** PRIO-GRID Month (PGM)  
**Created:** 2025-10-17

## Overview

The UN FAO postprocessor prepares VIEWS conflict prediction data for delivery to the United Nations Food and Agriculture Organization. It enriches ensemble forecast outputs with geographic metadata and uploads both historical observations and forecasts to Appwrite cloud storage for consumption by the FAO API.

## Purpose

This postprocessor:
1. **Fetches historical data** from Viewser (VIEWS data repository)
2. **Downloads ensemble forecasts** from the production datastore
3. **Enriches data with geographic metadata** (coordinates, country codes, GAUL admin boundaries)
4. **Validates metadata completeness** before saving
5. **Uploads processed data** to Appwrite for API consumption

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UN FAO Postprocessor Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Viewser    │    │   Appwrite   │    │   Mapper     │              │
│  │ (Historical) │    │ (Forecasts)  │    │ (Metadata)   │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────────────────────────────────────────────┐              │
│  │              UNFAOPostProcessorManager               │              │
│  │   _read() → _transform() → _validate() → _save()    │              │
│  └──────────────────────────────────────────────────────┘              │
│                            │                                            │
│                            ▼                                            │
│  ┌──────────────────────────────────────────────────────┐              │
│  │            Appwrite UNFAO Bucket                     │              │
│  │   historical_dataset_{timestamp}.parquet             │              │
│  │   forecast_dataset_{timestamp}.parquet               │              │
│  └──────────────────────────────────────────────────────┘              │
│                            │                                            │
│                            ▼                                            │
│  ┌──────────────────────────────────────────────────────┐              │
│  │                  FAO API (views-faoapi)              │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Configuration

### Meta Configuration (`configs/config_meta.py`)

```python
meta_config = {
    "name": "un_fao",
    "algorithm": "postprocessor",
    "targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    "level": "pgm",
    "ensemble": "orange_ensemble"  # Example Source ensemble for forecasts
}
```

### Hyperparameters (`configs/config_hyperparameters.py`)

```python
hyperparameters = {
    "steps": [*range(1, 36 + 1, 1)]  # 36-month forecast horizon
}
```

### Partitions (`configs/config_partitions.py`)

| Partition | Train Range | Test Range |
|-----------|-------------|------------|
| Calibration | 121-444 | 445-492 |
| Validation | 121-492 | 493-540 |
| Forecasting | 121 to (current month - 1) | (current month) to (current month + steps) |

### Data Query (`configs/config_queryset.py`)

Fetches historical conflict data from Viewser:

| Column | Source | Description |
|--------|--------|-------------|
| `lr_ged_sb` | `ged_sb_best_sum_nokgi` | State-based conflict fatalities |
| `lr_ged_ns` | `ged_ns_best_sum_nokgi` | Non-state conflict fatalities |
| `lr_ged_os` | `ged_os_best_sum_nokgi` | One-sided violence fatalities |

## Geographic Metadata

The postprocessor enriches each PRIO-GRID cell with:

| Column | Description |
|--------|-------------|
| `pg_xcoord` | PRIO-GRID X coordinate (longitude) |
| `pg_ycoord` | PRIO-GRID Y coordinate (latitude) |
| `country_iso_a3` | ISO Alpha-3 country code |
| `admin1_gaul0_code` | GAUL Level 0 (country) code |
| `admin1_gaul0_name` | GAUL Level 0 name |
| `admin1_gaul1_code` | GAUL Level 1 (province) code |
| `admin1_gaul1_name` | GAUL Level 1 name |
| `admin2_gaul2_code` | GAUL Level 2 (district) code |
| `admin2_gaul2_name` | GAUL Level 2 name |

## Output Files

Uploaded to Appwrite UNFAO bucket:

| File | Category | Description |
|------|----------|-------------|
| `historical_dataset_{timestamp}.parquet` | historical | Past observations with metadata |
| `forecast_dataset_{timestamp}.parquet` | forecast | Ensemble predictions with metadata |

### Forecast Targets

The forecast dataset includes probabilistic predictions:

- `pred_ln_sb_best` - Log fatalities (state-based), best estimate distribution
- `pred_ln_ns_best` - Log fatalities (non-state), best estimate distribution
- `pred_ln_os_best` - Log fatalities (one-sided), best estimate distribution
- `pred_ln_sb_prob` - Probability distribution (state-based)
- `pred_ln_ns_prob` - Probability distribution (non-state)
- `pred_ln_os_prob` - Probability distribution (one-sided)

## Environment Variables

Required in `.env` (located in the ensemble directory):

```bash
# Appwrite Configuration
APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1
APPWRITE_DATASTORE_PROJECT_ID=your_project_id
APPWRITE_DATASTORE_API_KEY=your_api_key

# Production Forecasts Bucket (source)
APPWRITE_PROD_FORECASTS_BUCKET_ID=prod_forecasts_bucket
APPWRITE_PROD_FORECASTS_BUCKET_NAME=prod-forecasts
APPWRITE_PROD_FORECASTS_COLLECTION_ID=prod_collection_id
APPWRITE_PROD_FORECASTS_COLLECTION_NAME=prod-collection

# UNFAO Bucket (destination)
APPWRITE_UNFAO_BUCKET_ID=unfao_bucket_id
APPWRITE_UNFAO_BUCKET_NAME=unfao-predictions
APPWRITE_UNFAO_COLLECTION_ID=unfao_collection_id
APPWRITE_UNFAO_COLLECTION_NAME=unfao-collection

# Metadata Database
APPWRITE_METADATA_DATABASE_ID=metadata_db_id
APPWRITE_METADATA_DATABASE_NAME=metadata
```

## Usage

### Running the Postprocessor

```bash
# Using the run script
./run.sh

# Or directly with Python
python main.py
```

### Programmatic Usage

```python
from pathlib import Path
from views_postprocessing.unfao.managers import UNFAOPostProcessorManager
from views_pipeline_core.managers.postprocessor.postprocessor import PostprocessorPathManager

model_path = PostprocessorPathManager(Path(__file__))
manager = UNFAOPostProcessorManager(model_path=model_path)
manager.run(args=None)
```

## Directory Structure

```
postprocessors/un_fao/
├── main.py                 # Entry point
├── run.sh                  # Shell runner with conda setup
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── configs/
│   ├── config_meta.py          # Model metadata
│   ├── config_deployment.py    # Deployment status
│   ├── config_hyperparameters.py # Steps configuration
│   ├── config_partitions.py    # Train/test splits
│   ├── config_queryset.py      # Viewser query
│   └── config_sweep.py         # Sweep configuration
├── data/                   # Local data storage
├── artifacts/              # Model artifacts
├── logs/                   # Execution logs
├── reports/                # Generated reports
├── notebooks/              # Development notebooks
├── appwrite_cache/         # Local Appwrite cache
└── wandb/                  # WandB logs
```

## Dependencies

- `views-postprocessing>=0.1.0` - Core postprocessing utilities
- `views-pipeline-core` - Pipeline infrastructure
- `pandas` - Data manipulation
- `polars` - High-performance dataframes
- `wandb` - Experiment tracking

## Deployment Status

**Current Status:** `shadow`

| Status | Description |
|--------|-------------|
| `shadow` | Not yet active, testing phase |
| `deployed` | Active and serving FAO API |
| `baseline` | Reference implementation |
| `deprecated` | No longer supported |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing ensemble config | `ensemble` not in meta config | Add `"ensemble": "orange_ensemble"` to config_meta.py |
| Appwrite connection error | Invalid credentials | Check `.env` in the views-models directory |
| Missing metadata columns | Mapper failed | Verify shapefiles exist in views-postprocessing |
| Empty forecast dataframe | No data in production bucket | Ensure ensemble has uploaded forecasts |

## Notes

- The postprocessor requires WandB authentication (`wandb.login()`)
- Historical data is fetched fresh from Viewser on each run
- Forecasts are downloaded from the production ensemble bucket
- All timestamps use format `YYYYMMDD_HHMMSS`