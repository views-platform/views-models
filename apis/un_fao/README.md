# UN FAO API

**Name:** `un_fao`  
**Type:** API  
**Level of Analysis:** PRIO-GRID Month (PGM)  
**Created:** 2025-10-09

## Overview

The UN FAO API serves VIEWS conflict prediction data to the United Nations Food and Agriculture Organization via a FastAPI REST interface. It provides endpoints for accessing historical observations and probabilistic forecasts at multiple geographic aggregation levels (PRIO-GRID, country, GAUL administrative boundaries).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         UN FAO API Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │    Client    │───►│  FAOApi      │───►│   Appwrite   │              │
│  │  (UN FAO)    │◄───│  Manager     │◄───│   Storage    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                             │                                           │
│                             ▼                                           │
│                      ┌──────────────┐                                   │
│                      │ FAO_PGMDataset│                                  │
│                      │ - HDI/MAP    │                                   │
│                      │ - Aggregation│                                   │
│                      └──────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-level geographic aggregation**: PRIO-GRID, country (ISO A3), GAUL levels 0/1/2
- **Probabilistic predictions**: Full posterior distributions with HDI and MAP estimates
- **Statistically correct aggregation**: Element-wise distribution summation
- **Per-user authentication**: Appwrite API key validation
- **Caching**: Manager, dataframe, and file-level caching
- **Multi-worker support**: Production deployment with uvicorn workers

## Configuration

### Meta Configuration (`configs/config_meta.py`)

```python
meta_config = {
    "name": "un_fao",
    "algorithm": "API",
    "historical_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    "level": "pgm"
}
```

### Deployment Configuration (`configs/config_deployment.py`)

```python
deployment_config = {
    "deployment_status": "shadow",
    "port": 8000,
    "host": "0.0.0.0",
    "action": "start",
    "workers": 4,
    "reload": "false"
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `port` | 8000 | Server port |
| `host` | 0.0.0.0 | Bind address |
| `workers` | 4 | Uvicorn worker processes |
| `reload` | false | Auto-reload on code changes |

## API Endpoints

### Data Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | List all available endpoints |
| `GET /health` | Health check and cache stats |
| `GET /data/historical/latest` | Latest historical observations |
| `GET /data/forecast/latest` | Latest forecast predictions |
| `GET /{level}/data/{category}/subset` | Filtered data subset |
| `GET /{level}/analysis/{category}/hdi-map` | HDI and MAP estimates |

### File Management

| Endpoint | Description |
|----------|-------------|
| `GET /files/{bucket_id}` | List files in bucket |
| `GET /files/{bucket_id}/{file_id}/info` | File metadata |
| `GET /files/{bucket_id}/{file_id}/download` | Download file |
| `GET /files/{bucket_id}/{file_id}/cached` | Get cached file |

### Cache Management

| Endpoint | Description |
|----------|-------------|
| `GET /cache/stats` | Cache statistics |
| `DELETE /cache` | Clear cache |

### Geographic Levels

| Level | Description | Entity ID Format |
|-------|-------------|------------------|
| `pg` | PRIO-GRID cells | Integer |
| `country` | Countries | ISO Alpha-3 (e.g., `SOM`) |
| `gaul0` | Admin Level 0 | Integer |
| `gaul1` | Admin Level 1 | Integer |
| `gaul2` | Admin Level 2 | Integer |

## Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `time_ids` | string | Comma-separated month IDs |
| `features` | string | Comma-separated feature names |
| `sample_idx` | string | Comma-separated sample indices |
| `entity_ids` | string | Entity IDs (ISO3 or integers) |
| `with_metadata` | bool | Include geographic metadata |
| `aggregate` | bool | Aggregate to specified level |
| `alpha` | float | HDI credibility level (default: 0.9) |
| `enforce_non_negative` | bool | Floor MAP at 0 |
| `force_refresh` | bool | Bypass cache |

## Authentication

All requests require an Appwrite API key in the header:

```bash
curl -H "X-API-Key: your_appwrite_api_key" http://localhost:8000/health
```

## Usage

### Running the API

```bash
# Using the run script (recommended)
./run.sh

# Or directly with Python
python main.py
```

### Example Requests

```bash
# Get latest forecast data
curl -H "X-API-Key: $API_KEY" "http://localhost:8000/data/forecast/latest"

# Get country-level HDI for Somalia
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/country/analysis/forecast/hdi-map?entity_ids=SOM&alpha=0.95"

# Get aggregated data for multiple countries
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/country/data/forecast/subset?entity_ids=SOM,ETH,KEN&aggregate=true"
```

### Python Client Example

```python
import requests

API_URL = "http://localhost:8000"
headers = {"X-API-Key": "your_api_key"}

# Get HDI/MAP for forecast
response = requests.get(
    f"{API_URL}/country/analysis/forecast/hdi-map",
    headers=headers,
    params={
        "entity_ids": "SOM,ETH",
        "alpha": "0.9",
        "aggregate": "true"
    }
)
data = response.json()
```

## Directory Structure

```
apis/un_fao/
├── main.py                 # Entry point
├── run.sh                  # Shell runner with conda setup
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── configs/
│   ├── config_meta.py          # API metadata
│   └── config_deployment.py    # Server configuration
├── data/                   # Local data storage
├── artifacts/              # API artifacts
├── cache/                  # Local cache directory
├── appwrite_cache/         # Appwrite file cache
├── logs/                   # Execution logs
├── reports/                # Generated reports
├── notebooks/              # Development notebooks
└── wandb/                  # WandB logs
```

## Dependencies

- `views-faoapi` - Core API implementation (from GitHub)
- FastAPI - Web framework
- Uvicorn - ASGI server
- Pandas/NumPy - Data manipulation
- Appwrite SDK - Cloud storage

## Environment Variables

Required in the postprocessor's ensemble `.env`:

```bash
APPWRITE_ENDPOINT=https://cloud.appwrite.io/v1
APPWRITE_DATASTORE_PROJECT_ID=your_project_id
APPWRITE_UNFAO_BUCKET_ID=unfao_predictions
```

## Deployment Status

**Current Status:** `shadow`

| Status | Description |
|--------|-------------|
| `shadow` | Testing phase, not public |
| `deployed` | Active and serving clients |
| `baseline` | Reference implementation |
| `deprecated` | No longer supported |

## Response Format

### Success Response

```json
{
  "success": true,
  "data": {
    "dataframe": [...],
    "shape": [rows, cols],
    "columns": ["col1", "col2", ...],
    "category": "forecast",
    "level": "country"
  }
}
```

### HDI-MAP Response

```json
{
  "success": true,
  "data": {
    "hdi_map": [
      {
        "country_iso_a3": "SOM",
        "pred_ln_sb_best_lower": 2.34,
        "pred_ln_sb_best_upper": 5.67,
        "pred_ln_sb_best_map": 3.45,
        "pred_ln_sb_best_min": 0.12,
        "pred_ln_sb_best_max": 8.91
      }
    ],
    "parameters": {
      "alpha": 0.9,
      "aggregate": true
    }
  }
}
```

## Related Components

- **[views-faoapi](https://github.com/views-platform/views-faoapi)** - Core API library
- **[postprocessors/un_fao](../postprocessors/un_fao/)** - Data preparation pipeline

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Invalid API key | Verify `X-API-Key` header |
| 404 No data | Empty bucket | Run postprocessor first |
| Connection refused | Server not running | Start with `./run.sh` |
| Port in use | Another process | Change port in config_deployment.py |

## Documentation

For detailed API documentation, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`