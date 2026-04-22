# shining_codex

Country-month N-BEATS model — datafactory consumer clone of `novel_heuristics`.

## Overview

| Field | Value |
|-------|-------|
| Algorithm | N-BEATS (`NBEATSModel`) |
| Level of analysis | Country-month (`cm`) |
| Target | `lr_ged_sb` (state-based fatalities, country sum) |
| Data source | views-datafactory (zarr over HTTP) |
| Manager | `DartsForecastingModelManager` (views-r2darts2) |

## Data source

Unlike `novel_heuristics` which fetches from PRIO's PostgreSQL via viewser,
`shining_codex` fetches from the VIEWS data factory zarr store. The factory
sums grid-cell UCDP fatalities per country per month using `gaul0_code` as
the grouping key (`output_format="country_month"`).

**Features (parity with novel_heuristics):**

| Factory name | Model name | Role | Description |
|-------------|------------|------|-------------|
| `ged_sb_best` | `lr_ged_sb` | Target | State-based fatalities (country sum) |
| `ged_sb_best` | `lr_ged_sb_dep` | Feature | Same data, legacy `_dep` naming convention |

## Setup

```bash
# ~/.netrc credentials for Hetzner zarr store
machine 204.168.219.108
    login views
    password <ask-team>
```

## Usage

```bash
# Via run.sh (creates/activates conda env automatically)
bash run.sh --run_type calibration

# Or directly
python main.py --run_type calibration
python main.py --run_type validation
python main.py --run_type forecasting
```

## Relationship to novel_heuristics

`shining_codex` is a direct clone of `novel_heuristics` with the viewser
dependency replaced by views-datafactory. Same hyperparameters, same
architecture, same partitions. Only the data source differs.

Scope: state-based fatalities only (matching novel_heuristics active
features). ns/os violence types, WDI, V-DEM, and topic model features
from the commented-out sections are not yet available via datafactory.
