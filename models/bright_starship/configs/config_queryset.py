"""Data specification for bright_starship (datafactory consumer).

This replaces the viewser Queryset pattern used in other models.
Instead of connecting to PRIO's PostgreSQL via viewser, bright_starship
fetches from the VIEWS data factory via load_dataset().

load_dataset() is a unified API that auto-detects the storage backend
(local npy directory vs zarr store) from the path. The URL below
points to a zarr store served over HTTP, but the function returns a
pandas DataFrame — zarr is the storage format, DataFrame is the
consumer interface.

Data flow:
  zarr store (HTTP) → numpy grid [T,H,W,C] → flatten + region filter
  → DataFrame with MultiIndex (month_id, priogrid_gid)

On first run, main.py calls fetch_data() which:
  1. Calls load_dataset() — opens zarr, subsets by time + region
  2. Renames factory column names to VIEWSER convention
  3. Derives row/col grid coordinates from priogrid_gid
  4. Fills NaN with 0.0 (defensive — see note below)
  5. Saves as {run_type}_viewser_df.parquet in data/raw/

Subsequent runs use the cached parquet directly — delete it to re-fetch.

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see README.md for setup)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)
logger = logging.getLogger(__name__)

# Data source URL — load_dataset() detects zarr vs npy from the path.
# Zarr over HTTP requires ~/.netrc credentials (see README.md).
ZARR_URL = DEFAULT_REMOTE.zarr_url

# 13,110 PRIO-GRID cells matching VIEWSER's Africa + Middle East coverage
REGION = "africa_me_legacy"

# UCDP field names as stored in the zarr store
FACTORY_FEATURES = ["ged_sb_best", "ged_ns_best", "ged_os_best", "gaul0_code"]

# Factory name → VIEWSER name (so downstream model code doesn't change)
FEATURE_RENAME = {
    "ged_sb_best": "lr_sb_best",   # state-based fatalities (best estimate)
    "ged_ns_best": "lr_ns_best",   # non-state fatalities
    "ged_os_best": "lr_os_best",   # one-sided violence fatalities
    "gaul0_code": "c_id",          # FAO GAUL country code → identity column
}

# PRIO-GRID is 720 columns wide (0.5° global grid). Used to derive row/col.
NCOL = 720


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface).

    NOTE: Not used at runtime. main.py's _ensure_data() calls fetch_data()
    directly, so the parquet exists before HydranetManager ever loads this.
    Kept for interface compatibility with views-pipeline-core.
    """
    return {
        "name": model_name,
        "source": "views-datafactory",  # "views-datafactory" or "viewser"
        "zarr_url": ZARR_URL,
        "region": REGION,               # any datafactory_query region name
        "loa": "priogrid_month",        # "priogrid_month" or "country_month"
        "features": FEATURE_RENAME,
    }


def fetch_data(
    run_type: str,
    output_dir: Path,
    partitions: dict,
) -> Path:
    """Fetch data from the factory and save as VIEWSER-compatible parquet.

    Called by main.py's _ensure_data() when the cached parquet is missing.

    Data flow:
        1. load_dataset() opens the zarr store over HTTP
        2. Subsets by time range (lazy — only requested months are downloaded)
        3. Subsets by region (africa_me_legacy = 13,110 PRIO-GRID cells)
        4. Returns DataFrame with MultiIndex (month_id, priogrid_gid)
        5. This function renames columns, derives row/col, fills NaN, saves

    Args:
        run_type: "calibration", "validation", or "forecasting".
        output_dir: Directory for the parquet file (typically data/raw/).
        partitions: Dict from config_partitions.generate() with structure
            {run_type: {"train": (start, end), "test": (start, end)}}.

    Returns:
        Path to the saved parquet file.

    Output contract:
        - MultiIndex: (month_id, priogrid_gid)
        - Columns: lr_sb_best, lr_ns_best, lr_os_best, c_id, row, col
        - No NaN values (filled with 0.0)
        - dtypes: float32 for event columns (from zarr), float64 for row/col/c_id
    """
    from datafactory_query import load_dataset

    bounds = partitions[run_type]
    start = bounds["train"][0]
    end = bounds["test"][1]

    logger.info(
        "Fetching %s data from %s (months %d-%d, region=%s)",
        run_type, ZARR_URL, start, end, REGION,
    )

    df = load_dataset(
        region=REGION,
        start=start,
        end=end,
        features=FACTORY_FEATURES,
        output_format="dataframe",  # also: "feature_frame" (numpy + identifiers)
        data_dir=ZARR_URL,          # auto-detects zarr vs npy from path
    )

    df = df.rename(columns=FEATURE_RENAME)

    # HydraNet expects row/col but the zarr store only has priogrid_gid.
    # PRIO-GRID definition: row = (pgid-1)//720 + 1, col = (pgid-1)%720 + 1.
    pgids = df.index.get_level_values("priogrid_gid")
    df["row"] = ((pgids - 1) // NCOL + 1).astype(np.float64)
    df["col"] = ((pgids - 1) % NCOL + 1).astype(np.float64)

    # Defensive: assembled grid has no NaN (events=0.0, admin=-1.0),
    # but the VIEWSER parquet contract requires no NaN. Cheap insurance.
    df = df.fillna(0.0)
    df = df.sort_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_type}_viewser_df.parquet"
    df.to_parquet(out_path)

    logger.info(
        "Saved %s: %d rows, %.1f MB",
        out_path, len(df), out_path.stat().st_size / 1e6,
    )
    return out_path
