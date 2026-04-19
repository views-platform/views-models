"""Data specification for bright_starship.

Data is fetched from the views-datafactory zarr store on Hetzner,
not from viewser. On first run, if the parquet cache is missing,
fetch_data() pulls from the remote store, renames columns to
VIEWSER convention, and saves the cache. Subsequent runs read
the cached parquet directly.

Prerequisites:
    - pip install views-datafactory (for datafactory_query)
    - ~/.netrc entry for the Hetzner server (see README.md)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)
logger = logging.getLogger(__name__)

ZARR_URL = "http://204.168.219.108/grid.zarr"
REGION = "africa_me_legacy"

FACTORY_FEATURES = ["ged_sb_best", "ged_ns_best", "ged_os_best", "gaul0_code"]

FEATURE_RENAME = {
    "ged_sb_best": "lr_sb_best",
    "ged_ns_best": "lr_ns_best",
    "ged_os_best": "lr_os_best",
    "gaul0_code": "c_id",
}

NCOL = 720


def generate():
    """Return data source descriptor for bright_starship.

    Unlike viewser-based models, this model's data comes from
    views-datafactory. This function returns a descriptor dict
    (not a viewser Queryset) documenting the data contract.
    """
    return {
        "name": model_name,
        "source": "views-datafactory",
        "zarr_url": ZARR_URL,
        "region": REGION,
        "loa": "priogrid_month",
        "features": FEATURE_RENAME,
    }


def fetch_data(
    run_type: str,
    output_dir: Path,
    partitions: dict,
) -> Path:
    """Fetch data from the Hetzner zarr store and save as parquet.

    Called by main.py when the cached parquet is missing.

    Args:
        run_type: Partition name (calibration, validation, forecasting).
        output_dir: Directory to save the parquet file.
        partitions: Partition dict from config_partitions.generate().

    Returns:
        Path to the saved parquet file.
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
        output_format="dataframe",
        data_dir=ZARR_URL,
    )

    df = df.rename(columns=FEATURE_RENAME)

    pgids = df.index.get_level_values("priogrid_gid")
    df["row"] = ((pgids - 1) // NCOL + 1).astype(np.float64)
    df["col"] = ((pgids - 1) % NCOL + 1).astype(np.float64)

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
