"""Data specification for shining_codex (datafactory consumer — country-month).

This replaces the viewser Queryset pattern used in novel_heuristics.
Instead of connecting to PRIO's PostgreSQL via viewser, shining_codex
fetches from the VIEWS data factory via load_dataset() with
output_format="country_month", which sums grid-cell features per
(month_id, country_id) using gaul0_code as the grouping key.

Data flow:
  zarr store (HTTP) → numpy grid [T,H,W,C] → flatten + region filter
  → groupby(month_id, gaul0_code).sum()
  → DataFrame with MultiIndex (month_id, country_id)

On first run, main.py calls fetch_data() which:
  1. Calls load_dataset(output_format="country_month")
  2. Renames factory column names to novel_heuristics convention
  3. Fills NaN with 0.0
  4. Saves as {run_type}_viewser_df.parquet in data/raw/

Subsequent runs use the cached parquet directly — delete it to re-fetch.

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108 (see datafactory docs)
"""

from __future__ import annotations

import logging
from pathlib import Path

from datafactory_query.defaults import DEFAULT_REMOTE
from views_pipeline_core.managers.model import ModelPathManager

model_name = ModelPathManager.get_model_name_from_path(__file__)
logger = logging.getLogger(__name__)

ZARR_URL = DEFAULT_REMOTE.zarr_url

REGION = "africa_me_legacy"

FACTORY_FEATURES = ["ged_sb_best"]

FEATURE_RENAME = {
    "ged_sb_best": "lr_ged_sb",
}


def generate():
    """Data source descriptor (satisfies ModelPathManager.get_queryset() interface).

    NOTE: Not used at runtime. main.py's _ensure_data() calls fetch_data()
    directly, so the parquet exists before DartsForecastingModelManager ever
    loads this. Kept for interface compatibility with views-pipeline-core.
    """
    return {
        "name": model_name,
        "source": "views-datafactory",
        "zarr_url": ZARR_URL,
        "region": REGION,
        "loa": "country_month",
        "features": FEATURE_RENAME,
    }


def fetch_data(
    run_type: str,
    output_dir: Path,
    partitions: dict,
) -> Path:
    """Fetch country-month data from the factory and save as parquet.

    Called by main.py's _ensure_data() when the cached parquet is missing.

    Args:
        run_type: "calibration", "validation", or "forecasting".
        output_dir: Directory for the parquet file (typically data/raw/).
        partitions: Dict from config_partitions.generate().

    Returns:
        Path to the saved parquet file.

    Output contract (parity with novel_heuristics):
        - MultiIndex: (month_id, country_id)
        - Columns: lr_ged_sb_dep (feature), lr_ged_sb (target)
        - Both columns contain identical data (country-sum of ged_sb_best)
        - lr_ged_sb_dep is legacy naming for "dependent variable as feature"
        - No NaN values (filled with 0.0)
    """
    from datafactory_query import load_dataset

    bounds = partitions[run_type]
    start = bounds["train"][0]
    end = bounds["test"][1]

    logger.info(
        "Fetching %s cm data from %s (months %d-%d, region=%s)",
        run_type, ZARR_URL, start, end, REGION,
    )

    df = load_dataset(
        region=REGION,
        start=start,
        end=end,
        features=FACTORY_FEATURES,
        output_format="country_month",
        data_dir=ZARR_URL,
    )

    df = df.rename(columns=FEATURE_RENAME)
    df["lr_ged_sb_dep"] = df["lr_ged_sb"]
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
