import logging
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_hydranet.manager.hydranet_manager import HydranetManager

logger = logging.getLogger(__name__)

try:
    model_path = ModelPathManager(Path(__file__))
except FileNotFoundError as fnf_error:
    raise RuntimeError(
        f"File not found: {fnf_error}. Check the file path and try again."
    )
except PermissionError as perm_error:
    raise RuntimeError(
        f"Permission denied: {perm_error}. Check your permissions and try again."
    )
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")


def _ensure_data(run_type: str) -> None:
    """Fetch data from Hetzner zarr store if the parquet cache is missing."""
    raw_dir = model_path.data_raw
    parquet = raw_dir / f"{run_type}_viewser_df.parquet"
    if parquet.exists():
        logger.info("Using cached %s", parquet)
        return

    logger.info("Cache miss for %s — fetching from Hetzner", run_type)
    from configs.config_queryset import fetch_data
    from configs.config_partitions import generate as generate_partitions

    partitions = generate_partitions()
    fetch_data(run_type, raw_dir, partitions)


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    _ensure_data(args.run_type)

    # Phase 1 workaround for views-pipeline-core C-51: get_data() hardcodes
    # viewser as sole data source. bright_starship fetches from views-datafactory
    # via _ensure_data(), so the parquet cache is already populated. Setting
    # saved=True routes get_data() through the cache-read path (dataloaders.py:
    # 1253-1257), bypassing _fetch_data_from_viewser() which would crash on our
    # dict descriptor. Remove once Phase 2 lands (DataFetchStrategy dispatch).
    args.saved = True

    manager = HydranetManager(model_path=model_path)

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
