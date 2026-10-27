from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.ensemble import (
    EnsemblePathManager,
    PredictionFrameEnsembleManager,
)

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    # No reconciler: chrome_crate is the side that is reconciled AGAINST, so it declares
    # `reconciliation: None` and injects nothing. chrome_box loads this ensemble's cm
    # forecast by name (`reconcile_with`) and scales its own grid draws to it.
    #
    # PredictionFrameEnsembleManager because the pool is distributional — 8 members x 128
    # draws concatenated on the sample axis. Equal counts on both sides are what put the
    # reconciliation in ALIGNED_DRAWS rather than POINT_BROADCAST; the mode is derived
    # from this frame's sample_count, so thinning here silently changes chrome_box's mode.
    manager = PredictionFrameEnsembleManager(
        ensemble_path=ensemble_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    manager.execute_single_run(args)
