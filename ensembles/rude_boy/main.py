import warnings
from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager

warnings.filterwarnings("ignore")

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = EnsembleManager(
        ensemble_path=ensemble_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    manager.execute_single_run(args)

