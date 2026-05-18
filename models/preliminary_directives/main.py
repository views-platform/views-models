from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

from views_r2darts2 import DartsForecastingModelManager, apply_nbeats_patch

apply_nbeats_patch()

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()
    
    manager = DartsForecastingModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
    )

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)