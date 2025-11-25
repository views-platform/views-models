import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_pipeline_core.managers.model import ModelPathManager
from views_baseline.manager.baseline_manager import BaselineForecastingModelManager

warnings.filterwarnings("ignore")

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    if args.sweep:
        print("No Sweep Run for Baseline Models")
    else:
        BaselineForecastingModelManager(
            model_path=model_path,
            wandb_notifications=args.wandb_notifications,
            use_prediction_store=args.prediction_store,
        ).execute_single_run(args)
