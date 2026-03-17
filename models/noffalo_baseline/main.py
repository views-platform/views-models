import wandb
import warnings
from pathlib import Path
from views_impact.cli import ImpactModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_baseline.manager.baseline_manager import BaselineImpactModelManager

warnings.filterwarnings("ignore")

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ImpactModelArgs.parse_args()

    manager = BaselineImpactModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )
    
    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
