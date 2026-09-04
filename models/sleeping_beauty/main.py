from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.model import ModelPathManager
from views_markov.manager.markovmodel_manager import MarkovModelManager

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    manager = MarkovModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )
    
    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)