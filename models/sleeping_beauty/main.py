import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_pipeline_core.managers.model import ModelPathManager

# Import your model manager class here
from views_markov.manager.markovmodel_manager import MarkovModelManager

warnings.filterwarnings("ignore")

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

if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    manager = MarkovModelManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )
    
    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
