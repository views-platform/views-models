import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_pipeline_core.logging.utils import setup_logging
from views_pipeline_core.managers.model import ModelPathManager
from views_stepshifter.manager.stepshifter_manager import StepshifterManager

warnings.filterwarnings("ignore")

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"An unexpected error occurred: {e}.")

logger = setup_logging(logging_path=model_path.logging)


if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    if args.sweep:
        StepshifterManager(model_path=model_path).execute_sweep_run(args)
    else:
        StepshifterManager(model_path=model_path).execute_single_run(args)
