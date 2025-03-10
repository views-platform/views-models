import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_pipeline_core.managers.log import LoggingManager
from views_pipeline_core.managers.model import ModelPathManager
from views_stepshifter.manager.stepshifter_manager import StepshifterManager
from multiprocessing import set_start_method


warnings.filterwarnings("ignore")

try:
    model_path = ModelPathManager(Path(__file__))
    logger = LoggingManager(model_path).get_logger()
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
    set_start_method('spawn')
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    if args.sweep:
        StepshifterManager(model_path=model_path).execute_sweep_run(args)
    else:
        StepshifterManager(model_path=model_path).execute_single_run(args)
