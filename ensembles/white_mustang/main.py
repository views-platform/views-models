import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_pipeline_core.logging.utils import setup_logging
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager

warnings.filterwarnings("ignore")

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"An unexpected error occurred: {e}.")

logger = setup_logging(logging_path=ensemble_path.logging)


if __name__ == "__main__":
    wandb.login()
    args = parse_args()
    validate_arguments(args)

    EnsembleManager(ensemble_path=ensemble_path).execute_single_run(args)
