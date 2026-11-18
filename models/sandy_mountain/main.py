import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.model import ModelPathManager
from views_skstepshifter.manager import SkStepshifterManager

# Narrowed from a bare `warnings.filterwarnings("ignore")` (#366). The blanket form
# silenced EVERY category in the process that consumes the data — including
# views-datafactory's coverage `UserWarning`, which under their ADR-047 is the only signal
# distinguishing an observed zero from a zero-filled gap. Their own words: a filled month
# is "structurally indistinguishable from months where the source observed zero events".
#
# The noise this was hiding is real, so it is silenced BY NAME rather than removed. What is
# no longer silenced is the category nobody considered — which is the one that mattered.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    args = ForecastingModelArgs.parse_args()

    manager = SkStepshifterManager(
        model_path=model_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
    )

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
