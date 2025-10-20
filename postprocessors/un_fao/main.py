import wandb
import warnings
from pathlib import Path
from views_postprocessing.managers.unfao import UNFAOPostProcessorManager
from views_pipeline_core.managers.postprocessor import PostprocessorPathManager

# Import your model manager class here
# E.g. from views_stepshifter.manager.stepshifter_manager import StepshifterManager

warnings.filterwarnings("ignore")

try:
    model_path = PostprocessorPathManager(Path(__file__))
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
    args = None
    # validate_arguments(args)

    manager = UNFAOPostProcessorManager(
        model_path=model_path,
    )
    
    manager.run(args=args)
