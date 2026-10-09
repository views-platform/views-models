import wandb
from pathlib import Path
from views_postprocessing.crafd.managers import CRAFDPostProcessorManager
from views_pipeline_core.managers.postprocessor.postprocessor import PostprocessorPathManager

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

    manager = CRAFDPostProcessorManager(
        model_path=model_path,
    )

    manager.run(args=args)
