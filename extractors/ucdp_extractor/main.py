import wandb
import warnings
from pathlib import Path
from views_pipeline_core.cli.utils import parse_args, validate_arguments
from views_graphdb.manager import UCDPExtractorManager
from views_graphdb.manager.extractor import ExtractorPathManager

warnings.filterwarnings("ignore")

try:
    extractor_path = ExtractorPathManager(Path(__file__))
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

    UCDPExtractorManager(model_path=extractor_path).run(args)