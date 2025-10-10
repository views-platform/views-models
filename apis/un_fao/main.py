import wandb
import warnings
from pathlib import Path
from views_faoapi.managers.api import APIPathManager
from views_faoapi.managers.api import FAOApiManager

warnings.filterwarnings("ignore")

try:
    model_path = APIPathManager(Path(__file__))
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
    # args = parse_args()

    manager = FAOApiManager(
        model_path=model_path,
        wandb_notifications=False,
    )
    
    manager.run()