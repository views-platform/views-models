from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager

# Import your model manager class here
from views_hydranet.manager.hydranet_manager import HydranetManager

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
    args = ForecastingModelArgs.parse_args()
    
    manager = HydranetManager(model_path=model_path)

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
