import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # The operator 'aten::_standard_gamma' is not currently implemented for the MPS device.

import warnings
from pathlib import Path
from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers import ModelPathManager
from views_impact.manager.model import ImpactModelManager

warnings.filterwarnings("ignore")

try:
    model_path = ModelPathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()
    if args.sweep:
        ImpactModelManager(model_path=model_path, wandb_notifications=False).execute_sweep_run(args)
    else:
        ImpactModelManager(model_path=model_path, wandb_notifications=False).execute_single_run(args)
