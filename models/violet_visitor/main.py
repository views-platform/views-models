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

    # ----------------------------------------------------------------------------------
    # GUARD (2026-06-19): the --report/-re stage OOM-kills this box (~18 GB host RAM) —
    # root-caused to the views-pipeline-core report/publish tail, NOT this model
    # (eval-only peaks ~2.4 GB). Tracked: views-pipeline-core#181 / views-hydranet C-116
    # (#124). Block --report until that's fixed so an eval run can't be accidentally
    # OOM-killed again. Re-run WITHOUT -re (eval persists predictions + metrics — all
    # mcr_readout / the #110 decision rule need). Remove this guard when #181 lands.
    # Deliberate override (only if you know the fix is in): ALLOW_RE_REPORT=1.
    # ----------------------------------------------------------------------------------
    import os
    import sys

    if getattr(args, "report", False) and not os.environ.get("ALLOW_RE_REPORT"):
        sys.exit(
            "BLOCKED: --report/-re triggers the pipeline-core report stage that OOMs this "
            "box (~18 GB; see views-pipeline-core#181 / C-116). Re-run WITHOUT -re. "
            "Override with ALLOW_RE_REPORT=1 only if the upstream fix has landed."
        )

    manager = HydranetManager(model_path=model_path)

    if args.sweep:
        manager.execute_sweep_run(args)
    else:
        manager.execute_single_run(args)
