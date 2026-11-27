import sys
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.ensemble import (
    EnsemblePathManager,
    PredictionFrameEnsembleManager,
)

# Composition root (ADR-014): bootstrap the repo root so the reconciliation wiring
# layer is importable (run.sh is immutable, so PYTHONPATH cannot be set there).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconciliation import build_reconciler_for_run  # noqa: E402

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
except FileNotFoundError as fnf_error:
    raise RuntimeError(f"File not found: {fnf_error}. Check the file path and try again.")
except PermissionError as perm_error:
    raise RuntimeError(f"Permission denied: {perm_error}. Check your permissions and try again.")
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")


if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    # chrome_box reconciles (pgm_cm) with chrome_crate — inject the concrete reconciler
    # at the composition root (EPIC #172). The geography source is DERIVED from the
    # ensemble's own data, not assumed: these members are datafactory consumers, so this
    # resolves to the GAUL `gaul0_code` provider (#601), never viewser's `country_id`.
    # The two id systems do not overlap (C-49), and the factory refuses an unknown source
    # rather than falling back.
    reconciler = build_reconciler_for_run(Path(__file__).resolve().parent)

    # ⛔ PredictionFrameEnsembleManager, NOT EnsembleManager. The DataFrame path refuses
    # `pgm_cm` outright — it implements point reconciliation only, and publishing there
    # would return unreconciled predictions. Draw-aware reconciliation lives on this path.
    manager = PredictionFrameEnsembleManager(
        ensemble_path=ensemble_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
        reconciler=reconciler,
    )

    manager.execute_single_run(args)
