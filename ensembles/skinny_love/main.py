import sys
from pathlib import Path

from views_pipeline_core.cli import ForecastingModelArgs
from views_pipeline_core.managers.ensemble import EnsemblePathManager, EnsembleManager

# Composition root (ADR-014): bootstrap the repo root so the reconciliation wiring
# layer is importable (run.sh is immutable, so PYTHONPATH cannot be set there).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reconciliation import build_reconciler_for_run  # noqa: E402

try:
    ensemble_path = EnsemblePathManager(Path(__file__))
except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}. Check the logs for details.")

if __name__ == "__main__":
    args = ForecastingModelArgs.parse_args()

    # skinny_love reconciles (pgm_cm_point) with pink_ponyclub — inject the concrete
    # reconciler at the composition root (EPIC #172).
    reconciler = build_reconciler_for_run(Path(__file__).resolve().parent)

    manager = EnsembleManager(
        ensemble_path=ensemble_path,
        wandb_notifications=args.wandb_notifications,
        use_prediction_store=args.prediction_store,
        reconciler=reconciler,
    )

    manager.execute_single_run(args)
