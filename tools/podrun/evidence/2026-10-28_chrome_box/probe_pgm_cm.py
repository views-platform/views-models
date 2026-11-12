"""PROBE — not a fix. Is the pgm_cm reconciliation broken only by the loader's format
assumption, or is there more behind it?

views-pipeline-core's `load_cm_frame` calls `_get_generated_predictions_data_file_paths`,
which requires a `.parquet` FILE. A PredictionFrame ensemble writes a DIRECTORY
(`predictions_forecasting_*/<target>/{y_pred.npy,identifiers.npz}`). Measured on this pod:

    file-based (what load_cm_frame calls):  EMPTY
    dir-based  (_get_generated_pf_prediction_paths):  ['predictions_forecasting_20261028_120904']

So the cm partner's forecast is present and the loader cannot see it. This monkeypatches a
PF-aware loader in front of the DataFrame one and runs chrome_box, purely to learn whether
anything ELSE fails once the format assumption is removed. It edits no installed file.
"""
import runpy
import sys
from pathlib import Path

from views_pipeline_core.managers.ensemble import prediction_frame_ensemble as pfe
from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager
from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf

_original = pfe.load_cm_frame


def load_cm_frame(cm_model: str, target: str, run_type: str):
    """PF directories first; fall back to the shipped DataFrame loader."""
    try:
        dirs = EnsemblePathManager(cm_model)._get_generated_pf_prediction_paths(run_type=run_type)
    except OSError:
        dirs = []
    for d in dirs:
        leaf = Path(d) / target
        if (leaf / "y_pred.npy").is_file():
            frame = load_pf(leaf, level="cm")
            print(f"[probe] loaded cm frame for {cm_model}/{target} from {leaf} "
                  f"shape={frame.y_pred.shape}", flush=True)
            return frame
    print(f"[probe] no PF dir for {cm_model}/{target}; falling back to the shipped loader",
          flush=True)
    return _original(cm_model, target, run_type)


pfe.load_cm_frame = load_cm_frame
print("[probe] load_cm_frame patched (PF-aware)", flush=True)

sys.argv = ["main.py", "-r", "forecasting", "-f", "-sa"]
runpy.run_path("main.py", run_name="__main__")
