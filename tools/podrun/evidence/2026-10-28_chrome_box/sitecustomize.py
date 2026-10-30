"""PROBE — not a fix. Patches BOTH known defects so chrome_box can be run past them,
purely to learn whether a THIRD defect lies behind them. Edits no installed file.

  1. views-pipeline-core#562 - load_cm_frame looks for .parquet, PF writes a directory.
  2. views-models#647        - the GAUL mapping stops at the store horizon (564).
"""
from pathlib import Path

import numpy as np


def _patch_loader():
    from views_pipeline_core.managers.ensemble import prediction_frame_ensemble as pfe
    from views_pipeline_core.managers.ensemble.ensemble import EnsemblePathManager
    from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf

    original = pfe.load_cm_frame

    def load_cm_frame(cm_model, target, run_type):
        try:
            dirs = EnsemblePathManager(cm_model)._get_generated_pf_prediction_paths(
                run_type=run_type
            )
        except OSError:
            dirs = []
        for d in dirs:
            leaf = Path(d) / target
            if (leaf / "y_pred.npy").is_file():
                frame = load_pf(leaf, level="cm")
                print(
                    f"[probe] cm frame {cm_model}/{target} <- {leaf.name} "
                    f"n_rows={frame.n_rows} samples={frame.sample_count}",
                    flush=True,
                )
                return frame
        print(f"[probe] no PF dir for {cm_model}/{target}; shipped loader", flush=True)
        return original(cm_model, target, run_type)

    pfe.load_cm_frame = load_cm_frame
    print("[probe] #562 load_cm_frame patched (PF-aware)", flush=True)


def _patch_mapping():
    # sitecustomize runs before main.py inserts the repo root, so do it here.
    import sys
    if "/workspace/views-models" not in sys.path:
        sys.path.insert(0, "/workspace/views-models")
    from reconciliation.country_mapping import CountryMapping
    from reconciliation import datafactory_country_mapping_provider as dp

    original = dp.DatafactoryCountryMappingProvider.build

    def build(self):
        m = original(self)  # keeps BOTH refusals intact
        gids, vals = m.map_keys[:, 1], m.map_vals
        uniq, first = np.unique(gids, return_index=True)
        codes = vals[first]
        months = np.arange(self._start_month, self._end_month + 1, dtype=np.int64)
        keys = np.empty((months.size * uniq.size, 2), dtype=np.int64)
        keys[:, 0] = np.repeat(months, uniq.size)
        keys[:, 1] = np.tile(uniq, months.size)
        print(
            f"[probe] #647 mapping broadcast: {len(m)} keys "
            f"({np.unique(m.map_keys[:, 0]).size} months) -> {keys.shape[0]} keys "
            f"({months.size} months x {uniq.size} cells, {months[0]}..{months[-1]})",
            flush=True,
        )
        return CountryMapping(keys, np.tile(codes, months.size))

    dp.DatafactoryCountryMappingProvider.build = build
    print("[probe] #647 mapping provider patched (broadcast)", flush=True)


for name, fn in (("loader", _patch_loader), ("mapping", _patch_mapping)):
    try:
        fn()
    except Exception as exc:
        print(f"[probe] {name} PATCH FAILED: {type(exc).__name__}: {exc}", flush=True)
