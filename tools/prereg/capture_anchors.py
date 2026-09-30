"""Record what a delivery PRODUCED, before anything queries the API.

Pre-registration v3's P8 — *the values served ARE the values we produced* — asks for anchor
values written down "before the publish". That ordering is unachievable as written: pooling and
publishing are a single invocation, so the pooled numbers never exist un-published.

What the rule protects is narrower and IS achievable: anchors must not be chosen after seeing
what the API returns. So this reads the pooled frame on disk — the exact artefact that was
published — and is run automatically at completion, before any API call. Running it by hand
afterwards preserves the property only if nobody has looked; running it from the runner removes
that "only if".

WHAT IT RECORDS, and why both. The `tower_point` estimate, because that is the statistic faoapi
serves and therefore the thing P8 compares. AND the raw draws, because on 2026-09-30 the point
estimate was zero for every cell in the delivery — so comparing it would have passed while
proving nothing. The draws discriminate where the point estimate cannot: a cell with 3 non-zero
draws of 128 and a specific spike is a fingerprint. A probe that can only DETECT a disagreement
is worth less than one that can also DIAGNOSE it.

The cell SELECTION is seeded, so it is reproducible and provably not steered by the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20260930
N_RANDOM = 20
N_TOP = 5


def _load_pooled(target_dir: Path):
    import views_frames as vf

    y = np.load(target_dir / "y_pred.npy")
    ids = np.load(target_dir / "identifiers.npz")
    index = vf.SpatioTemporalIndex(time=ids["time"], unit=ids["unit"], level=vf.SpatialLevel.PGM)
    return vf.PredictionFrame(y_pred=y, index=index), y, ids


def anchors_for(target_dir: Path, rng: np.random.Generator) -> dict:
    from views_frames_summarize import tower_point

    frame, draws, ids = _load_pooled(target_dir)
    point = np.asarray(tower_point(frame).values).squeeze()

    picked = rng.choice(len(point), size=min(N_RANDOM, len(point)), replace=False)
    top = np.argsort(point)[-N_TOP:][::-1]
    cells = []
    for i in list(picked) + list(top):
        i = int(i)
        cells.append({
            "month_id": int(ids["time"][i]),
            "priogrid_id": int(ids["unit"][i]),
            # repr() rather than float(): full precision, no rounding at the boundary the
            # comparison will be made across.
            "tower_point": repr(float(point[i])),
            "draws": [repr(float(x)) for x in draws[i]],
        })
    return {
        "rows": int(draws.shape[0]),
        "samples": int(draws.shape[1]),
        "y_pred_sha256": hashlib.sha256((target_dir / "y_pred.npy").read_bytes()).hexdigest(),
        "point_nonzero": int((point > 0).sum()),
        "point_max": repr(float(point.max())),
        "cells": cells,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pooled_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("rehearsal", "production"), required=True)
    args = p.parse_args(argv)

    import views_frames as vf

    targets = sorted(d for d in args.pooled_dir.iterdir() if d.is_dir() and d.name.startswith("lr_"))
    if not targets:
        print(f"no lr_* targets under {args.pooled_dir}")
        return 2

    rng = np.random.default_rng(SEED)
    rec = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "pooled_run": args.pooled_dir.name,
        "purpose": "prereg v3 P8 — values PRODUCED, recorded before any API query",
        # Recorded because P8 claimed to remove the 'compared the wrong thing' failure mode BY
        # CONSTRUCTION, on the basis that both sides run the same version. They float
        # independently inside `views-frames>=1.10.2,<2`, and the SERVING version is not
        # observable from outside (faoapi's /version reports the app, not its dependencies).
        # So the version is written down instead of assumed.
        "views_frames_version": getattr(vf, "__version__", "unknown"),
        "estimator": "views_frames_summarize.tower_point",
        "selection": f"seeded rng({SEED}): {N_RANDOM} random cells + {N_TOP} highest by point estimate",
        "anchors": {t.name: anchors_for(t, rng) for t in targets},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    total = sum(len(v["cells"]) for v in rec["anchors"].values())
    print(f"wrote {args.out}: {len(rec['anchors'])} target(s), {total} anchor cells, "
          f"views_frames {rec['views_frames_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
