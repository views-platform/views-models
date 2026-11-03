"""Report the health of a pooled posterior, so a degenerate one cannot be missed.

WHY THIS EXISTS. The 2026-09-30 rehearsal delivered successfully and served nothing useful: the
pooled posterior had all 128 draws at zero in 98.2% of cells, and `tower_point` — the estimator
faoapi serves — correctly returned 0 for every one of 2,333,448 rows. Every structural check
passed. The manifest was valid, the coverage was right, the findability guard was green. Nothing
in the delivery said the numbers were empty, and it was found by hand at 4am only because
someone happened to be computing anchor values for an unrelated probe.

For a REHEARSAL that outcome is expected — 40 lessons is an undertrained model. For a
PRODUCTION run it means the delivery is worthless, and the partner would be the one to notice.
The numbers are identical; only the mode differs, and the runner knows the mode.

WHAT IT CANNOT DO. It cannot prevent anything. Pooling and publishing are a single invocation,
so by the time a posterior can be inspected it has already been published. This reports; the
refusal, if one is ever wanted, has to live where the publish decision is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_pooled(target_dir: Path):
    """Load a pooled PredictionFrame written as y_pred.npy + identifiers.npz.

    Deliberately NOT `PredictionFrame.load`: that expects `values.npy`, and the ensemble writes
    `y_pred.npy`. The mismatch cost twenty minutes on 2026-09-30 and is worth stating here rather
    than rediscovering.
    """
    import views_frames as vf

    y = np.load(target_dir / "y_pred.npy")
    ids = np.load(target_dir / "identifiers.npz")
    index = vf.SpatioTemporalIndex(
        time=ids["time"], unit=ids["unit"], level=vf.SpatialLevel.PGM
    )
    return vf.PredictionFrame(y_pred=y, index=index), y, ids


def health(target_dir: Path) -> dict:
    from views_frames_summarize import tower_point

    frame, draws, _ = _load_pooled(target_dir)
    point = np.asarray(tower_point(frame).values).squeeze()
    row_max = draws.max(axis=1)
    return {
        "target": target_dir.name,
        "rows": int(draws.shape[0]),
        "samples": int(draws.shape[1]),
        "draws_nonzero_fraction": float((draws > 0).mean()),
        "rows_all_zero_fraction": float((row_max == 0).mean()),
        "rows_with_any_signal": int((row_max > 0).sum()),
        "max_draw": float(draws.max()),
        "point_nonzero": int((point > 0).sum()),
        "point_max": float(point.max()),
    }


def verdict(stats: dict) -> tuple[str, str]:
    """DEGENERATE / SPARSE / OK, with the reason in the operator's words.

    The DEGENERATE threshold is deliberately the starkest fact available — the served point
    estimate is zero EVERYWHERE — rather than a tuned fraction. A threshold that needs
    calibration is a threshold that gets argued with; this one cannot be.
    """
    if stats["point_nonzero"] == 0:
        return "DEGENERATE", (
            "the served point estimate is zero for EVERY one of "
            f"{stats['rows']:,} cells. Whatever is delivered, a consumer reads nothing from it."
        )
    if stats["rows_all_zero_fraction"] > 0.999:
        return "SPARSE", (
            f"{stats['rows_all_zero_fraction']:.3%} of cells have no signal in any draw; "
            f"only {stats['rows_with_any_signal']:,} cells carry anything."
        )
    return "OK", (
        f"{stats['point_nonzero']:,} cells carry a non-zero point estimate "
        f"(max {stats['point_max']:.6g})."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pooled_dir", type=Path, help="predictions_forecasting_<ts> of the ensemble")
    p.add_argument("--mode", choices=("rehearsal", "production"), required=True)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    targets = sorted(d for d in args.pooled_dir.iterdir() if d.is_dir() and d.name.startswith("lr_"))
    if not targets:
        print(f"no lr_* target directories under {args.pooled_dir}", file=sys.stderr)
        return 2

    report, worst = [], "OK"
    order = {"OK": 0, "SPARSE": 1, "DEGENERATE": 2}
    for t in targets:
        stats = health(t)
        stats["verdict"], stats["reason"] = verdict(stats)
        report.append(stats)
        if order[stats["verdict"]] > order[worst]:
            worst = stats["verdict"]

    print("=== posterior health ===")
    for s in report:
        print(
            f"  {s['target']:<14} {s['verdict']:<11} "
            f"rows={s['rows']:,} samples={s['samples']} "
            f"all-zero={s['rows_all_zero_fraction']:.2%} "
            f"point_nonzero={s['point_nonzero']:,} max_draw={s['max_draw']:.6g}"
        )
        print(f"    {s['reason']}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"mode": args.mode, "verdict": worst, "targets": report}, indent=1))

    if worst == "DEGENERATE":
        banner = "#" * 78
        print(banner)
        if args.mode == "rehearsal":
            print("### POSTERIOR IS DEGENERATE — EXPECTED for a rehearsal at low lesson count.")
            print("### The chain is proven; the NUMBERS are not. Do not read these as forecasts.")
        else:
            print("### POSTERIOR IS DEGENERATE ON A PRODUCTION RUN.")
            print("### The delivery is structurally valid and serves nothing. It is already")
            print("### published — this cannot unpublish it. Supersede it and investigate the")
            print("### MODELS: a valid manifest over an empty posterior passes every other check.")
        print(banner)
        # Non-zero ONLY for production: a rehearsal is expected to look like this, and failing
        # the run it was designed to produce would train the operator to ignore the exit code.
        return 1 if args.mode == "production" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
