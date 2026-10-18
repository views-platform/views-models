#!/usr/bin/env python3
"""Bring the GATE posterior home from a pod, at h=1, in tens of megabytes.

WHY THIS EXISTS. `pod_run_model.sh` archives `-path '*/lr_*'` — the magnitude body — and
leaves `by_{sb,ns,os}_best`, the gate, on `/workspace` to die with the pod. It then reports
`STATUS: OK`, and its own `N_DRAWS`/`N_ARCHIVED` guard counts only the half it archives, so
nothing notices (views-models#558, register C-161). On 2026-10-13 that cost the composition
arm its pre-registered readout; the gate was recovered only because it was sliced by hand on
the pod before the pod was cleared, with a script that then died with it. This is that script,
committed, so the next arm does not depend on someone improvising at 2am.

WHY h=1 AND NOT THE WHOLE THING. The full gate archive measured 1.3 GB (4.8x compression —
gate draws are high-entropy floats, unlike the lr draws' 194x). Over a link measured at
43-48 KB/s that is 7.7 hours. The h=1 slice is 1/36 of the rows and came home in minutes.
Every endpoint the arms pre-register — median |gate - p|, median implied w, gate AP — is
evaluated at h=1 on the 13 out-of-sample months, so the slice is the readout, not a
compromise.

HOW IT COMES HOME WITH NO CHANGE TO THE RUNNER. It writes into `$OUT`
(`/workspace/deliver/<model>`), which is the directory the operator already rsyncs. The
runner is production and is not modified by this; nothing here touches it.

USAGE, on the pod, after `pod_run_model.sh` reports OK:

    python3 /workspace/views-models/tools/podrun/slice_gate_posterior.py <model>

Then rsync `$OUT` as usual. Verify locally against `$OUT/gate_h1.sha256`.

IT REFUSES rather than writing a partial slice: no `by_*` directories at all, a target
missing from an origin, a row count that is not the land-cell count, a draw count that
disagrees between targets, any non-finite value, or an archive whose entry count does not
match what was written. A silent partial slice is the failure this file exists to prevent, so
it may not be one of its own outcomes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Global land, PRIO-GRID 0.5 degree. The arms are all REGION="land" and the runner refuses
#: anything else, so a different count means the slice is not of the run we think it is.
EXPECTED_ROWS = 64818

#: zstd level and thread count. NOT `-T0`: on the 128-core pod that is one thread per core,
#: and it died in 2.3 s leaving a truncated 1 MB archive from 182 MB of input — the same
#: shape of failure as the pod's own OOM later that day (#558 addendum). `-T4` completed
#: cleanly. Bounded threads, deliberately.
ZSTD_LEVEL = "-12"
ZSTD_THREADS = "-T4"

GATE_PREFIX = "by_"


class Refuse(SystemExit):
    """Exit non-zero with a message that names what was examined."""

    def __init__(self, message: str) -> None:
        super().__init__(f"!!! slice_gate_posterior: {message}")


def _latest_predictions_dir(repo: Path, model: str) -> Path:
    generated = repo / "models" / model / "data" / "generated"
    if not generated.is_dir():
        raise Refuse(f"no such directory: {generated}")
    candidates = sorted(generated.glob("predictions_calibration_*"))
    if not candidates:
        raise Refuse(
            f"no predictions_calibration_* directory under {generated}. Run the model first; "
            f"this script slices an existing posterior and never produces one."
        )
    return candidates[-1]


def _origin_dirs(source: Path) -> list[Path]:
    origins = sorted(
        (d for d in source.iterdir() if d.is_dir() and d.name.startswith("origin_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not origins:
        raise Refuse(f"no origin_* directories under {source}")
    return origins


def _gate_targets(origin: Path) -> list[Path]:
    return sorted(d for d in origin.iterdir() if d.is_dir() and d.name.startswith(GATE_PREFIX))


def slice_one(target_dir: Path, out_dir: Path, expected_rows: int) -> dict:
    """Write the h=1 rows of one origin/target, and report what was written."""
    y_path, id_path = target_dir / "y_pred.npy", target_dir / "identifiers.npz"
    for path in (y_path, id_path):
        if not path.is_file():
            raise Refuse(f"{path} is missing, so this target cannot be sliced")

    y = np.load(y_path)
    ids = np.load(id_path)
    if "time" not in ids or "unit" not in ids:
        raise Refuse(f"{id_path} has keys {sorted(ids.files)}; expected 'time' and 'unit'")
    time, unit = ids["time"], ids["unit"]

    if y.shape[0] != time.shape[0]:
        raise Refuse(
            f"{target_dir}: y_pred has {y.shape[0]} rows but identifiers have "
            f"{time.shape[0]} — they do not describe the same array"
        )

    h1 = int(time.min())
    mask = time == h1
    rows = int(mask.sum())
    if rows != expected_rows:
        raise Refuse(
            f"{target_dir}: the first forecast month ({h1}) has {rows} rows, expected "
            f"{expected_rows} land cells. Either this is not a global-land run or the "
            f"identifiers are not what this script assumes — refusing rather than slicing "
            f"an array of unknown shape."
        )

    y_h1 = np.ascontiguousarray(y[mask])
    n_nonfinite = int((~np.isfinite(y_h1)).sum())
    if n_nonfinite:
        raise Refuse(
            f"{target_dir}: {n_nonfinite} non-finite values in the h=1 slice. A gate "
            f"probability cannot be NaN or inf; this is a defect in the run, not in the slice."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_pred_h1.npy", y_h1)
    np.savez(out_dir / "identifiers_h1.npz", time=time[mask], unit=unit[mask])

    return {
        "h1_month": h1,
        "rows": rows,
        "shape": list(y_h1.shape),
        "draws": int(y_h1.shape[1]),
        "min": float(y_h1.min()),
        "max": float(y_h1.max()),
        "n_nonfinite": n_nonfinite,
    }


def _archive(tree: Path, archive: Path, expected_files: int) -> None:
    """tar + zstd the slice, then verify the archive holds what was written."""
    cmd = [
        "tar",
        "-I",
        f"zstd {ZSTD_LEVEL} {ZSTD_THREADS}",
        "-cf",
        str(archive),
        "-C",
        str(tree.parent),
        tree.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Refuse(f"archiving failed: {result.stderr.strip() or result.returncode}")

    # Count what is INSIDE the archive, not what we meant to put there. The runner's own
    # guard counted the set it selected and so could not see the half it dropped (C-161);
    # this counts the archive's members against the files on disk.
    listing = subprocess.run(
        ["tar", "-I", "zstd", "-tf", str(archive)], capture_output=True, text=True
    )
    if listing.returncode != 0:
        raise Refuse(f"the archive cannot be listed back: {listing.stderr.strip()}")
    members = [line for line in listing.stdout.splitlines() if not line.endswith("/")]
    if len(members) != expected_files:
        raise Refuse(
            f"the archive holds {len(members)} files but {expected_files} were written. "
            f"Refusing to leave a partial archive that reports success."
        )


def _parse(argv: list[str]) -> tuple[str, Path, Path, int]:
    """model, repo, out, expected_rows — the last three overridable, explicitly.

    Overrides exist so this file can be tested against a small fixture tree instead of a
    149 MB array, and they are CLI arguments rather than environment variables so that what
    a run used is visible in the command that ran it. The defaults are the pod's real paths
    and the real land-cell count, and `tests/test_slice_gate_posterior.py` pins them.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="slice_gate_posterior.py", add_help=True)
    parser.add_argument("model")
    parser.add_argument("--repo", type=Path, default=None, help="views-models checkout root")
    parser.add_argument("--out", type=Path, default=None, help="the rsynced deliver directory")
    parser.add_argument(
        "--expected-rows", type=int, default=EXPECTED_ROWS, help="land cells per month"
    )
    args = parser.parse_args(argv[1:])
    repo = args.repo or Path(__file__).resolve().parent.parent.parent
    out = args.out or Path("/workspace/deliver") / args.model
    return args.model, repo, out, args.expected_rows


def main(argv: list[str]) -> int:
    model, repo, out, expected_rows = _parse(argv)
    if not out.is_dir():
        raise Refuse(
            f"{out} does not exist. This runs ON THE POD, after pod_run_model.sh, and writes "
            f"into the directory the operator rsyncs. Off-pod there is nothing to slice."
        )

    source = _latest_predictions_dir(repo, model)
    print(f"source:  {source}")
    print(f"out:     {out}")

    tree = out / "gate_h1"
    if tree.exists():
        raise Refuse(
            f"{tree} already exists. Remove it deliberately rather than having this script "
            f"merge a new slice into an old one."
        )

    manifest: dict[str, dict] = {}
    files_written = 0
    draw_counts: set[int] = set()

    for origin in _origin_dirs(source):
        targets = _gate_targets(origin)
        if not targets:
            raise Refuse(
                f"{origin} has no {GATE_PREFIX}* directories. The gate posterior is the "
                f"whole point of this script — if HydraNet did not write one, there is "
                f"nothing here to bring home and the arm has no readout."
            )
        manifest[origin.name] = {}
        for target in targets:
            stats = slice_one(target, tree / origin.name / target.name, expected_rows)
            manifest[origin.name][target.name] = stats
            draw_counts.add(stats["draws"])
            files_written += 2
            print(
                f"  {origin.name}/{target.name}: h1={stats['h1_month']} "
                f"rows={stats['rows']} draws={stats['draws']} "
                f"range=[{stats['min']:.6g}, {stats['max']:.6g}]"
            )

    if len(draw_counts) != 1:
        raise Refuse(
            f"targets disagree on the draw count: {sorted(draw_counts)}. One of them is not "
            f"from this run."
        )

    target_names = {name for per_origin in manifest.values() for name in per_origin}
    for origin_name, per_origin in manifest.items():
        missing = target_names - set(per_origin)
        if missing:
            raise Refuse(f"{origin_name} is missing {sorted(missing)}, which other origins have")

    (tree / "SLICE_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    files_written += 1

    archive = out / "gate_h1.tar.zst"
    _archive(tree, archive, files_written)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / "gate_h1.sha256").write_text(f"{digest}  gate_h1.tar.zst\n")

    print()
    print(f"origins:   {len(manifest)}")
    print(f"targets:   {sorted(target_names)}")
    print(f"files:     {files_written}")
    print(f"archive:   {archive}  ({archive.stat().st_size} bytes)")
    print(f"sha256:    {digest}")
    print()
    print("Both gate_h1/ and gate_h1.tar.zst are inside $OUT, so the usual rsync brings them")
    print("home. Verify locally with:  sha256sum -c gate_h1.sha256")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via tests/ against a fixture tree
    try:
        sys.exit(main(sys.argv))
    except Refuse as refusal:
        print(refusal.args[0], file=sys.stderr)
        sys.exit(1)
