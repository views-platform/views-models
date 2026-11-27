#!/usr/bin/env python3
"""Bring the GATE posterior home from a pod, at h=1 AND h=36, in tens of megabytes.

WHY THIS EXISTS. `pod_run_model.sh` archives `-path '*/lr_*'` — the emitted forecast — and
leaves `by_{sb,ns,os}_best`, the gate, on `/workspace` to die with the pod.

⛔ `lr_*` IS NOT THE UNGATED MAGNITUDE BODY. This docstring said "the magnitude body" until
2026-11-02, and it was wrong for every member of the roster. Composition is applied at emit
time (views-hydranet `utils/config_initializer.py:222`), so under `soft_gate` and
`threshold_gate` the gate has already been multiplied into the array that is written
(`utils/hydranet_inference.py:1093`). Only `self_zeroed` writes the family's own draw, and no
ROSTER member uses it — five are `soft_gate`, three are `threshold_gate`. So `lr_*` is the
*composed* forecast and `by_*` is the gate that produced it; they are not two halves of one
posterior, and anyone reasoning about this slice from the old sentence was reasoning about the
wrong pair of arrays. Caught by the views-hydranet seat 2026-11-01, after the wrong sentence
had been repeated to Simon as fact. It then reports
`STATUS: OK`, and its own `N_DRAWS`/`N_ARCHIVED` guard counts only the half it archives, so
nothing notices (views-models#558, register C-161). On 2026-10-13 that cost the composition
arm its pre-registered readout; the gate was recovered only because it was sliced by hand on
the pod before the pod was cleared, with a script that then died with it. This is that script,
committed, so the next arm does not depend on someone improvising at 2am.

WHY TWO ENDPOINTS AND NOT THE WHOLE THING. The full gate archive measured 1.3 GB (4.8x
compression — gate draws are high-entropy floats, unlike the lr draws' 194x). Over a link
measured at 43-48 KB/s that is 7.7 hours. Each endpoint slice is 1/36 of the rows; two of them
are 1/18, and the h=1 slice alone came home in minutes.

⚠️ **h=36 WAS ADDED 2026-11-02 and the file was h=1-only before it** (#686). EXP-02
pre-registers endpoints at BOTH the near and the far horizon, and the near one alone cannot
speak to the far one: a gate that is well calibrated at one month ahead and has collapsed by
three years ahead looks identical in an h=1 readout. Slicing only h=1 and calling it "the
readout" was correct for the composition arm, whose endpoints were all h=1, and is not
correct for EXP-02.

⛔ **h36 IS DERIVED AS `first + 35`, NEVER AS `time.max()`.** They coincide only when the
window really is 36 contiguous steps. On a short or gappy window `time.max()` silently returns
whatever the last month happens to be and labels it `h36` — so a readout pre-registered at the
36th step would be computed at the 30th and reported under the pre-registered name. That is a
wrong number on a pre-registered endpoint, which is worse than a missing one, because nothing
downstream can tell. So the window is CHECKED to be exactly `range(first, first + steps)` and
the slice REFUSES otherwise. Raised by the views-hydranet seat 2026-11-01.

HOW IT COMES HOME WITH NO CHANGE TO THE RUNNER. It writes into `$OUT`
(`/workspace/deliver/<model>`), which is the directory the operator already rsyncs. The
runner is production and is not modified by this; nothing here touches it.

USAGE, on the pod, after `pod_run_model.sh` reports OK:

    python3 /workspace/views-models/tools/podrun/slice_gate_posterior.py <model>

`tools/podrun/pod_run_exp02_arm.sh` calls it for you and refuses to report OK without it —
prefer that for an EXP-02 arm, so that "the gate came home" is a property of the run rather
than something an operator has to remember at teardown (#686, C-161).

Then rsync `$OUT` as usual. Verify locally against `$OUT/gate_endpoints.sha256`.

IT REFUSES rather than writing a partial slice: no `by_*` directories at all, a target
missing from an origin, a forecast window that is not exactly `steps` contiguous months, a row
count that is not the land-cell count, a draw count that disagrees between targets or between
endpoints, any non-finite value, or an archive whose entry count does not match what was
written. A silent partial slice is the failure this file exists to prevent, so it may not be
one of its own outcomes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Global land, PRIO-GRID 0.5 degree. The arms are all REGION="land" and the runner refuses
#: anything else, so a different count means the slice is not of the run we think it is.
EXPECTED_ROWS = 64818

#: The forecast window every arm declares: 36 steps, h=1 .. h=36. The far endpoint is derived
#: from the near one and this number, never read off the data — see the docstring.
EXPECTED_STEPS = 36

#: zstd level and thread count. NOT `-T0`: on the 128-core pod that is one thread per core,
#: and it died in 2.3 s leaving a truncated 1 MB archive from 182 MB of input — the same
#: shape of failure as the pod's own OOM later that day (#558 addendum). `-T4` completed
#: cleanly. Bounded threads, deliberately.
ZSTD_LEVEL = "-12"
ZSTD_THREADS = "-T4"

GATE_PREFIX = "by_"

#: The slice tree and its archive. Named for what they hold: `gate_h1` was the name until
#: 2026-11-02 and would now be a lie about the contents, which is precisely the defect class
#: the docstring above is a monument to. The guide and
#: tests/test_falsification_pod_teardown_completeness.py were updated with it.
TREE_NAME = "gate_endpoints"


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


def endpoint_months(time: np.ndarray, steps: int, *, where: str) -> dict[str, int]:
    """The month id of each pre-registered endpoint, or a refusal.

    ⛔ The far endpoint is `first + steps - 1`. It is NOT `time.max()`, and the difference is
    invisible in the output: on a 30-month window `time.max()` names month 30 and the manifest
    would call it `h36`. So the window is verified to be exactly the contiguous run
    `first .. first + steps - 1` with no gaps, no duplicates beyond the per-month cell block,
    and no extra months, and anything else refuses here rather than being labelled.
    """
    months = np.unique(time)
    first = int(months[0])
    expected = np.arange(first, first + steps, dtype=months.dtype)
    if months.shape != expected.shape or not np.array_equal(months, expected):
        raise Refuse(
            f"{where}: the forecast window is not {steps} contiguous months. Found "
            f"{months.shape[0]} distinct months spanning {first}..{int(months[-1])}; expected "
            f"exactly {first}..{first + steps - 1}. The far endpoint is derived as "
            f"first+{steps - 1}, so on this window it cannot be derived at all — and "
            f"time.max() would have returned {int(months[-1])} and labelled it h{steps}, "
            f"which is a wrong number under a pre-registered name. Refusing."
        )
    return {"h1": first, f"h{steps}": first + steps - 1}


def slice_one(target_dir: Path, out_dir: Path, expected_rows: int, steps: int) -> dict:
    """Write each endpoint's rows for one origin/target, and report what was written."""
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

    endpoints = endpoint_months(time, steps, where=str(target_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    per_endpoint: dict[str, dict] = {}
    units_at: dict[str, np.ndarray] = {}
    for label, month in endpoints.items():
        mask = time == month
        rows = int(mask.sum())
        if rows != expected_rows:
            raise Refuse(
                f"{target_dir}: month {month} ({label}) has {rows} rows, expected "
                f"{expected_rows} land cells. Either this is not a global-land run or the "
                f"identifiers are not what this script assumes — refusing rather than "
                f"slicing an array of unknown shape."
            )

        y_slice = np.ascontiguousarray(y[mask])
        n_nonfinite = int((~np.isfinite(y_slice)).sum())
        if n_nonfinite:
            raise Refuse(
                f"{target_dir}: {n_nonfinite} non-finite values in the {label} slice. A gate "
                f"probability cannot be NaN or inf; this is a defect in the run, not in the "
                f"slice."
            )

        np.save(out_dir / f"y_pred_{label}.npy", y_slice)
        np.savez(out_dir / f"identifiers_{label}.npz", time=time[mask], unit=unit[mask])
        units_at[label] = np.sort(unit[mask])

        per_endpoint[label] = {
            "month": month,
            "rows": rows,
            "shape": list(y_slice.shape),
            "draws": int(y_slice.shape[1]),
            "min": float(y_slice.min()),
            "max": float(y_slice.max()),
            "n_nonfinite": n_nonfinite,
        }

    # ⛔ THE ENDPOINTS MUST CARRY THE SAME SUPPORT, not merely the same NUMBER of rows.
    # The per-endpoint check above compares a COUNT against `expected_rows`, and two endpoints
    # can both hold 64,818 rows over different sets of grid cells. Downstream,
    # `score_v2_horizons._support_keys` derives support from the (month, unit) keys, so a unit
    # set that differs between endpoints shrinks the support silently — the readout still
    # computes, on fewer cells than it reports. Raised by the views-hydranet seat 2026-11-02;
    # the count check genuinely did not cover it.
    #
    # This is the aggregate-assertion trap from the other direction: a count is one number per
    # endpoint and cannot say WHICH cells it counted.
    if len(units_at) > 1:
        labels = sorted(units_at)
        reference = units_at[labels[0]]
        for label in labels[1:]:
            if not np.array_equal(reference, units_at[label]):
                only_ref = np.setdiff1d(reference, units_at[label])
                only_other = np.setdiff1d(units_at[label], reference)
                raise Refuse(
                    f"{target_dir}: {labels[0]} and {label} hold the same number of rows but "
                    f"NOT the same grid cells — {len(only_ref)} cell(s) only in {labels[0]} "
                    f"(e.g. {only_ref[:5].tolist()}), {len(only_other)} only in {label} "
                    f"(e.g. {only_other[:5].tolist()}). Support is derived from the "
                    f"(month, unit) keys downstream, so this would shrink it without saying "
                    f"so. A row COUNT cannot see this, which is why it is checked separately."
                )

    # One array sliced two ways cannot legitimately disagree on its draw count; if it does,
    # the mask is selecting across something other than the time axis and both slices are
    # suspect. Checked here rather than only across targets, because the cross-target check
    # below compares one number per target and so cannot see a within-target disagreement.
    draws = {stats["draws"] for stats in per_endpoint.values()}
    if len(draws) != 1:
        raise Refuse(
            f"{target_dir}: the endpoints disagree on the draw count ({sorted(draws)}) "
            f"although they are slices of one array — the mask is not selecting on time."
        )

    return {"endpoints": per_endpoint, "draws": draws.pop(), "files": 2 * len(per_endpoint)}


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
    lines = listing.stdout.splitlines()
    # ⚠️ Directories are excluded by their trailing slash, which is a CONVENTION of the tar
    # build doing the listing, not a guarantee. If this tar did not mark directories that way
    # the filter would be a no-op and the count below would compare files-plus-directories
    # against files — a mismatch blamed on a partial archive. The tree always contains at
    # least its own root directory, so at least one line MUST end in "/"; if none does, the
    # instrument is not measuring what it reports (ADR-025). Raised by the views-hydranet
    # seat 2026-11-01, which read the filter as counting directories.
    if not any(line.endswith("/") for line in lines):
        raise Refuse(
            f"the archive listing has no directory entry among its {len(lines)} lines, so "
            f"this tar does not mark directories with a trailing slash and the file count "
            f"below cannot distinguish files from directories. Refusing rather than "
            f"comparing two different quantities."
        )
    members = [line for line in lines if not line.endswith("/")]
    if len(members) != expected_files:
        raise Refuse(
            f"the archive holds {len(members)} files but {expected_files} were written. "
            f"Refusing to leave a partial archive that reports success."
        )


def _slice_all(
    source: Path,
    tree: Path,
    expected_rows: int,
    steps: int,
    manifest: dict[str, dict],
    draw_counts: set[int],
) -> None:
    """Slice every origin/target into `tree`, filling `manifest` and `draw_counts` as it goes.

    Separated from `main` so the caller can remove a half-written tree on any refusal; the
    mutable arguments are deliberate, because what was written before the refusal is exactly
    what the caller needs to clean up and report.
    """
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
            stats = slice_one(target, tree / origin.name / target.name, expected_rows, steps)
            manifest[origin.name][target.name] = stats["endpoints"]
            draw_counts.add(stats["draws"])
            shown = " ".join(
                f"{label}={per['month']}" for label, per in stats["endpoints"].items()
            )
            print(
                f"  {origin.name}/{target.name}: {shown} "
                f"rows={next(iter(stats['endpoints'].values()))['rows']} "
                f"draws={stats['draws']}"
            )


def _parse(argv: list[str]) -> tuple[str, Path, Path, int, int]:
    """model, repo, out, expected_rows, steps — the last four overridable, explicitly.

    Overrides exist so this file can be tested against a small fixture tree instead of a
    149 MB array, and they are CLI arguments rather than environment variables so that what
    a run used is visible in the command that ran it. The defaults are the pod's real paths,
    the real land-cell count and the real 36-step window, and
    `tests/test_slice_gate_posterior.py` pins them.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="slice_gate_posterior.py", add_help=True)
    parser.add_argument("model")
    parser.add_argument("--repo", type=Path, default=None, help="views-models checkout root")
    parser.add_argument("--out", type=Path, default=None, help="the rsynced deliver directory")
    parser.add_argument(
        "--expected-rows", type=int, default=EXPECTED_ROWS, help="land cells per month"
    )
    parser.add_argument(
        "--steps", type=int, default=EXPECTED_STEPS,
        help="months in the forecast window; the far endpoint is first+steps-1",
    )
    args = parser.parse_args(argv[1:])
    repo = args.repo or Path(__file__).resolve().parent.parent.parent
    out = args.out or Path("/workspace/deliver") / args.model
    return args.model, repo, out, args.expected_rows, args.steps


def main(argv: list[str]) -> int:
    model, repo, out, expected_rows, steps = _parse(argv)
    if not out.is_dir():
        raise Refuse(
            f"{out} does not exist. This runs ON THE POD, after pod_run_model.sh, and writes "
            f"into the directory the operator rsyncs. Off-pod there is nothing to slice."
        )

    source = _latest_predictions_dir(repo, model)
    print(f"source:  {source}")
    print(f"out:     {out}")
    print(f"window:  {steps} months -> endpoints h1 and h{steps}")

    tree = out / TREE_NAME
    if tree.exists():
        raise Refuse(
            f"{tree} already exists. Remove it deliberately rather than having this script "
            f"merge a new slice into an old one."
        )

    manifest: dict[str, dict] = {}
    files_written = 0
    draw_counts: set[int] = set()

    # ⛔ A REFUSAL MUST NOT LEAVE A PARTIAL TREE BEHIND. Found by
    # `test_the_far_endpoint_is_derived_from_the_window_not_read_off_the_data` 2026-11-02:
    # when the third target of the second origin refuses, the first origin has already been
    # written, so `$OUT/gate_endpoints/` holds a real-looking subset. Nothing downstream would
    # be fooled — there is no archive and no `.sha256` — but the rsync carries it home, and
    # the docstring above promises a partial slice is not one of this script's outcomes. So
    # it is removed on the way out, and the promise is kept by the code rather than by the
    # sentence. The archive and digest are written after this block, so there is nothing else
    # to undo.
    try:
        _slice_all(source, tree, expected_rows, steps, manifest, draw_counts)
    except BaseException:
        shutil.rmtree(tree, ignore_errors=True)
        raise
    files_written = sum(
        2 * len(per_target)
        for per_origin in manifest.values()
        for per_target in per_origin.values()
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

    archive = out / f"{TREE_NAME}.tar.zst"
    _archive(tree, archive, files_written)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out / f"{TREE_NAME}.sha256").write_text(f"{digest}  {TREE_NAME}.tar.zst\n")

    print()
    print(f"origins:   {len(manifest)}")
    print(f"targets:   {sorted(target_names)}")
    print(f"endpoints: h1, h{steps}")
    print(f"files:     {files_written}")
    print(f"archive:   {archive}  ({archive.stat().st_size} bytes)")
    print(f"sha256:    {digest}")
    print()
    print(f"Both {TREE_NAME}/ and {TREE_NAME}.tar.zst are inside $OUT, so the usual rsync")
    print(f"brings them home. Verify locally with:  sha256sum -c {TREE_NAME}.sha256")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via tests/ against a fixture tree
    try:
        sys.exit(main(sys.argv))
    except Refuse as refusal:
        print(refusal.args[0], file=sys.stderr)
        sys.exit(1)
