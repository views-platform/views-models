#!/usr/bin/env python3
"""Publish an ALREADY-POOLED, already-reconciled forecast to the production shelf.

WHY THIS EXISTS. `un_crafd` does not read a forecast from local disk — it reads the Appwrite
`production_forecasts` shelf (views-postprocessing `crafd/managers/crafd.py:222` ->
`_build_prod_forecasts_store`). So between a pod run and a CRAF'd wire there is a publish, and
there was no way to do it:

  * `publish_sampled_forecast` has exactly ONE caller in pipeline-core —
    `prediction_frame_ensemble.py:764`, inside `_forecast_ensemble` — so publishing normally
    happens only as part of a forecast run;
  * a forecast run re-pools from the eight members' artifacts, which are 26.7 GiB and
    deliberately left on the pod (`tools/podrun/pod_run_chrome_box.sh:66-70`);
  * publishing needs Appwrite credentials, and RunPod ground rule 5 forbids a publish
    credential on rented hardware.

⇒ The pod has the members and may not hold the credential; this laptop has the credential and
31 GB of RAM against a measured ~64 GiB pooling peak. Neither machine can re-pool-and-publish.
This script does neither: it loads the pool the pod already wrote and publishes it as-is.

⭐ IT INVENTS NOTHING. `publish_sampled_forecast` is keyword-only and documents `run_id` and
`generated_at` as "injectable (§10.2)", and `load_pf` is pipeline-core's own reader. This file
wires the two together and adds refusals. It does not aggregate, reconcile, or wrap the store.

⛔ PUBLISHING IS IRREVERSIBLE AND OUTWARD-FACING, so `--publish` is opt-in: the default run
reports what it would do and constructs no store at all. The refusals below are not defensive
programming, they are the four ways this could put a wrong forecast on a shelf a UN partner
reads, and each one has a prior in this repository's register.

USAGE

    # the default: read, check, report. No credential needed, no store constructed.
    python -m tools.shelf.publish_pooled_forecast \\
        --pool ensembles/chrome_box/data/generated/pod_delivery_20261102/parquet/predictions_forecasting_20261102_143000

    # the real thing, after reading the above
    python -m tools.shelf.publish_pooled_forecast --pool <same> --publish

It needs the nine shelf variables (`PredictionStoreConfig.from_environment()` fails loud and
names the missing one): APPWRITE_ENDPOINT, APPWRITE_DATASTORE_PROJECT_ID,
APPWRITE_DATASTORE_API_KEY, APPWRITE_PROD_FORECASTS_{BUCKET_ID,BUCKET_NAME,COLLECTION_ID,
COLLECTION_NAME}, APPWRITE_METADATA_DATABASE_{ID,NAME}.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: The three pgm targets `chrome_box` pools, in `config_meta.py`'s order. Named rather than
#: globbed: a glob would publish whatever happens to be on disk, and a pool missing a target
#: is exactly the case `_refuse_a_partial_pool` exists to stop.
TARGETS = ("lr_sb_best", "lr_ns_best", "lr_os_best")

#: `prediction_frame_ensemble.py:781` builds the run identity as
#: f"{name}_{run_type}_{timestamp}". The wire's staging directory is named after it and
#: views-postprocessing's `source_selection` matches on it, so it is reproduced exactly rather
#: than invented. The timestamp comes from the pool directory, which is the only place on this
#: machine that still knows it.
POOL_DIR = re.compile(r"^predictions_(?P<run_type>[a-z]+)_(?P<ts>\d{8}_\d{6})$")


class Refuse(SystemExit):
    """Exit non-zero with a message that names what was examined."""

    def __init__(self, message: str) -> None:
        super().__init__(f"!!! publish_pooled_forecast: {message}")


def run_identity(pool: Path, ensemble: str) -> tuple[str, str]:
    """(run_id, run_type) from the pool directory's own name.

    ⛔ Derived, never passed in. `run_id` is the wire identity: views-postprocessing's
    `source_selection` resolves the newest manifest and then requires every declared target to
    carry the same one, so a hand-typed id that disagrees with the shards fails loud there —
    after an irreversible upload. The directory name is the one record of the timestamp that
    survives the rsync.
    """
    m = POOL_DIR.match(pool.name)
    if not m:
        raise Refuse(
            f"{pool.name!r} is not a pool directory name. Expected "
            f"'predictions_<run_type>_<YYYYMMDD_HHMMSS>' — that prefix is a cross-repo "
            f"contract (`model_path.py` matches `predictions_<run_type>`), and the run_id is "
            f"derived from the timestamp in it. Point --pool at the directory that holds the "
            f"per-target subdirectories, not at its parent."
        )
    return f"{ensemble}_{m['run_type']}_{m['ts']}", m["run_type"]


def refuse_a_partial_pool(pool: Path, targets: tuple[str, ...]) -> None:
    """All targets or none.

    A pool short of a target publishes a manifest naming fewer streams than the delivery
    declares, and `deliveries/un_crafd.py`'s `REQUIRE.targets` names all three. The pod's own
    runner guards the mirror of this case — its `POOLABLE` gate refuses to pool 7 members of 8
    because nothing downstream notices a short pool — and this is that gate one hop later.
    """
    missing = [
        t for t in targets
        if not (pool / t / "y_pred.npy").is_file()
        or not (pool / t / "identifiers.npz").is_file()
    ]
    if missing:
        raise Refuse(
            f"these targets have no y_pred.npy + identifiers.npz under {pool}: {missing}. "
            f"Refusing to publish a partial run — a manifest naming fewer streams than "
            f"`REQUIRE.targets` declares is a short delivery that reports success."
        )


def refuse_unless_the_residual_says_fully(pool: Path) -> Path:
    """⛔ THE GUARD THAT MATTERS, and the only one that is about truth rather than shape.

    This script passes `reconciled=True`, which becomes `provenance.reconciled` in every shard
    header and is what `deliveries/un_crafd.py`'s `REQUIRE.reconciled` asserts. It is a claim
    that reaches a UN partner.

    On 2026-10-30 that claim would have been FALSE on a run that reported `STATUS: OK`:
    4.023% / 7.321% / 2.943% of the fatality mass of lr_sb_best / lr_ns_best / lr_os_best was
    never placed on the grid, with `scaling_failures: 0` and nothing in the delivery saying so
    (views-frames#283, register C-192 at Tier 1, C-145 for the field that asserted it).

    So the flag is not taken on trust. It is read from `RECONCILIATION_RESIDUAL`, the file the
    residual check itself wrote, which must carry the `✅ FULLY RECONCILED` verdict — meaning
    unplaced mass 0 for every target and no scaling failure. ⚠️ Deliberately NOT read from the
    MANIFEST (prose that has been wrong in both directions) and NOT recomputed here (a second
    opinion that can disagree with the one the delivery carries).

    Searched upward from the pool, because the pod writes the residual beside `parquet/` rather
    than inside it.
    """
    for directory in (pool, *pool.parents[:3]):
        candidate = directory / "RECONCILIATION_RESIDUAL"
        if candidate.is_file():
            text = candidate.read_text()
            if "✅ FULLY RECONCILED" in text:
                return candidate
            verdict = (
                "PARTIALLY — the residual is non-zero"
                if "UP TO " in text
                else "UNREADABLE — neither verdict line is present"
            )
            raise Refuse(
                f"{candidate} does not say FULLY RECONCILED; it says {verdict}. This script "
                f"publishes with reconciled=True, which `REQUIRE.reconciled` asserts and which "
                f"reaches a UN partner. Refusing (C-192, C-145). Re-run "
                f"`python -m reconciliation.residual --ensemble ensembles/<e>` and fix the "
                f"cause; do not pass this guard by deleting the file."
            )
    raise Refuse(
        f"no RECONCILIATION_RESIDUAL found at or above {pool}. ⛔ Absence is not a pass "
        f"(ADR-025): without it nothing here can establish that reconciled=True is true, and "
        f"that flag is the delivery's central claim. Bring the file home with the pool."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="publish_pooled_forecast",
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pool", required=True, type=Path,
        help="the predictions_<run_type>_<ts>/ directory holding the per-target subdirectories",
    )
    ap.add_argument("--ensemble", default="chrome_box", help="provenance; also the run_id prefix")
    ap.add_argument("--level", default="pgm", help='spatial level ("pgm"/"cm")')
    ap.add_argument(
        "--publish", action="store_true",
        help="actually upload. Without it nothing is sent and no store is constructed.",
    )
    ap.add_argument(
        "--targets", default=None,
        help=(
            "comma-separated subset to publish, for RESUMING after a failed leg. Default: "
            "all three. ⛔ Only use it to resume: see the note in main()."
        ),
    )
    args = ap.parse_args(argv)

    # ⛔ RESUME, NOT SELECTION. The manifest is uploaded LAST per (run, target) and is the
    # commit marker — consumers must ignore unmanifested shards (ADR-013 §3.2) — so a target
    # whose leg died is invisible rather than partial, and re-publishing it is safe. A target
    # that COMMITTED is a different matter: `_upload` has no skip-if-exists, so re-running it
    # re-uploads 36 shards for no reason. Hence resuming names the remaining targets instead
    # of redoing the lot.
    #
    # ⚠️ What this does NOT relax: `refuse_a_partial_pool` below still requires ALL THREE
    # target directories to exist on disk, and the draw-count agreement is still checked
    # across all three. The subset governs what is UPLOADED, never what is verified — a
    # narrowed check is how a short delivery passes (ADR-025).
    selected = TARGETS
    if args.targets:
        selected = tuple(t.strip() for t in args.targets.split(",") if t.strip())
        unknown = [t for t in selected if t not in TARGETS]
        if unknown:
            raise Refuse(f"unknown target(s) {unknown}; known: {list(TARGETS)}")

    pool = args.pool.resolve()
    if not pool.is_dir():
        raise Refuse(f"{pool} is not a directory")

    run_id, run_type = run_identity(pool, args.ensemble)
    refuse_a_partial_pool(pool, TARGETS)
    residual = refuse_unless_the_residual_says_fully(pool)

    print(f"pool:      {pool}")
    print(f"run_id:    {run_id}   (derived from the directory name)")
    print(f"run_type:  {run_type}")
    print(f"level:     {args.level}")
    print(f"residual:  {residual}  — ✅ FULLY RECONCILED")
    print()

    # ⛔ Load ALL targets before publishing ANY. A load failure after a partial upload leaves
    # the shelf holding a run that can never be completed, and `source_selection` requires
    # every declared target to resolve under one manifest. mmap because the pool is ~27 GiB
    # against 31 GB of RAM; `load_pf` takes the per-target directory, not the run directory.
    from views_pipeline_core.managers.prediction.prediction_frame_io import load_pf

    frames = {}
    for target in TARGETS:
        frame = load_pf(pool / target, level=args.level, mmap=True)
        frames[target] = frame
        rows, draws = frame.values.shape
        print(f"  {target:12s} rows={rows:>10,}  draws={draws:>5,}")

    draw_counts = {f.values.shape[1] for f in frames.values()}
    if len(draw_counts) != 1:
        raise Refuse(
            f"targets disagree on the draw count: {sorted(draw_counts)}. One of them is not "
            f"from this run, and the shelf would carry a run whose streams cannot be pooled."
        )
    print(f"\n  all three agree at {draw_counts.pop():,} draws")

    if not args.publish:
        print()
        print("DRY RUN — nothing was sent and no store was constructed.")
        print("Every check above passed. Re-run with --publish to upload.")
        return 0

    # Only now is a credential touched. `from_environment()` fails loud and names the missing
    # variable; nothing here re-implements that check or defaults around it.
    from views_pipeline_core.configs.prediction_store import PredictionStoreConfig
    from views_pipeline_core.managers.ensemble import EnsemblePathManager
    from views_pipeline_core.managers.ensemble.sampled_forecast_publisher import (
        publish_sampled_forecast,
    )
    from views_pipeline_core.modules.datastore import DatastoreModule

    config = PredictionStoreConfig.from_environment()
    path_manager = EnsemblePathManager(ensemble_name_or_path=args.ensemble, validate=False)
    datastore = DatastoreModule(
        appwrite_file_manager_config=config.to_appwrite_config(path_manager)
    )

    if selected != TARGETS:
        print(f"\n⚠️  RESUMING — uploading only {list(selected)}; "
              f"{[t for t in TARGETS if t not in selected]} already committed or skipped.")
    for target in selected:
        print(f"\npublishing {target} …")
        manifest = publish_sampled_forecast(
            datastore,
            frames[target],
            ensemble_name=args.ensemble,
            internal_target=target,
            run_id=run_id,
            level=args.level,
            reconciled=True,
        )
        print(f"  manifest: {manifest}")

    print(f"\n{len(selected)} of {len(TARGETS)} targets published under run_id {run_id}.")
    print("Read it back from the shelf by that run_id before trusting it.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refuse as refusal:
        print(refusal.args[0], file=sys.stderr)
        sys.exit(1)
