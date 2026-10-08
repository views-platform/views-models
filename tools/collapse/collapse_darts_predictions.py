"""Collapse the r2darts2 `dataframe` output to the researchers' point-prediction parquet.

**For views-models#533, epic #532.** The darts sibling of `collapse_predictions.py`. Turns what a
`-r calibration -t -e` run leaves on disk for a `prediction_format: "dataframe"` model —

    <model>/data/generated/predictions_<run_type>_<ts>_<NN>.parquet   (13 of them, _00.._12)
        index   (month_id, priogrid_id)
        columns pred_<target>, EVERY CELL A PYTHON LIST of sample values

— into the shape `views-platform/ensemble-updater` reads (views-models#505):

    month_id | priogrid_id | pred_lr_ged_sb | pred_lr_ged_ns | pred_lr_ged_os

flat `int64` keys, one `float` scalar per cell.

## Why this is mandatory rather than cosmetic

`views_r2darts2/transformers/darts_bridge.py::prediction_frames_to_dataframe` writes "a list of
sample values (length 1 for deterministic, length S for probabilistic)" into every cell. The
consumer does not collapse it. ADR-023 records what it does instead:

    `_as_float_prediction_array` in `ensemble-updater` takes `float(x[0])` on a list cell —
    **one draw, silently**, which is the exact failure this ADR is meant to prevent.

So handing the run's parquets over unconverted does not raise. It delivers draw zero as if it were
the answer. For a deterministic model that happens to be right; for anything else it is a wrong
number with no error attached. This module exists to make that outcome unreachable.

## Why not `collapse_predictions.py`

That one is HydraNet-shaped on four independent axes, each fatal here: it requires
`origin_<i>/<target>/{y_pred.npy,identifiers.npz}`; it raises on `draws.shape[1] < 2` ("a posterior
cube always carries D x K >= 2") where nine of these eleven models carry exactly one; it hard-codes
`TARGETS = ("lr_sb_best", ...)` with no override; and its default is pinned against the eight
HydraNets' declarations by `tests/test_roster_conformance.py`. Nothing in `tools/collapse` reads a
parquet. ADR-023's own Consequences already counts the cost of a third place knowing the HydraNet
layout (register C-152) — so this is a sibling, not a generalisation.

## What this does NOT do, deliberately

- **No scale conversion.** `--min-plausible-max` refuses a suspiciously small per-target maximum
  instead of "correcting" it. See the note on that threshold below: it is inherited and, for darts,
  not yet validated against a real run.
- **No reindexing, no fill, no sort.** Rows are emitted in the order the model wrote them.
- **No renaming by default.** Darts emits canonical `pred_lr_ged_*` (views-models#151).
  `ensemble-updater`'s `target_column` is configurable, and views-models#505 puts renaming "in the
  converter at send-off" — so `--rename` exists and is off unless asked for.
- **No writing over its own input.** Unlike the HydraNet path, the source and the deliverable share
  one filename pattern here, so an in-place default would destroy the run's output. The destination
  is a distinct directory and a collision is refused.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

#: Flat key columns the consumer joins on (views-models#505: "flat column, not an index").
KEY_COLUMNS: tuple[str, str] = ("month_id", "priogrid_id")

#: 36 months x 64 818 priogrid cells, the global-land `REGION = "land"` grid. One origin's rows.
EXPECTED_ROWS = 2_333_448

#: `test (457, 504)` is 48 months, `max(steps)` is 36, so 48 - 36 + 1 rolling origins. Identical
#: across all eleven models (`config_partitions.py` is one blob) and asserted by the engine's
#: `_resolve_total_sequence_number`.
EXPECTED_ORIGINS = 13

#: Applied to the FRAME maximum, not per target — see `_check_scale`. **Now validated against a
#: real darts run**, which is what the previous note here asked for: dark_river at global pgm
#: (2026-10-07, the first such run in existence) reached 283.68 on `lr_ged_sb`, 21.56 on
#: `lr_ged_ns` and 14.94 on `lr_ged_os`. Per target this threshold REFUSED that correct output
#: after 87 minutes of GPU time; against the frame maximum it has 20x headroom.
#: `--min-plausible-max 0` disables the check.
MIN_PLAUSIBLE_MAX = 12.0

#: Legacy HydraNet spelling, for `--rename`. Canonical on the left (views-models#151).
RENAME_TO_HYDRANET: dict[str, str] = {
    "pred_lr_ged_sb": "pred_lr_sb_best",
    "pred_lr_ged_ns": "pred_lr_ns_best",
    "pred_lr_ged_os": "pred_lr_os_best",
}

_SEQUENCE_SUFFIX = re.compile(r"_(\d{2})$")


class DartsCollapseError(RuntimeError):
    """Raised when the inputs are not what the specification says they must be."""


def _scalars_from_list_cells(column: pd.Series, source: Path, name: str) -> np.ndarray:
    """One list-per-cell column -> one float64 scalar per row.

    Length 1 is **unwrapped**, not averaged: a deterministic model's single value is the value,
    and calling it a mean would imply a posterior that does not exist. Length S > 1 is the
    arithmetic mean in count space — ADR-021's INVERT-then-COLLAPSE ordering means the values
    are already counts when they reach here, and `float64` before summing because float32 sums
    make the answer depend on the draw count (`collapse_predictions.py:154`).
    """
    try:
        stacked = np.asarray(column.to_list(), dtype=np.float64)
    except ValueError as exc:
        # numpy (pinned <2 by viewser) refuses a ragged nested sequence rather than building an
        # object array. Ragged cells mean the sample count varies row to row, which no collapse
        # can reconcile — the alternative to raising is averaging different-sized posteriors
        # together.
        raise DartsCollapseError(
            f"{source}: column '{name}' has cells of differing length, so the sample count "
            f"varies row to row and there is no single collapse to apply ({exc})"
        ) from exc

    if stacked.ndim == 1:
        # Already scalar. Not what views_r2darts2 0.2.x writes, but a future version might, and
        # silently treating a scalar as a one-element list is the kind of guess that hides a
        # format change. Accept it, and say which shape was found if anything else fails.
        values = stacked
    elif stacked.ndim == 2:
        if stacked.shape[1] == 0:
            raise DartsCollapseError(
                f"{source}: column '{name}' has empty cells — no sample to collapse"
            )
        values = stacked[:, 0] if stacked.shape[1] == 1 else stacked.mean(axis=1)
    else:
        raise DartsCollapseError(
            f"{source}: column '{name}' stacks to shape {stacked.shape}; expected one value or "
            f"one list of values per row"
        )

    if not np.isfinite(values).all():
        n = int((~np.isfinite(values)).sum())
        raise DartsCollapseError(
            f"{source}: column '{name}' has {n} non-finite value(s) after collapse; refusing to "
            f"average them away"
        )
    if (values < 0).any():
        worst = float(values.min())
        raise DartsCollapseError(
            f"{source}: column '{name}' has negative values (min {worst:.4g}) — these are "
            f"counts, not log space"
        )
    return values


def _keys_as_columns(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Return a frame with month_id and priogrid_id as flat columns, or raise.

    `prediction_frames_to_dataframe` ends with `df.set_index([time_id, entity_id])`, so the keys
    normally arrive as an index. They may equally arrive as columns depending on how the parquet
    was written. Both are handled explicitly and a frame carrying neither is refused — guessing
    which of its columns are the keys is exactly the "magic discovery" that lets a shape change
    pass as data.
    """
    missing_as_columns = [k for k in KEY_COLUMNS if k not in frame.columns]
    if not missing_as_columns:
        return frame

    index_names = [n for n in (frame.index.names or []) if n is not None]
    if all(k in index_names for k in KEY_COLUMNS):
        return frame.reset_index()

    raise DartsCollapseError(
        f"{source}: cannot find {KEY_COLUMNS} as columns or as index levels "
        f"(columns={list(frame.columns)}, index names={index_names})"
    )


def collapse_parquet(
    source: Path,
    *,
    min_plausible_max: float = MIN_PLAUSIBLE_MAX,
    expected_rows: int | None = EXPECTED_ROWS,
) -> pd.DataFrame:
    """One run parquet -> one DataFrame of point predictions.

    Every `pred_*` column present is collapsed; the target set is read off the file rather than
    hard-coded, so a model declaring different targets needs no change here
    (views-models#151 — derive from the data, do not force a uniform value).
    """
    if not source.is_file():
        raise DartsCollapseError(f"missing {source}")

    frame = _keys_as_columns(pd.read_parquet(source), source)

    prediction_columns = [c for c in frame.columns if c.startswith("pred_")]
    if not prediction_columns:
        raise DartsCollapseError(
            f"{source}: no 'pred_*' column (columns={list(frame.columns)}); this is not a "
            f"prediction frame — the metric frames and the run log live beside them"
        )

    if expected_rows is not None and len(frame) != expected_rows:
        raise DartsCollapseError(
            f"{source}: {len(frame)} rows, expected {expected_rows} "
            f"(36 months x 64 818 global-land cells). A short frame means the run did not cover "
            f"the grid it claims to; pass --expect-rows 0 only if the partition geometry changed"
        )

    out = pd.DataFrame(
        {
            "month_id": frame["month_id"].to_numpy(dtype="int64"),
            "priogrid_id": frame["priogrid_id"].to_numpy(dtype="int64"),
        }
    )
    for name in prediction_columns:
        out[name] = _scalars_from_list_cells(frame[name], source, name)

    duplicated = out.duplicated(list(KEY_COLUMNS))
    if duplicated.any():
        first = out.loc[duplicated, list(KEY_COLUMNS)].iloc[0]
        raise DartsCollapseError(
            f"{source}: {int(duplicated.sum())} duplicate (month_id, priogrid_id) row(s), first "
            f"at month {int(first.month_id)} cell {int(first.priogrid_id)}. ensemble-updater "
            f"joins on that pair, so a repeat silently wins or loses the join (ADR-023)."
        )

    _check_scale(out, source, min_plausible_max)
    return out


def _check_scale(frame: pd.DataFrame, source: Path, min_plausible_max: float) -> None:
    """Refuse a FRAME whose magnitude says it never left transformed space.

    **Frame-level, not per-target — and the first real run is why.** This guard was imported
    per-target from `collapse_predictions._check_scale`, whose reasoning is sound for HydraNet:
    there, a per-target scaler REGISTRY can leave one target in log1p space while its siblings
    invert correctly, so a combined maximum is carried over the threshold by a healthy sibling
    while the corrupted one ships.

    r2darts2 has no such registry. One `target_scaler` chain is applied to all targets, so the
    inverse either ran for the frame or did not. The failure mode the per-target form exists to
    catch cannot occur here, and keeping it imports a false positive instead:

        dark_river, 2026-10-07, the first pgm r2darts2 run in existence
            pred_lr_ged_sb  max 283.68   <- plainly counts
            pred_lr_ged_ns  max  21.56
            pred_lr_ged_os  max  14.94   <- REFUSED, "below 12" on some origins

        It refused at origin 06 with `pred_lr_ged_os` at 11.46, after 87 minutes of GPU time,
        on output that was correct.

    `sb` at 283 settles the frame: were it transformed, the underlying value would be
    astronomical. One-sided and non-state violence are simply rarer, and a deterministic point
    model shrinks hard toward the mean — measured against observed data the same run under-
    predicts totals by 2.5-4x. A small maximum on a rare target is the model being timid, not
    the scaler being broken, and no magnitude test can separate those two for a single column.

    **What this gives up, stated plainly.** If a future engine does acquire per-target scaling,
    this check will not see one target left behind. The trigger to revisit is exactly that: an
    r2darts2 release that scales targets independently. Until then, per-target here is a guard
    that fires on correct data, which is worse than one that is narrower and true.
    """
    if min_plausible_max <= 0:
        return
    predictions = [c for c in frame.columns if c.startswith("pred_")]
    if predictions:
        hi = max(float(frame[c].max()) for c in predictions)
        col = max(predictions, key=lambda c: float(frame[c].max()))
        if hi < min_plausible_max:
            raise DartsCollapseError(
                f"{source}: the largest value in the WHOLE frame is {hi:.4g} "
                f"(in '{col}'), below {min_plausible_max:g}. Every target here shares one "
                f"scaler chain, so if none of them reaches a plausible count, the inverse "
                f"transform did not run. Investigate upstream — do NOT expm1 here "
                f"(views-models#505). For calibration: the first real pgm darts run "
                f"(dark_river, 2026-10-07) reached 283.68 on lr_ged_sb while its rarest "
                f"target peaked at 14.94, so a frame maximum in the hundreds is normal and "
                f"one in single digits is not."
            )


def _discover(generated: Path, run_type: str) -> tuple[str, list[tuple[int, Path]]]:
    """Find the latest run's prediction parquets. Returns (timestamp, [(sequence, path)])."""
    prefix = f"predictions_{run_type}_"
    found: dict[str, dict[int, Path]] = {}
    for path in generated.glob(f"{prefix}*.parquet"):
        match = _SEQUENCE_SUFFIX.search(path.stem)
        if match is None:
            continue
        timestamp = path.stem[len(prefix) : match.start()]
        found.setdefault(timestamp, {})[int(match.group(1))] = path

    if not found:
        raise DartsCollapseError(
            f"no {prefix}<ts>_<NN>.parquet under {generated}. A `dataframe`-format model writes "
            f"these directly; a `prediction_frame` model writes {prefix}<ts>/ directories "
            f"instead, which is what collapse_predictions.py reads"
        )

    # A second `-e` on the same artifact writes a second set beside the first, and the timestamp
    # is the artifact's rather than the run's. Lexicographically greatest is newest, which is the
    # rule ensemble-updater applies to the same filenames.
    timestamp = max(found)
    by_sequence = found[timestamp]
    return timestamp, [(seq, by_sequence[seq]) for seq in sorted(by_sequence)]


def convert_model(
    model_dir: Path,
    run_type: str = "calibration",
    out_dir: Path | None = None,
    *,
    min_plausible_max: float = MIN_PLAUSIBLE_MAX,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_origins: int | None = EXPECTED_ORIGINS,
    rename: bool = False,
) -> list[Path]:
    """Convert every origin of a model's latest prediction set. Returns the files written."""
    generated = model_dir / "data" / "generated"
    timestamp, sequences = _discover(generated, run_type)

    numbers = [seq for seq, _ in sequences]
    expected_range = list(range(len(numbers)))
    if numbers != expected_range:
        raise DartsCollapseError(
            f"{generated}: run {timestamp} has sequence numbers {numbers}, which is not a "
            f"contiguous _00.._{len(numbers) - 1:02d}. ensemble-updater raises FileNotFoundError "
            f"naming any origin it cannot find, so a gap must not be converted quietly"
        )
    if expected_origins is not None and len(numbers) != expected_origins:
        raise DartsCollapseError(
            f"{generated}: run {timestamp} has {len(numbers)} origin(s), expected "
            f"{expected_origins} (test window 48 months - 36 steps + 1). Pass "
            f"--expect-origins 0 only if the partition geometry changed"
        )

    # The source and the deliverable share one filename pattern, so the destination must differ
    # from the directory being read or the conversion destroys its own input.
    destination = out_dir if out_dir is not None else generated / f"delivery_{run_type}_{timestamp}"
    destination = destination.resolve()
    if destination == generated.resolve():
        raise DartsCollapseError(
            f"refusing to write the deliverable into {generated}: the run's own parquets use the "
            f"same names and would be overwritten. Choose a different --out-dir"
        )
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for sequence, source in sequences:
        frame = collapse_parquet(
            source, min_plausible_max=min_plausible_max, expected_rows=expected_rows
        )
        if rename:
            frame = frame.rename(columns=RENAME_TO_HYDRANET)
        path = destination / f"predictions_{run_type}_{timestamp}_{sequence:02d}.parquet"
        if path.resolve() == source.resolve():
            raise DartsCollapseError(f"refusing to overwrite the input {source}")
        frame.to_parquet(path, index=False)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("model_dir", type=Path, help="models/<model>")
    parser.add_argument("--run-type", default="calibration")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <model>/data/generated/delivery_<run_type>_<ts>/ — never the source dir",
    )
    parser.add_argument("--min-plausible-max", type=float, default=MIN_PLAUSIBLE_MAX)
    parser.add_argument(
        "--expect-rows", type=int, default=EXPECTED_ROWS, help="0 disables the row-count check"
    )
    parser.add_argument(
        "--expect-origins", type=int, default=EXPECTED_ORIGINS, help="0 disables the origin count"
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="emit the legacy HydraNet column spelling (pred_lr_sb_best, ...) instead of the "
        "canonical pred_lr_ged_* names",
    )
    args = parser.parse_args(argv)

    written = convert_model(
        args.model_dir,
        run_type=args.run_type,
        out_dir=args.out_dir,
        min_plausible_max=args.min_plausible_max,
        expected_rows=args.expect_rows or None,
        expected_origins=args.expect_origins or None,
        rename=args.rename,
    )
    for path in written:
        print(path)
    print(f"{len(written)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
