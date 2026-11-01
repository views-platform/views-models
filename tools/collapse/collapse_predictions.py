"""Collapse HydraNet posterior draws to point predictions, and write the researchers' parquet.

**One-off, for views-models#505.** Turns what a `-r calibration -t -e` run leaves on disk —

    <model>/data/generated/predictions_<run_type>_<ts>/origin_<i>/<target>/
        y_pred.npy        (rows, draws)  float32, RAW COUNTS, gate x body composed
        identifiers.npz   time[rows] int32 (month_id), unit[rows] int32 (priogrid_id)

— into one parquet per origin, the shape `views-platform/ensemble-updater` reads:

    month_id | priogrid_id | pred_lr_sb_best | pred_lr_ns_best | pred_lr_os_best

## What this does NOT do, deliberately

- **No scale conversion.** The draws are already counts; hydranet applies `expm1` through its
  scaler registry before the frame is saved (views-models#505). Anything that looks like log
  space here is a bug upstream, not something to fix by transforming — so
  `--min-plausible-max` refuses a suspiciously small per-target maximum instead of
  silently "correcting" it.
- **No gate arithmetic.** The draws are composed already (views-hydranet `vhy_069`,
  `compose_samples`).
- **No reindexing, no fill, no sort.** Rows are emitted in the order the model wrote them.
  Reordering would hide a misalignment between targets rather than surface it.

## The collapse

Two vocabularies, deliberately kept apart.

**Aggregate methods** — `arithmetic_mean` and `median`, the two views-hydranet `vhy_021`
defines. These are contract vocabulary: the only names a model may *declare* in
`aggregate_method`. Never hard-coded here; an unknown name is refused rather than defaulted.
All eight of the roster declare `arithmetic_mean`, and `tests/test_roster_conformance.py`
fails if that stops being true.

The mean is also the estimator the pipeline's own design points at:
`feature_scaler.py:199` — *"Essential for accurate Arithmetic Mean collapse (ADR 021)"* — INVERT
before COLLAPSE, so the mean is taken in count space. A better point estimate exists
(`gate x mu`, ledger M70) but is unobtainable without replacing `main.py`.

**Estimators** — the wider set this tool can compute, adding `q95` and `conditional_mean`
(`E[y|y>0]`). These are *not* aggregate methods and no model declares them. They exist for the
views-models#505 selector experiment: the same posterior cube submitted three times as three
apparent models, so the ensemble selector's own criteria decide between a mean that
under-predicts the total ~5x and two estimators that do not. Which estimator produced a frame
is not recoverable from the parquet — it is recorded by the output directory the caller picks,
and by nothing else.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

#: The three regression targets the researchers asked for. `by_*` are gate probabilities,
#: not fatalities, and are deliberately absent.
TARGETS: tuple[str, ...] = ("lr_sb_best", "lr_ns_best", "lr_os_best")


def _arithmetic_mean(draws: np.ndarray) -> np.ndarray:
    return draws.mean(axis=1)


def _median(draws: np.ndarray) -> np.ndarray:
    return np.median(draws, axis=1)


def _q95(draws: np.ndarray) -> np.ndarray:
    """The 95th percentile across draws, linearly interpolated (numpy's default).

    At the roster's D x K = 16 this sits between the 15th and 16th order statistics, so it is
    a coarse tail estimate — the upper draws are a small sample and one of them moves it. That
    is a property of 16 draws, not of the quantile: it is reported, not corrected for.
    """
    return np.quantile(draws, 0.95, axis=1)


def _conditional_mean(draws: np.ndarray) -> np.ndarray:
    """`E[y|y>0]` — the mean over the positive draws only.

    Identical to `E[y] / P(y>0)`: a zero draw adds nothing to the sum, so
    `sum / n_positive == (sum / D) / (n_positive / D)`. Both forms are the conditional
    intensity; this one is written as the division that cannot divide by a probability of zero.

    A cell every draw calls silent has no conditional intensity to report, and gets **0.0**,
    not NaN. `ensemble-updater` joins and scores these columns, so a NaN would propagate into
    a metric instead of announcing itself. Where every draw is positive this equals the
    arithmetic mean; it is never below it.
    """
    n_positive = (draws > 0).sum(axis=1)
    return np.where(n_positive > 0, draws.sum(axis=1) / np.maximum(n_positive, 1), 0.0)


#: Every point estimator this tool can compute, by its command-line name.
ESTIMATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "arithmetic_mean": _arithmetic_mean,
    "median": _median,
    "q95": _q95,
    "conditional_mean": _conditional_mean,
}

# views-hydranet ADR-021 defines exactly these two and rejects anything else
# (`volume_handler.py::collapse_to_point`). We implement the same two, under the same names,
# so a model's declared `aggregate_method` can be passed straight through.
#
# `q95` and `conditional_mean` are deliberately NOT here. They are alternative estimators for
# the views-models#505 experiment, not contract vocabulary, and a model that declared one would
# be a config error `tests/test_roster_conformance.py` must keep catching.
AGGREGATE_METHODS: tuple[str, ...] = ("arithmetic_mean", "median")
DEFAULT_AGGREGATE_METHOD = "arithmetic_mean"

#: A collapsed count below this maximum, across a whole origin, means the values are almost
#: certainly still in log1p space — a real global-land origin reaches the hundreds. Refuse
#: rather than ship silently-wrong numbers (views-models#505).
MIN_PLAUSIBLE_MAX = 12.0


class CollapseError(RuntimeError):
    """Raised when the inputs are not what the specification says they must be."""


def _load_target(target_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (draws, month_id, priogrid_id) for one target directory, or raise."""
    y_path, ids_path = target_dir / "y_pred.npy", target_dir / "identifiers.npz"
    for p in (y_path, ids_path):
        if not p.is_file():
            raise CollapseError(f"missing {p}")

    draws = np.load(y_path)
    if draws.ndim != 2:
        raise CollapseError(f"{y_path}: expected (rows, draws), got shape {draws.shape}")
    if draws.shape[1] < 2:
        raise CollapseError(
            f"{y_path}: only {draws.shape[1]} draw(s) — nothing to collapse; "
            f"a posterior cube always carries D x K >= 2"
        )
    if not np.isfinite(draws).all():
        n = int((~np.isfinite(draws)).sum())
        raise CollapseError(f"{y_path}: {n} non-finite value(s); refusing to average them away")
    if (draws < 0).any():
        raise CollapseError(f"{y_path}: negative values — these are counts, not log space")

    with np.load(ids_path) as ids:
        for key in ("time", "unit"):
            if key not in ids.files:
                raise CollapseError(f"{ids_path}: no '{key}' array (found {ids.files})")
        month, unit = ids["time"], ids["unit"]

    if month.shape != (draws.shape[0],) or unit.shape != (draws.shape[0],):
        raise CollapseError(
            f"{target_dir}: identifiers {month.shape}/{unit.shape} do not match "
            f"{draws.shape[0]} prediction rows"
        )
    return draws, month, unit


def collapse_origin(
    origin_dir: Path,
    targets: tuple[str, ...] = TARGETS,
    estimator: str = DEFAULT_AGGREGATE_METHOD,
) -> pd.DataFrame:
    """One origin directory -> one DataFrame of point predictions.

    Every target must be present, must carry the same number of draws, and must be indexed
    identically row-for-row. A mismatch is an error, never a merge: silently joining on keys
    would paper over a misalignment that changes which cell a number belongs to.
    """
    try:
        collapse = ESTIMATORS[estimator]
    except KeyError:
        raise CollapseError(
            f"unknown estimator {estimator!r}; this tool implements "
            f"{', '.join(sorted(ESTIMATORS))} — of which views-hydranet ADR-021 defines "
            f"only {', '.join(AGGREGATE_METHODS)} as a declarable aggregate_method"
        ) from None
    if not origin_dir.is_dir():
        raise CollapseError(f"not a directory: {origin_dir}")

    frame: pd.DataFrame | None = None
    reference: tuple[np.ndarray, np.ndarray] | None = None
    n_draws: int | None = None

    for target in targets:
        draws, month, unit = _load_target(origin_dir / target)

        if n_draws is None:
            n_draws = draws.shape[1]
        elif draws.shape[1] != n_draws:
            raise CollapseError(
                f"{origin_dir}: '{target}' has {draws.shape[1]} draws but an earlier target "
                f"has {n_draws}; these cannot be from one run"
            )

        if reference is None:
            reference = (month, unit)
            frame = pd.DataFrame(
                {"month_id": month.astype("int64"), "priogrid_id": unit.astype("int64")}
            )
        else:
            ref_month, ref_unit = reference
            if month.shape != ref_month.shape:
                raise CollapseError(
                    f"{origin_dir}: '{target}' has {month.shape[0]} rows but the first target "
                    f"has {ref_month.shape[0]}; these cannot be from one run"
                )
            if not (np.array_equal(month, ref_month) and np.array_equal(unit, ref_unit)):
                bad = int((month != ref_month).sum() + (unit != ref_unit).sum())
                raise CollapseError(
                    f"{origin_dir}: '{target}' is not row-aligned with the first target "
                    f"({bad} differing identifier entries). Refusing to join."
                )

        wide = draws.astype("float64")  # float32 sums make the answer depend on the draw count
        frame[f"pred_{target}"] = collapse(wide)

    assert frame is not None  # targets is non-empty by construction

    duplicated = frame.duplicated(["month_id", "priogrid_id"])
    if duplicated.any():
        first = frame.loc[duplicated, ["month_id", "priogrid_id"]].iloc[0]
        raise CollapseError(
            f"{origin_dir}: {int(duplicated.sum())} duplicate (month_id, priogrid_id) row(s), "
            f"first at month {int(first.month_id)} cell {int(first.priogrid_id)}. "
            "ensemble-updater joins on that pair, so a repeat silently wins or loses the join. "
            "Every target agreeing on a duplicated identifier is still a duplicate, which is "
            "why the row-alignment check above cannot see it."
        )
    return frame


def _check_scale(frame: pd.DataFrame, origin_dir: Path, min_plausible_max: float) -> None:
    """Refuse a target whose magnitude says it never left log1p space.

    Checked PER TARGET, not on the three flattened together. A single target can be left in
    log1p space while its siblings are correctly inverted — an upstream registry mismatch does
    not have to hit all three — and a combined maximum is then carried over the threshold by a
    healthy sibling while the corrupted one ships as log1p(count). That is exactly the
    "plausible-looking but wrong" parquet this guard exists to stop, and the flattened form
    could not see it.

    The trade-off is deliberate: a genuinely tiny target trips a false alarm and stops the
    conversion. That is the safe direction. The message names the column, so the next step is
    to look upstream — never to expm1 here.
    """
    for col in (c for c in frame.columns if c.startswith("pred_")):
        hi = float(frame[col].max())
        if hi < min_plausible_max:
            raise CollapseError(
                f"{origin_dir}: largest collapsed '{col}' is {hi:.4g}, below "
                f"{min_plausible_max:g}. Counts at global land reach the hundreds; this looks "
                f"like log1p space. Investigate upstream — do NOT expm1 here (views-models#505)."
            )


def convert_model(
    model_dir: Path,
    run_type: str = "calibration",
    out_dir: Path | None = None,
    targets: tuple[str, ...] = TARGETS,
    min_plausible_max: float = MIN_PLAUSIBLE_MAX,
    estimator: str = DEFAULT_AGGREGATE_METHOD,
) -> list[Path]:
    """Convert every origin of a model's latest prediction directory. Returns files written.

    When several `predictions_<run_type>_<ts>/` exist — a second `-e` on the same artifact
    writes a second one — the lexicographically greatest timestamp wins, which is the newest,
    and is the same rule `ensemble-updater` applies to filenames.
    """
    generated = model_dir / "data" / "generated"
    candidates = sorted(p for p in generated.glob(f"predictions_{run_type}_*") if p.is_dir())
    if not candidates:
        raise CollapseError(f"no predictions_{run_type}_* directory under {generated}")
    source = candidates[-1]
    timestamp = source.name[len(f"predictions_{run_type}_") :]

    origins = sorted(
        (p for p in source.glob("origin_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not origins:
        raise CollapseError(f"no origin_* directories under {source}")

    destination = out_dir if out_dir is not None else generated
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for origin in origins:
        index = int(origin.name.split("_")[1])
        frame = collapse_origin(origin, targets, estimator)
        _check_scale(frame, origin, min_plausible_max)
        path = destination / f"predictions_{run_type}_{timestamp}_{index:02d}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("model_dir", type=Path, help="models/<model>")
    parser.add_argument("--run-type", default="calibration")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-plausible-max", type=float, default=MIN_PLAUSIBLE_MAX)
    parser.add_argument(
        "--estimator",
        "--aggregate-method",  # the name before q95/conditional_mean existed; still accepted
        dest="estimator",
        default=DEFAULT_AGGREGATE_METHOD,
        choices=sorted(ESTIMATORS),
        help=(
            "arithmetic_mean or median must match the model's own `aggregate_method` (all "
            "eight declare arithmetic_mean); q95 and conditional_mean are the views-models#505 "
            "selector experiment and are declared by no model"
        ),
    )
    args = parser.parse_args(argv)

    written = convert_model(
        args.model_dir,
        run_type=args.run_type,
        out_dir=args.out_dir,
        min_plausible_max=args.min_plausible_max,
        estimator=args.estimator,
    )
    for path in written:
        print(path)
    print(f"{len(written)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
