"""The converter for views-models#505 must refuse anything it cannot vouch for.

Two kinds of test. The **contract** tests fix what a correct conversion produces. The
**mutation** tests corrupt an input the way a real run could corrupt it and require the
converter to raise — because the failure mode that matters here is not a crash, it is a
plausible-looking parquet with the wrong numbers in it. Researchers cannot tell those apart
by looking, and neither can `ensemble-updater`: it would score them and report metrics.

Every fixture is synthetic and tiny. `tests/test_collapse_on_real_predictions.py` runs the
same code over the real HydraNet output on this machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.collapse.collapse_predictions import (
    AGGREGATE_METHODS,
    DEFAULT_AGGREGATE_METHOD,
    TARGETS,
    CollapseError,
    collapse_origin,
    convert_model,
)

ROWS, DRAWS = 40, 8


def _write_target(origin: Path, target: str, draws: np.ndarray, month=None, unit=None) -> None:
    d = origin / target
    d.mkdir(parents=True, exist_ok=True)
    n = draws.shape[0]
    np.save(d / "y_pred.npy", draws.astype("float32"))
    np.savez(
        d / "identifiers.npz",
        time=(np.arange(n, dtype="int32") % 10 + 500) if month is None else month,
        unit=(np.arange(n, dtype="int32") // 10 + 62000) if unit is None else unit,
    )


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """One well-formed origin: three targets, identical identifiers, counts."""
    o = tmp_path / "predictions_calibration_20260101_000000" / "origin_0"
    rng = np.random.default_rng(0)
    for i, target in enumerate(TARGETS):
        draws = rng.gamma(shape=0.2, scale=60.0, size=(ROWS, DRAWS)) * (i + 1)
        draws[rng.random((ROWS, DRAWS)) < 0.7] = 0.0  # the zero-inflation a gate produces
        _write_target(o, target, draws)
    return o


# ── contract ──────────────────────────────────────────────────────────────────────


def test_columns_and_order_are_the_specification(origin):
    df = collapse_origin(origin)
    assert list(df.columns) == [
        "month_id", "priogrid_id",
        "pred_lr_sb_best", "pred_lr_ns_best", "pred_lr_os_best",
    ]
    assert df["month_id"].dtype == "int64" and df["priogrid_id"].dtype == "int64"
    assert len(df) == ROWS


def test_the_value_is_the_arithmetic_mean_of_the_draws(origin):
    df = collapse_origin(origin)
    raw = np.load(origin / "lr_sb_best" / "y_pred.npy")
    np.testing.assert_allclose(df["pred_lr_sb_best"].to_numpy(), raw.mean(axis=1), rtol=1e-6)


def test_identifiers_are_passed_through_unreordered(origin):
    df = collapse_origin(origin)
    with np.load(origin / "lr_sb_best" / "identifiers.npz") as ids:
        np.testing.assert_array_equal(df["month_id"].to_numpy(), ids["time"])
        np.testing.assert_array_equal(df["priogrid_id"].to_numpy(), ids["unit"])


def test_zeros_survive_the_collapse(origin):
    """A cell whose every draw is zero must stay zero — not become NaN or be dropped."""
    raw = np.load(origin / "lr_sb_best" / "y_pred.npy")
    all_zero = (raw == 0).all(axis=1)
    assert all_zero.any(), "fixture should contain all-zero cells"
    df = collapse_origin(origin)
    assert (df.loc[all_zero, "pred_lr_sb_best"] == 0).all()


def test_convert_model_writes_one_parquet_per_origin(tmp_path, origin):
    model = tmp_path / "m"
    generated = model / "data" / "generated"
    generated.mkdir(parents=True)
    src = generated / "predictions_calibration_20260101_000000"
    for i in range(13):  # 13, as the deliverable has — origin_10 sorts before origin_2 as text
        for target in TARGETS:
            o = src / f"origin_{i}"
            rng = np.random.default_rng(i)
            _write_target(o, target, rng.gamma(0.2, 60.0, size=(ROWS, DRAWS)))
    written = convert_model(model, out_dir=tmp_path / "out")
    assert [p.name for p in written] == [
        f"predictions_calibration_20260101_000000_{i:02d}.parquet" for i in range(13)
    ]
    assert list(pd.read_parquet(written[0]).columns)[:2] == ["month_id", "priogrid_id"]


def test_the_latest_timestamp_wins(tmp_path):
    """A second `-e` leaves two prediction dirs; the newer must be the one converted."""
    model = tmp_path / "m"
    generated = model / "data" / "generated"
    for ts, value in (("20260101_000000", 5.0), ("20260202_000000", 500.0)):
        for target in TARGETS:
            _write_target(
                generated / f"predictions_calibration_{ts}" / "origin_0",
                target, np.full((ROWS, DRAWS), value),
            )
    written = convert_model(model, out_dir=tmp_path / "out")
    assert "20260202_000000" in written[0].name
    assert pd.read_parquet(written[0])["pred_lr_sb_best"].iloc[0] == pytest.approx(500.0)


# ── mutations: each must raise ────────────────────────────────────────────────────


def test_mutation_target_directory_missing(origin):
    import shutil
    shutil.rmtree(origin / "lr_os_best")
    with pytest.raises(CollapseError, match="missing"):
        collapse_origin(origin)


def test_mutation_identifiers_differ_between_targets(origin):
    """The trap this guards: joining misaligned targets would move numbers between cells."""
    d = origin / "lr_ns_best"
    with np.load(d / "identifiers.npz") as ids:
        unit = ids["unit"].copy()
        month = ids["time"].copy()
    unit[3] += 1
    np.savez(d / "identifiers.npz", time=month, unit=unit)
    with pytest.raises(CollapseError, match="not row-aligned"):
        collapse_origin(origin)


def test_mutation_row_count_differs_between_targets(origin):
    rng = np.random.default_rng(1)
    _write_target(origin, "lr_ns_best", rng.gamma(0.2, 60.0, size=(ROWS - 1, DRAWS)))
    with pytest.raises(CollapseError, match="39 rows but the first target has 40"):
        collapse_origin(origin)


def test_mutation_draw_count_differs_between_targets(origin):
    rng = np.random.default_rng(2)
    _write_target(origin, "lr_ns_best", rng.gamma(0.2, 60.0, size=(ROWS, DRAWS + 4)))
    with pytest.raises(CollapseError, match="draws"):
        collapse_origin(origin)


def test_mutation_nan_in_the_draws(origin):
    d = origin / "lr_sb_best"
    draws = np.load(d / "y_pred.npy")
    draws[7, 2] = np.nan
    np.save(d / "y_pred.npy", draws)
    with pytest.raises(CollapseError, match="non-finite"):
        collapse_origin(origin)


def test_mutation_negative_values(origin):
    d = origin / "lr_sb_best"
    draws = np.load(d / "y_pred.npy")
    draws[1, 1] = -0.5
    np.save(d / "y_pred.npy", draws)
    with pytest.raises(CollapseError, match="negative"):
        collapse_origin(origin)


def test_mutation_already_collapsed_single_draw(origin):
    """A (rows, 1) array means someone collapsed upstream; averaging it again hides that."""
    rng = np.random.default_rng(3)
    _write_target(origin, "lr_sb_best", rng.gamma(0.2, 60.0, size=(ROWS, 1)))
    with pytest.raises(CollapseError, match="nothing to collapse"):
        collapse_origin(origin)


def test_mutation_log_space_values_are_refused(tmp_path):
    """The failure this exists for: log1p values look fine and score as nonsense."""
    model = tmp_path / "m"
    for target in TARGETS:
        _write_target(
            model / "data" / "generated" / "predictions_calibration_20260101_000000" / "origin_0",
            target, np.full((ROWS, DRAWS), 2.3),   # log1p(9) — plausible-looking, wrong scale
        )
    with pytest.raises(CollapseError, match="log1p space"):
        convert_model(model, out_dir=tmp_path / "out")


def test_mutation_identifiers_file_missing_a_key(origin):
    d = origin / "lr_sb_best"
    with np.load(d / "identifiers.npz") as ids:
        np.savez(d / "identifiers.npz", time=ids["time"])   # 'unit' dropped
    with pytest.raises(CollapseError, match="no 'unit'"):
        collapse_origin(origin)


def test_mutation_no_origin_directories(tmp_path):
    model = tmp_path / "m"
    (model / "data" / "generated" / "predictions_calibration_20260101_000000").mkdir(parents=True)
    with pytest.raises(CollapseError, match="no origin_"):
        convert_model(model, out_dir=tmp_path / "out")


def test_the_mean_accumulates_in_float64(origin):
    """Draws are stored float32; summing in float32 makes the answer depend on the draw count.

    The difference is ~1e-6 on counts — irrelevant to a researcher, but a float64 accumulator
    costs nothing and makes the number reproducible from the stored array alone.
    """
    df = collapse_origin(origin)
    raw = np.load(origin / "lr_sb_best" / "y_pred.npy")
    assert raw.dtype == np.float32
    np.testing.assert_array_equal(
        df["pred_lr_sb_best"].to_numpy(), raw.astype("float64").mean(axis=1)
    )


def test_mutation_identifiers_shorter_than_the_draws(origin):
    """A truncated identifiers.npz would silently label the wrong cells if we zipped them."""
    d = origin / "lr_sb_best"
    np.savez(
        d / "identifiers.npz",
        time=np.arange(ROWS - 5, dtype="int32"),
        unit=np.arange(ROWS - 5, dtype="int32"),
    )
    with pytest.raises(CollapseError, match="do not match"):
        collapse_origin(origin)


def test_mutation_identifiers_longer_than_the_draws(origin):
    """The other direction: extra identifier rows mean the arrays are not from one write."""
    d = origin / "lr_os_best"
    np.savez(
        d / "identifiers.npz",
        time=np.arange(ROWS + 3, dtype="int32"),
        unit=np.arange(ROWS + 3, dtype="int32"),
    )
    with pytest.raises(CollapseError, match="do not match"):
        collapse_origin(origin)


def test_the_default_method_is_the_one_the_roster_declares():
    """All eight HydraNets declare `arithmetic_mean`; `tests/test_roster_conformance.py`
    fails if that stops being true. This pins the other half of the agreement."""
    assert DEFAULT_AGGREGATE_METHOD == "arithmetic_mean"
    assert set(AGGREGATE_METHODS) == {"arithmetic_mean", "median"}


def test_median_is_available_and_is_not_the_mean(origin):
    """views-hydranet ADR-021 allows median. If a model ever declares it we must honour it."""
    raw = np.load(origin / "lr_sb_best" / "y_pred.npy")
    df = collapse_origin(origin, aggregate_method="median")
    np.testing.assert_allclose(
        df["pred_lr_sb_best"].to_numpy(), np.median(raw.astype("float64"), axis=1), rtol=1e-12
    )
    assert not np.allclose(df["pred_lr_sb_best"], raw.astype("float64").mean(axis=1))


def test_mutation_unknown_aggregate_method_is_refused(origin):
    """A typo must not fall back to the mean and ship an estimator nobody chose."""
    with pytest.raises(CollapseError, match="unknown aggregate_method"):
        collapse_origin(origin, aggregate_method="geometric_mean")


def test_mutation_log_space_in_ONE_target_only_is_refused(tmp_path):
    """The failure the flattened check could not see.

    `test_mutation_log_space_values_are_refused` puts all three targets in log1p space, so a
    guard on the maximum of the three flattened together passes it just as well as a per-target
    one — it cannot tell the two designs apart. An upstream scaler mismatch does not have to hit
    all three targets. When it hits one, a healthy sibling carries the combined maximum over the
    threshold and the corrupted target ships as log1p(count).
    """
    model = tmp_path / "m"
    origin = model / "data" / "generated" / "predictions_calibration_20260101_000000" / "origin_0"
    counts = np.full((ROWS, DRAWS), 900.0)       # honest counts
    logged = np.full((ROWS, DRAWS), 6.8)         # log1p(900) — the corrupted one
    _write_target(origin, "lr_sb_best", counts)
    _write_target(origin, "lr_ns_best", logged)
    _write_target(origin, "lr_os_best", counts)
    with pytest.raises(CollapseError, match="pred_lr_ns_best"):
        convert_model(model, out_dir=tmp_path / "out")


def test_the_scale_guard_names_the_offending_target(tmp_path):
    """Refusing is only useful if it says which target to go and look at."""
    model = tmp_path / "m"
    origin = model / "data" / "generated" / "predictions_calibration_20260101_000000" / "origin_0"
    for target in TARGETS:
        _write_target(origin, target, np.full((ROWS, DRAWS), 900.0))
    _write_target(origin, "lr_os_best", np.full((ROWS, DRAWS), 2.0))
    with pytest.raises(CollapseError) as exc:
        convert_model(model, out_dir=tmp_path / "out")
    assert "pred_lr_os_best" in str(exc.value)
    assert "log1p" in str(exc.value)


def test_mutation_duplicate_identifier_rows_are_refused(origin):
    """All three targets agreeing on a duplicated key is still a duplicate.

    Cross-target row alignment compares the targets to each other, so it is blind to a
    duplicate they share. `ensemble-updater` joins on (priogrid_id, month_id); a repeated pair
    silently wins or loses that join.
    """
    for target in TARGETS:
        d = origin / target
        with np.load(d / "identifiers.npz") as ids:
            month, unit = ids["time"].copy(), ids["unit"].copy()
        month[5], unit[5] = month[4], unit[4]      # row 5 now repeats row 4
        np.savez(d / "identifiers.npz", time=month, unit=unit)
    with pytest.raises(CollapseError, match="duplicate"):
        collapse_origin(origin)
