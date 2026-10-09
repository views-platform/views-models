"""The darts converter for views-models#533 must refuse anything it cannot vouch for.

Same two kinds of test as `tests/test_collapse_predictions.py`. The **contract** tests fix what a
correct conversion produces. The **mutation** tests corrupt an input the way a real run could and
require the converter to raise.

The single most important test in this file is
`test_a_multi_sample_cell_becomes_the_mean_and_NOT_the_first_draw`. Everything else here guards
against a crash or a refusal; that one guards against the delivery being quietly wrong. ADR-023
records the consumer's behaviour: `ensemble-updater`'s `_as_float_prediction_array` "takes
`float(x[0])` on a list cell — **one draw, silently**". If this converter did the same, nothing
downstream would notice: the numbers would be plausible, the schema valid, the metrics computable.
So the fixture deliberately makes draw zero an outlier, and the assertion is that the output is
the mean and is *not* that outlier.

Every fixture is synthetic and tiny. The real row count (2 333 448) is checked by the production
default and switched off here with `expected_rows=None`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.collapse.collapse_darts_predictions import (
    EXPECTED_ORIGINS,
    EXPECTED_ROWS,
    MIN_PLAUSIBLE_MAX,
    RENAME_TO_HYDRANET,
    DartsCollapseError,
    collapse_parquet,
    convert_model,
)

ROWS = 40
TARGETS = ("pred_lr_ged_sb", "pred_lr_ged_ns", "pred_lr_ged_os")


def _cells(values: np.ndarray) -> list[list[float]]:
    """(rows, samples) -> the list-in-cell format views_r2darts2 writes."""
    return [list(map(float, row)) for row in np.atleast_2d(values)]


def _write_parquet(
    path: Path,
    columns: dict[str, object],
    *,
    month: np.ndarray | None = None,
    unit: np.ndarray | None = None,
    as_index: bool = True,
) -> Path:
    """Write one run parquet the way `prediction_frames_to_dataframe` leaves it."""
    n = len(next(iter(columns.values())))
    frame = pd.DataFrame(columns)
    frame["month_id"] = np.arange(n, dtype="int64") % 10 + 500 if month is None else month
    frame["priogrid_id"] = np.arange(n, dtype="int64") // 10 + 62000 if unit is None else unit
    if as_index:
        frame = frame.set_index(["month_id", "priogrid_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def _counts(rng: np.random.Generator, rows: int, samples: int, scale: float = 60.0) -> np.ndarray:
    """Zero-inflated, heavy-tailed counts — the shape a conflict field actually has."""
    v = rng.gamma(shape=0.2, scale=scale, size=(rows, samples))
    v[rng.random((rows, samples)) < 0.7] = 0.0
    v[0, :] = scale * 8  # guarantee the per-target maximum clears MIN_PLAUSIBLE_MAX
    return v


@pytest.fixture
def deterministic(tmp_path: Path) -> Path:
    """One well-formed origin from a `num_samples: 1` model — every cell a list of one."""
    rng = np.random.default_rng(0)
    cols = {t: _cells(_counts(rng, ROWS, 1, scale=60.0 * (i + 1))) for i, t in enumerate(TARGETS)}
    return _write_parquet(tmp_path / "predictions_calibration_20260101_000000_00.parquet", cols)


def _write_run(
    generated: Path,
    timestamp: str = "20260101_000000",
    n_origins: int = EXPECTED_ORIGINS,
    samples: int = 1,
    sequences: list[int] | None = None,
) -> Path:
    """A whole run's worth of parquets under <model>/data/generated."""
    rng = np.random.default_rng(1)
    for seq in sequences if sequences is not None else range(n_origins):
        cols = {
            t: _cells(_counts(rng, ROWS, samples, scale=60.0 * (i + 1)))
            for i, t in enumerate(TARGETS)
        }
        _write_parquet(
            generated / f"predictions_calibration_{timestamp}_{seq:02d}.parquet", cols
        )
    return generated


# ── contract ──────────────────────────────────────────────────────────────────────


def test_columns_and_order_are_the_specification(deterministic):
    df = collapse_parquet(deterministic, expected_rows=None)
    assert list(df.columns) == ["month_id", "priogrid_id", *TARGETS]
    assert df["month_id"].dtype == "int64" and df["priogrid_id"].dtype == "int64"
    for t in TARGETS:
        assert df[t].dtype == "float64"
    assert len(df) == ROWS


def test_the_keys_come_out_as_flat_columns_not_an_index(deterministic):
    """views-models#505: 'flat column, not an index'. The run writes them as an index."""
    raw = pd.read_parquet(deterministic)
    assert list(raw.index.names) == ["month_id", "priogrid_id"], "fixture is not index-shaped"
    df = collapse_parquet(deterministic, expected_rows=None)
    assert df.index.names == pd.RangeIndex(0).names
    assert "month_id" in df.columns and "priogrid_id" in df.columns


def test_keys_already_flat_are_accepted_too(tmp_path):
    rng = np.random.default_rng(2)
    cols = {t: _cells(_counts(rng, ROWS, 1)) for t in TARGETS}
    p = _write_parquet(tmp_path / "predictions_calibration_20260101_000000_00.parquet", cols,
                       as_index=False)
    df = collapse_parquet(p, expected_rows=None)
    assert len(df) == ROWS and "month_id" in df.columns


def test_a_single_sample_cell_is_unwrapped_not_averaged(tmp_path):
    """A deterministic model's one value IS the value. Averaging it would imply a posterior."""
    values = np.array([[3.5], [0.0], [480.0], [1.25]])
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(values)},
    )
    df = collapse_parquet(p, expected_rows=None, min_plausible_max=0)
    assert df[TARGETS[0]].to_list() == [3.5, 0.0, 480.0, 1.25]


def test_a_multi_sample_cell_becomes_the_mean_and_NOT_the_first_draw(tmp_path):
    """THE test this module exists for.

    Draw zero is a deliberate outlier. `ensemble-updater` would take exactly that value and
    report metrics on it without error (ADR-023). A converter that forwarded `x[0]`, or that
    forgot to collapse at all, passes every other test in this file and fails this one.
    """
    samples = np.tile(np.array([0.0, 100.0, 200.0, 300.0]), (4, 1))
    samples[:, 0] = 999.0  # the outlier a `float(x[0])` consumer would publish
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(samples)},
    )
    df = collapse_parquet(p, expected_rows=None)

    expected = float(np.mean([999.0, 100.0, 200.0, 300.0]))
    assert df[TARGETS[0]].to_list() == [expected] * 4
    assert not np.allclose(df[TARGETS[0]], 999.0), "the first draw was forwarded, not the mean"


def test_the_mean_accumulates_in_float64(tmp_path):
    """float32 sums make the answer depend on the draw count (collapse_predictions.py:154)."""
    samples = np.full((2, 600), 0.1, dtype="float64")
    samples[:, 0] = 20.0
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(samples)},
    )
    df = collapse_parquet(p, expected_rows=None, min_plausible_max=0)
    assert df[TARGETS[0]].iloc[0] == pytest.approx(samples[0].mean(), rel=0, abs=1e-12)


def test_zeros_survive_the_collapse(deterministic):
    df = collapse_parquet(deterministic, expected_rows=None)
    assert (df[TARGETS[0]] == 0).any(), "the zero-inflated majority vanished"
    assert (df[list(TARGETS)] >= 0).all().all()


def test_identifiers_are_passed_through_unreordered(deterministic):
    raw = pd.read_parquet(deterministic).reset_index()
    df = collapse_parquet(deterministic, expected_rows=None)
    assert df["month_id"].to_list() == raw["month_id"].to_list()
    assert df["priogrid_id"].to_list() == raw["priogrid_id"].to_list()


def test_the_target_set_is_read_off_the_file_not_hard_coded(tmp_path):
    """views-models#151: derive from the data. A model with other targets needs no change here."""
    rng = np.random.default_rng(3)
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {"pred_something_else": _cells(_counts(rng, ROWS, 1))},
    )
    df = collapse_parquet(p, expected_rows=None)
    assert list(df.columns) == ["month_id", "priogrid_id", "pred_something_else"]


def test_convert_model_writes_one_parquet_per_origin(tmp_path):
    _write_run(tmp_path / "data" / "generated")
    written = convert_model(tmp_path, expected_rows=None)
    assert len(written) == EXPECTED_ORIGINS
    assert [p.name[-11:] for p in written] == [f"_{i:02d}.parquet" for i in range(13)]
    for p in written:
        assert p.is_file()
        assert list(pd.read_parquet(p).columns) == ["month_id", "priogrid_id", *TARGETS]


def test_the_deliverable_does_not_overwrite_the_run_output(tmp_path):
    """Source and deliverable share one filename pattern — the HydraNet path has no such clash."""
    generated = _write_run(tmp_path / "data" / "generated", n_origins=2)
    before = {p.name: p.read_bytes() for p in generated.glob("*.parquet")}
    written = convert_model(tmp_path, expected_rows=None, expected_origins=None)
    assert all(p.parent != generated for p in written), "wrote into the source directory"
    after = {p.name: p.read_bytes() for p in generated.glob("*.parquet")}
    assert before == after, "the run's own parquets were modified"


def test_writing_into_the_source_directory_is_refused(tmp_path):
    generated = _write_run(tmp_path / "data" / "generated", n_origins=2)
    with pytest.raises(DartsCollapseError, match="refusing to write"):
        convert_model(tmp_path, out_dir=generated, expected_rows=None, expected_origins=None)


def test_the_latest_timestamp_wins(tmp_path):
    """A second `-e` on the same artifact writes a second set beside the first."""
    generated = tmp_path / "data" / "generated"
    _write_run(generated, timestamp="20260101_000000", n_origins=2)
    _write_run(generated, timestamp="20260201_000000", n_origins=2)
    written = convert_model(tmp_path, expected_rows=None, expected_origins=None)
    assert all("20260201_000000" in p.name for p in written)


def test_rename_is_off_by_default_and_available_on_request(tmp_path):
    _write_run(tmp_path / "data" / "generated", n_origins=1)
    plain = convert_model(tmp_path, expected_rows=None, expected_origins=None)
    assert list(pd.read_parquet(plain[0]).columns)[2:] == list(TARGETS)

    renamed = convert_model(
        tmp_path,
        out_dir=tmp_path / "renamed",
        expected_rows=None,
        expected_origins=None,
        rename=True,
    )
    assert list(pd.read_parquet(renamed[0]).columns)[2:] == [
        RENAME_TO_HYDRANET[t] for t in TARGETS
    ]


def test_the_production_defaults_are_the_real_run_geometry():
    """If the partition is bumped these must be revisited, not quietly wrong."""
    assert EXPECTED_ROWS == 36 * 64_818
    assert EXPECTED_ORIGINS == 48 - 36 + 1


# ── mutation ──────────────────────────────────────────────────────────────────────


def test_mutation_nan_in_a_cell(tmp_path):
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(np.array([[500.0], [np.nan]]))},
    )
    with pytest.raises(DartsCollapseError, match="non-finite"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_negative_values(tmp_path):
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(np.array([[500.0], [-1.0]]))},
    )
    with pytest.raises(DartsCollapseError, match="negative"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_ragged_cells(tmp_path):
    """A varying sample count row to row has no single collapse to apply."""
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: [[1.0, 2.0], [3.0]]},
    )
    with pytest.raises(DartsCollapseError, match="differing length"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_empty_cells(tmp_path):
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: [[], []]},
    )
    with pytest.raises(DartsCollapseError, match="empty cells"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_duplicate_identifier_rows_are_refused(tmp_path):
    """ensemble-updater joins on the pair, so a repeat silently wins or loses the join."""
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(np.array([[500.0], [400.0]]))},
        month=np.array([500, 500], dtype="int64"),
        unit=np.array([62000, 62000], dtype="int64"),
    )
    with pytest.raises(DartsCollapseError, match="duplicate"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_wrong_row_count(deterministic):
    with pytest.raises(DartsCollapseError, match="rows, expected"):
        collapse_parquet(deterministic, expected_rows=EXPECTED_ROWS)


def test_mutation_log_space_values_are_refused(tmp_path):
    rng = np.random.default_rng(4)
    small = rng.random((ROWS, 1)) * 5.0  # log1p of a few hundred is ~5-6
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(small)},
    )
    with pytest.raises(DartsCollapseError, match="inverse transform did not run"):
        collapse_parquet(p, expected_rows=None)


def test_a_single_small_target_beside_a_large_one_is_ACCEPTED(tmp_path):
    """The guarantee this converter deliberately gives up, asserted so it cannot be lost.

    The HydraNet converter checks the scale PER TARGET, because its per-target scaler registry
    can leave one target in log1p space while its siblings invert correctly. r2darts2 has no
    such registry — one `target_scaler` chain covers every target — so that failure cannot
    occur here, and the per-target form instead refuses correct output.

    It did exactly that on the first real run: `dark_river` at global pgm, 2026-10-07, refused
    at origin 06 because `pred_lr_ged_os` peaked at 11.46, after 87 minutes of GPU time, on a
    frame whose `pred_lr_ged_sb` reached 283.68. One-sided violence is rare and a deterministic
    point model shrinks hard; small is the model being timid, not the scaler being broken.

    So this asserts the NEW behaviour, not the old: a rare target may be small beside a large
    sibling. **The trigger to restore the per-target check is an r2darts2 release that scales
    targets independently** — at which point this test should fail and be rewritten, which is
    the point of asserting it rather than deleting the old one.
    """
    rng = np.random.default_rng(5)
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {
            TARGETS[0]: _cells(_counts(rng, ROWS, 1)),      # plainly counts
            TARGETS[1]: _cells(rng.random((ROWS, 1)) * 5.0),  # rare and shrunk
        },
    )
    df = collapse_parquet(p, expected_rows=None)
    assert len(df) == ROWS
    assert df[TARGETS[1]].max() < 12.0, "fixture no longer exercises the small-target case"


def test_a_frame_where_NO_target_reaches_a_plausible_count_is_still_refused(tmp_path):
    """The guarantee that is kept: if the inverse transform did not run, nothing is large."""
    rng = np.random.default_rng(9)
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {
            TARGETS[0]: _cells(rng.random((ROWS, 1)) * 5.0),
            TARGETS[1]: _cells(rng.random((ROWS, 1)) * 4.0),
            TARGETS[2]: _cells(rng.random((ROWS, 1)) * 6.0),
        },
    )
    with pytest.raises(DartsCollapseError) as exc:
        collapse_parquet(p, expected_rows=None)
    msg = str(exc.value)
    assert "WHOLE frame" in msg, "the refusal does not say it is frame-level"
    assert "one scaler chain" in msg, "the refusal does not give its reason"


def test_the_scale_guard_can_be_disabled_deliberately(tmp_path):
    rng = np.random.default_rng(6)
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(rng.random((ROWS, 1)) * 5.0)},
    )
    assert len(collapse_parquet(p, expected_rows=None, min_plausible_max=0)) == ROWS


def test_mutation_no_prediction_column(tmp_path):
    """A metric frame or an eval parquet must not be mistaken for predictions."""
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {"Brier_cls_sample": [0.1, 0.2]},
    )
    with pytest.raises(DartsCollapseError, match="no 'pred_\\*' column"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_keys_are_neither_columns_nor_index(tmp_path):
    frame = pd.DataFrame({TARGETS[0]: [[1.0], [2.0]], "wrong_key": [1, 2]})
    p = tmp_path / "predictions_calibration_20260101_000000_00.parquet"
    frame.to_parquet(p, index=False)
    with pytest.raises(DartsCollapseError, match="cannot find"):
        collapse_parquet(p, expected_rows=None)


def test_mutation_missing_file(tmp_path):
    with pytest.raises(DartsCollapseError, match="missing"):
        collapse_parquet(tmp_path / "nope.parquet", expected_rows=None)


def test_mutation_a_gap_in_the_sequence_numbers(tmp_path):
    """ensemble-updater raises FileNotFoundError naming a missing origin. So must we."""
    _write_run(tmp_path / "data" / "generated", sequences=[0, 1, 3])
    with pytest.raises(DartsCollapseError, match="contiguous"):
        convert_model(tmp_path, expected_rows=None, expected_origins=None)


def test_mutation_wrong_number_of_origins(tmp_path):
    _write_run(tmp_path / "data" / "generated", n_origins=4)
    with pytest.raises(DartsCollapseError, match="origin"):
        convert_model(tmp_path, expected_rows=None)


def test_mutation_no_prediction_parquets_at_all(tmp_path):
    (tmp_path / "data" / "generated").mkdir(parents=True)
    with pytest.raises(DartsCollapseError, match="prediction_frame"):
        convert_model(tmp_path, expected_rows=None)


def test_a_prediction_frame_layout_is_not_silently_accepted(tmp_path):
    """`prediction_format: "prediction_frame"` writes DIRECTORIES of the same name (#492)."""
    generated = tmp_path / "data" / "generated"
    (generated / "predictions_calibration_20260101_000000" / "origin_0").mkdir(parents=True)
    with pytest.raises(DartsCollapseError, match="collapse_predictions.py"):
        convert_model(tmp_path, expected_rows=None)


def test_the_scale_refusal_carries_the_measurement_that_calibrates_it(tmp_path):
    """The threshold used to be unvalidated; it no longer is, and the message must say what by.

    The previous version of this test asserted the message admitted it had "never been
    validated against a darts run" — correct then, and the honest thing to say when nothing
    had been measured. `dark_river` measured it on 2026-10-07: frame maximum 283.68 against a
    rarest-target maximum of 14.94. An operator who trips this guard now needs that number, not
    an apology: it is what tells them whether their frame maximum of 9 is a broken scaler or a
    very timid model.
    """
    assert MIN_PLAUSIBLE_MAX == 12.0
    rng = np.random.default_rng(7)
    p = _write_parquet(
        tmp_path / "predictions_calibration_20260101_000000_00.parquet",
        {TARGETS[0]: _cells(rng.random((ROWS, 1)) * 5.0)},
    )
    with pytest.raises(DartsCollapseError) as exc:
        collapse_parquet(p, expected_rows=None)
    message = str(exc.value)
    assert "283.68" in message, "the refusal does not carry the measured reference point"
    assert "dark_river" in message, "the refusal does not say which run calibrated it"
    assert "one scaler chain" in message, "the refusal does not justify being frame-level"
