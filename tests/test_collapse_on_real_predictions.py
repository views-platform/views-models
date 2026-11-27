"""Run the converter over whatever real HydraNet output this machine happens to hold.

The synthetic suite proves the guards fire. This one proves the converter survives contact with
the arrays a real run writes — 16 draws, 13 time-shifted origins, a field that is 98% zeros, and
the `by_*` targets sitting in the same directory as the `lr_*` ones we actually want.

It skips when there is no prediction output, so a clean checkout and CI stay green
(see views-models#505). It is not a substitute for eyeballing
`python -m tools.collapse.plot_collapse_audit`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.collapse.collapse_predictions import TARGETS, collapse_origin

MODELS = Path(__file__).resolve().parent.parent / "models"


def _origins() -> list[Path]:
    found = []
    for model in sorted(MODELS.iterdir()) if MODELS.is_dir() else []:
        runs = sorted((model / "data" / "generated").glob("predictions_*_*"))
        for run in runs[-1:]:
            found.extend(sorted(p for p in run.glob("origin_*") if (p / TARGETS[0]).is_dir()))
    return found


REAL = _origins()
pytestmark = pytest.mark.skipif(not REAL, reason="no prediction output on this machine")


@pytest.fixture(scope="module")
def sample() -> tuple[Path, pd.DataFrame]:
    origin = REAL[0]
    return origin, collapse_origin(origin)


def test_real_output_has_the_delivered_shape(sample):
    _, frame = sample
    assert list(frame.columns) == ["month_id", "priogrid_id"] + [f"pred_{t}" for t in TARGETS]
    assert (frame[[f"pred_{t}" for t in TARGETS]] >= 0).all().all()
    assert np.isfinite(frame[[f"pred_{t}" for t in TARGETS]].to_numpy()).all()


def test_every_cell_month_pair_appears_once(sample):
    """`ensemble-updater` keys on (unit, month). A duplicate key silently wins or loses a join."""
    _, frame = sample
    assert not frame.duplicated(["month_id", "priogrid_id"]).any()


def test_the_field_is_mostly_zero_but_not_entirely(sample):
    """A field of all zeros, or one with no zeros, means the gate did not run. Both have shipped."""
    origin, frame = sample
    zero = float((frame["pred_lr_sb_best"] == 0).mean())
    assert 0.5 < zero < 0.9999, f"{origin}: {zero:.4f} of cells are zero"


def test_the_collapse_is_not_just_the_first_draw(sample):
    """If mean == draw 0 everywhere, the draw axis is degenerate and D x K bought nothing."""
    origin, frame = sample
    draws = np.load(origin / "lr_sb_best" / "y_pred.npy", mmap_mode="r")
    first = np.asarray(draws[:, 0], dtype="float64")
    assert not np.allclose(frame["pred_lr_sb_best"].to_numpy(), first)


def test_by_targets_are_not_picked_up(sample):
    """`by_*` (single-cause decompositions) sit beside `lr_*`; only `lr_*` is the deliverable."""
    origin, frame = sample
    if (origin / "by_sb_best").is_dir():
        assert not [c for c in frame.columns if c.startswith("pred_by_")]


def test_every_origin_on_this_machine_converts():
    """The converter must not depend on which model or gate type produced the draws."""
    failures = []
    for origin in REAL:
        try:
            collapse_origin(origin)
        except Exception as exc:  # noqa: BLE001 - we want the full list, not the first
            failures.append(f"{origin}: {exc}")
    assert not failures, "\n".join(failures)
