"""S5 (#178) — the parity / conservation gate.

Proves the *wired* reconciler (factory → views_frames_reconcile ReconciliationModule)
reconciles grid forecasts to CM country totals (the conservation invariant) and
preserves the all-zero-country edge case — i.e. the wiring computes the right thing.
A real-viewser-geography build is exercised skip-when-unavailable.
"""
import re

import numpy as np
import pytest

from tests.live_deadline import (
    VIEWSER_DEADLINE_SECONDS,
    DeadlineExceeded,
    deadline,
)

# reconciliation/__init__.py imports composition, which needs pipeline-core's Reconciler
# port (3.0.0+, unreleased). Guard BEFORE any reconciliation import: placed after one,
# collection ERRORS instead of skipping, which is what turned CI red (ADR-005 §skip).
pytest.importorskip("views_pipeline_core.domain.reconciliation_port")

from reconciliation.composition import build_reconciler_for_run
from reconciliation.country_mapping import CountryMapping
from reconciliation.reconciler_factory import build_reconciler

pytestmark = pytest.mark.green

SpatialLevel = pytest.importorskip("views_frames").SpatialLevel
prediction_frame_from_arrays = pytest.importorskip(
    "views_frames_reconcile.frames"
).prediction_frame_from_arrays


class _FixedProvider:
    def __init__(self, map_keys, map_vals):
        self._mapping = CountryMapping(map_keys, map_vals)

    def build(self) -> CountryMapping:
        return self._mapping


def _frames(pg_unit, pg_country, pg_vals, cm_unit, cm_vals, month=480):
    n_pg, n_cm = len(pg_unit), len(cm_unit)
    pgm = prediction_frame_from_arrays(
        np.full(n_pg, month, dtype=np.int64),
        np.asarray(pg_unit, dtype=np.int64),
        np.asarray(pg_vals, dtype=np.float32),
        level=SpatialLevel.PGM,
    )
    cm = prediction_frame_from_arrays(
        np.full(n_cm, month, dtype=np.int64),
        np.asarray(cm_unit, dtype=np.int64),
        np.asarray(cm_vals, dtype=np.float32),
        level=SpatialLevel.CM,
    )
    map_keys = np.stack(
        [np.full(n_pg, month, dtype=np.int64), np.asarray(pg_unit, dtype=np.int64)], axis=1
    )
    return pgm, cm, map_keys, np.asarray(pg_country, dtype=np.int64)


def test_wired_reconciler_conserves_grid_to_cm_country_totals():
    # grids 100,101 -> country 10; grid 200 -> country 20; 2 samples.
    pgm, cm, mk, mv = _frames(
        pg_unit=[100, 101, 200], pg_country=[10, 10, 20],
        pg_vals=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        cm_unit=[10, 20], cm_vals=[[8.0, 12.0], [10.0, 12.0]],
    )
    reconciler = build_reconciler(480, 480, provider=_FixedProvider(mk, mv))
    out = reconciler.reconcile(cm, pgm).values  # (3, 2), pgm row order
    np.testing.assert_allclose(out[[0, 1]].sum(axis=0), [8.0, 12.0], rtol=1e-5)  # country 10
    np.testing.assert_allclose(out[2], [10.0, 12.0], rtol=1e-5)  # country 20


def _frames_version() -> tuple[int, ...]:
    """The INSTALLED views-frames version, or (0,) when it cannot be read.

    (0,) selects the pre-3.0.0 branch, which asserts the mass LOSS — so an unreadable
    version can never make this test claim the fix is present when it is not (ADR-025).
    """
    try:
        import importlib.metadata as md

        return tuple(int(n) for n in re.findall(r"\d+", md.version("views-frames"))[:3])
    except Exception:
        return (0,)


def test_an_all_zero_country_draw_is_filled_from_a_donor_on_frames_3():
    """What happens to a country whose grid is all-zero in one draw — by VERSION.

    ⛔ This test ASSERTED THE LOSS until 2026-11-02. It was named
    `test_all_zero_country_sample_stays_zero` and pinned `out[:, 0] == [0, 0]` as a
    "documented edge case", which encodes the defect as the expectation. The mass it was
    content to watch vanish is the same 2.9-7.3% that register C-192 carries at Tier 1 and
    that views-frames#283 fixed — so on the stack the pod lane now floors, this test fails,
    and it fails by asserting that the fix did not happen. Found by crafdapi's /falsify,
    views-models#675.

    ⚠️ Both branches assert, and neither skips. CI resolves views-frames 2.0.0 today while
    the pod lane floors 3.0.0, so a test that skipped on the version it did not like would
    report a pass for a behaviour nobody checked — which is the whole of ADR-025.
    """
    pgm, cm, mk, mv = _frames(
        pg_unit=[100, 101], pg_country=[10, 10],
        pg_vals=[[0.0, 2.0], [0.0, 4.0]],
        cm_unit=[10], cm_vals=[[5.0, 12.0]],
    )
    reconciler = build_reconciler(480, 480, provider=_FixedProvider(mk, mv))
    out = reconciler.reconcile(cm, pgm).values

    # Draw 1's grid is non-zero, so it must conserve on every version.
    np.testing.assert_allclose(out[:, 1].sum(), 12.0, rtol=1e-5)

    if _frames_version() >= (3, 0, 0):
        # Donor fill: draw 0's all-zero grid takes the pattern of another draw of the same
        # (country, month), so the country's 5.0 reaches the grid instead of disappearing.
        np.testing.assert_allclose(out[:, 0].sum(), 5.0, rtol=1e-5)
        assert (out[:, 0] >= 0).all(), "donor fill must not produce negative cells"
    else:
        # Pre-3.0.0: multiplicative scaling cannot lift an all-zero grid, so the mass is
        # lost. Asserted so the old stack stays covered, and labelled as the defect it is
        # rather than as a "documented edge case".
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
        assert out[:, 0].sum() == 0.0, (
            "pre-3.0.0 loses this country's 5.0 entirely — views-frames#283, C-192"
        )


@pytest.mark.red
@pytest.mark.live
def test_real_viewser_geography_wires_end_to_end():
    try:
        import viewser  # noqa: F401
    except ImportError as e:
        pytest.skip(f"viewser unavailable: {e}")
    from pathlib import Path

    reconciler_port = pytest.importorskip(
        "views_pipeline_core.domain.reconciliation_port"
    ).Reconciler
    skinny_love = Path(__file__).resolve().parent.parent / "ensembles" / "skinny_love"
    try:
        # Bounded — see tests/live_deadline.py and #409.
        with deadline(VIEWSER_DEADLINE_SECONDS, "viewser geography fetch"):
            reconciler = build_reconciler_for_run(skinny_love)
    except DeadlineExceeded as e:  # before the bare Exception — it is one
        pytest.skip(str(e))
    except Exception as e:  # network/credentials/schema
        pytest.skip(f"viewser geography fetch failed: {type(e).__name__}: {e}")
    assert isinstance(reconciler, reconciler_port)
