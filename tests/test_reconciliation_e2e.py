"""S5 (#178) — the parity / conservation gate.

Proves the *wired* reconciler (factory → views_frames_reconcile ReconciliationModule)
reconciles grid forecasts to CM country totals (the conservation invariant) and
preserves the all-zero-country edge case — i.e. the wiring computes the right thing.
A real-viewser-geography build is exercised skip-when-unavailable.
"""
import numpy as np
import pytest

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


def test_all_zero_country_sample_stays_zero():
    # country 10 grids are all-zero for sample 0 -> stays zero (documented edge case);
    # sample 1 still conserves to the cm total.
    pgm, cm, mk, mv = _frames(
        pg_unit=[100, 101], pg_country=[10, 10],
        pg_vals=[[0.0, 2.0], [0.0, 4.0]],
        cm_unit=[10], cm_vals=[[5.0, 12.0]],
    )
    reconciler = build_reconciler(480, 480, provider=_FixedProvider(mk, mv))
    out = reconciler.reconcile(cm, pgm).values
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(out[:, 1].sum(), 12.0, rtol=1e-5)


@pytest.mark.red
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
        reconciler = build_reconciler_for_run(skinny_love)
    except Exception as e:  # network/credentials/schema
        pytest.skip(f"viewser geography fetch failed: {type(e).__name__}: {e}")
    assert isinstance(reconciler, reconciler_port)
