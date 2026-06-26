"""S4 (#177) — composition: window sizing + build_reconciler_for_run."""
import numpy as np
import pytest

from reconciliation.composition import (
    _WINDOW_BUFFER_MONTHS,
    _forecast_window,
    build_reconciler_for_run,
)
from reconciliation.country_mapping import CountryMapping

pytestmark = pytest.mark.green

Reconciler = pytest.importorskip(
    "views_pipeline_core.domain.reconciliation_port"
).Reconciler
pytest.importorskip("views_frames_reconcile")


def test_forecast_window_is_union_of_test_ranges_plus_buffer():
    partitions = {
        "calibration": {"train": (121, 456), "test": (457, 504)},
        "validation": {"train": (121, 504), "test": (505, 552)},
        "forecasting": {"train": (121, 557), "test": (558, 594)},
    }
    start, end = _forecast_window(partitions)
    assert start == 457
    assert end == 594 + _WINDOW_BUFFER_MONTHS


def test_forecast_window_requires_test_ranges():
    with pytest.raises(ValueError):
        _forecast_window({"calibration": {"train": (1, 2)}})


class _FakeProvider:
    def build(self) -> CountryMapping:
        return CountryMapping(
            np.array([[480, 100]], dtype=np.int64), np.array([10], dtype=np.int64)
        )


def test_build_reconciler_for_run_returns_the_port(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "config_partitions.py").write_text(
        "def generate():\n"
        "    return {'forecasting': {'train': (121, 557), 'test': (558, 594)}}\n"
    )
    rec = build_reconciler_for_run(tmp_path, provider=_FakeProvider())
    assert isinstance(rec, Reconciler)
