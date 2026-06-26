"""S3 (#176) — reconciler_factory.

Verifies the factory returns a valid `Reconciler` (the port), uses the injected
provider, exposes the provider-selection seam, and fails on an unknown source.
Needs views-postprocessing (dev-installed) to construct the concrete.
"""
import numpy as np
import pytest

from reconciliation.country_mapping import CountryMapping
from reconciliation.reconciler_factory import _PROVIDERS, build_reconciler

pytestmark = pytest.mark.green

Reconciler = pytest.importorskip(
    "views_pipeline_core.domain.reconciliation_port"
).Reconciler
pytest.importorskip("views_frames_reconcile")


class _FakeProvider:
    def __init__(self):
        self.built = False

    def build(self) -> CountryMapping:
        self.built = True
        return CountryMapping(
            np.array([[480, 100], [480, 101]], dtype=np.int64),
            np.array([10, 10], dtype=np.int64),
        )


def test_build_reconciler_returns_the_port_type():
    rec = build_reconciler(480, 481, provider=_FakeProvider())
    assert isinstance(rec, Reconciler)  # runtime_checkable: has reconcile(cm, pgm)


def test_uses_the_injected_provider():
    provider = _FakeProvider()
    build_reconciler(480, 481, provider=provider)
    assert provider.built


def test_unknown_source_fails_loud():
    with pytest.raises(ValueError):
        build_reconciler(480, 481, source="datafactory")  # not registered yet


def test_provider_selection_seam_is_open_for_extension():
    # viewser is the only registered source today; a datafactory provider slots
    # in here without changing build_reconciler's signature or its callers.
    assert "viewser" in _PROVIDERS
