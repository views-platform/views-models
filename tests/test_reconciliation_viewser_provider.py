"""S2 (#175) — ViewserCountryMappingProvider.

Build logic is unit-tested with an injected fake fetch (no viewser); a real
viewser fetch is exercised in a skip-when-unavailable integration test.
"""
import numpy as np
import pandas as pd
import pytest

from reconciliation.country_mapping import CountryMapping
from reconciliation.country_mapping_provider import CountryMappingProvider
from reconciliation.viewser_country_mapping_provider import ViewserCountryMappingProvider

pytestmark = pytest.mark.green


def _meta_df(rows):
    """rows: list of (month_id, priogrid_gid, country_id)."""
    idx = pd.MultiIndex.from_tuples(
        [(m, g) for m, g, _ in rows], names=["month_id", "priogrid_gid"]
    )
    return pd.DataFrame({"country_id": [c for *_, c in rows]}, index=idx)


def test_builds_country_mapping_from_fetch():
    fetch = lambda s, e: _meta_df(  # noqa: E731
        [(480, 100, 10), (480, 101, 20), (481, 100, 10), (481, 101, 20)]
    )
    cm = ViewserCountryMappingProvider(480, 481, fetch_country_metadata=fetch).build()
    assert isinstance(cm, CountryMapping)
    np.testing.assert_array_equal(
        cm.map_keys, np.array([[480, 100], [480, 101], [481, 100], [481, 101]])
    )
    np.testing.assert_array_equal(cm.map_vals, np.array([10, 20, 10, 20]))


def test_first_country_per_grid_is_time_invariant_parity():
    # gid 100 maps to country 10 in month 480, then 99 in 481. Parity with
    # views-reporting takes .first() per grid → 10 for BOTH rows.
    fetch = lambda s, e: _meta_df([(480, 100, 10), (481, 100, 99)])  # noqa: E731
    cm = ViewserCountryMappingProvider(480, 481, fetch_country_metadata=fetch).build()
    np.testing.assert_array_equal(cm.map_vals, np.array([10, 10]))


def test_satisfies_provider_port():
    fetch = lambda s, e: _meta_df([(480, 100, 10)])  # noqa: E731
    provider = ViewserCountryMappingProvider(480, 480, fetch_country_metadata=fetch)
    assert isinstance(provider, CountryMappingProvider)


@pytest.mark.red
def test_viewser_fetch_integration():
    try:
        import viewser  # noqa: F401
    except ImportError as e:
        pytest.skip(f"viewser unavailable: {e}")
    try:
        cm = ViewserCountryMappingProvider(480, 481).build()
    except Exception as e:  # network/credentials/schema
        pytest.skip(f"viewser fetch failed: {type(e).__name__}: {e}")
    assert len(cm) > 0
    assert cm.map_keys.shape[1] == 2
