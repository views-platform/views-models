"""S1 (#174) — `CountryMapping` value invariants + `CountryMappingProvider` port.

Pure value/port contracts — no viewser/postprocessing dependency.
"""
import numpy as np
import pytest

from reconciliation.country_mapping import CountryMapping
from reconciliation.country_mapping_provider import CountryMappingProvider

pytestmark = pytest.mark.green


def _valid_arrays():
    keys = np.array([[480, 100], [480, 101], [481, 100]], dtype=np.int64)
    vals = np.array([10, 10, 10], dtype=np.int64)
    return keys, vals


def test_valid_mapping_constructs_and_holds_arrays():
    keys, vals = _valid_arrays()
    cm = CountryMapping(keys, vals)
    assert len(cm) == 3
    np.testing.assert_array_equal(cm.map_keys, keys)
    np.testing.assert_array_equal(cm.map_vals, vals)


def test_rejects_keys_not_M_by_2():
    with pytest.raises(ValueError):
        CountryMapping(np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64))


def test_rejects_vals_misaligned_with_keys():
    keys, _ = _valid_arrays()
    with pytest.raises(ValueError):
        CountryMapping(keys, np.array([10, 10], dtype=np.int64))


def test_rejects_non_integer_arrays():
    keys, vals = _valid_arrays()
    with pytest.raises(ValueError):
        CountryMapping(keys.astype(float), vals)


def test_is_frozen():
    cm = CountryMapping(*_valid_arrays())
    with pytest.raises(Exception):
        cm.map_vals = np.array([1], dtype=np.int64)


def test_provider_port_is_runtime_checkable():
    class _Provider:
        def build(self) -> CountryMapping:
            return CountryMapping(*_valid_arrays())

    class _NotAProvider:
        pass

    assert isinstance(_Provider(), CountryMappingProvider)
    assert not isinstance(_NotAProvider(), CountryMappingProvider)
