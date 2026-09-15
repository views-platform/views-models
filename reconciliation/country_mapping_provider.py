"""The `CountryMappingProvider` port (EPIC #172 / ADR-014).

The geography source differs by data source — VIEWS `country_id` (viewser) vs
`gaul0_code` (views-datafactory) — and both coexist during the migration. So the
source is a port: concrete providers (`ViewserCountryMappingProvider` now, a
datafactory provider later) are selected per ensemble, derived from its data
source (ADR-013). A provider encapsulates *how* to obtain the mapping (window,
region, fetch); `build()` takes no arguments and returns the value.

This module is the stable abstraction (SAP/SDP): it imports nothing concrete —
not viewser, not views-postprocessing.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from reconciliation.country_mapping import CountryMapping


@runtime_checkable
class CountryMappingProvider(Protocol):
    """Port: build the `(time, priogrid_gid) -> country_id` mapping for a
    reconciliation run, in the country-id system matching the ensemble's data
    source. Implementations hold whatever inputs they need (window, region)."""

    def build(self) -> CountryMapping:
        """Return the geography mapping covering the run's grid x time universe."""
        ...
