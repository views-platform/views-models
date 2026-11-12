"""Viewser country-mapping provider — the parity source (EPIC #172 / S2 #175).

Builds `(time, priogrid_gid) -> VIEWS country_id` for a forecast window, matching
the **current** reconciliation behaviour bit-for-bit. The current path
(views-reporting `metadata.build_country_to_grids_cache` → `get_country_id` →
`build_pg_metadata_cache`) sources `country_id` from a `priogrid_month` queryset
column `Column("country_id", from_loa="country_month", from_column="country_id")`
and takes the **first** country per grid (`.groupby(grid).first()`, time-invariant).
We reproduce exactly that, so swapping the wiring does not change the numbers.

The viewser fetch is injected (a callable) so the build logic is unit-testable
without viewser, and the viewser/pandas coupling stays isolated to one default.
A datafactory provider (`gaul0_code`) is a separate class (different id system).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from reconciliation.country_mapping import CountryMapping

# (start_month, end_month) -> DataFrame indexed by (month_id, priogrid_gid) with a
# "country_id" column (VIEWS country_id).
FetchCountryMetadata = Callable[[int, int], "object"]


# ─── TRANSITIONAL (C-89) ──────────────────────────────────────────────────────
# This provider depends on viewser + pandas — both being PHASED OUT for
# views-datafactory / views-frames. It is the parity source for *viewser*-sourced
# ensembles only; a datafactory `gaul0_code` provider (#196) is selected per
# ensemble for datafactory-sourced ones. Do NOT extend the viewser/pandas surface
# here — and the whole reconciliation algorithm is slated to move to a
# `views-frames-reconciler` sister package. See risk register C-89.
# ──────────────────────────────────────────────────────────────────────────────
class ViewserCountryMappingProvider:
    """Builds the VIEWS-`country_id` geography mapping for a forecast window."""

    def __init__(
        self,
        start_month: int,
        end_month: int,
        fetch_country_metadata: Optional[FetchCountryMetadata] = None,
    ) -> None:
        self._start_month = start_month
        self._end_month = end_month
        self._fetch = fetch_country_metadata or self._fetch_from_viewser

    def build(self) -> CountryMapping:
        df = self._fetch(self._start_month, self._end_month)
        # Time-invariant canonical country per grid — parity with views-reporting's
        # build_country_to_grids_cache (.groupby(grid)["country_id"].first()).
        gids_idx = df.index.get_level_values(1)
        grid_country = df.groupby(gids_idx)["country_id"].first()

        months = np.asarray(df.index.get_level_values(0), dtype=np.int64)
        gids = np.asarray(df.index.get_level_values(1), dtype=np.int64)
        map_keys = np.stack([months, gids], axis=1)
        map_vals = grid_country.reindex(gids).to_numpy().astype(np.int64)
        return CountryMapping(map_keys, map_vals)

    @staticmethod
    def _fetch_from_viewser(start_month: int, end_month: int):
        """The parity fetch — the same column the current reconciliation path uses."""
        from viewser import Column, Queryset

        queryset = Queryset("recon_pg_country_map", "priogrid_month").with_column(
            Column("country_id", from_loa="country_month", from_column="country_id")
        )
        return queryset.publish().fetch(start_date=start_month, end_date=end_month)
