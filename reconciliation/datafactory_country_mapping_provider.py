"""Datafactory country-mapping provider — GAUL `gaul0_code` (EPIC #172 / #196 / S2b #601).

Builds `(time, priogrid_gid) -> gaul0_code` for a forecast window, for
**datafactory-sourced** ensembles. A separate class from
`ViewserCountryMappingProvider` because the id system differs: VIEWS `country_id`
and FAO GAUL `gaul0_code` do not overlap, so the two cannot share an
implementation and must never be substituted for one another (C-49). #196
deferred this provider until a datafactory-sourced reconciling ensemble existed;
epic #595's `chrome_box` is the first, so it is due.

⛔ **This provider is not parity-preserving, by construction.** Grouping by
`gaul0_code` attributes cells to different countries than viewser's `country_id`,
so it cannot be validated by diffing against the viewser provider — #196 is
explicit about that. Its ground truth is `datafactory_query.load_region_pgids`,
which resolves a region to its pgid set from the GAUL admin Parquet files and is
an independent path from the feature fetch this provider reads.

Four things it refuses rather than papers over, because each one would otherwise
surface as a plausible wrong reconciled number:

1. **A cell with no country.** `gaul0_code <= 0` means GAUL gives the cell no
   country, so it cannot be summed into one. `grid_to_country_month` drops those
   silently unless the caller declares `feature_agg_types`. Encountering one here
   means the region was `land` (64,818) rather than `land_gaul` (64,742), and the
   76-cell difference would go on to break epic #595's acceptance criterion
   without ever raising. Refuse and say so.
2. **A cell whose country changes over the window.** GAUL 2024 polygons are
   static per cell, so this cannot legitimately happen. The viewser provider
   collapses time with `.groupby(grid).first()`, but it does that for documented
   bit-for-bit parity with views-reporting — a reason this provider does not have.
   Taking `.first()` here would silently pick one of two contradictory answers.
3. **A missing feature.** If the fetch returns no `gaul0_code` column, the mapping
   cannot be built at all, and an empty mapping reconciles everything to nothing.
4. **A fetch with no rows.** The same hazard through the other door: the column is
   present and empty. Both refusals below pass vacuously on zero rows, so this one is
   checked explicitly rather than inferred from them.

The fetch is injected (a callable) so the build logic is unit-testable without
network or credentials, exactly as `ViewserCountryMappingProvider` does it.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from reconciliation.country_mapping import CountryMapping

#: The datafactory feature carrying the FAO GAUL country code per grid cell.
#: Every epic #595 member renames it in its own queryset (`"gaul0_code": "c_id"`);
#: this provider reads it under its factory name, before any renaming.
GAUL_CODE_FEATURE = "gaul0_code"

#: `land` minus the cells GAUL gives no country. See the module docstring.
DEFAULT_REGION = "land_gaul"

# (start_month, end_month, region) -> a views-frames FeatureFrame whose
# `feature_names` include `gaul0_code`, indexed by (time, priogrid_gid).
FetchGaulCodes = Callable[[int, int, str], "object"]


class DatafactoryCountryMappingProvider:
    """Builds the GAUL-`gaul0_code` geography mapping for a forecast window."""

    def __init__(
        self,
        start_month: int,
        end_month: int,
        region: str = DEFAULT_REGION,
        fetch_gaul_codes: Optional[FetchGaulCodes] = None,
    ) -> None:
        self._start_month = start_month
        self._end_month = end_month
        self._region = region
        self._fetch = fetch_gaul_codes or self._fetch_from_datafactory

    def build(self) -> CountryMapping:
        frame = self._fetch(self._start_month, self._end_month, self._region)

        names = list(frame.feature_names)
        if GAUL_CODE_FEATURE not in names:
            raise ValueError(
                f"the datafactory fetch returned no {GAUL_CODE_FEATURE!r} feature "
                f"(got {names}); the grid->country mapping cannot be built, and an "
                f"empty mapping would reconcile every cell to nothing"
            )

        # `frame.index.time` is deliberately NOT read. The mapping is keyed on the
        # window the caller asked for, not the months the store happened to return —
        # see `_broadcast_over_the_requested_window` (#647).
        gids = np.asarray(frame.index.unit, dtype=np.int64)

        # A fetch with the column but no rows. Sibling of the check above, and it has to
        # be here rather than left to the refusals below: both of those pass VACUOUSLY on
        # zero rows, `np.unique` returns empty, and `build` would hand back a mapping with
        # no keys — the §3 hazard arriving by the other door. Before #647 a short fetch at
        # least died loudly in `views_frames.index._remap`; keying on the requested window
        # removed that accident, so the refusal has to be deliberate (ADR-003, ADR-025).
        if gids.size == 0:
            raise ValueError(
                f"the datafactory fetch returned the {GAUL_CODE_FEATURE!r} feature but "
                f"ZERO rows for region {self._region!r}, months "
                f"{self._start_month}..{self._end_month}. The grid->country mapping would "
                f"have no keys, and an empty mapping reconciles every cell to nothing. "
                f"Check the region name resolves to a pgid set ('land_gaul' is 64,742 "
                f"cells) and that the store covers the window."
            )

        # `FeatureFrame.values` is (rows, features, samples) — the trailing axis is the
        # draw dimension, length 1 for a non-sample frame. Geography is not stochastic, so
        # anything else means the fetch returned a forecast where a lookup was asked for.
        values = np.asarray(frame.values)
        if values.ndim != 3 or values.shape[2] != 1:
            raise ValueError(
                f"the datafactory fetch returned a frame of shape {values.shape}; a "
                f"grid->country lookup must have exactly one sample per cell "
                f"(expected (rows, features, 1)). Geography is not stochastic — a draw "
                f"dimension here means a forecast was fetched instead of a lookup."
            )
        codes = values[:, names.index(GAUL_CODE_FEATURE), 0]
        codes = np.rint(codes.astype(np.float64)).astype(np.int64)

        self._refuse_cells_without_a_country(gids, codes)
        self._refuse_cells_whose_country_moves(gids, codes)

        return self._broadcast_over_the_requested_window(gids, codes)

    def _broadcast_over_the_requested_window(
        self, gids: np.ndarray, codes: np.ndarray
    ) -> CountryMapping:
        """One code per cell, repeated across every month the CALLER asked for (#647).

        **The defect this fixes.** This used to key the mapping on the months the fetch
        happened to return: `CountryMapping(np.stack([times, gids], axis=1), codes)`.
        `datafactory_query.load_dataset` does not refuse a window its store cannot
        cover — it returns the months it has and emits a `UserWarning`. The datafactory
        zarr store held 457..564; `chrome_box` asks for 457..622 (the union of its test
        ranges plus `composition._WINDOW_BUFFER_MONTHS = 24`). So the mapping was built
        58 months short, reconciliation ran fine for three steps, and month 565 — step 4
        of 36 — died in `views_frames.index._remap` with
        `(time, unit) (565, 49182) has no entry in the injected mapping`.

        A warning that something is 58 months short, swallowed into a mapping that looks
        complete, is the shape ADR-003 exists to stop. The mapping now covers the window
        that was *requested*, so a short fetch cannot silently shorten it.

        **Why broadcasting is sound rather than a papering-over.** The time axis is
        redundant here, and this class already proves it: GAUL 2024 polygons are static
        per cell, and `_refuse_cells_whose_country_moves` has just refused the frame if
        any cell reported more than one code across the window. After that check a code
        is a property of the *cell*, not of `(cell, month)`, so repeating it across the
        requested months adds no information and invents none. Taking `codes[first]` is
        safe for exactly the same reason — not a `.first()` collapse of contradictory
        values, which §2 of the module docstring refuses by name, but a read of the one
        value that survived the refusal.

        Note what this does NOT do: it does not widen the *cell* set. Every requested
        month gets precisely the cells the fetch returned, so a fetch short in cells
        still fails loudly in `_remap`, with the missing cell named.

        **What it costs, measured rather than guessed.** The mapping grows from the
        store's 108 months to the requested 166: 64,742 cells x 166 months = 10,747,172
        rows, and at 24 bytes a row (two int64 keys plus an int64 value) that is 246.0
        MiB, up 85.9 MiB. Against the 68.2 GiB peak measured for `chrome_box`'s pool on
        2026-10-28 that is 0.12%, so it does not move the box class — but it is a real
        increase and is written down rather than waved past.

        ⚠️ Known waste, deliberately left: the fetch still asks for the whole window
        (166 months on `chrome_box`) to read a static feature. One month would do. That
        is a change to what `_fetch_from_datafactory` requests, and the store's region
        resolution is the kind of thing to alter on its own evidence, not in passing —
        trigger to revisit: the next time this fetch is slow enough to notice, or the
        next time the store's horizon moves.
        """
        cells, first_row_of_each = np.unique(gids, return_index=True)
        code_of_cell = codes[first_row_of_each]

        months = np.arange(self._start_month, self._end_month + 1, dtype=np.int64)
        keys = np.empty((months.size * cells.size, 2), dtype=np.int64)
        keys[:, 0] = np.repeat(months, cells.size)
        keys[:, 1] = np.tile(cells, months.size)

        return CountryMapping(keys, np.tile(code_of_cell, months.size))

    def _refuse_cells_without_a_country(
        self, gids: np.ndarray, codes: np.ndarray
    ) -> None:
        """`gaul0_code <= 0` cannot be summed into a country. Refuse (see §1)."""
        bad = codes <= 0
        if not bad.any():
            return
        offenders = np.unique(gids[bad])
        raise ValueError(
            f"{offenders.size} grid cell(s) have no GAUL country "
            f"({GAUL_CODE_FEATURE} <= 0), e.g. priogrid_gid "
            f"{offenders[:5].tolist()}. They cannot be reconciled to a country, and "
            f"`grid_to_country_month` drops them SILENTLY unless the caller declares "
            f"`feature_agg_types`. Region is {self._region!r} — if that is 'land' "
            f"(64,818 cells), use 'land_gaul' (64,742), whose 76-cell difference is "
            f"exactly these cells."
        )

    def _refuse_cells_whose_country_moves(
        self, gids: np.ndarray, codes: np.ndarray
    ) -> None:
        """GAUL polygons are static per cell, so a cell's code cannot change (see §2)."""
        order = np.lexsort((codes, gids))
        g, c = gids[order], codes[order]
        # A cell is inconsistent iff two adjacent rows share the gid and differ in code.
        moved = (g[1:] == g[:-1]) & (c[1:] != c[:-1])
        if not moved.any():
            return
        offenders = np.unique(g[1:][moved])
        raise ValueError(
            f"{offenders.size} grid cell(s) report more than one {GAUL_CODE_FEATURE} "
            f"across the window, e.g. priogrid_gid {offenders[:5].tolist()}. GAUL 2024 "
            f"polygons are static per cell, so this is a data defect, not a boundary "
            f"change. Refusing rather than taking the first value: the viewser provider "
            f"collapses time that way for documented parity with views-reporting, which "
            f"is not a reason that applies here, and picking one of two contradictory "
            f"country attributions produces a plausible wrong reconciled number."
        )

    def _fetch_from_datafactory(self, start_month: int, end_month: int, region: str):
        """The real fetch. Needs `~/.netrc` for the remote zarr store (ADR-026)."""
        from datafactory_query import load_dataset
        from datafactory_query.defaults import DEFAULT_REMOTE

        return load_dataset(
            region=region,
            start=start_month,
            end=end_month,
            features=[GAUL_CODE_FEATURE],
            output_format="feature_frame",
            data_dir=DEFAULT_REMOTE.zarr_url,
        )
