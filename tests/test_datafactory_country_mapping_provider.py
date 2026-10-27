"""Guards for the datafactory `gaul0_code` geography provider (#601, epic #595 S2b).

⛔ **Why these tests do not diff against the viewser provider.** #196 is explicit that this
provider is not parity-preserving: grouping by `gaul0_code` attributes cells to different
countries than viewser's `country_id`, and the two id systems do not overlap (C-49). A parity
test would therefore encode a false equivalence. The ground truth used instead is
`datafactory_query.load_region_pgids("land_gaul")`, which reads a JSON bundled **inside the
installed package** — no network, no credentials, no sibling checkout — and is an independent
path from the feature fetch the provider reads.

⚠️ **What is proved here and what is not.** These guards prove the provider covers the right
*cells* and refuses the three states that would otherwise yield a plausible wrong reconciled
number. They do **not** prove the `gaul0_code` *values* are the right countries: that needs a
live datafactory fetch with credentials, which is #599's launch gate, not a unit test. Stated
rather than implied, because "covers every cell" reads like "is correct" and is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ⛔ The frame types come from `views_frames`, NOT from `datafactory_adapters`, which
# only re-exports them. `run_tests.yml` installs `views_pipeline_core==3.3.0` and nothing
# else, which resolves views-frames 2.0.0 — verified in a venv built exactly like that
# job, where `FeatureFrame`, `SpatioTemporalIndex` and `SpatialLevel` are all top-level.
# Importing them from the datafactory side instead is what made this whole module ERROR
# at collection in CI, and reaching for a skip would have silenced 24 guards to fix 5.
from views_frames import FeatureFrame, SpatialLevel, SpatioTemporalIndex

from reconciliation import source_detection
from reconciliation.datafactory_country_mapping_provider import (
    DEFAULT_REGION,
    GAUL_CODE_FEATURE,
    DatafactoryCountryMappingProvider,
)
from reconciliation.reconciler_factory import _PROVIDERS


def _load_region_pgids():
    """`datafactory_query` only — ⛔ and ONLY for the guards that genuinely need it.

    `run_tests.yml` installs `views_pipeline_core==3.3.0` and nothing else, so the package
    is absent there; a module-level import ERRORED this file at collection. A module-level
    `importorskip` would have been worse: it silences all 25 guards to accommodate 6, and a
    skipped guard reads as a pass, which is the thing ADR-025 forbids. So the skip is
    per-guard, and `.github/workflows/datafactory_geography.yml` is where the 6 actually
    run — pinned to the same `views-datafactory>=1.13.0,<2.0.0` floor the sixteen declare.
    `test_the_skipped_guards_have_a_job_that_runs_them` below keeps that honest.
    """
    pytest.importorskip(
        "datafactory_query",
        reason="absent in run_tests.yml by design; see datafactory_geography.yml",
    )
    from datafactory_query import load_region_pgids

    return load_region_pgids


def _frame(times, gids, codes, feature_names=(GAUL_CODE_FEATURE,), extra=None):
    """A real views-frames FeatureFrame, not a stand-in — the provider reads
    `.feature_names`, `.index.time`, `.index.unit` and `.values`, so a mock would
    prove nothing about whether it reads them correctly."""
    cols = [np.asarray(codes, dtype=np.float64)]
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float64))
    values = np.stack(cols, axis=1)
    index = SpatioTemporalIndex(
        time=np.asarray(times, dtype=np.int64),
        unit=np.asarray(gids, dtype=np.int32),
        level=SpatialLevel.PGM,
    )
    return FeatureFrame.from_2d(
        y_features_2d=values, index=index, feature_names=list(feature_names)
    )


def _frame_with_columns(times, gids, columns):
    """A frame with several named features, in the given order. Needed because a
    one-or-two-column fixture cannot distinguish `names.index(...)` from a constant
    index: with the gaul column second of two, 1 and -1 coincide."""
    values = np.stack(
        [np.asarray(v, dtype=np.float64) for v in columns.values()], axis=1
    )
    index = SpatioTemporalIndex(
        time=np.asarray(times, dtype=np.int64),
        unit=np.asarray(gids, dtype=np.int32),
        level=SpatialLevel.PGM,
    )
    return FeatureFrame.from_2d(
        y_features_2d=values, index=index, feature_names=list(columns)
    )


class _RecordingFetch:
    """Records the ``(start, end, region)`` the provider asks for.

    ⛔ An independent guard audit of this file found that every guard injected
    ``lambda s, e, r: frame``, discarding all three arguments, so **nothing pinned what
    the provider fetched** — only what it did with the answer. Three separate production
    mutations survived on that: hardcoding the constructor's default region to ``land``,
    discarding the caller's region, and calling the fetch with literals. That matters
    specifically because the ``gaul0_code <= 0`` refusal exists to catch "the region was
    ``land``, not ``land_gaul``" — and the provider could have been asking for ``land``
    all along.
    """

    def __init__(self, frame):
        self._frame = frame
        self.calls: list[tuple[int, int, str]] = []

    def __call__(self, start, end, region):
        self.calls.append((start, end, region))
        return self._frame


def _provider_and_fetch(frame, region=None, start=500, end=501):
    """The provider plus the recorder it fetches through. ``region=None`` means *do not
    pass one*, so the constructor default is what gets exercised."""
    fetch = _RecordingFetch(frame)
    kwargs = {} if region is None else {"region": region}
    provider = DatafactoryCountryMappingProvider(
        start, end, fetch_gaul_codes=fetch, **kwargs
    )
    return provider, fetch


def _provider(frame, region=DEFAULT_REGION, start=500, end=501):
    return _provider_and_fetch(frame, region=region, start=start, end=end)[0]


class TestTheMappingItBuilds:
    def test_keys_and_values_line_up_with_the_frame_row_for_row(self):
        frame = _frame([500, 500, 501, 501], [1001, 1002, 1001, 1002], [231, 94, 231, 94])
        mapping = _provider(frame).build()
        assert mapping.map_keys.tolist() == [[500, 1001], [500, 1002], [501, 1001], [501, 1002]]
        assert mapping.map_vals.tolist() == [231, 94, 231, 94]

    def test_the_gaul_column_is_read_by_name_not_by_position(self):
        """A second feature ahead of nothing is not enough; the column must be located
        by name. With `gaul0_code` second, a positional read returns the other feature."""
        # ⛔ THREE columns, with gaul0_code in the MIDDLE. The earlier version of this
        # guard put it second of two, where `names.index(...)` and the literal `-1`
        # coincide — so it caught a positional read spelled `0` and missed one spelled
        # `-1`, which an audit duly found. At index 1 of 3, neither 0 nor -1 works.
        frame = _frame_with_columns(
            [500, 500],
            [1001, 1002],
            {
                "a_decoy_before": [7, 7],
                GAUL_CODE_FEATURE: [231, 94],
                "a_decoy_after": [9, 9],
            },
        )
        assert _provider(frame).build().map_vals.tolist() == [231, 94]

    def test_a_float_coded_column_becomes_integers_without_truncating_down(self):
        """Zarr hands back floats. 230.9999 is country 231, not 230."""
        frame = _frame([500, 500], [1001, 1002], [230.9999, 94.00001])
        assert _provider(frame).build().map_vals.tolist() == [231, 94]


class TestWhatItAsksFor:
    def test_the_default_window_and_region_reach_the_fetch_unchanged(self):
        """Pins the plumbing on the INJECTED path, independently of the real fetch: the
        constructor's window and its `land_gaul` default must be what gets asked for.
        Without this, `DEFAULT_REGION` is pinned as a constant by its own guard while
        nothing checks that anything uses it."""
        frame = _frame([500], [1001], [231])
        provider, fetch = _provider_and_fetch(frame, region=None, start=500, end=536)
        provider.build()
        assert fetch.calls == [(500, 536, "land_gaul")]


class TestWhatItRefuses:
    def test_a_cell_with_no_gaul_country_refuses_and_names_land_gaul(self):
        frame = _frame([500, 500], [1001, 1002], [231, -1])
        with pytest.raises(ValueError, match=r"no GAUL country"):
            _provider(frame, region="land").build()

    def test_the_refusal_tells_the_operator_which_region_to_use(self):
        frame = _frame([500], [1002], [-1])
        with pytest.raises(ValueError) as e:
            _provider(frame, region="land").build()
        assert "land_gaul" in str(e.value) and "64,742" in str(e.value)
        assert "1002" in str(e.value), "the offending cell must be named, not just counted"

    def test_zero_is_refused_as_well_as_minus_one(self):
        """`grid_to_country_month` drops `country_id <= 0`, not just -1."""
        frame = _frame([500], [1001], [0])
        with pytest.raises(ValueError, match=r"no GAUL country"):
            _provider(frame).build()

    def test_a_cell_whose_country_changes_refuses_instead_of_taking_the_first(self):
        """⛔ TIME-MAJOR rows, two cells — not one cell's two months adjacent.

        The earlier fixture had a single cell, so its two contradictory rows were already
        neighbours and the `lexsort` was never load-bearing: an audit deleted the sort
        outright and this guard stayed green. Real frames arrive time-major, so a cell's
        months are a whole grid apart — 64,742 rows on `land_gaul`. Here cell 1001 is
        country 231 in month 500 and 94 in month 501, with cell 1002 interleaved between
        them, so only a sort that groups by cell can see the contradiction.
        """
        frame = _frame(
            [500, 500, 501, 501], [1001, 1002, 1001, 1002], [231, 94, 94, 94]
        )
        with pytest.raises(ValueError, match=r"more than one gaul0_code"):
            _provider(frame).build()

    def test_a_sampled_frame_refuses_because_geography_is_not_stochastic(self):
        """`FeatureFrame.values` carries a trailing draw axis. A lookup must have exactly
        one sample per cell; more means a forecast was fetched where geography was asked
        for, and taking draw 0 would silently reconcile against one draw of a posterior."""
        index = SpatioTemporalIndex(
            time=np.array([500, 500], dtype=np.int64),
            unit=np.array([1001, 1002], dtype=np.int32),
            level=SpatialLevel.PGM,
        )
        # (N=2, F=1, S=2) — two draws per cell. The library's own ADR-012 shape.
        sampled = FeatureFrame(
            y_features=np.array([[[231.0, 231.0]], [[94.0, 94.0]]]),
            index=index,
            feature_names=[GAUL_CODE_FEATURE],
        )
        with pytest.raises(ValueError, match=r"exactly one sample per cell"):
            _provider(sampled).build()

    def test_a_missing_gaul_feature_refuses_rather_than_mapping_nothing(self):
        frame = _frame([500], [1001], [231], feature_names=("ged_sb_best",))
        with pytest.raises(ValueError, match=r"no 'gaul0_code' feature"):
            _provider(frame).build()

    def test_neighbouring_cells_in_different_countries_are_not_mistaken_for_a_moving_cell(self):
        """⛔ This guard was WEAK and a mutation proved it. It first used the same code for
        every row, so the inconsistency check could not fire either way — dropping the
        per-cell grouping (`g[1:] == g[:-1]`) left it green. The discriminating case is the
        ORDINARY one: two different cells in two different countries, each stable over the
        window. Sorted by (cell, code) those rows sit adjacent and differ in code, so
        without the grouping condition every such boundary reads as a cell that moved —
        which on real data is 64,742 cells in ~200 countries, i.e. always."""
        frame = _frame(
            [500, 500, 501, 501], [1001, 1002, 1001, 1002], [231, 94, 231, 94]
        )
        assert _provider(frame).build().map_vals.tolist() == [231, 94, 231, 94]


class TestCoverageAgainstTheBundledGroundTruth:
    def test_the_default_region_is_land_gaul_not_land(self):
        assert DEFAULT_REGION == "land_gaul"

    def test_the_ground_truth_is_the_region_epic_595_was_built_on(self):
        """Pins the two figures the epic, #601 and every member queryset quote, from the
        package's own bundled list rather than from any of those documents."""
        load_region_pgids = _load_region_pgids()
        assert len(load_region_pgids("land_gaul")) == 64_742
        assert len(load_region_pgids("land")) == 64_818
        assert len(load_region_pgids("land") - load_region_pgids("land_gaul")) == 76

    def test_the_mapping_covers_every_cell_in_land_gaul(self):
        """At real scale: one month over all 64,742 cells. Coverage is the acceptance
        criterion #601 states; the VALUES are not proved here (see the module docstring)."""
        load_region_pgids = _load_region_pgids()
        gids = np.array(sorted(load_region_pgids("land_gaul")), dtype=np.int64)
        frame = _frame(np.full(gids.size, 500), gids, np.full(gids.size, 231))
        mapping = _provider(frame).build()
        assert set(mapping.map_keys[:, 1].tolist()) == set(gids.tolist())
        assert len(mapping) == gids.size

    def test_the_76_cells_land_excludes_are_exactly_what_the_provider_refuses(self):
        """Ties the refusal to the ground truth: build on `land`, with the 76 carrying
        -1 as the real data does, and the provider must refuse — naming 76."""
        load_region_pgids = _load_region_pgids()
        land = np.array(sorted(load_region_pgids("land")), dtype=np.int64)
        gaul = sorted(load_region_pgids("land_gaul"))
        # ⛔ TWO months, so rows != cells. With one month the figure in the message
        # cannot discriminate a cell count from a row count, and an audit showed both
        # `np.unique(gids[bad])` -> `gids[bad]` and `offenders.size` -> `bad.sum()`
        # surviving. Over two months the 76 cells are 152 rows, so only a count of
        # DISTINCT cells still reads 76.
        gids = np.concatenate([land, land])
        times = np.concatenate([np.full(land.size, 500), np.full(land.size, 501)])
        codes = np.where(np.isin(gids, gaul), 231, -1)
        frame = _frame(times, gids, codes)
        with pytest.raises(ValueError) as e:
            _provider(frame, region="land").build()
        assert "76 grid cell(s) have no GAUL country" in str(e.value)
        assert "152" not in str(e.value), "cells were counted as rows"


class TestWhatTheRealFetchAsksFor:
    """⛔ `_fetch_from_datafactory` had NO guard of any kind. Every other test injects a
    fake fetch, so three mutations of the real one — asking for the wrong feature, swapping
    `start` and `end`, and hardcoding `region="land"` — all left the suite green. The whole
    point of the `gaul0_code <= 0` refusal is to catch a run that asked for `land`; if the
    only code path that ever asks is itself unguarded, the refusal is guarding a stub.

    `load_dataset` is patched where it is looked up, so no network and no credentials.
    """

    def _capture(self, monkeypatch):
        _load_region_pgids()  # same per-guard skip; see its docstring
        import datafactory_query

        seen = {}

        def _fake_load_dataset(**kwargs):
            seen.update(kwargs)
            return _frame([500], [1001], [231])

        monkeypatch.setattr(datafactory_query, "load_dataset", _fake_load_dataset)
        return seen

    def test_it_asks_for_the_gaul_feature_over_the_callers_window_and_region(
        self, monkeypatch
    ):
        seen = self._capture(monkeypatch)
        DatafactoryCountryMappingProvider(500, 536).build()

        assert seen["features"] == [GAUL_CODE_FEATURE]
        assert (seen["start"], seen["end"]) == (500, 536), "the window must not be reordered"
        assert seen["region"] == DEFAULT_REGION

    def test_an_explicit_region_reaches_the_fetch_instead_of_the_default(
        self, monkeypatch
    ):
        """The caller's region must not be silently replaced by the default — a mutation
        doing exactly that survived, and it would have made every run fetch land_gaul
        however it was configured, which hides a misconfiguration rather than refusing it."""
        seen = self._capture(monkeypatch)
        DatafactoryCountryMappingProvider(500, 501, region="Ethiopia").build()
        assert seen["region"] == "Ethiopia"

    def test_it_asks_for_a_lookup_not_a_forecast(self, monkeypatch):
        seen = self._capture(monkeypatch)
        DatafactoryCountryMappingProvider(500, 501).build()
        assert seen["output_format"] == "feature_frame"


class TestTheRegistryWiring:
    def test_the_registry_key_is_the_detector_s_own_constant(self):
        """⛔ The one-line registry entry is the whole defect surface of #601. The key
        must be `source_detection.DATAFACTORY` -- "views-datafactory". `reconciler_factory`'s
        comment said to add "datafactory", which would raise the very ValueError #601 fixes."""
        assert source_detection.DATAFACTORY == "views-datafactory"
        assert _PROVIDERS[source_detection.DATAFACTORY] is DatafactoryCountryMappingProvider

    def test_the_wrong_key_is_not_registered(self):
        assert "datafactory" not in _PROVIDERS

    def test_both_sources_are_registered_and_distinct(self):
        from reconciliation.viewser_country_mapping_provider import (
            ViewserCountryMappingProvider,
        )

        assert _PROVIDERS[source_detection.VIEWSER] is ViewserCountryMappingProvider
        assert _PROVIDERS[source_detection.DATAFACTORY] is not ViewserCountryMappingProvider

    def test_build_reconciler_resolves_the_source_through_the_registry(self, monkeypatch):
        """#601's first acceptance line, and ⛔ it was DECORATIVE as first written.

        It passed `provider=`, which makes `build_reconciler` short-circuit before the
        registry lookup — so `source` was never read, and an audit deleted the whole
        `DATAFACTORY` registry entry with this guard still green. The fetch is now
        stubbed on the class instead, so the registry lookup and the construction both
        actually happen.
        """
        from reconciliation.country_mapping import CountryMapping
        from reconciliation.reconciler_factory import build_reconciler

        seen = {}

        def _stub_build(self):
            seen["region"] = self._region
            seen["window"] = (self._start_month, self._end_month)
            return CountryMapping(np.array([[500, 1001]]), np.array([231]))

        monkeypatch.setattr(DatafactoryCountryMappingProvider, "build", _stub_build)
        reconciler = build_reconciler(500, 501, source=source_detection.DATAFACTORY)

        assert reconciler is not None
        # The provider the registry chose was constructed with the caller's window and
        # the land_gaul default -- not with literals, and not with `land`.
        assert seen == {"region": "land_gaul", "window": (500, 501)}

    def test_an_unknown_source_still_refuses_and_names_both_known_ones(self):
        """The name says "names both known ones" and the body did not check that. An
        audit stripped `known sources: {sorted(_PROVIDERS)}` from the message and this
        guard stayed green — the operator's only clue as to what IS accepted."""
        from reconciliation.reconciler_factory import build_reconciler

        with pytest.raises(ValueError, match=r"Unknown reconciliation geography source") as e:
            build_reconciler(500, 501, source="gaul")
        assert source_detection.VIEWSER in str(e.value)
        assert source_detection.DATAFACTORY in str(e.value)


def test_the_skipped_guards_have_a_job_that_runs_them():
    """⛔ The six guards above that need `datafactory_query` SKIP in `run_tests.yml`, and
    a skipped guard reads as a pass. This asserts the job that actually runs them exists
    and still names this file — so the skip cannot quietly become permanent.

    The precedent is specific: `test_roster_configs_load.py` carries
    `importorskip("views_hydranet")` and had therefore always skipped in `run_tests.yml`,
    which is how two production ensemble members sat unloadable from August to September
    with every test green (#404, #458 §1b). A dedicated job was the fix there, and this
    keeps that fix from decaying here.
    """
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "datafactory_geography.yml"
    )
    assert workflow.exists(), (
        "the job that runs the datafactory-dependent guards is gone; they now skip in "
        "run_tests.yml and run NOWHERE. Restore it or move those guards."
    )
    text = workflow.read_text(encoding="utf-8")
    assert Path(__file__).name in text, (
        f"{workflow.name} no longer names {Path(__file__).name}, so the guards it exists "
        f"to run are not being run by it."
    )
    assert "views-datafactory>=1.13.0" in text, (
        "the job's views-datafactory floor no longer matches the floor the sixteen #595 "
        "members declare; the guards would run against a version the models do not use."
    )


def test_the_provider_satisfies_the_port():
    """⚠️ An ATTRIBUTE-PRESENCE check only, not a contract guard. `CountryMappingProvider`
    is a `runtime_checkable` Protocol, and `isinstance` against one of those checks that
    the named attributes exist — not their signatures. `build` could grow a required
    parameter and this would stay green (an audit confirmed it does). The signature is
    pinned instead by every guard that calls `build()` with no arguments."""
    from reconciliation.country_mapping_provider import CountryMappingProvider

    assert isinstance(
        DatafactoryCountryMappingProvider(500, 501, fetch_gaul_codes=lambda s, e, r: None),
        CountryMappingProvider,
    )
