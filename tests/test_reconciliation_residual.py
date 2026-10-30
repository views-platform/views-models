"""`reconciliation/residual.py` must measure the mass reconciliation could not place.

The module exists because the 2026-10-30 `chrome_box` delivery lost **2.9%-7.3%** of the
country forecast's fatalities to countries whose grid cells were all zero, and reported
`reconciled: against chrome_crate` with no qualification. Every property below is either
that arithmetic or one of the ways a verifier can silently measure less than it claims.

Fixture, small enough to compute by hand: 3 grid cells, 2 countries, 2 months, 2 draws.

    gid 1 -> country 10      months 1 and 2
    gid 2 -> country 10
    gid 3 -> country 20
"""

from pathlib import Path

import numpy as np
import pytest

# ⛔ ABOVE the reconciliation import, and that ordering is the whole point.
# `reconciliation/__init__.py` imports `composition`, which imports
# `views_pipeline_core.domain.reconciliation_port` at module level — a symbol that exists
# only from pipeline-core 3.0.0. CI installs the PUBLISHED core, so importing any
# `reconciliation.*` submodule fails there, including this one, whose own module-level
# imports are just numpy and the stdlib. Below the import this guard never executes and
# collection ERRORS rather than skipping, which aborts the entire pytest run and not only
# this file. `tests/test_reconciliation_skip_is_truthful.py` enforces the ordering; it
# caught this file the first time the full suite ran against it.
pytest.importorskip("views_pipeline_core.domain.reconciliation_port")

from reconciliation.residual import (  # noqa: E402  (must follow the guard)
    TargetResidual,
    main,
    measure_target,
    render,
)

pytestmark = pytest.mark.beige

# (time, gid) rows, in the order the grid forecast carries them
PGM_TIME = np.array([1, 1, 1, 2, 2, 2])
PGM_UNIT = np.array([1, 2, 3, 1, 2, 3])
# (time, country) rows for the country forecast
CM_TIME = np.array([1, 1, 2, 2])
CM_UNIT = np.array([10, 20, 10, 20])

MAP_KEYS = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
MAP_VALS = np.array([10, 10, 20, 10, 10, 20])


def _measure(pgm, cm, **kw) -> TargetResidual:
    return measure_target(
        "lr_sb_best",
        np.asarray(pgm, dtype=np.float32), PGM_TIME, PGM_UNIT,
        np.asarray(cm, dtype=np.float32), CM_TIME, CM_UNIT,
        MAP_KEYS, MAP_VALS, **kw,
    )


def test_a_perfectly_reconciled_forecast_has_no_residual() -> None:
    """Grid sums equal country totals draw for draw: nothing unplaced, nothing failed."""
    pgm = [[1, 2], [3, 4], [5, 5],        # month 1: c10 -> [4,6], c20 -> [5,5]
           [2, 2], [2, 2], [9, 1]]        # month 2: c10 -> [4,4], c20 -> [9,1]
    cm = [[4, 6], [5, 5], [4, 4], [9, 1]]
    r = _measure(pgm, cm)
    assert (r.short_triples, r.scaling_failures) == (0, 0)
    assert r.unplaced_mass == 0.0
    assert r.unplaced_fraction == 0.0
    assert (r.countries, r.months, r.draws, r.triples) == (2, 2, 2, 8)
    assert r.cm_mass == pytest.approx(4 + 6 + 5 + 5 + 4 + 4 + 9 + 1)


def test_a_country_whose_grid_is_all_zero_is_counted_as_unplaced_mass() -> None:
    """⛔ The finding this module was written for.

    Country 20, month 1 has a single cell and it is zero in both draws, while the
    country forecast says 7 fatalities in draw 0 and none in draw 1. `0 x anything = 0`,
    so draw 0's 7 fatalities are absent from the grid product. Draw 1 agrees (both zero)
    and must NOT be counted — a residual that counted honest zeros would be noise.
    """
    pgm = [[1, 2], [3, 4], [0, 0],
           [2, 2], [2, 2], [9, 1]]
    cm = [[4, 6], [7, 0], [4, 4], [9, 1]]
    r = _measure(pgm, cm)
    assert r.short_triples == 1, "exactly one (country,month,draw) is short"
    assert r.unplaced_mass == pytest.approx(7.0)
    assert r.scaling_failures == 0, "a zero grid is not a scaling failure"
    assert r.unplaced_fraction == pytest.approx(7.0 / r.cm_mass)


def test_a_grid_that_is_non_zero_and_still_disagrees_is_a_scaling_failure() -> None:
    """The other half, and the one that must be fatal: the grid sums to 4 where the
    country says 40, with nothing zero anywhere. Scaling simply did not happen. This
    was 0 across 22,339,584 triples on 2026-10-30; if it is ever non-zero the run is
    delivering wrong country totals."""
    pgm = [[1, 2], [3, 4], [5, 5],
           [2, 2], [2, 2], [9, 1]]
    cm = [[40, 6], [5, 5], [4, 4], [9, 1]]
    r = _measure(pgm, cm)
    assert r.scaling_failures == 1
    assert r.short_triples == 0
    assert r.unplaced_mass == 0.0
    assert r.worst_scaling_error == pytest.approx(36 / 40)


def test_mismatched_draw_counts_are_refused_not_compared() -> None:
    """aligned-draws pairs draw s to draw s. With 2 grid draws and 1 country draw there
    is no pairing, and anything this function returned would be invented."""
    with pytest.raises(ValueError, match="draws"):
        measure_target(
            "t", np.ones((6, 2), np.float32), PGM_TIME, PGM_UNIT,
            np.ones((4, 1), np.float32), CM_TIME, CM_UNIT, MAP_KEYS, MAP_VALS,
        )


def test_a_grid_cell_with_no_country_is_refused_not_dropped() -> None:
    """⛔ ADR-025. A cell absent from the mapping belongs to no country total, so
    skipping it would shrink the comparison and make the residual look SMALLER than it
    is — a verifier reporting a cleaner result than it established."""
    holey_keys = MAP_KEYS[1:]
    holey_vals = MAP_VALS[1:]
    with pytest.raises(ValueError, match="no country"):
        measure_target(
            "t", np.ones((6, 2), np.float32), PGM_TIME, PGM_UNIT,
            np.ones((4, 2), np.float32), CM_TIME, CM_UNIT, holey_keys, holey_vals,
        )


def test_a_country_in_one_forecast_but_not_the_other_is_refused() -> None:
    """Comparing the intersection would hide precisely the countries most likely to be
    short — the small ones that may have no grid cells at all."""
    with pytest.raises(ValueError, match="one forecast but not the other"):
        measure_target(
            "t", np.ones((6, 2), np.float32), PGM_TIME, PGM_UNIT,
            np.ones((4, 2), np.float32), CM_TIME, np.array([10, 99, 10, 99]),
            MAP_KEYS, MAP_VALS,
        )


def test_the_rendered_block_states_the_loss_rather_than_burying_it() -> None:
    """The block an operator and a MANIFEST both read. It must name the percentage and
    must not let `reconciled` stand unqualified — that unqualified word in a MANIFEST is
    the defect this module answers."""
    r = TargetResidual("lr_sb_best", 202, 36, 1024, 7_446_528, 164_429, 0, 0.0,
                       319_789_635.0, 12_866_485.0)
    block = render([r], ["ensemble:      chrome_box"])
    assert "RECONCILIATION RESIDUAL" in block
    assert "4.023%" in block, "the loss percentage must appear, not just the raw mass"
    assert "EXCEPT" in block, "the block must qualify the word 'reconciled'"
    assert "scaling_failures total: 0" in block


def test_unplaced_mass_is_reported_but_a_scaling_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exit code is what makes the runner `die`, so it is the loudness that matters.

    Unplaced mass must NOT fail the run: it is a property of multiplicative scaling and
    a three-hour forecast is still correct everywhere else. A scaling failure MUST fail
    it. Getting these the wrong way round would either throw away good forecasts or
    deliver wrong country totals.
    """
    import reconciliation.residual as mod

    short = TargetResidual("t", 2, 2, 2, 8, 1, 0, 0.0, 38.0, 7.0)
    broken = TargetResidual("t", 2, 2, 2, 8, 0, 1, 0.9, 38.0, 0.0)

    monkeypatch.setattr(mod, "measure_run", lambda *a, **k: ([short], ["n"]))
    out = tmp_path / "RESIDUAL"
    assert main(["--ensemble", "x", "--out", str(out)]) == 0
    assert "EXCEPT" in out.read_text()

    monkeypatch.setattr(mod, "measure_run", lambda *a, **k: ([broken], ["n"]))
    assert main(["--ensemble", "x"]) == 1
