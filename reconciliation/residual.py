"""What reconciliation could not place.

Reconciliation scales a grid forecast so that the cells inside each country add up to
that country's own forecast. Scaling is multiplication, so there is one thing it cannot
do: **if every cell in a country is zero, no multiplier produces a positive total.**
``0 x anything = 0``. The country's fatalities are then simply absent from the grid
product, and the two forecasts disagree for that (country, month, draw).

This is not rare. Measured on the 2026-10-30 `chrome_box` delivery — 202 countries,
36 months, 1,024 draws, 3 targets — it accounts for **100%** of all disagreement, and
costs **2.9% to 7.3% of the forecast's fatality mass** depending on the target. 94-96%
of grid cells are zero, so a country with few cells can draw all-zero across its whole
area while the country ensemble draws a positive count.

⛔ **The reason this module exists is that nothing reported it.** The run exited 0, wrote
`STATUS: OK`, and its MANIFEST said `reconciled: against chrome_crate` with no
qualification — in the artefact a UN partner receives. A recipient who summed our grid
forecast by country and compared it to our country forecast would find a 3-7% gap and no
explanation. A delivery may be approximate; it may not be quietly approximate.

Two numbers come out of this, and they mean opposite things:

* ``unplaced_mass`` — country fatalities with nowhere to go, because the grid was zero.
  An inherent limit of multiplicative scaling. Report it; it is not a bug.
* ``scaling_failures`` — the grid sum differs from the country total **while the grid sum
  is non-zero**. There is no excuse for this one: the scaling simply did not work. It was
  **0 across 22,339,584 triples** on 2026-10-30, and if it is ever non-zero the run must
  fail rather than deliver.

The redistribution question — *which* cells should receive a zero-grid country's mass
(population? historical conflict? the mean of the other draws?) — is a modelling choice,
is upstream in views-frames, and is not answered here. Filed with this measurement as
**views-frames#283**. ⚠️ Not views-frames#145, which pipeline-core's `reconcile_frames`
docstring still points at for the `aligned-draws` approximation: that issue is CLOSED.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

# A country total is "met" when the grid sums to within this relative distance of it.
# Float32 sums over ~64k cells do not land exactly; 0.1% is far below any reporting
# resolution and far above float32 accumulation error.
TOLERANCE = 1e-3

# (time, priogrid_gid) packed into one int64 for a sorted lookup. Grid ids are < 10^7.
_KEY_STRIDE = 10_000_000


@dataclass(frozen=True)
class TargetResidual:
    """What one target's reconciliation achieved, and what it could not."""

    target: str
    countries: int
    months: int
    draws: int
    triples: int
    short_triples: int
    scaling_failures: int
    worst_scaling_error: float
    cm_mass: float
    unplaced_mass: float

    @property
    def unplaced_fraction(self) -> float:
        return self.unplaced_mass / self.cm_mass if self.cm_mass else 0.0

    @property
    def short_fraction(self) -> float:
        return self.short_triples / self.triples if self.triples else 0.0

    def line(self) -> str:
        """One MANIFEST-width line. Deliberately states the loss, not just the success."""
        return (
            f"{self.target:12s} unplaced {self.unplaced_mass:>15,.0f} of "
            f"{self.cm_mass:>15,.0f} ({100 * self.unplaced_fraction:6.3f}% of mass) "
            f"in {self.short_triples:>9,} of {self.triples:>11,} triples "
            f"({100 * self.short_fraction:5.3f}%); scaling_failures {self.scaling_failures}"
        )


def measure_target(
    target: str,
    pgm_values: NDArray,
    pgm_time: NDArray,
    pgm_unit: NDArray,
    cm_values: NDArray,
    cm_time: NDArray,
    cm_unit: NDArray,
    map_keys: NDArray,
    map_vals: NDArray,
    *,
    tolerance: float = TOLERANCE,
) -> TargetResidual:
    """Compare the reconciled grid forecast to the country forecast, draw for draw.

    The contract under test is the one `views_pipeline_core.modules.reconciliation.
    reconcile_frames` states for ``mode=aligned-draws``: *"a cm forecast that already
    carries S draws is scaled draw-for-draw."* So for every (country, month, draw) the
    grid cells in that country must sum to the country value for that same draw.

    ⛔ Refuses rather than reporting a clean result it did not establish (ADR-025): a
    draw-count mismatch, a grid cell with no country, or a country present in one
    forecast and not the other all raise. Each of those would otherwise silently shrink
    what is compared and make the residual look smaller than it is.
    """
    pgm_values, cm_values = np.asarray(pgm_values), np.asarray(cm_values)
    if pgm_values.shape[1] != cm_values.shape[1]:
        raise ValueError(
            f"{target}: grid carries {pgm_values.shape[1]} draws and country forecast "
            f"carries {cm_values.shape[1]}. aligned-draws pairs draw s to draw s, so "
            f"these cannot be compared — and should never have been reconciled."
        )
    draws = int(pgm_values.shape[1])

    keys = np.asarray(map_keys)[:, 0].astype(np.int64) * _KEY_STRIDE + np.asarray(map_keys)[
        :, 1
    ].astype(np.int64)
    order = np.argsort(keys)
    keys, vals = keys[order], np.asarray(map_vals)[order]

    months = np.unique(np.asarray(pgm_time))
    triples = short = failures = 0
    worst = 0.0
    cm_mass = unplaced = 0.0
    countries_seen: set[int] = set()

    for month in months:
        grid_rows = np.asarray(pgm_time) == month
        unit = np.asarray(pgm_unit)[grid_rows]
        value = np.asarray(pgm_values[grid_rows], dtype=np.float64)

        probe = month.astype(np.int64) * _KEY_STRIDE + unit.astype(np.int64)
        at = np.searchsorted(keys, probe)
        found = (at < keys.size) & (keys[np.clip(at, 0, keys.size - 1)] == probe)
        if not found.all():
            raise ValueError(
                f"{target}, month {int(month)}: {int((~found).sum())} grid cells have no "
                f"country in the mapping. Their fatalities belong to no country total, so "
                f"any residual computed here would understate the true one."
            )
        country = vals[at]

        by_country = np.argsort(country, kind="stable")
        country, value = country[by_country], value[by_country]
        uniq, first = np.unique(country, return_index=True)
        grid_total = np.add.reduceat(value, first, axis=0)

        cm_rows = np.asarray(cm_time) == month
        cm_u = np.asarray(cm_unit)[cm_rows]
        cm_v = np.asarray(cm_values[cm_rows], dtype=np.float64)
        by_unit = np.argsort(cm_u)
        cm_u, cm_v = cm_u[by_unit], cm_v[by_unit]

        missing = set(cm_u.tolist()) ^ set(uniq.tolist())
        if missing:
            raise ValueError(
                f"{target}, month {int(month)}: {len(missing)} countries appear in one "
                f"forecast but not the other (e.g. {sorted(missing)[:5]}). Comparing only "
                f"the intersection would hide exactly the mass this function measures."
            )
        countries_seen.update(uniq.tolist())

        # An honest zero (country 0, grid 0) needs no special case: 0 / 1e-12 == 0, so
        # it is already below tolerance. A mutation that deleted an explicit guard here
        # survived the suite, which is how the guard was identified as dead.
        gap = np.abs(grid_total - cm_v) / np.maximum(np.abs(cm_v), 1e-12)
        off = gap > tolerance

        triples += gap.size
        cm_mass += float(cm_v.sum())
        zero_grid = off & (grid_total == 0)
        short += int(zero_grid.sum())
        unplaced += float(cm_v[zero_grid].sum())

        real = off & (grid_total > 0)
        failures += int(real.sum())
        if real.any():
            worst = max(worst, float(np.where(real, gap, 0.0).max()))

    return TargetResidual(
        target=target,
        countries=len(countries_seen),
        months=int(months.size),
        draws=draws,
        triples=triples,
        short_triples=short,
        scaling_failures=failures,
        worst_scaling_error=worst,
        cm_mass=cm_mass,
        unplaced_mass=unplaced,
    )


def newest_prediction_dir(generated: Path) -> Path:
    """The most recent ``predictions_forecasting_*`` under a `data/generated`.

    Refuses on none. The names are timestamped, so "newest" is the lexicographic
    maximum — and the chosen path is printed, because picking one of several silently
    is how a verifier ends up measuring last week's forecast.
    """
    found = sorted(Path(generated).glob("predictions_forecasting_*"))
    if not found:
        raise FileNotFoundError(
            f"{generated} holds no predictions_forecasting_* directory — there is no "
            f"forecast here to measure."
        )
    return found[-1]


def load_target(prediction_dir: Path, target: str) -> tuple[NDArray, NDArray, NDArray]:
    """``(values, time, unit)`` for one target, values memory-mapped (they are ~9 GB)."""
    values = np.load(Path(prediction_dir) / target / "y_pred.npy", mmap_mode="r")
    ids = np.load(Path(prediction_dir) / target / "identifiers.npz")
    return values, ids["time"], ids["unit"]


def targets_in(prediction_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(prediction_dir).iterdir() if p.is_dir())


def measure_run(
    ensemble_dir: Path, *, ensembles_root: Path | None = None
) -> tuple[list[TargetResidual], list[str]]:
    """Measure a finished reconciling run against its declared partner.

    Uses `composition.mapping_provider_for_run`, which is the same entry point
    `build_reconciler_for_run` uses, so the mapping here is the run's own.
    """
    from reconciliation.composition import mapping_provider_for_run, partner_of

    ensemble_dir = Path(ensemble_dir).resolve()
    root = Path(ensembles_root) if ensembles_root else ensemble_dir.parent
    partner = partner_of(ensemble_dir)

    pgm_dir = newest_prediction_dir(ensemble_dir / "data" / "generated")
    cm_dir = newest_prediction_dir(root / partner / "data" / "generated")

    notes = [
        f"ensemble:      {ensemble_dir.name}",
        f"partner:       {partner} (declared reconcile_with)",
        f"grid forecast: {pgm_dir}",
        f"country fcst:  {cm_dir}",
    ]

    pgm_targets, cm_targets = targets_in(pgm_dir), targets_in(cm_dir)
    if set(pgm_targets) != set(cm_targets):
        raise ValueError(
            f"targets differ: grid has {pgm_targets}, country has {cm_targets}. "
            f"Reconciliation pairs them by name; a mismatch is not measurable."
        )

    mapping = mapping_provider_for_run(ensemble_dir).build()
    notes.append(
        f"mapping:       {len(mapping.map_vals):,} (time,gid)->country rows, "
        f"{len(np.unique(mapping.map_vals))} countries"
    )

    out = []
    for target in pgm_targets:
        pv, pt, pu = load_target(pgm_dir, target)
        cv, ct, cu = load_target(cm_dir, target)
        out.append(
            measure_target(
                target, pv, pt, pu, cv, ct, cu, mapping.map_keys, mapping.map_vals
            )
        )
    return out, notes


def render(results: Iterable[TargetResidual], notes: Iterable[str]) -> str:
    """The operator-facing block. Loud by construction: the loss is a heading, not a
    footnote, and the sentence a MANIFEST would otherwise have printed is stated in
    full so nobody has to reconstruct it."""
    results = list(results)
    lines = ["=== RECONCILIATION RESIDUAL ===", *notes, ""]
    worst = max((r.unplaced_fraction for r in results), default=0.0)
    failures = sum(r.scaling_failures for r in results)

    if failures:
        lines += [
            "⛔ SCALING FAILED. The grid sum differs from the country total where the grid",
            "   is NOT zero. This is not the zero-grid limit — the scaling did not work.",
            "",
        ]
    lines += [
        f"⚠️  UP TO {100 * worst:.3f}% OF THE COUNTRY FORECAST'S FATALITIES ARE NOT ON THE",
        "   GRID. Multiplicative scaling cannot lift a country whose every cell is zero,",
        "   so that country's fatalities are absent from the grid product. This delivery",
        "   is reconciled EXCEPT for the mass below, and may not be described otherwise.",
        "",
    ]
    lines += [r.line() for r in results]
    lines += ["", f"scaling_failures total: {failures}"]
    if failures:
        lines.append(
            f"worst non-zero-grid error: "
            f"{max(r.worst_scaling_error for r in results):.3e}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ensemble", required=True, type=Path, help="e.g. ensembles/chrome_box")
    ap.add_argument("--out", type=Path, help="also write the block here")
    args = ap.parse_args(argv)

    results, notes = measure_run(args.ensemble)
    block = render(results, notes)
    print(block)
    if args.out:
        args.out.write_text(block + "\n")

    # Unplaced mass is reported, never fatal — it is a property of the method. A scaling
    # failure IS fatal: it means the reconciler did not do the one thing it exists for,
    # and a forecast whose country totals are wrong must not be delivered.
    return 1 if sum(r.scaling_failures for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
