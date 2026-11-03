"""Does the forecast add up? — pre-registered aggregate calibration and dependence diagnostics.

THE QUESTION THIS ANSWERS. Sum the served tower-map surface over a conflict-affected country and
you get far less than history records. That looks like bad calibration and mostly is not: it is
the wrong arithmetic. `tower_point` is a MODE-like summary — its own docstring says it returns 0
"when the zero atom dominates, by density" — and for a zero-inflated, heavy-tailed variable the
mode is ~0 almost everywhere while the mean is not. Expectation is linear, modes are not:
``E[sum] = sum E[x]`` always holds, ``mode(sum) != sum mode(x)`` for skewed data. So summing MAP
surfaces is guaranteed to undershoot even for a perfectly calibrated model, and tells you nothing.

The right object is the distribution of the TOTAL, obtained by summing the DRAWS:

    for each draw d:   total_d = sum over cells of x[cell, d]

That yields a posterior over the aggregate, which can be compared with what was observed.

THE FAILURE MODE WORTH LOOKING FOR, which the above sets up. Every cell's marginal can be
perfectly calibrated while the TOTAL is badly calibrated, because the total's spread depends on
the DEPENDENCE between cells. If each draw is a coherent joint scenario, totals have realistic
spread. If the draws are independent per-cell marginals stitched together, summing ~64,818 of
them washes out the tail — variances add instead of covariances accumulating — and country totals
become far too narrow and too low. That is checkable and this tool checks it:

    dependence_ratio = Var_d(total)  /  sum_cells Var_d(cell)

    ~1.0  cells behave independently across draws — the tail of the aggregate is understated
    >1.0  positive dependence — draws carry joint structure, which is what a scenario should be
    <1.0  negative dependence, which for conflict would itself want explaining

PRE-REGISTERED FALSIFIERS — stated before the numbers are looked at, which is the only time such
a statement is worth anything:

  F1  dependence_ratio within [0.9, 1.1] for most (unit, target). The draws are then effectively
      independent marginals and every aggregate interval is too narrow, regardless of how good
      the per-cell marginals look.
  F2  with actuals: PIT values concentrated near 1.0 — the observed total sits above almost the
      whole predictive distribution. This is the timid-under-prediction signature, and it is the
      one the platform already has reason to expect: MSLE rewards timidity, which is why MCR
      (yhat/ybar) exists as a magnitude guardrail and why ranking on MSLE alone is refused.
  F3  with actuals: coverage of the central 90% interval far below 0.90 across units.
  F4  MCR far below 1.0 — the predicted mean total is a fraction of the observed mean total.

WHAT IT CANNOT DO. A FORECASTING run predicts months with no observations yet, so F2-F4 need a
partition where actuals exist (calibration or validation). Against a forecast the tool reports the
predictive totals and the dependence ratio only, and says so rather than inventing a comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_pooled(target_dir: Path):
    """y_pred.npy + identifiers.npz — the layout the ensemble writes, not PredictionFrame.load's."""
    y = np.load(target_dir / "y_pred.npy", mmap_mode="r")
    ids = np.load(target_dir / "identifiers.npz")
    return y, ids["time"], ids["unit"]


def _unit_map(path: Path | None, cells: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map each cell to an aggregation unit. Without a mapping every cell is one global unit —
    still informative, and it needs nothing we might not have on a pod."""
    if path is None:
        return np.zeros(len(cells), dtype=np.int64), {0: "GLOBAL"}
    import pandas as pd

    m = pd.read_parquet(path)
    cols = [c for c in m.columns if c != "priogrid_id"]
    if not cols:
        raise SystemExit(f"{path} has no unit column beside priogrid_id")
    lookup = dict(zip(m["priogrid_id"].to_numpy(), m[cols[0]].to_numpy()))
    names = sorted({str(v) for v in lookup.values()})
    code = {n: i for i, n in enumerate(names)}
    return (
        np.array([code.get(str(lookup.get(c, "UNMAPPED")), -1) for c in cells], dtype=np.int64),
        {i: n for n, i in code.items()} | {-1: "UNMAPPED"},
    )


def aggregate(target_dir: Path, unit_of_cell: np.ndarray, unit_names: dict,
              actuals: dict | None = None) -> list[dict]:
    y, time, _ = _load_pooled(target_dir)
    out = []
    for month in sorted(set(int(t) for t in time)):
        month_rows = time == month
        for ucode in sorted(set(int(u) for u in unit_of_cell)):
            sel = month_rows & (unit_of_cell == ucode)
            n = int(sel.sum())
            if n == 0:
                continue
            block = np.asarray(y[sel], dtype=np.float64)      # (cells, draws)
            totals = block.sum(axis=0)                        # (draws,)
            # Dependence: variance of the SUM against the sum of per-cell variances. Equal iff
            # the cells are uncorrelated across draws.
            per_cell_var = block.var(axis=1, ddof=1).sum()
            total_var = float(totals.var(ddof=1))
            row = {
                "month_id": month,
                "unit": unit_names.get(ucode, str(ucode)),
                "cells": n,
                "draws": int(block.shape[1]),
                "total_mean": float(totals.mean()),
                "total_median": float(np.median(totals)),
                "total_q05": float(np.quantile(totals, 0.05)),
                "total_q95": float(np.quantile(totals, 0.95)),
                "total_max": float(totals.max()),
                "sum_of_cell_map_like_zeros": int((block.max(axis=1) == 0).sum()),
                "dependence_ratio": (total_var / per_cell_var) if per_cell_var > 0 else None,
            }
            # Scored HERE, where the draw totals are still in hand. Computing PIT later would
            # mean carrying 128 floats per aggregate through the report purely to re-derive a
            # number available for free at this point.
            obs = (actuals or {}).get((month, row["unit"]))
            if obs is not None:
                obs = float(obs)
                row["observed_total"] = obs
                # `<=` not `<`: a zero-inflated predictive puts real mass exactly at 0, and an
                # observed 0 should not be scored as sitting below all of it.
                row["pit"] = float(np.mean(totals <= obs))
                row["covered_90"] = bool(row["total_q05"] <= obs <= row["total_q95"])
                row["mcr"] = (row["total_mean"] / obs) if obs > 0 else None
            out.append(row)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pooled_dir", type=Path)
    p.add_argument("--units", type=Path, default=None,
                   help="parquet mapping priogrid_id -> unit (e.g. the un_fao GAUL sidecar). "
                        "Without it, one GLOBAL unit.")
    p.add_argument("--actuals", type=Path, default=None,
                   help="parquet with month_id, unit, observed_total. Absent for a FORECASTING "
                        "run, where no observation exists yet.")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    targets = sorted(d for d in args.pooled_dir.iterdir() if d.is_dir() and d.name.startswith("lr_"))
    if not targets:
        print(f"no lr_* targets under {args.pooled_dir}")
        return 2

    _, _, cells = _load_pooled(targets[0])
    unit_of_cell, unit_names = _unit_map(args.units, cells)

    actuals = None
    if args.actuals is not None:
        import pandas as pd
        a = pd.read_parquet(args.actuals)
        missing = {"month_id", "unit", "observed_total"} - set(a.columns)
        if missing:
            raise SystemExit(f"--actuals is missing columns: {sorted(missing)}")
        actuals = {(int(r.month_id), str(r.unit)): float(r.observed_total) for r in a.itertuples()}

    report = {}
    for t in targets:
        report[t.name] = aggregate(t, unit_of_cell, unit_names, actuals)

    print("=== aggregate calibration ===")
    if args.actuals is None:
        print("  no --actuals: this is the FORECASTING case. Predictive totals and the")
        print("  dependence ratio only — PIT, coverage and MCR need observations that do not")
        print("  exist yet. Run against calibration or validation to score those.")
    for name, rows in report.items():
        ratios = [r["dependence_ratio"] for r in rows if r["dependence_ratio"] is not None]
        means = [r["total_mean"] for r in rows]
        print(f"  {name}: {len(rows)} (unit, month) aggregates")
        if means:
            print(f"    predictive total per aggregate — mean {np.mean(means):,.1f}, "
                  f"max {np.max(means):,.1f}")
        if ratios:
            med = float(np.median(ratios))
            near_one = float(np.mean([(0.9 <= x <= 1.1) for x in ratios]))
            print(f"    dependence_ratio median {med:.3f}; "
                  f"{near_one:.0%} of aggregates in [0.9, 1.1]")
            if near_one > 0.5:
                print("    *** F1: draws behave as INDEPENDENT per-cell marginals. Aggregate")
                print("    *** intervals are too narrow no matter how good the marginals are.")
        scored = [r for r in rows if "pit" in r]
        if scored:
            pits = [r["pit"] for r in scored]
            cov = float(np.mean([r["covered_90"] for r in scored]))
            mcrs = [r["mcr"] for r in scored if r["mcr"] is not None]
            print(f"    scored against actuals: {len(scored)} aggregates")
            print(f"      PIT median {np.median(pits):.3f}; share above 0.95: "
                  f"{np.mean([x > 0.95 for x in pits]):.0%}")
            print(f"      90% interval coverage {cov:.0%} (want ~0.90)")
            if mcrs:
                print(f"      MCR median {np.median(mcrs):.3f} (want ~1.0)")
            if np.mean([x > 0.95 for x in pits]) > 0.25:
                print("      *** F2: observed totals sit above the predictive distribution")
                print("      *** in a large share of aggregates — timid under-prediction.")
            if cov < 0.75:
                print("      *** F3: 90% intervals cover far less than 90%.")
            if mcrs and np.median(mcrs) < 0.5:
                print("      *** F4: predicted mean total is under half the observed.")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=1, default=str))
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
