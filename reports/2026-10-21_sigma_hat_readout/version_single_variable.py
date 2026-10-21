"""Pod five: the single-variable version test, and sigma-hat on a CLEAN same-version cell.

silver   = seed 42, views-hydranet 0.1.3, git 2e73de3, weight_decay 0.1
silver014= seed 42, views-hydranet 0.1.4, git 2e73de3, weight_decay 0.1   <- same everything else
So silver -> silver014 isolates the library version. Nothing else moved.

The as-launched M1 cell was {silver(0.1.3), bronze(0.1.4), cobalt(0.1.4)} -- it spans the
version boundary, which is C-167. Replacing silver with silver014 gives seeds 42/43/44 all on
0.1.4: the cell the prereg actually asked for.

Common bin support is prereg-mandated: a bin enters a fit only where EVERY arm in the cell has
0 < p < 1 (and, for the n>=1000 variant, n >= 1000 in every arm).
"""
import sys
import numpy as np
sys.path.insert(0, "/home/simon/Documents/scripts/views_platform/views-models/reports/2026-10-21_sigma_hat_readout")
from sigma_run_to_run import bin_table, logit, W5_THRESHOLD, FEASIBILITY

R = "/home/simon/Documents/scripts/views_platform/views-models/reports/2026-10-21_sigma_hat_readout/results"
TAB = {n: bin_table(f"{R}/p2_result_{n}.npz")
       for n in ("amber", "copper", "silver", "silver014", "bronze", "cobalt", "indigo")}


def common_support(names, min_n=None):
    """bin lower-edges usable in EVERY named arm."""
    keys = None
    for n in names:
        ok = {r[0] for r in TAB[n] if 0 < r[4] < 1 and (min_n is None or r[2] >= min_n)}
        keys = ok if keys is None else (keys & ok)
    return sorted(keys)


def fit_on(name, edges, weighted=False):
    rows = [r for r in TAB[name] if r[0] in edges]
    s = logit(np.array([r[3] for r in rows]))
    p = logit(np.array([r[4] for r in rows]))
    w = np.sqrt(np.array([r[2] for r in rows], float)) if weighted else None
    b, a = np.polyfit(s, p, 1, w=w)
    return a, b, len(rows)


EST = [("n>=1000", 1000, False), ("all_bins", None, False), ("sqrt_n", None, True)]

print("=" * 94)
print("A.  SINGLE-VARIABLE TEST -- silver(0.1.3) vs silver014(0.1.4), seed 42, git 2e73de3 both")
print("=" * 94)
for lbl, mn, wt in EST:
    e = common_support(["silver", "silver014"], mn)
    a0, b0, n0 = fit_on("silver", e, wt)
    a1, b1, n1 = fit_on("silver014", e, wt)
    print(f"  {lbl:<9} ({len(e)} common bins)   "
          f"silver b={b0:.4f} a={a0:+.4f}    silver014 b={b1:.4f} a={a1:+.4f}    "
          f"db={b1-b0:+.4f} da={a1-a0:+.4f}")

print()
print("  signed error (gate - observed p) per common bin, all-bins support:")
e = common_support(["silver", "silver014"], None)
print(f"    {'bin':<14}{'n(silv)':>9}{'silver':>11}{'silver014':>12}")
for lo in e:
    r0 = [r for r in TAB["silver"] if r[0] == lo][0]
    r1 = [r for r in TAB["silver014"] if r[0] == lo][0]
    print(f"    [{lo:.2f},{r0[1]:.2f}){'':<4}{r0[2]:>9}{r0[5]:>+11.4f}{r1[5]:>+12.4f}")
m0 = np.median([abs([r for r in TAB['silver'] if r[0]==lo][0][5]) for lo in e])
m1 = np.median([abs([r for r in TAB['silver014'] if r[0]==lo][0][5]) for lo in e])
print(f"    median |gate-p|:  silver {m0:.4f}   silver014 {m1:.4f}")
s0 = [np.sign([r for r in TAB['silver'] if r[0]==lo][0][5]) for lo in e]
s1 = [np.sign([r for r in TAB['silver014'] if r[0]==lo][0][5]) for lo in e]
print(f"    crosses zero?     silver {len(set(s0))>1}   silver014 {len(set(s1))>1}")

for title, cell in (("B.  CLEAN sigma-hat -- seeds 42/43/44 ALL on 0.1.4", ["silver014","bronze","cobalt"]),
                    ("C.  AS-LAUNCHED cell (spans the 0.1.3/0.1.4 boundary -- C-167)", ["silver","bronze","cobalt"])):
    print()
    print("=" * 94)
    print(title)
    print("=" * 94)
    sds = []
    for lbl, mn, wt in EST:
        e = common_support(cell, mn)
        bs = [fit_on(n, e, wt)[1] for n in cell]
        sd = float(np.array(bs).std(ddof=1))
        sds.append(sd)
        print(f"  {lbl:<9} ({len(e)} common bins)  b = [" + ", ".join(f"{x:.4f}" for x in bs) + "]"
              f"   mean {np.mean(bs):.4f}   sigma-hat_b = {sd:.4f}"
              f"   {'OVER' if sd > W5_THRESHOLD else 'under'} W-5 {W5_THRESHOLD}"
              f" / {'over' if sd > FEASIBILITY else 'under'} {FEASIBILITY}")
    lo, hi = min(sds), max(sds)
    print(f"  family: {lo:.4f} .. {hi:.4f}   straddles W-5? "
          f"{'YES -> NULL' if lo <= W5_THRESHOLD <= hi else 'no -> ROBUST ' + ('ABOVE' if lo > W5_THRESHOLD else 'BELOW')}")
    print(f"  straddles 0.0356 feasibility? {'YES -> NULL' if lo <= FEASIBILITY <= hi else 'no'}")
