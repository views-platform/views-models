"""#572 readout: M1-M4 on COMMON BIN SUPPORT, signed error alongside absolute, sqrt-2 shown.

Common bin support (prereg, non-optional): a bin enters the fit only if EVERY model has it with
n>=1 and 0 < observed p < 1. Fitting each model on its own bins makes b differences partly a
difference in which bins were fitted -- the pseudo-replication error's cousin.
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from sigma_run_to_run import bin_table, logit, W5_THRESHOLD, FEASIBILITY

# One directory for the sweep artifacts: `results/`. These three read
# "/p2_results" while sweep_all.sh wrote there and the per-head harness and
# version_single_variable.py both used "results" -- which is the directory that
# actually holds the published npz. Two names for one artifact class meant a
# regeneration left the headline single-variable test reading the stale set.
SP = sys.argv[1] + "/results"
ARMS = [("amber", 42, "0.1.2", 0.1, 1.0), ("copper", 42, "0.1.3", 0.1, 2.0),
        ("silver", 42, "0.1.3", 0.1, 1.0), ("bronze", 43, "0.1.4", 0.1, 1.0),
        ("cobalt", 44, "0.1.4", 0.1, 1.0), ("indigo", 42, "0.1.4", 0.0, 1.0)]
TAB = {n: {r[0]: r for r in bin_table(f"{SP}/p2_result_{n}.npz")} for n, *_ in ARMS}

usable = [{lo for lo, r in t.items() if 0 < r[4] < 1} for t in TAB.values()]
COMMON = sorted(set.intersection(*usable))
big = [lo for lo in COMMON if all(TAB[n][lo][2] >= 1000 for n, *_ in ARMS)]
print(f"common support      : {len(COMMON)} bins  {['%.2f' % b for b in COMMON]}")
print(f"common & n>=1000    : {len(big)} bins  {['%.2f' % b for b in big]}")


def fit(n, bins, weighted):
    rs = [TAB[n][lo] for lo in bins]
    x, y = logit(np.array([r[3] for r in rs])), logit(np.array([r[4] for r in rs]))
    w = np.sqrt(np.array([r[2] for r in rs], float)) if weighted else None
    b, a = np.polyfit(x, y, 1, w=w) if w is not None else np.polyfit(x, y, 1)
    return a, b


VAR = [("common n>=1000", big, False), ("common all", COMMON, False),
       ("common sqrt-n", COMMON, True)]
F = {n: {lab: fit(n, bins, w) for lab, bins, w in VAR} for n, *_ in ARMS}

for q, lbl in ((1, "b  (SLOPE)  -- honest gate = 1.000"), (0, "a  (LEVEL)  -- honest gate = 0.000")):
    print("\n" + "=" * 88)
    print(lbl)
    print(f"{'arm':<8}{'seed':>5}{'ver':>8}{'wd':>5}{'posw':>6}   " + "".join(f"{v:>17}" for v, _, _ in VAR))
    print("=" * 88)
    for n, s, ver, wd, pw in ARMS:
        print(f"{n:<8}{s:>5}{ver:>8}{wd:>5}{pw:>6}   " + "".join(f"{F[n][v][q]:>+17.4f}" for v, _, _ in VAR))

print("\n" + "=" * 88)
print("SIGNED error (gate - observed p) on common support -- prereg requires it beside |.|")
print("=" * 88)
print(f"{'arm':<8}" + "".join(f"{('[%.2f)' % lo):>10}" for lo in COMMON))
for n, *_ in ARMS:
    print(f"{n:<8}" + "".join(f"{TAB[n][lo][5]:>+10.4f}" for lo in COMMON))
print(f"\n{'arm':<8}{'median |gate-p| (common)':>26}{'median signed':>16}")
for n, *_ in ARMS:
    e = np.array([TAB[n][lo][5] for lo in COMMON])
    print(f"{n:<8}{np.median(np.abs(e)):>26.4f}{np.median(e):>+16.4f}")

print("\n" + "=" * 88)
print("M1  σ̂_b  -- raw spread and the sqrt-2 conversion, per the acceptance criteria")
print("=" * 88)
SETS = [("as launched {silver42,bronze43,cobalt44}  MIXED VERSION", ["silver","bronze","cobalt"]),
        ("same-version pair {bronze43,cobalt44}  0.1.4 only",       ["bronze","cobalt"])]
for v, _, _ in VAR:
    print(f"\n  --- {v} ---")
    for lab, names in SETS:
        bs = np.array([F[n][v][1] for n in names])
        sd = bs.std(ddof=1)
        extra = f"   raw |diff| = {abs(bs[0]-bs[1]):.4f}  -> /sqrt(2) = {abs(bs[0]-bs[1])/np.sqrt(2):.4f}" if len(bs) == 2 else \
                f"   range = {bs.max()-bs.min():.4f}"
        print(f"      n={len(bs)} df={len(bs)-1}  σ̂_b = {sd:.4f}{extra}")
        print(f"          {lab}")
        # ⛔ These thresholds are IMPORTED, not retyped. This line hardcoded 0.0356 —
        # retired by prereg Amendment 10 — so re-running the readout reproduced the exact
        # mis-reading the record retracts. A threshold with a provenance must have exactly
        # one definition; sigma_run_to_run.py is it.
        print(f"          vs W-5 {W5_THRESHOLD}: {'FIRES' if sd > W5_THRESHOLD else 'under'}    "
              f"vs feasibility {FEASIBILITY}: {'exceeded' if sd > FEASIBILITY else 'under'}")
