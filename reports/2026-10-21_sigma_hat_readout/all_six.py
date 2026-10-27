"""The check I should have run before advising the purchase.

amber is silver's EXACT config (seed 42, weight_decay 0.1) on views-hydranet 0.1.2.
If the level (a) tracks the library version, amber and silver sit together and apart from
the 0.1.4 arms. If amber sits with the 0.1.4 arms, silver is simply a noisy run and the
version story collapses -- which would mean sigma-hat is honest after all.
"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from sigma_run_to_run import bin_table, fits

# One directory for the sweep artifacts: `results/`. These three read
# "/p2_results" while sweep_all.sh wrote there and the per-head harness and
# version_single_variable.py both used "results" -- which is the directory that
# actually holds the published npz. Two names for one artifact class meant a
# regeneration left the headline single-variable test reading the stale set.
SP = sys.argv[1] + "/results"
# name, seed, version, weight_decay, pos_weight
ARMS = [("amber",  42, "0.1.2", 0.1, 1.0), ("copper", 42, "0.1.3", 0.1, 2.0),
        ("silver", 42, "0.1.3", 0.1, 1.0), ("bronze", 43, "0.1.4", 0.1, 1.0),
        ("cobalt", 44, "0.1.4", 0.1, 1.0), ("indigo", 42, "0.1.4", 0.0, 1.0)]
T = {n: fits(bin_table(f"{SP}/p2_result_{n}.npz")) for n, *_ in ARMS}
V = ["n>=1000", "all_bins", "sqrt_n_weighted"]

for q, lbl in ((0, "a  (LEVEL)"), (1, "b  (SLOPE)")):
    print("=" * 92)
    print(f"{lbl}")
    print(f"{'arm':<8}{'seed':>5}{'ver':>8}{'wd':>5}{'posw':>6}   " + "".join(f"{v:>18}" for v in V))
    print("=" * 92)
    for n, s, ver, wd, pw in ARMS:
        mark = "  <-- 0.1.4" if ver == "0.1.4" else ""
        print(f"{n:<8}{s:>5}{ver:>8}{wd:>5}{pw:>6}   "
              + "".join(f"{T[n][v][q]:>+18.4f}" for v in V) + mark)
    print()

print("=" * 92)
print("the test: amber is silver's config on 0.1.2.  Does the LEVEL split by version?")
print("=" * 92)
for v in V:
    a = {n: T[n][v][0] for n, *_ in ARMS}
    old = [a["amber"], a["copper"], a["silver"]]
    new = [a["bronze"], a["cobalt"], a["indigo"]]
    # ⛔ Two-sided. This read `min(old) > max(new)` only, so a perfectly clean separation
    # running the OTHER way would have printed "NO clean split" -- a one-sided criterion
    # answering a two-sided question. The docstring calls this "the check I should have run
    # before advising the purchase"; a check that can only see one direction is half a check.
    if min(old) > max(new):
        sep = "CLEAN SPLIT (<=0.1.3 above)"
    elif min(new) > max(old):
        sep = "CLEAN SPLIT (0.1.4 above)"
    else:
        sep = "NO clean split"
    print(f"  {v:<17} <=0.1.3: [{', '.join(f'{x:+.4f}' for x in old)}]   "
          f"0.1.4: [{', '.join(f'{x:+.4f}' for x in new)}]   -> {sep}")

print()
print("=" * 92)
print("sigma-hat_b candidates")
print("=" * 92)
for v in V:
    b = {n: T[n][v][1] for n, *_ in ARMS}
    sets = {
        "as-launched {silver,bronze,cobalt}  (mixed version)": ["silver", "bronze", "cobalt"],
        "same-version pair {bronze,cobalt}   (df=1)":          ["bronze", "cobalt"],
        "same-seed-42 {amber,silver,indigo}  (mixed wd+ver)":  ["amber", "silver", "indigo"],
    }
    print(f"  --- {v} ---")
    for lab, names in sets.items():
        arr = np.array([b[n] for n in names])
        print(f"      {lab:<52} sigma_b = {arr.std(ddof=1):.4f}")
