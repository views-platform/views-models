"""Does the LEVEL (a) track the library version or the seed?

σ̂_b over {silver,bronze,cobalt} is dominated by silver. silver is also the only arm on
views_hydranet 0.1.3 (bronze/cobalt/indigo are 0.1.4), so 'being silver' and 'being 0.1.3'
are perfectly confounded in the σ̂ set.

indigo breaks that confound in one direction: it shares silver's SEED (42) but bronze/cobalt's
VERSION (0.1.4). It differs in weight_decay (0.0 vs 0.1), which is its own confound -- so this
discriminates, it does not prove.
"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from sigma_run_to_run import bin_table, fits

SP = sys.argv[1] + "/p2_results"
ARMS = [("silver", 42, "0.1.3", 0.1), ("bronze", 43, "0.1.4", 0.1),
        ("cobalt", 44, "0.1.4", 0.1), ("indigo", 42, "0.1.4", 0.0)]
T = {n: fits(bin_table(f"{SP}/p2_result_{n}.npz")) for n, _, _, _ in ARMS}
V = ["n>=1000", "all_bins", "sqrt_n_weighted"]

print("="*94)
print(f"{'arm':<9}{'seed':>5}{'ver':>8}{'wd':>5}   " + "".join(f"{v+' a':>19}" for v in V))
print("="*94)
for n, s, ver, wd in ARMS:
    print(f"{n:<9}{s:>5}{ver:>8}{wd:>5}   " + "".join(f"{T[n][v][0]:>+19.4f}" for v in V))
print(f"\n{'arm':<9}{'seed':>5}{'ver':>8}{'wd':>5}   " + "".join(f"{v+' b':>19}" for v in V))
for n, s, ver, wd in ARMS:
    print(f"{n:<9}{s:>5}{ver:>8}{wd:>5}   " + "".join(f"{T[n][v][1]:>19.4f}" for v in V))

print("\n" + "="*94)
print("σ̂_b stratified: does silver carry the variance?")
print("="*94)
for v in V:
    three = np.array([T[n][v][1] for n in ("silver","bronze","cobalt")])
    two   = np.array([T[n][v][1] for n in ("bronze","cobalt")])
    print(f"  {v:<17} σ̂_b all three (df=2) = {three.std(ddof=1):.4f}   "
          f"σ̂_b 0.1.4 pair only (df=1) = {two.std(ddof=1):.4f}   "
          f"ratio = {three.std(ddof=1)/two.std(ddof=1):.1f}x")

print("\n" + "="*94)
print("the discriminating test: indigo shares silver's SEED but bronze/cobalt's VERSION")
print("="*94)
for v in V:
    a = {n: T[n][v][0] for n, _, _, _ in ARMS}
    side = "with silver (seed)" if a["indigo"] > 0 else "with bronze/cobalt (version)"
    print(f"  {v:<17} silver {a['silver']:+.4f} | bronze {a['bronze']:+.4f} "
          f"cobalt {a['cobalt']:+.4f} | indigo {a['indigo']:+.4f}  -> indigo sides {side}")
