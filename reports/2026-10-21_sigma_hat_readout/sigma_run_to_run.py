"""Run-to-run sd of the gate calibration slope b  -- epic #565's M1 (σ̂_b).

NAME: deliberately NOT "sigma". views-hydranet's e1_sigma.py computes the ACROSS-DRAW logit sd
(D=4, floors 0.85/1.20). This computes the ACROSS-RUN sd of a fitted slope (threshold ~0.0601).
The two differ by ~20x in scale and answer different questions; conflating them was a live risk.

bin_table() and logit() are a deliberate verbatim copy of
  views-hydranet/reports/2026-10-16_pos_weight_arm_outcome/evidence/verify_slope.py
Importing that module is not possible -- it loads amber/copper npz at module level. Copying keeps
this read-only with respect to the other seat's repo. Extraction belongs in views-hydranet.

⛔ The prereg (2026-10-18_weight_decay_arms_prereg.md:676,682) names σ̂_b and its thresholds but
does NOT pin the estimator -- bin filter and weighting are unspecified, and verify_slope.py offers
four. So every variant is reported. If the family straddles a threshold, the reading is a NULL.
"""
import sys
import numpy as np

BINS = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
W5_THRESHOLD = 0.0601      # prereg:682  -- amber-vs-copper resolvability
# ⛔ Amendment 10 (:1018) RETIRES 0.0356 as "1.81x too harsh" -- it was derived for n=1 vs n=2
# and the approved set is n=3 vs n=2. Use the live boundaries; 0.0356 is kept only so an old
# number in the record can be traced to the reading that produced it.
FEASIBILITY_POOLED = 0.0645        # prereg:1015 -- n=2 vs n=3, df=3, 2.905*S
FEASIBILITY_CONSERVATIVE = 0.0477  # prereg:1016 -- sigma-hat from the 3 controls only, df=2
FEASIBILITY = FEASIBILITY_POOLED   # the default reading
FEASIBILITY_RETIRED = 0.0356       # prereg:397 -- superseded, do not report against this


def bin_table(path):
    d = np.load(path, allow_pickle=True)
    gm = d["gate"].mean(axis=0).ravel()
    yy = (d["truth"].ravel() > 0).astype(float)
    idx = np.digitize(gm, BINS) - 1
    out = []
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        s, p = float(gm[m].mean()), float(yy[m].mean())
        iw = s * (1 - p) / (p * (1 - s)) if 0 < p < 1 else np.nan
        out.append((BINS[b], BINS[b + 1], n, s, p, s - p, iw))
    return out


def logit(x):
    return np.log(x / (1 - x))


def fits(R):
    """b under every estimator verify_slope.py offers. Returns {variant: (a, b, n_bins)}."""
    usable = [r for r in R if 0 < r[4] < 1]
    out = {}
    for lbl, sub in (("n>=1000", [r for r in usable if r[2] >= 1000]), ("all_bins", usable)):
        if len(sub) < 2:
            out[lbl] = (np.nan, np.nan, len(sub))
            continue
        b, a = np.polyfit(logit(np.array([r[3] for r in sub])),
                          logit(np.array([r[4] for r in sub])), 1)
        out[lbl] = (a, b, len(sub))
    wt = np.sqrt(np.array([r[2] for r in usable], float))
    b, a = np.polyfit(logit(np.array([r[3] for r in usable])),
                      logit(np.array([r[4] for r in usable])), 1, w=wt)
    out["sqrt_n_weighted"] = (a, b, len(usable))
    return out


if __name__ == "__main__":
    RUNS = [(n, p) for n, p in (("silver(42)", sys.argv[1]), ("bronze(43)", sys.argv[2]),
                                ("cobalt(44)", sys.argv[3]))]
    tabs = {n: fits(bin_table(p)) for n, p in RUNS}
    variants = ["n>=1000", "all_bins", "sqrt_n_weighted"]

    print("=" * 86)
    print("M1  σ̂_b — run-to-run sd of the gate calibration slope over 3 seed replicates (df=2)")
    print("=" * 86)
    for v in variants:
        print(f"\n--- estimator: {v} ---")
        bs, as_ = [], []
        for n, _ in RUNS:
            a, b, nb = tabs[n][v]
            print(f"    {n:<12} a={a:+.4f}  b={b:.4f}   ({nb} bins)")
            bs.append(b)
            as_.append(a)
        bs = np.array(bs)
        as_ = np.array(as_)
        sd_b = float(bs.std(ddof=1))
        sd_a = float(as_.std(ddof=1))
        print(f"    mean b = {bs.mean():.4f}     ⛔ σ̂_b = {sd_b:.4f}   (sd of a, for reference: {sd_a:.4f})")
        w5 = "EXCEEDED -> W-5 FIRES" if sd_b > W5_THRESHOLD else "under"
        fe = "exceeded" if sd_b > FEASIBILITY else "under"
        print(f"    vs W-5 {W5_THRESHOLD}: {w5}        vs 0.0356 feasibility: {fe}")

    print("\n" + "=" * 86)
    sds = {v: float(np.array([tabs[n][v][1] for n, _ in RUNS]).std(ddof=1)) for v in variants}
    lo, hi = min(sds.values()), max(sds.values())
    print(f"σ̂_b across estimators: {lo:.4f} .. {hi:.4f}")
    straddles = lo <= W5_THRESHOLD <= hi
    print(f"family straddles W-5 ({W5_THRESHOLD})?  {'YES -> READING IS A NULL' if straddles else 'no'}")
    if not straddles:
        print(f"W-5 verdict is ROBUST to the estimator: σ̂_b is {'ABOVE' if lo > W5_THRESHOLD else 'BELOW'} 0.0601 under all three")
    print("=" * 86)
