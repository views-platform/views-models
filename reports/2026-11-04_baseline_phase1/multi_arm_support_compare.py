import numpy as np, sys
from pathlib import Path
base = Path(sys.argv[1])
ARMS = [("plain_burlap", "A  empirical resample"),
        ("short_linen",  "B  interp-quantile, short window"),
        ("plain_linen",  "B  interp-quantile, mid window"),
        ("long_linen",   "B  interp-quantile, long window"),
        ("sheer_velvet", "C  KDE, SJ bandwidth"),
        ("even_tweed",   "E  empirical-KDE mixture, a=0.5")]

Y = {m: np.load(base/m/'origin_12'/'lr_sb_best'/'y_pred.npy') for m, _ in ARMS}
S = next(iter(Y.values())).shape[1]
n = next(iter(Y.values())).shape[0]
assert all(v.shape == (n, S) for v in Y.values())
print(f"matched at n={S} draws, {n:,} cell-horizons, origin_12 / lr_sb_best\n")

def holes(y):
    m = y.max(axis=1)
    edges = (np.arange(1, 10)[None, :] * m[:, None] / 10.0)
    binned = (y[:, :, None] > edges[:, None, :]).sum(axis=2)
    occ = np.zeros((y.shape[0], 10), dtype=bool)
    occ[np.repeat(np.arange(y.shape[0]), S), binned.ravel()] = True
    out = np.empty(y.shape[0], dtype=int)
    for i, o in enumerate(occ):
        idx = np.flatnonzero(o)
        out[i] = 0 if idx.size < 2 else int(np.max(np.diff(idx)) - 1)
    return out, occ.sum(axis=1)

print(f"{'arm':<34} {'all-zero':>9} {'eligible':>9} {'distinct':>9} {'occ.dec':>8} {'hole':>6} {'hole>=3':>8}")
for m, label in ARMS:
    y = Y[m]; mx = y.max(axis=1); sel = mx >= 10
    sub = y[sel]
    h, occ = holes(sub)
    dist = np.array([np.unique(r).size for r in sub])
    print(f"{label:<34} {100*(mx==0).mean():>8.2f}% {sel.sum():>9,} {np.median(dist):>9.0f} "
          f"{np.median(occ):>8.0f} {np.median(h):>6.0f} {100*np.mean(h>=3):>7.1f}%")

# paired against the control on cells eligible in BOTH
ctrl = "plain_burlap"
selC = Y[ctrl].max(axis=1) >= 10
print(f"\n=== each arm paired against {ctrl} (A), on cells eligible in both ===")
print(f"{'arm':<34} {'n paired':>9} {'novel interior':>15} {'median#':>8} {'hole<A':>8} {'hole>A':>8}")
for m, label in ARMS:
    if m == ctrl: continue
    both = selC & (Y[m].max(axis=1) >= 10)
    a, b = Y[ctrl][both], Y[m][both]
    ha, _ = holes(a); hb, _ = holes(b)
    novel = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        atoms = np.unique(a[i]); bd = np.unique(b[i])
        inner = bd[(bd > atoms.min()) & (bd < atoms.max())]
        novel[i] = np.isin(inner, atoms, invert=True).sum()
    print(f"{label:<34} {both.sum():>9,} {100*np.mean(novel>0):>14.1f}% {np.median(novel):>8.0f} "
          f"{100*np.mean(hb<ha):>7.1f}% {100*np.mean(hb>ha):>7.1f}%")
print("\nnovel interior = rows drawing >=1 value inside A's range but absent from A's support")
