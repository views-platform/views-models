import numpy as np, sys
from pathlib import Path
base = Path(sys.argv[1])

def load(m):
    return np.load(base/m/'origin_12'/'lr_sb_best'/'y_pred.npy')

A = load('plain_burlap')   # arm A: empirical resampling
B = load('short_linen')    # arm B: interpolated quantile
S = A.shape[1]
assert A.shape == B.shape, (A.shape, B.shape)
print(f"matched at n={S} draws, {A.shape[0]:,} cell-horizons, origin_12 / lr_sb_best\n")

def stats(y, name):
    mx = y.max(axis=1)
    sel = mx >= 10
    sub, m = y[sel], mx[sel]
    distinct = np.array([np.unique(r).size for r in sub])
    edges = (np.arange(1, 10)[None, :] * m[:, None] / 10.0)
    binned = (sub[:, :, None] > edges[:, None, :]).sum(axis=2)
    occ = np.zeros((sub.shape[0], 10), dtype=bool)
    occ[np.repeat(np.arange(sub.shape[0]), S), binned.ravel()] = True
    def hole(o):
        i = np.flatnonzero(o)
        return 0 if i.size < 2 else int(np.max(np.diff(i)) - 1)
    holes = np.array([hole(o) for o in occ])
    print(f"{name}")
    print(f"  all-zero rows                 : {100*(mx==0).mean():.2f}%")
    print(f"  eligible rows (max >= 10)     : {sel.sum():,}")
    print(f"  distinct draws / row (median) : {np.median(distinct):.0f} of {S}")
    print(f"  occupied deciles (median)     : {np.median(occ.sum(axis=1)):.0f} of 10")
    print(f"  largest interior hole (median): {np.median(holes):.0f}   p90 {np.percentile(holes,90):.0f}")
    print(f"  rows with NO interior hole    : {100*np.mean(holes==0):.1f}%")
    print(f"  rows with hole >= 3 deciles   : {100*np.mean(holes>=3):.1f}%")
    return sel, distinct, holes

selA, dA, hA = stats(A, "A  plain_burlap  (empirical resampling)")
print()
selB, dB, hB = stats(B, "B  short_linen   (interpolated quantile)")

# paired comparison on the rows eligible in BOTH arms — same cells, same draw budget
both = selA & selB
print(f"\n=== paired on the {both.sum():,} cell-horizons eligible in BOTH arms ===")
sa, sb = A[both], B[both]
da = np.array([np.unique(r).size for r in sa])
db = np.array([np.unique(r).size for r in sb])
print(f"distinct draws / row : A median {np.median(da):.0f}   B median {np.median(db):.0f}   B>A in {100*np.mean(db>da):.1f}% of rows")

def holes_of(y):
    m = y.max(axis=1)
    edges = (np.arange(1, 10)[None, :] * m[:, None] / 10.0)
    binned = (y[:, :, None] > edges[:, None, :]).sum(axis=2)
    occ = np.zeros((y.shape[0], 10), dtype=bool)
    occ[np.repeat(np.arange(y.shape[0]), S), binned.ravel()] = True
    out = np.empty(y.shape[0], dtype=int)
    for i, o in enumerate(occ):
        idx = np.flatnonzero(o)
        out[i] = 0 if idx.size < 2 else int(np.max(np.diff(idx)) - 1)
    return out

ha, hb = holes_of(sa), holes_of(sb)
print(f"largest interior hole: A median {np.median(ha):.0f}   B median {np.median(hb):.0f}")
print(f"  B strictly smaller hole than A : {100*np.mean(hb<ha):.1f}% of rows")
print(f"  equal                          : {100*np.mean(hb==ha):.1f}%")
print(f"  B strictly LARGER hole than A  : {100*np.mean(hb>ha):.1f}%")

# the decisive test: values B draws that A's support cannot contain
print("\n=== interior values B produces that A's own draw support does not ===")
novel = np.zeros(sa.shape[0])
for i in range(sa.shape[0]):
    atoms = np.unique(sa[i])
    bd = np.unique(sb[i])
    inner = bd[(bd > atoms.min()) & (bd < atoms.max())]
    novel[i] = np.isin(inner, atoms, invert=True).sum()
print(f"rows where B draws >=1 interior value absent from A : {100*np.mean(novel>0):.1f}%")
print(f"median count of such values per row                 : {np.median(novel):.0f}")
print(f"p90                                                 : {np.percentile(novel,90):.0f}")
