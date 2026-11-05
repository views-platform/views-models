import numpy as np, sys
from pathlib import Path

root = Path(sys.argv[1]); label = sys.argv[2]
d = sorted(root.glob('predictions_calibration_*'))[0]
y = np.load(d/'origin_12'/'lr_sb_best'/'y_pred.npy')   # (rows, draws)
S = y.shape[1]

mx = y.max(axis=1)
print(f"=== {label}: origin_12 / lr_sb_best, {y.shape[0]:,} cell-horizons x {S} draws ===")
print(f"rows all-zero draws            : {(mx == 0).sum():,}  ({100*(mx==0).mean():.2f}%)")

# a gap is only *possible* where the draw range spans enough room
sel = mx >= 10
print(f"rows with max draw >= 10       : {sel.sum():,}  ({100*sel.mean():.3f}%)  <- the only rows where a bulk/tail gap can exist")
if sel.sum() == 0:
    raise SystemExit

sub = y[sel]
m = mx[sel]

# integer-valued draws => atoms of the empirical sample; non-integer => interpolated
frac_int = float(np.mean(sub == np.round(sub)))
print(f"draws that are exact integers  : {100*frac_int:.2f}%   (an empirical resampler is ~100%)")

# distinct draws per row
distinct = np.array([np.unique(r).size for r in sub])
print(f"distinct draw values per row   : median {np.median(distinct):.0f} of {S}   p10 {np.percentile(distinct,10):.0f}   p90 {np.percentile(distinct,90):.0f}")

# decile occupancy of [0, max]: how much of the row's own range carries any draw
edges = (np.arange(1, 10)[None, :] * m[:, None] / 10.0)
binned = (sub[:, :, None] > edges[:, None, :]).sum(axis=2)       # 0..9 per draw
occ = np.zeros((sub.shape[0], 10), dtype=bool)
rows = np.repeat(np.arange(sub.shape[0]), S)
occ[rows, binned.ravel()] = True
n_occ = occ.sum(axis=1)

# largest run of consecutive EMPTY deciles strictly inside the occupied span
def biggest_hole(o):
    idx = np.flatnonzero(o)
    if idx.size < 2: return 0
    return int(np.max(np.diff(idx)) - 1)
holes = np.array([biggest_hole(o) for o in occ])

print(f"occupied deciles of own range  : median {np.median(n_occ):.0f} of 10   p10 {np.percentile(n_occ,10):.0f}   p90 {np.percentile(n_occ,90):.0f}")
print(f"largest interior empty run     : median {np.median(holes):.0f} deciles   p90 {np.percentile(holes,90):.0f}   max {holes.max()}")
print(f"rows with NO interior hole     : {100*np.mean(holes == 0):.1f}%")
print(f"rows with a hole >= 3 deciles  : {100*np.mean(holes >= 3):.1f}%   <- bulk and tail with nothing between")

# worked example: the row with the largest max
i = int(np.argmax(m))
ex = np.unique(sub[i])
print(f"\nworked example (largest max, {ex.size} distinct of {S} draws):")
print("  " + np.array2string(ex[:14], precision=2, separator=', ') + (" ... " + np.array2string(ex[-4:], precision=2, separator=', ') if ex.size > 18 else ""))
