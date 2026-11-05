"""The three tests views-hydranet specified, plus the fine-grid integral my 8-point table could not
support.

(1) FINE-GRID THRESHOLD INTEGRAL. CRPS = integral of BS(t) dt. My earlier table sampled 8 thresholds
    with wildly unequal spacing and I compared HEIGHTS as if they were contributions. Compute BS(t)
    on a dense grid, integrate properly (trapezoid), and report per-region CONTRIBUTION and
    per-region integrated ABSOLUTE spread.
(2) RANK CROSSING. Does arm rank at low t differ from arm rank at high t? Spearman between the
    per-arm BS ordering at each threshold and at t=0. If orderings decorrelate, differences CANCEL in
    the integral and no single scalar can represent the comparison -- a stronger finding than
    dilution, and it has a different remedy.
(3) RESTRICTED AP. Rank the arms by AP on exactly the 25,601 cells where every arm is
    non-degenerate. If SCRPS agrees with that restricted AP, SCRPS's verdict reversal is a selection
    effect on the active subset, not a magnitude signal.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/tf'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]
# dense grid: every integer 0..200, then geometric to 2000 (draws max 1526)
GRID = np.unique(np.concatenate([np.arange(0, 201), np.round(np.geomspace(201, 2000, 60))]))

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'],457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

def load(model):
    d = WORK/model
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    tf = ROOT/'archives_s64'/f'{model}_s64.tar.zst'
    top = subprocess.run(['tar','-I','zstd -d','-tf',str(tf)],capture_output=True,text=True).stdout.split('\n')[0].strip('/')
    sub = f'{top}/origin_12/lr_sb_best'
    subprocess.run(['tar','-I','zstd -d','-xf',str(tf),'-C',str(d),f'{sub}/y_pred.npy',f'{sub}/identifiers.npz'],check=True)
    y = np.load(d/sub/'y_pred.npy'); z = np.load(d/sub/'identifiers.npz')
    k = z['time'].astype(np.int64)*10_000_000 + z['unit'].astype(np.int64)
    idx = np.clip(np.searchsorted(key,k),0,key.size-1); ok = key[idx]==k
    return y[ok].astype(np.float64), act[idx[ok]], d

def gmd(yy):
    m = yy.shape[1]; xs = np.sort(yy, axis=1)
    i = np.arange(1, m+1, dtype=np.float64)[None,:]
    return (2.0/(m*m))*np.sum((2*i-m-1)*xs, axis=1)

def ap(score, label):
    o2 = np.argsort(-score, kind='stable'); lab = label[o2].astype(np.float64)
    tp = np.cumsum(lab); prec = tp/np.arange(1, lab.size+1); n = lab.sum()
    return float(np.sum(prec*lab)/n) if n else float('nan')

# pass 1: the common non-degenerate mask (same as the paired SCRPS run)
mask = None
for model, label in ARMS:
    yy, truth, d = load(model)
    nd = gmd(yy) > 0
    mask = nd if mask is None else (mask & nd)
    del yy; shutil.rmtree(d)
print(f"common non-degenerate cells: {mask.sum():,}", flush=True)

curves, aps = {}, {}
for model, label in ARMS:
    yy, truth, d = load(model)
    bs = np.empty(GRID.size)
    for j, th in enumerate(GRID):
        p = (yy <= th).mean(axis=1)
        bs[j] = np.mean((p - (truth <= th))**2)
    curves[model] = (label, bs)
    aps[model] = (label, ap((yy[mask] > 0).mean(axis=1), truth[mask] > 0))
    print(f"  {label:<18} done", flush=True)
    del yy; shutil.rmtree(d)

B = np.array([curves[m][1] for m,_ in ARMS])           # (10, G)
labels = [curves[m][0] for m,_ in ARMS]
integral = np.trapz(B, GRID, axis=1)
absspread = B.max(axis=0) - B.min(axis=0)
tot = np.trapz(absspread, GRID)

print(f"\n### (1) FINE-GRID INTEGRAL — {GRID.size} thresholds, trapezoid")
print(f"{'region':>16}{'contribution':>15}{'share':>9}{'abs spread':>13}{'share':>9}")
for lo, hi in [(0,1),(1,10),(10,25),(25,100),(100,1000),(1000,2000)]:
    s = (GRID >= lo) & (GRID <= hi)
    if s.sum() < 2: continue
    c = np.trapz(B.mean(axis=0)[s], GRID[s]); a = np.trapz(absspread[s], GRID[s])
    print(f"{f'[{lo}, {hi}]':>16}{c:>15.4f}{100*c/np.trapz(B.mean(axis=0),GRID):>8.1f}%{a:>13.4f}{100*a/tot:>8.1f}%")
print(f"\n  mean CRPS from the integral: {integral.mean():.5f}   (direct per-cell CRPS was ~0.1268)")
print(f"  integrated absolute spread / integral = {100*tot/np.trapz(B.mean(axis=0),GRID):.2f}%")
print(f"  pooled CRPS relative spread (measured earlier) = 4.0%")

print(f"\n### (2) RANK CROSSING — Spearman of per-arm BS ordering vs the ordering at t=0")
from scipy.stats import spearmanr
r0 = np.argsort(np.argsort(B[:,0]))
for th in (0,1,2,5,10,25,50,100,200,500,1000):
    j = int(np.argmin(np.abs(GRID-th)))
    rj = np.argsort(np.argsort(B[:,j]))
    rho = spearmanr(r0, rj).statistic
    best = labels[int(np.argmin(B[:,j]))]
    print(f"  t<={GRID[j]:>6.0f}   rho vs t=0 {rho:>6.3f}   best arm: {best}")

print(f"\n### (3) RESTRICTED AP on the same {mask.sum():,} cells — hydranet's selection-effect separator")
print(f"{'arm':<18}{'AP (restricted)':>17}")
for m,_ in sorted(ARMS, key=lambda a: -aps[a[0]][1]):
    print(f"{aps[m][0]:<18}{aps[m][1]:>17.5f}")
scrps_order = ["A  resample","C  KDE sj","B  interp mid","E  mix a=0.75","C  KDE silverman",
               "C  KDE fixed1.0","E  mix a=0.25","E  mix a=0.50","B  interp short","B  interp long"]
ap_order = [aps[m][0] for m,_ in sorted(ARMS, key=lambda a: -aps[a[0]][1])]
rs = spearmanr([scrps_order.index(l) for l in ap_order], list(range(10))).statistic
print(f"\n  Spearman(SCRPS rank, restricted-AP rank) = {rs:.3f}")
print(f"  if ~1.0 -> SCRPS adds little beyond occurrence on the subset where it is defined")
