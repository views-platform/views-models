"""Conditional size, per views-baseline's correction.

My earlier size_ratio = median over {truth>0} of (mean of ALL draws / truth). The mean of all draws
mixes occurrence into a magnitude statistic: mean_all ~ P(fire) * E[size | fires]. This script
separates them:
  p_fire_pos   = mean over {truth>0} of P(draw > 0)            <- occurrence, on conflict cells
  cond_mean    = median over {truth>0} of (mean of POSITIVE draws / truth)
  cond_median  = median over {truth>0} of (median of POSITIVE draws / truth)
  size_ratio   = the old, mixed statistic, retained for comparison
Cells where NO draw is positive have undefined conditional size and are excluded from cond_*;
their count is reported rather than silently dropped.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/cs'); WORK.mkdir(parents=True, exist_ok=True)
TARGET, ORIGIN = 'lr_sb_best', 'origin_12'
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id',TARGET])
t = t.filter(pc.greater_equal(t['month_id'], 457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t[TARGET].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

res = {}
for model, label in ARMS:
    d = WORK/model
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    tf = ROOT/'archives_s64'/f'{model}_s64.tar.zst'
    top = subprocess.run(['tar','-I','zstd -d','-tf',str(tf)],capture_output=True,text=True).stdout.split('\n')[0].strip('/')
    sub = f'{top}/{ORIGIN}/lr_sb_best'
    subprocess.run(['tar','-I','zstd -d','-xf',str(tf),'-C',str(d),f'{sub}/y_pred.npy',f'{sub}/identifiers.npz'],check=True)
    y = np.load(d/sub/'y_pred.npy'); z = np.load(d/sub/'identifiers.npz')
    k = z['time'].astype(np.int64)*10_000_000 + z['unit'].astype(np.int64)
    idx = np.clip(np.searchsorted(key,k),0,key.size-1); ok = key[idx]==k
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]
    posm = truth > 0
    Y = yy[posm]; T = truth[posm]                 # conflict cells only
    firing = Y > 0
    nfire = firing.sum(axis=1)
    p_fire = (nfire / Y.shape[1])
    have = nfire > 0
    # conditional size: mean / median of POSITIVE draws only
    sums = np.where(firing, Y, 0.0).sum(axis=1)
    cmean = np.full(Y.shape[0], np.nan); cmean[have] = sums[have]/nfire[have]
    Ym = np.where(firing, Y, np.nan)
    cmed = np.nanmedian(Ym, axis=1)
    res[model] = dict(label=label, n_pos=int(posm.sum()),
        no_fire=int((~have).sum()),
        p_fire=float(p_fire.mean()),
        cond_mean=float(np.median(cmean[have]/T[have])),
        cond_med=float(np.median(cmed[have]/T[have])),
        mixed=float(np.median(yy[posm].mean(axis=1)/T)))
    print(f"  {label:<18} done", flush=True)
    del y, yy, Y, Ym
    shutil.rmtree(d)

print(f"\nlr_sb_best / origin_12 / S=64 — conflict cells (truth>0): {list(res.values())[0]['n_pos']:,}\n")
h=f"{'arm':<18}{'P(fire)':>9}{'cond_mean':>11}{'cond_med':>10}{'old mixed':>11}{'no-fire cells':>14}"
print(h); print("-"*len(h))
for m,r in sorted(res.items(), key=lambda kv:-kv[1]['cond_mean']):
    print(f"{r['label']:<18}{r['p_fire']:>9.4f}{r['cond_mean']:>11.4f}{r['cond_med']:>10.4f}{r['mixed']:>11.4f}{r['no_fire']:>13,}")
print("\n=== dynamic range ===")
for k2,n in [('p_fire','P(fire)'),('cond_mean','cond_mean'),('cond_med','cond_med'),('mixed','old mixed')]:
    v=np.array([r[k2] for r in res.values()])
    print(f"  {n:<11} min {v.min():.4f} max {v.max():.4f}  spread {100*(v.max()-v.min())/v.min():>7.1f}%  range {v.max()/v.min():.2f}x")
