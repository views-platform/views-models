"""Sensitivity of every battery number to the single month-504 outlier cell (lr_sb_best = 113,395).
If one cell moves the ranking, every conclusion drawn from these aggregates is about that cell.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/os'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]

def crps(d, y):
    m = d.shape[1]; x = np.sort(d, axis=1)
    i = np.arange(1, m+1, dtype=np.float64)[None, :]; yv = y[:, None]
    return (2.0/(m*m))*np.sum((x-yv)*(m*(yv<x)-i+0.5), axis=1)

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'],457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy(); mon = t['month_id'].to_numpy()
o = np.argsort(key); key, act, mon = key[o], act[o], mon[o]

res = {}
for model, label in ARMS:
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
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]; mm = mon[idx[ok]]
    c = crps(yy, truth)
    out = truth > 50000                      # the single outlier
    m504 = mm == 504
    res[model] = (label, float(c.mean()), float(c[~out].mean()), float(c[~m504].mean()),
                  float(c[out].sum()), float(c.sum()), int(out.sum()))
    print(f"  {label:<18} done", flush=True)
    del y, yy, c; shutil.rmtree(d)

print(f"\n{'arm':<18}{'CRPS as-is':>12}{'excl 1 cell':>13}{'excl mo504':>12}{'outlier % of total CRPS':>25}")
print("-"*82)
for m,(lab,a,b,cc,osum,tot,n) in sorted(res.items(), key=lambda kv: kv[1][1]):
    print(f"{lab:<18}{a:>12.5f}{b:>13.5f}{cc:>12.5f}{100*osum/tot:>24.1f}%")

print("\n=== RANK CHANGES ===")
for nm, i in (("as-is",1), ("excl the 1 outlier cell",2), ("excl all of month 504",3)):
    order = [res[m][0] for m in sorted(res, key=lambda m: res[m][i])]
    print(f"  {nm:<26} best -> worst: {' | '.join(o.split('  ')[0]+o.split('  ')[1][:9] for o in order)}")
for nm, i in (("as-is",1), ("excl 1 cell",2), ("excl mo504",3)):
    v = np.array([res[m][i] for m in res])
    print(f"  spread {nm:<14} {100*(v.max()-v.min())/v.min():.1f}%")
