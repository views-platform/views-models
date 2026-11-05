"""SCRPS on the INTERSECTION of cells where EVERY arm has a non-degenerate forecast.

The unpaired run scored arms on 32,026 to 54,086 cells each, and the arm penalised hardest was the
one scored on the fewest. Paired on a common support, the comparison is valid. Two passes: pass 1
builds the intersection mask, pass 2 scores on it.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/sp'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]

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

def gmd_of(yy):
    m = yy.shape[1]; xs = np.sort(yy, axis=1)
    i = np.arange(1, m+1, dtype=np.float64)[None,:]
    return (2.0/(m*m)) * np.sum((2*i - m - 1) * xs, axis=1)

print("pass 1: intersection of non-degenerate support", flush=True)
mask = None
for model, label in ARMS:
    yy, truth, d = load(model)
    nd = gmd_of(yy) > 0
    mask = nd if mask is None else (mask & nd)
    print(f"  {label:<18} nondeg {nd.sum():>8,}  running intersection {mask.sum():>8,}", flush=True)
    del yy; shutil.rmtree(d)

print(f"\npass 2: SCRPS on the common {mask.sum():,} cells ({100*mask.mean():.3f}% of all)", flush=True)
res = {}
for model, label in ARMS:
    yy, truth, d = load(model)
    yy, truth = yy[mask], truth[mask]
    g = gmd_of(yy)
    e = np.abs(yy - truth[:,None]).mean(axis=1)
    sc = e/g + 0.5*np.log(g)
    m2 = yy.shape[1]; xs = np.sort(yy, axis=1)
    i = np.arange(1, m2+1, dtype=np.float64)[None,:]
    cr = (2.0/(m2*m2))*np.sum((xs - truth[:,None])*(m2*(truth[:,None] < xs) - i + 0.5), axis=1)
    res[model] = (label, float(sc.mean()), float(cr.mean()))
    del yy; shutil.rmtree(d)

print(f"\n{'arm':<18}{'SCRPS':>12}{'CRPS':>12}   (same {mask.sum():,} cells)")
print("-"*56)
for m2,(lab,s,c) in sorted(res.items(), key=lambda kv: kv[1][1]):
    print(f"{lab:<18}{s:>12.5f}{c:>12.4f}")
vs = np.array([r[1] for r in res.values()]); vc = np.array([r[2] for r in res.values()])
print(f"\n  SCRPS spread {100*(vs.max()-vs.min())/abs(vs.min()):>6.1f}%   range {vs.max()/vs.min():.3f}x")
print(f"  CRPS  spread {100*(vc.max()-vc.min())/vc.min():>6.1f}%   range {vc.max()/vc.min():.3f}x   (on the same cells)")
print(f"\n  unpaired SCRPS ordering had B interp short LAST on 32,026 cells; paired ordering above.")
