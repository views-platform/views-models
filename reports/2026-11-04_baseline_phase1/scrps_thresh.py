"""Two tests datafactory specified.

(1) SCRPS (Bolin & Wallin 2023), pre-registered prediction: it WIDENS the spread on this arm set.
    SCRPS(F,y) = E|X-y| / E|X-X'| + 0.5*log E|X-X'|   (minimised)
    Ensemble forms, both O(m log m) via sorted draws:
      E|X-y|   = mean_i |x_i - y|
      E|X-X'|  = (2/m^2) * sum_i (2i - m - 1) * x_(i)      [Gini mean difference]
    NOTE: E|X-X'| = 0 for any DEGENERATE forecast (all draws equal). SCRPS is then undefined
    (division by zero, log 0). On a 98%-zero field that is most cells, so the share undefined is
    itself a result and is reported rather than silently dropped.

(2) Threshold decomposition (Gneiting & Ranjan 2011): CRPS = integral of Brier scores over
    thresholds. Per threshold t: BS(t) = mean over cells of (Fhat(t) - 1{y<=t})^2, where
    Fhat(t) = fraction of draws <= t. Localises WHERE in the value range the dynamic range lives.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/st'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]
THRESH = [0, 1, 2, 5, 10, 25, 100, 1000]

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'],457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

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
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]
    m = yy.shape[1]
    xs = np.sort(yy, axis=1)
    i = np.arange(1, m+1, dtype=np.float64)[None,:]
    # Gini mean difference = E|X-X'|
    gmd = (2.0/(m*m)) * np.sum((2*i - m - 1) * xs, axis=1)
    e_abs = np.abs(yy - truth[:,None]).mean(axis=1)
    nondeg = gmd > 0
    sc = np.full(yy.shape[0], np.nan)
    sc[nondeg] = e_abs[nondeg]/gmd[nondeg] + 0.5*np.log(gmd[nondeg])
    # threshold Brier
    bs = {}
    for th in THRESH:
        p = (yy <= th).mean(axis=1)
        bs[th] = float(np.mean((p - (truth <= th).astype(np.float64))**2))
    res[model] = dict(label=label, n=int(ok.sum()), nondeg=int(nondeg.sum()),
                      scrps=float(np.nanmean(sc[nondeg])), bs=bs)
    print(f"  {label:<18} nondegenerate {100*nondeg.mean():.2f}%", flush=True)
    del y, yy, xs; shutil.rmtree(d)

n = list(res.values())[0]['n']; nd = list(res.values())[0]['nondeg']
print(f"\n### (1) SCRPS — scored on the {nd:,} of {n:,} cells ({100*nd/n:.2f}%) with a NON-DEGENERATE forecast")
print(f"{'arm':<18}{'SCRPS':>12}{'nondeg cells':>14}")
for m2,r in sorted(res.items(), key=lambda kv: kv[1]['scrps']):
    print(f"{r['label']:<18}{r['scrps']:>12.5f}{r['nondeg']:>14,}")
v = np.array([r['scrps'] for r in res.values()])
print(f"\n  SCRPS spread {100*(v.max()-v.min())/abs(v.min()):.1f}%   range {v.max()/v.min():.3f}x")
print(f"  for comparison: CRPS as-is 4.0% | CRPS excl the outlier cell 6.5% | AP 4.4%")

print(f"\n### (2) THRESHOLD DECOMPOSITION — Brier score at each threshold, x1000")
hdr = f"{'arm':<18}" + "".join(f"{('t<='+str(th)):>10}" for th in THRESH)
print(hdr); print("-"*len(hdr))
for m2,r in res.items():
    print(f"{r['label']:<18}" + "".join(f"{1000*r['bs'][th]:>10.4f}" for th in THRESH))
print(f"\n{'spread %':<18}" + "".join(
    f"{100*(max(r['bs'][th] for r in res.values())-min(r['bs'][th] for r in res.values()))/min(r['bs'][th] for r in res.values()):>9.1f}%"
    for th in THRESH))
