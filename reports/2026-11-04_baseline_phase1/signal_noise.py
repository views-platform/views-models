"""Signal-to-noise for each metric, per views-hydranet's reframing:
a metric's dynamic range is only meaningful against its OWN noise on the same population.

All ten arms declare seed 42, so no seed replicates exist and seed noise cannot be measured here.
Substitute: SPLIT-HALF draw noise. For each arm, compute every metric on draws 0..31 and again on
draws 32..63 — same model, same data, same cells, different draw subsample. The between-half spread
is a within-arm noise estimate.

CAVEAT built in and reported: halves are S=32 while the headline numbers are S=64, so split-half
noise OVERSTATES the S=64 noise by roughly sqrt(2) for sampling-limited quantities. The ratio below
is therefore a CONSERVATIVE (low) estimate of signal-to-noise.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/sn'); WORK.mkdir(parents=True, exist_ok=True)
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

def crps(d, y):
    m = d.shape[1]; x = np.sort(d, axis=1)
    i = np.arange(1, m+1, dtype=np.float64)[None,:]; yv = y[:,None]
    return (2.0/(m*m))*np.sum((x-yv)*(m*(yv<x)-i+0.5), axis=1)

def ap(score, label):
    o2 = np.argsort(-score, kind='stable'); lab = label[o2].astype(np.float64)
    tp = np.cumsum(lab); prec = tp/np.arange(1, lab.size+1); n = lab.sum()
    return float(np.sum(prec*lab)/n) if n else float('nan')

def metrics(yy, truth):
    pf = (yy > 0).mean(axis=1)
    pos = truth > 0
    return dict(crps=float(crps(yy, truth).mean()),
                ap=ap(pf, pos),
                act_ratio=float(pf.mean()/pos.mean()))

full, halves = {}, {}
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
    full[model]   = (label, metrics(yy, truth))
    halves[model] = (label, metrics(yy[:, :32], truth), metrics(yy[:, 32:], truth))
    print(f"  {label:<18} done", flush=True)
    del y, yy; shutil.rmtree(d)

print(f"\n### SIGNAL vs NOISE — lr_sb_best / origin_12")
print(f"{'metric':<12}{'between-arm (signal)':>22}{'within-arm split-half (noise)':>31}{'S/N':>8}")
print("-"*74)
for mk, name in (('crps','CRPS'),('ap','AP'),('act_ratio','act_ratio')):
    v = np.array([full[m][1][mk] for m,_ in ARMS])
    sig = (v.max()-v.min())/v.min()
    # noise: mean |half1 - half2| / full, across arms
    noise = np.mean([abs(halves[m][1][mk]-halves[m][2][mk])/full[m][1][mk] for m,_ in ARMS])
    print(f"{name:<12}{100*sig:>21.2f}%{100*noise:>30.2f}%{sig/noise if noise>0 else float('inf'):>8.2f}x")
print(f"\n  noise = mean over arms of |metric(draws 0-31) - metric(draws 32-63)| / metric(all 64)")
print(f"  halves are S=32 vs headline S=64, so noise is OVERSTATED ~sqrt(2) and S/N is CONSERVATIVE")
print(f"\n### per-arm split-half detail (AP)")
for m,_ in ARMS:
    l, h1, h2 = halves[m]
    print(f"  {l:<18} h1 {h1['ap']:.5f}  h2 {h2['ap']:.5f}  full {full[m][1]['ap']:.5f}  |diff| {abs(h1['ap']-h2['ap']):.5f}")
