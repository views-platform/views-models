"""The metric battery from views-hydranet reports/2026-09-30_evaluation_learning_report.md
§2.2 (AP — dynamic range where CRPS has none), §2.4 (act_ratio, size_ratio-as-MEDIAN),
§2.3 (zero-share decomposition of the CRPS gap).

Method rule from §2.3: the reference's S must equal the arms' cube width. All ten arms here are
S=64, so it is satisfied. Stated because violating it silently is the documented trap.
"""
import numpy as np, subprocess, sys, shutil
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/bat'); WORK.mkdir(parents=True, exist_ok=True)
TARGET = 'lr_sb_best'
ORIGIN = 'origin_12'
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]

def crps_ensemble(draws, y):
    """CRPS via the sorted form: (2/m^2) sum_i (x_(i)-y)(m*1{y<x_(i)} - i + 0.5). O(m log m)."""
    m = draws.shape[1]
    x = np.sort(draws, axis=1)
    i = np.arange(1, m + 1, dtype=np.float64)[None, :]
    yv = y[:, None]
    return (2.0 / (m * m)) * np.sum((x - yv) * (m * (yv < x) - i + 0.5), axis=1)

# ---- actuals, test window only
print("loading actuals ...", flush=True)
t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id', 'priogrid_id', TARGET])
t = t.filter(pc.greater_equal(t['month_id'], 457))
mon = t['month_id'].to_numpy(); unit = t['priogrid_id'].to_numpy(); act = t[TARGET].to_numpy()
key_act = mon.astype(np.int64) * 10_000_000 + unit.astype(np.int64)
order = np.argsort(key_act); key_act, act = key_act[order], act[order]
print(f"  actuals: {act.shape[0]:,} rows, months 457+, prevalence(>0) = {100*(act>0).mean():.4f}%", flush=True)

def extract(model):
    d = WORK/model
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    tf = ROOT/'archives_s64'/f'{model}_s64.tar.zst'
    top = subprocess.run(['tar','-I','zstd -d','-tf',str(tf)], capture_output=True, text=True).stdout.split('\n')[0].strip('/')
    subprocess.run(['tar','-I','zstd -d','-xf',str(tf),'-C',str(d),
                    f'{top}/{ORIGIN}/{TARGET.replace("_best","")}_best/y_pred.npy',
                    f'{top}/{ORIGIN}/{TARGET.replace("_best","")}_best/identifiers.npz'], check=True)
    p = d/top/ORIGIN/f'{TARGET.replace("_best","")}_best'
    return np.load(p/'y_pred.npy'), np.load(p/'identifiers.npz')

def average_precision(score, label):
    o = np.argsort(-score, kind='stable')
    lab = label[o].astype(np.float64)
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, lab.size + 1)
    npos = lab.sum()
    return float(np.sum(prec * lab) / npos) if npos else float('nan')

rows = {}
for model, label in ARMS:
    y, z = extract(model)
    k = z['time'].astype(np.int64) * 10_000_000 + z['unit'].astype(np.int64)
    idx = np.searchsorted(key_act, k)
    ok = (idx < key_act.size)
    idx_ok = np.clip(idx, 0, key_act.size - 1)
    ok &= (key_act[idx_ok] == k)
    if ok.sum() == 0:
        print(f"{label}: NO ALIGNMENT — refusing to report (ADR-025)"); shutil.rmtree(WORK/model); continue
    yy = y[ok].astype(np.float64); truth = act[idx_ok[ok]]
    p_pos = (yy > 0).mean(axis=1)
    guess = yy.mean(axis=1)
    pos = truth > 0
    cr = crps_ensemble(yy, truth)
    rows[model] = dict(
        label=label, n=int(ok.sum()), aligned=100*ok.mean(),
        crps=float(cr.mean()),
        ap=average_precision(p_pos, pos),
        prevalence=float(pos.mean()),
        act_ratio=float(p_pos.mean() / pos.mean()) if pos.mean() else float('nan'),
        size_ratio=float(np.median(guess[pos] / truth[pos])) if pos.sum() else float('nan'),
        pos_mcr=float(np.mean(guess[pos] / truth[pos])) if pos.sum() else float('nan'),
        crps_zero=float(cr[~pos].sum()), crps_pos=float(cr[pos].sum()),
    )
    print(f"  {label:<18} aligned={rows[model]['aligned']:.1f}%  n={rows[model]['n']:,}", flush=True)
    del y, yy, cr
    shutil.rmtree(WORK/model)

np.save('/home/simon/.claude/jobs/3583e9b8/tmp/battery_rows.npy', rows, allow_pickle=True)
print("\nsaved")
