"""Two measurements views-baseline asked for.

(1) Conditional size STRATIFIED by truth magnitude. cond_med=1.0 on nine arms is suspected to be
    mostly integer ties (median positive draw 1 vs truth 1). Stratify: truth==1, 2..10, >10.
(2) RELIABILITY of P(fire): bin ALL cells by predicted P(draw>0), compare to realised frequency of
    truth>0 in each bin. This is the calibration axis the battery lacks.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/sr'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("short_linen","B  interp short"),("plain_linen","B  interp mid"),
        ("long_linen","B  interp long"),("sheer_velvet","C  KDE sj"),("plush_velvet","C  KDE silverman"),
        ("flat_velvet","C  KDE fixed1.0"),("light_tweed","E  mix a=0.25"),("even_tweed","E  mix a=0.50"),
        ("heavy_tweed","E  mix a=0.75")]
BINS = np.array([0.0, 1e-9, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.01])

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'], 457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

strat, rel = {}, {}
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
    pf = (yy > 0).mean(axis=1)

    # ---- (2) reliability over ALL cells
    bi = np.digitize(pf, BINS) - 1
    rows = []
    for b in range(len(BINS)-1):
        s = bi == b
        if s.sum() == 0: continue
        rows.append((BINS[b], BINS[b+1], int(s.sum()), float(pf[s].mean()), float((truth[s] > 0).mean())))
    rel[model] = (label, rows)
    # expected calibration error, weighted
    n = yy.shape[0]
    ece = sum(c*abs(p-f) for _,_,c,p,f in rows)/n

    # ---- (1) stratified conditional size
    posm = truth > 0
    Y, T = yy[posm], truth[posm]
    firing = Y > 0; nf = firing.sum(axis=1); have = nf > 0
    cmean = np.full(Y.shape[0], np.nan); cmean[have] = np.where(firing,Y,0.0).sum(axis=1)[have]/nf[have]
    cmed = np.nanmedian(np.where(firing,Y,np.nan), axis=1)
    buckets = {'truth==1': T==1, 'truth 2-10': (T>=2)&(T<=10), 'truth >10': T>10}
    st = {'ece': ece, 'n_pos': int(posm.sum())}
    for bn, bm in buckets.items():
        m = bm & have
        st[bn] = (int(bm.sum()), int((bm & ~have).sum()),
                  float(np.median(cmean[m]/T[m])) if m.sum() else float('nan'),
                  float(np.median(cmed[m]/T[m])) if m.sum() else float('nan'))
    strat[model] = (label, st)
    print(f"  {label:<18} done", flush=True)
    del y, yy, Y
    shutil.rmtree(d)

n_pos = list(strat.values())[0][1]['n_pos']
b1 = list(strat.values())[0][1]['truth==1'][0]
print(f"\n### (1) CONDITIONAL SIZE BY TRUTH MAGNITUDE — lr_sb_best/origin_12/S=64")
print(f"conflict cells {n_pos:,}; truth==1 is {b1:,} = {100*b1/n_pos:.1f}% of them\n")
print(f"{'arm':<18}{'t=1 n':>8}{'cmean':>8}{'cmed':>7}  |{'t2-10 n':>9}{'cmean':>8}{'cmed':>7}  |{'t>10 n':>8}{'cmean':>8}{'cmed':>7}")
print("-"*96)
for m,(label,st) in strat.items():
    a,b,c = st['truth==1'], st['truth 2-10'], st['truth >10']
    print(f"{label:<18}{a[0]:>8,}{a[2]:>8.3f}{a[3]:>7.2f}  |{b[0]:>9,}{b[2]:>8.3f}{b[3]:>7.2f}  |{c[0]:>8,}{c[2]:>8.3f}{c[3]:>7.2f}")

print(f"\n### (2) RELIABILITY OF P(fire) — predicted vs realised, all {len(act):,}-cell aligned set")
print(f"{'arm':<18}{'ECE':>9}")
for m,(label,st) in strat.items(): print(f"{label:<18}{st['ece']:>9.5f}")
print(f"\nper-bin, arm A and the two extremes:")
for m in ('plain_burlap','short_linen','plush_velvet'):
    label, rows = rel[m]
    print(f"\n  {label}")
    print(f"    {'bin':>14}{'n':>12}{'pred':>9}{'realised':>10}{'ratio':>8}")
    for lo,hi,c,p,f in rows:
        r = f/p if p>0 else float('nan')
        print(f"    {f'{lo:.2f}-{hi:.2f}':>14}{c:>12,}{p:>9.4f}{f:>10.4f}{r:>8.2f}")
