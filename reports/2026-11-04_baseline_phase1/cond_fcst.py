"""Two fixes views-baseline specified.

(A) SIZE CALIBRATION CONDITIONED ON THE FORECAST (Murphy-Winkler). Bin conflict-firing cells by the
    PREDICTED median positive draw, then compare to realised truth within each bin. My earlier table
    binned by observed truth, which yields the regression-to-the-mean signature even for a perfect
    forecast, so it could not show a defect.

(B) RELIABILITY OF P(fire) PER HORIZON. Their hypothesis: the window is fixed at train_end (#82), so
    a cell's P(fire) is constant across all 36 horizons while reality mean-reverts. That predicts the
    "too extreme" shape should WORSEN with horizon. Test h=1 vs h=18 vs h=36.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/cf'); WORK.mkdir(parents=True, exist_ok=True)
ARMS = [("plain_burlap","A  resample"),("plush_velvet","C  KDE silverman"),("short_linen","B  interp short")]
PBINS = np.array([0.0, 1e-9, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.01])
SBINS = np.array([0.0, 1.5, 2.5, 5.5, 10.5, 25.5, 100.5, 1e9])   # predicted median positive draw

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'], 457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

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
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]; tm = z['time'][ok]
    months = np.unique(tm); h = np.searchsorted(months, tm) + 1      # horizon 1..36
    pf = (yy > 0).mean(axis=1)
    firing = yy > 0
    cmed = np.nanmedian(np.where(firing, yy, np.nan), axis=1)

    print(f"\n{'='*86}\n{label}   (horizons {h.min()}..{h.max()}, months {months.min()}..{months.max()})")

    print(f"\n(A) SIZE CALIBRATION CONDITIONED ON THE FORECAST — cells with >=1 positive draw")
    sel = firing.any(axis=1)
    print(f"    {'predicted median draw':>22}{'n':>10}{'mean pred':>11}{'mean truth':>12}{'median truth':>14}{'ratio':>8}")
    bi = np.digitize(cmed[sel], SBINS) - 1
    P, T = cmed[sel], truth[sel]
    for b in range(len(SBINS)-1):
        s = bi == b
        if s.sum() < 30: continue
        lo, hi = SBINS[b], SBINS[b+1]
        rt = T[s].mean()/P[s].mean() if P[s].mean() > 0 else np.nan
        print(f"    {f'{lo:.1f}-{hi:.1f}' if hi<1e8 else f'>{lo:.1f}':>22}{s.sum():>10,}{P[s].mean():>11.2f}{T[s].mean():>12.2f}{np.median(T[s]):>14.1f}{rt:>8.2f}")

    print(f"\n(B) P(fire) RELIABILITY BY HORIZON — top bin is the test of the #82 mechanism")
    print(f"    {'horizon':>9}{'bin':>14}{'n':>10}{'pred':>9}{'realised':>10}{'ratio':>8}")
    for hh in (1, 18, 36):
        m = h == hh
        bi2 = np.digitize(pf[m], PBINS) - 1
        for b in (len(PBINS)-2, len(PBINS)-3, 2):
            s = bi2 == b
            if s.sum() < 20: continue
            p, f = pf[m][s].mean(), (truth[m][s] > 0).mean()
            print(f"    {hh:>9}{f'{PBINS[b]:.2f}-{PBINS[b+1]:.2f}':>14}{s.sum():>10,}{p:>9.4f}{f:>10.4f}{(f/p if p>0 else np.nan):>8.2f}")
    del y, yy
    shutil.rmtree(d)
