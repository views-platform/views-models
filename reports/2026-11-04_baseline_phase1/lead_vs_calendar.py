"""Lead vs calendar, for views-baseline ADR-024 / PR #128.

IDENTIFICATION. One artifact serves all 13 origins, so the resampling pool is fixed at
train_end=456. Therefore "months since the history ended" = target_month - 456, which is
COLLINEAR with calendar month. Those two cannot be separated.

What IS separable: horizon-from-origin. Month 480 appears at h=24 in origin_0, h=18 in origin_6,
h=12 in origin_12. So:

  variation across HORIZON at a fixed calendar month  -> an origin/horizon effect
  variation across CALENDAR MONTH pooled over horizon -> a lead-or-calendar effect (one variable)
  and the SHAPE of the calendar profile separates smooth decay from a localised shock.

Metric: top-bin (predicted P(fire) >= 0.70) realised/predicted ratio, pooled. Reported with and
without month 504 (the views-datafactory#533 cell). Month 514 is outside this pull.
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/lv'); WORK.mkdir(parents=True, exist_ok=True)
MODEL, LABEL = 'plain_burlap', 'A resample (control)'
OUTLIER_MONTH = 504

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best'])
t = t.filter(pc.greater_equal(t['month_id'],457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy()
o = np.argsort(key); key, act = key[o], act[o]

tf = ROOT/'archives_s64'/f'{MODEL}_s64.tar.zst'
top = subprocess.run(['tar','-I','zstd -d','-tf',str(tf)],capture_output=True,text=True).stdout.split('\n')[0].strip('/')

rows = []   # (origin, calendar_month, horizon, n, sum_pred, n_realised)
for org in range(13):
    d = WORK/f'o{org}'
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    sub = f'{top}/origin_{org}/lr_sb_best'
    subprocess.run(['tar','-I','zstd -d','-xf',str(tf),'-C',str(d),f'{sub}/y_pred.npy',f'{sub}/identifiers.npz'],check=True)
    y = np.load(d/sub/'y_pred.npy'); z = np.load(d/sub/'identifiers.npz')
    k = z['time'].astype(np.int64)*10_000_000 + z['unit'].astype(np.int64)
    idx = np.clip(np.searchsorted(key,k),0,key.size-1); ok = key[idx]==k
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]; tm = z['time'][ok]
    months = np.unique(tm); h_of = {m: i+1 for i, m in enumerate(months)}
    pf = (yy > 0).mean(axis=1)
    sel = pf >= 0.70
    for m in months:
        s = sel & (tm == m)
        if s.sum() == 0: continue
        rows.append((org, int(m), h_of[m], int(s.sum()), float(pf[s].sum()), int((truth[s] > 0).sum())))
    print(f"  origin_{org:>2} done", flush=True)
    del y, yy; shutil.rmtree(d)

R = np.array(rows, dtype=np.float64)
org_, mon_, h_, n_, sp_, nr_ = R.T

def ratio(mask):
    n = n_[mask].sum()
    if n == 0: return float('nan'), float('nan'), 0
    pred = sp_[mask].sum()/n; real = nr_[mask].sum()/n
    return pred, real, int(n)

for tag, keep in (("WITH month 504", np.ones(len(R), bool)), ("WITHOUT month 504", mon_ != OUTLIER_MONTH)):
    print(f"\n{'='*78}\n{LABEL} — top bin (predicted >= 0.70) — {tag}\n{'='*78}")

    print("\n(A) BY HORIZON-FROM-ORIGIN, pooled over all calendar months and origins")
    print(f"    {'horizon':>12}{'n':>9}{'pred':>8}{'realised':>10}{'ratio':>8}")
    for lo, hi, nm in ((1,12,'h1-12'),(13,24,'h13-24'),(25,36,'h25-36')):
        m = keep & (h_ >= lo) & (h_ <= hi)
        p, r, n = ratio(m)
        print(f"    {nm:>12}{n:>9,}{p:>8.4f}{r:>10.4f}{r/p:>8.2f}")

    print("\n(B) BY CALENDAR MONTH (= lead from 456, collinear), pooled over horizon and origin")
    print(f"    {'months':>12}{'lead':>9}{'n':>9}{'pred':>8}{'realised':>10}{'ratio':>8}")
    for lo, hi in ((457,472),(473,488),(489,504)):
        m = keep & (mon_ >= lo) & (mon_ <= hi)
        p, r, n = ratio(m)
        if n == 0: continue
        print(f"    {f'{lo}-{hi}':>12}{f'{lo-456}-{hi-456}':>9}{n:>9,}{p:>8.4f}{r:>10.4f}{r/p:>8.2f}")

    print("\n(C) THE IDENTIFYING TEST — ratio by horizon AT A FIXED CALENDAR MONTH BAND")
    print("    if the ratio moves across a row -> horizon/origin effect")
    print("    if it only moves down a column  -> calendar/lead effect")
    print(f"    {'cal months':>12}{'h1-12':>14}{'h13-24':>14}{'h25-36':>14}")
    for lo, hi in ((469,480),(481,492),(493,504)):
        cells = []
        for hlo, hhi in ((1,12),(13,24),(25,36)):
            m = keep & (mon_ >= lo) & (mon_ <= hi) & (h_ >= hlo) & (h_ <= hhi)
            p, r, n = ratio(m)
            cells.append(f"{r/p:.2f} (n={n:,})" if n > 50 else "—")
        print(f"    {f'{lo}-{hi}':>12}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}")

print(f"\n(D) PER-MONTH PROFILE — smooth decay, or a localised shock?")
print(f"    {'month':>7}{'lead':>6}{'n':>8}{'pred':>8}{'realised':>10}{'ratio':>8}")
for m in sorted(set(mon_.astype(int))):
    msk = mon_ == m
    p, r, n = ratio(msk)
    flag = '  <- #533 cell' if m == OUTLIER_MONTH else ''
    print(f"    {m:>7}{m-456:>6}{n:>8,}{p:>8.4f}{r:>10.4f}{r/p:>8.2f}{flag}")
