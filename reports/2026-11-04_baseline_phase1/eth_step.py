"""Is the 501-504 collapse Ethiopia-specific (a #533 aggregation artefact) or global
(real conflict dynamics / model decay)?

Hypothesis: UCDP event 463131 assigns a YEAR of Tigray fatalities to cell 148759 at month 504.
If the aggregation also strips events from the OTHER Ethiopian cells that would normally record
them, those cells read zero — and they are exactly the high-P(fire) cells. That would collapse the
top-bin realised rate in Ethiopia while leaving global prevalence normal.

Test: split the top-bin ratio by country. c_id 124 is the outlier cell's country.
If the step is Ethiopia-only -> my "temporal decay" campaign headline is a data artefact.
If it is global -> the step is real and the headline survives (as a step, not a decay).
"""
import numpy as np, subprocess, shutil
from pathlib import Path
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path('/home/simon/Documents/scripts/views_platform/views-models/reports/2026-11-04_baseline_phase1')
WORK = Path('/home/simon/.claude/jobs/3583e9b8/tmp/es'); WORK.mkdir(parents=True, exist_ok=True)
ETH = 124.0

t = pq.read_table(ROOT/'input_s64'/'calibration_datafactory_df.parquet',
                  columns=['month_id','priogrid_id','lr_sb_best','c_id'])
t = t.filter(pc.greater_equal(t['month_id'],457))
key = t['month_id'].to_numpy().astype(np.int64)*10_000_000 + t['priogrid_id'].to_numpy().astype(np.int64)
act = t['lr_sb_best'].to_numpy(); cid = t['c_id'].to_numpy()
o = np.argsort(key); key, act, cid = key[o], act[o], cid[o]

# context: what does Ethiopia's own event record look like across the window?
print("### Ethiopia (c_id 124) state-based events by month — is the record itself odd at 501-504?")
print(f"    {'month':>7}{'cells>0':>9}{'total events':>14}{'max cell':>11}")
for m in range(495, 505):
    s = (t['month_id'].to_numpy()[o] == m) & (cid == ETH)
    a = act[s]
    print(f"    {m:>7}{int((a>0).sum()):>9}{a.sum():>14,.0f}{a.max() if a.size else 0:>11,.0f}")

tf = ROOT/'archives_s64'/'plain_burlap_s64.tar.zst'
top = subprocess.run(['tar','-I','zstd -d','-tf',str(tf)],capture_output=True,text=True).stdout.split('\n')[0].strip('/')
rows = []
for org in range(13):
    d = WORK/f'o{org}'
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    sub = f'{top}/origin_{org}/lr_sb_best'
    subprocess.run(['tar','-I','zstd -d','-xf',str(tf),'-C',str(d),f'{sub}/y_pred.npy',f'{sub}/identifiers.npz'],check=True)
    y = np.load(d/sub/'y_pred.npy'); z = np.load(d/sub/'identifiers.npz')
    k = z['time'].astype(np.int64)*10_000_000 + z['unit'].astype(np.int64)
    idx = np.clip(np.searchsorted(key,k),0,key.size-1); ok = key[idx]==k
    yy = y[ok].astype(np.float64); truth = act[idx[ok]]; cc = cid[idx[ok]]; tm = z['time'][ok]
    pf = (yy > 0).mean(axis=1); sel = pf >= 0.70
    for m in np.unique(tm):
        for nm, cm in (('ETH', cc == ETH), ('rest', cc != ETH)):
            s = sel & (tm == m) & cm
            if s.sum() == 0: continue
            rows.append((int(m), nm == 'ETH', int(s.sum()), float(pf[s].sum()), int((truth[s] > 0).sum())))
    del y, yy; shutil.rmtree(d)
    print(f"  origin_{org} done", flush=True)

R = np.array([(a,b,c,d_,e) for a,b,c,d_,e in rows], dtype=np.float64)
mon_, is_eth, n_, sp_, nr_ = R.T

def rat(mask):
    n = n_[mask].sum()
    if n < 20: return None
    return sp_[mask].sum()/n, nr_[mask].sum()/n, int(n)

print(f"\n### TOP-BIN RATIO BY MONTH, ETHIOPIA vs REST OF WORLD")
print(f"    {'month':>7}  {'ETH n':>7}{'ETH ratio':>11}   {'rest n':>8}{'rest ratio':>12}")
for m in sorted(set(mon_.astype(int))):
    if m < 492: continue
    e = rat((mon_ == m) & (is_eth == 1)); r = rat((mon_ == m) & (is_eth == 0))
    es = f"{e[2]:>7,}{e[1]/e[0]:>11.2f}" if e else f"{'—':>7}{'—':>11}"
    rs = f"{r[2]:>8,}{r[1]/r[0]:>12.2f}" if r else f"{'—':>8}{'—':>12}"
    print(f"    {m:>7}  {es}   {rs}")

print(f"\n### POOLED: months 457-500 vs 501-504")
for nm, cm in (('ETHIOPIA', is_eth == 1), ('REST OF WORLD', is_eth == 0)):
    a = rat(cm & (mon_ <= 500)); b = rat(cm & (mon_ >= 501))
    if a and b:
        print(f"    {nm:<15} 457-500: {a[1]/a[0]:.2f} (n={a[2]:,})   501-504: {b[1]/b[0]:.2f} (n={b[2]:,})   change {b[1]/b[0] - a[1]/a[0]:+.2f}")
    else:
        print(f"    {nm:<15} insufficient n")
