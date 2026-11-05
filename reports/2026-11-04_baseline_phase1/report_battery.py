import numpy as np
rows = np.load('/home/simon/.claude/jobs/3583e9b8/tmp/battery_rows.npy', allow_pickle=True).item()
prev = list(rows.values())[0]['prevalence']
print(f"lr_sb_best, origin_12, S=64, n=2,333,448 cell-horizons, prevalence(y>0) = {100*prev:.4f}%\n")

def rng(key, lo_is_best=True):
    v = [r[key] for r in rows.values()]
    return max(v)/min(v) if min(v) > 0 else float('nan')

hdr = f"{'arm':<18}{'CRPS':>10}{'AP':>9}{'AP/prev':>9}{'act_ratio':>11}{'size_ratio':>11}{'pos_mcr':>9}{'zero-share':>11}"
print(hdr); print("-"*len(hdr))
for m, r in sorted(rows.items(), key=lambda kv: -kv[1]['ap']):
    zs = r['crps_zero']/(r['crps_zero']+r['crps_pos'])
    print(f"{r['label']:<18}{r['crps']:>10.5f}{r['ap']:>9.4f}{r['ap']/prev:>9.1f}x"
          f"{r['act_ratio']:>11.3f}{r['size_ratio']:>11.4f}{r['pos_mcr']:>9.2f}{100*zs:>10.1f}%")

print("\n=== DYNAMIC RANGE across the ten arms (the §2.2 test) ===")
for key, name in [('crps','CRPS'),('ap','AP'),('act_ratio','act_ratio'),('size_ratio','size_ratio'),('pos_mcr','pos_mcr')]:
    v = np.array([r[key] for r in rows.values()])
    print(f"  {name:<12} min {v.min():.5f}  max {v.max():.5f}  spread {100*(v.max()-v.min())/v.min():>8.1f}%   range {v.max()/v.min():.2f}x")
