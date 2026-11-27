"""P1b: is the written output actually RECONCILED, or merely written?

Reconciliation means: summing the pgm grid draws to country-month must reproduce the cm
country forecast, draw for draw (ALIGNED_DRAWS). Checks a bounded sample, mmap only.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/workspace/views-models")
sys.path.insert(0, str(REPO))
from reconciliation.datafactory_country_mapping_provider import (
    DatafactoryCountryMappingProvider,
)

TARGET = "lr_sb_best"
PGM = REPO / "ensembles/chrome_box/data/generated/predictions_forecasting_20261028_165332" / TARGET
CM = REPO / "ensembles/chrome_crate/data/generated/predictions_forecasting_20261028_120904" / TARGET

pgm_ids = np.load(PGM / "identifiers.npz")
cm_ids = np.load(CM / "identifiers.npz")
print("pgm identifier keys:", list(pgm_ids.keys()))
print("cm  identifier keys:", list(cm_ids.keys()))

pgm_y = np.load(PGM / "y_pred.npy", mmap_mode="r")
cm_y = np.load(CM / "y_pred.npy", mmap_mode="r")
print("pgm", pgm_y.shape, "cm", cm_y.shape)

pgm_t = np.asarray(pgm_ids["time"], dtype=np.int64)
pgm_u = np.asarray(pgm_ids["unit"], dtype=np.int64)
cm_t = np.asarray(cm_ids["time"], dtype=np.int64)
cm_u = np.asarray(cm_ids["unit"], dtype=np.int64)

# grid -> country, from the same provider the run used
p = DatafactoryCountryMappingProvider(int(pgm_t.min()), int(pgm_t.max()))
m = p.build()
g2c = dict(zip(m.map_keys[:, 1].tolist(), m.map_vals.tolist()))
codes = np.array([g2c.get(int(g), -1) for g in pgm_u], dtype=np.int64)
print("grid cells with no country in mapping:", int((codes < 0).sum()))

# Check a bounded sample: 3 months x 3 countries x draws 0 and 511
months = np.unique(pgm_t)[[0, 12, 35]]
rng = np.random.default_rng(0)
report = []
for t in months:
    cm_here = cm_u[cm_t == t]
    for c in rng.choice(cm_here, size=3, replace=False):
        sel = (pgm_t == t) & (codes == c)
        n = int(sel.sum())
        if n == 0:
            report.append((int(t), int(c), 0, None, None, None))
            continue
        cm_row = np.where((cm_t == t) & (cm_u == c))[0][0]
        for draw in (0, 511):
            got = float(np.asarray(pgm_y[sel, draw]).sum())
            want = float(cm_y[cm_row, draw])
            rel = abs(got - want) / max(abs(want), 1e-9)
            report.append((int(t), int(c), n, draw, want, got, rel))

print("%6s %8s %6s %5s %14s %14s %10s" % ("month","country","cells","draw","cm","sum(pgm)","rel.err"))
worst = 0.0
for r in report:
    if len(r) == 6:
        print(f"{r[0]:>6} {r[1]:>8} {r[2]:>6}   NO GRID CELLS FOR THIS COUNTRY")
        continue
    t, c, n, draw, want, got, rel = r
    worst = max(worst, rel)
    print(f"{t:>6} {c:>8} {n:>6} {draw:>5} {want:>14.4f} {got:>14.4f} {rel:>10.2e}")
print()
print(f"WORST relative error: {worst:.3e}")
print("VERDICT:", "RECONCILED" if worst < 1e-4 else "NOT RECONCILED")
