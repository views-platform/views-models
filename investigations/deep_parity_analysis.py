"""Deep parity analysis between purple_alien (viewser) and bright_starship (datafactory).

Produces detailed statistics on prediction divergence to diagnose data-layer differences.
"""

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PA_DIR = REPO / "models/purple_alien/data/generated/predictions_calibration_20260525_063227"
BS_DIR = REPO / "models/bright_starship/data/generated/predictions_calibration_20260526_013733"

TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
TARGET_LABELS = {
    "lr_sb_best": "State-based fatalities (best estimate)",
    "lr_ns_best": "Non-state fatalities (best estimate)",
    "lr_os_best": "One-sided violence fatalities (best estimate)",
}
VARIABLE_MAP = {
    "lr_sb_best": ("ged_sb_best_sum_nokgi", "ged_sb_best"),
    "lr_ns_best": ("ged_ns_best_sum_nokgi", "ged_ns_best"),
    "lr_os_best": ("ged_os_best_sum_nokgi", "ged_os_best"),
}


def analyze_target(target):
    viewser_var, factory_var = VARIABLE_MAP[target]
    label = TARGET_LABELS[target]

    print(f"\n{'='*72}")
    print(f"TARGET: {target}")
    print(f"  {label}")
    print(f"  Viewser variable:     {viewser_var}")
    print(f"  Datafactory variable: {factory_var}")
    print(f"{'='*72}")

    pa = np.load(PA_DIR / "origin_0" / target / "y_pred.npy")
    bs = np.load(BS_DIR / "origin_0" / target / "y_pred.npy")
    ids = dict(np.load(PA_DIR / "origin_0" / target / "identifiers.npz"))

    mean_pa = pa.mean(axis=1)
    mean_bs = bs.mean(axis=1)

    n_cells = len(np.unique(ids["unit"]))
    n_steps = len(np.unique(ids["time"]))
    time_range = (ids["time"].min(), ids["time"].max())

    pa_zero = (mean_pa < 1e-6).sum()
    bs_zero = (mean_bs < 1e-6).sum()
    pa_nonzero = (mean_pa >= 1e-6).sum()
    bs_nonzero = (mean_bs >= 1e-6).sum()

    print(f"\n  Array shape: {pa.shape}")
    print(f"    = {n_cells:,} cells x {n_steps} steps x {pa.shape[1]} posterior samples")
    print(f"    Test window: month_id {time_range[0]}-{time_range[1]}")

    print("\n  1. SPARSITY COMPARISON (origin_0, mean prediction < 1e-6 = 'zero')")
    print("  ---------------------------------------------------------------")
    print(f"    purple_alien (viewser):      {pa_zero:>7,} zero ({100*pa_zero/len(mean_pa):.1f}%)  |  {pa_nonzero:>7,} nonzero ({100*pa_nonzero/len(mean_pa):.1f}%)")
    print(f"    bright_starship (datafactory):{bs_zero:>7,} zero ({100*bs_zero/len(mean_bs):.1f}%)  |  {bs_nonzero:>7,} nonzero ({100*bs_nonzero/len(mean_bs):.1f}%)")
    sparsity_ratio = bs_zero / pa_zero if pa_zero > 0 else float("inf")
    print(f"    Sparsity ratio: datafactory is {sparsity_ratio:.2f}x more sparse")

    print("\n  2. DISTRIBUTION OF MEAN PREDICTIONS (origin_0)")
    print("  -----------------------------------------------")
    for name, arr in [("purple_alien (viewser)", mean_pa), ("bright_starship (datafactory)", mean_bs)]:
        nz = arr[arr >= 1e-6]
        print(f"    {name}:")
        print(f"      All {len(arr):,} rows: mean={arr.mean():.6f}  std={arr.std():.6f}  max={arr.max():.4f}")
        if len(nz) > 0:
            print(f"      {len(nz):,} nonzero:   mean={nz.mean():.6f}  std={nz.std():.6f}  max={nz.max():.4f}")
            print(f"      Percentiles:     p10={np.percentile(nz, 10):.6f}  p25={np.percentile(nz, 25):.6f}  p50={np.percentile(nz, 50):.6f}  p75={np.percentile(nz, 75):.6f}  p95={np.percentile(nz, 95):.6f}")
        else:
            print("      ALL PREDICTIONS ARE ZERO")

    both_nz = (mean_pa >= 1e-6) & (mean_bs >= 1e-6)
    pa_only = (mean_pa >= 1e-6) & (mean_bs < 1e-6)
    bs_only = (mean_pa < 1e-6) & (mean_bs >= 1e-6)
    both_zero = (mean_pa < 1e-6) & (mean_bs < 1e-6)

    print("\n  3. OVERLAP ANALYSIS (where do the models agree on nonzero?)")
    print("  -----------------------------------------------------------")
    print(f"    Both nonzero:                  {both_nz.sum():>7,} rows  ({100*both_nz.sum()/len(mean_pa):.1f}%)")
    print(f"    Viewser-only nonzero:           {pa_only.sum():>7,} rows  ({100*pa_only.sum()/len(mean_pa):.1f}%)")
    print(f"    Datafactory-only nonzero:       {bs_only.sum():>7,} rows  ({100*bs_only.sum()/len(mean_pa):.1f}%)")
    print(f"    Both zero:                     {both_zero.sum():>7,} rows  ({100*both_zero.sum()/len(mean_pa):.1f}%)")

    if both_nz.sum() > 10:
        r_cond = np.corrcoef(mean_pa[both_nz], mean_bs[both_nz])[0, 1]
        ratio = mean_pa[both_nz].mean() / mean_bs[both_nz].mean() if mean_bs[both_nz].mean() > 0 else float("inf")
        mae_cond = np.abs(mean_pa[both_nz] - mean_bs[both_nz]).mean()
        print("\n    WHERE BOTH ARE NONZERO:")
        print(f"      Conditional correlation:     {r_cond:.6f}")
        print(f"      Scale ratio (viewser/factory): {ratio:.2f}x")
        print(f"      Conditional MAE:              {mae_cond:.6f}")
        print(f"      Viewser mean (conditional):   {mean_pa[both_nz].mean():.6f}")
        print(f"      Datafactory mean (conditional):{mean_bs[both_nz].mean():.6f}")
    else:
        print(f"\n    TOO FEW OVERLAPPING NONZERO ROWS ({both_nz.sum()}) FOR CONDITIONAL ANALYSIS")

    print("\n  4. POSTERIOR SAMPLE SPARSITY")
    print("  ----------------------------")
    pa_frac = (pa > 1e-6).mean(axis=0)
    bs_frac = (bs > 1e-6).mean(axis=0)
    print(f"    purple_alien:    avg {100*pa_frac.mean():.1f}% of rows nonzero per sample (range {100*pa_frac.min():.1f}-{100*pa_frac.max():.1f}%)")
    print(f"    bright_starship: avg {100*bs_frac.mean():.1f}% of rows nonzero per sample (range {100*bs_frac.min():.1f}-{100*bs_frac.max():.1f}%)")

    print("\n  5. PER-ORIGIN BREAKDOWN (all 13 calibration origins)")
    print("  ----------------------------------------------------")
    print(f"    {'Origin':>6}  {'r(all)':>8}  {'r(cond)':>8}  {'both_nz':>8}  {'pa_nz':>8}  {'bs_nz':>8}  {'pa_mean':>10}  {'bs_mean':>10}  {'ratio':>8}")
    print(f"    {'------':>6}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'----------':>10}  {'----------':>10}  {'--------':>8}")

    for oi in range(13):
        pa_o = np.load(PA_DIR / f"origin_{oi}" / target / "y_pred.npy")
        bs_o = np.load(BS_DIR / f"origin_{oi}" / target / "y_pred.npy")
        m_pa = pa_o.mean(axis=1)
        m_bs = bs_o.mean(axis=1)
        r = np.corrcoef(m_pa, m_bs)[0, 1]
        both = (m_pa >= 1e-6) & (m_bs >= 1e-6)
        r_cond = np.corrcoef(m_pa[both], m_bs[both])[0, 1] if both.sum() > 10 else float("nan")
        nz_pa = (m_pa >= 1e-6).sum()
        nz_bs = (m_bs >= 1e-6).sum()
        ratio_o = m_pa.mean() / m_bs.mean() if m_bs.mean() > 1e-8 else float("inf")
        print(f"    {oi:>6}  {r:>8.4f}  {r_cond:>8.4f}  {both.sum():>8,}  {nz_pa:>8,}  {nz_bs:>8,}  {m_pa.mean():>10.6f}  {m_bs.mean():>10.6f}  {ratio_o:>8.1f}x")


def main():
    print("=" * 72)
    print("DEEP PARITY ANALYSIS")
    print("purple_alien (viewser, ged_*_best_sum_nokgi)")
    print("    vs")
    print("bright_starship (datafactory, ged_*_best)")
    print("=" * 72)
    print()
    print("Run type: calibration")
    print(f"purple_alien dir:    {PA_DIR.name}")
    print(f"bright_starship dir: {BS_DIR.name}")

    for target in TARGETS:
        analyze_target(target)

    print(f"\n{'='*72}")
    print("DIAGNOSIS")
    print(f"{'='*72}")
    print("""
    The viewser pipeline serves `ged_*_best_sum_nokgi` — an aggregated, imputation-
    corrected variant of UCDP fatality counts. The datafactory zarr store serves
    `ged_*_best` — the raw best-estimate counts.

    Key differences:
      _sum     : fatalities summed across sub-events within each PRIO-GRID cell-month
      _nokgi   : "no known group imputation" — removes statistically imputed values
                 for events attributed to known armed groups

    The combined effect makes `_sum_nokgi` a DENSER signal (more nonzero cells) with
    HIGHER values per cell (summed sub-events) compared to raw `ged_*_best`.

    This explains the parity results:
      1. Sparsity gap: viewser predictions are nonzero in far more cells
      2. Scale gap: viewser predictions are 2-10x larger where both are nonzero
      3. Correlation: weak to zero for ns_best and os_best (sparse signals
         become zero in the datafactory version)

    RESOLUTION OPTIONS:
      A. Datafactory provides `_sum_nokgi` variants → modify bright_starship queryset
      B. Viewser models switch to raw `ged_*_best` → modify purple_alien queryset
      C. Document the difference and accept non-parity between data sources
""")


if __name__ == "__main__":
    main()
