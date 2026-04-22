"""Audit: compare datafactory fetch against purple_alien's viewser data.

Fetches the calibration partition from Hetzner via the same code path
that bright_starship/main.py uses, then compares against purple_alien's
cached viewser parquet. This isolates the data question from the
training question.

Known expected differences:
  - c_id: factory uses GAUL codes, viewser uses its own country IDs.
    c_id is an identity column (not a training feature) — different
    codes are acceptable for M11.
  - float32 vs float64: zarr stores float32, viewser stores float64.
  - Event values: ~0.1% of cells differ due to UCDP annual data
    version difference (factory: v25.1, viewser: older). Documented
    in reports/consumer_parity_investigation.md.

Usage:
    cd views-datafactory
    uv run python ../views-models/models/bright_starship/scripts/audit_data_parity.py

Prerequisites:
    pip install views-datafactory
    ~/.netrc entry for 204.168.219.108
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from datafactory_query.defaults import DEFAULT_REMOTE

PURPLE_ALIEN_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "purple_alien" / "data" / "raw" / "calibration_viewser_df.parquet"
)

ZARR_URL = DEFAULT_REMOTE.zarr_url
REGION = "africa_me_legacy"
FACTORY_FEATURES = ["ged_sb_best", "ged_ns_best", "ged_os_best", "gaul0_code"]
FEATURE_RENAME = {
    "ged_sb_best": "lr_sb_best",
    "ged_ns_best": "lr_ns_best",
    "ged_os_best": "lr_os_best",
    "gaul0_code": "c_id",
}
NCOL = 720
CALIBRATION_TRAIN = (121, 444)
CALIBRATION_TEST = (445, 492)

EVENT_COLS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
IDENTITY_COLS = ["c_id", "row", "col"]


def fetch_from_hetzner() -> pd.DataFrame:
    """Fetch calibration partition from Hetzner — same logic as fetch_data()."""
    from datafactory_query import load_dataset

    start = CALIBRATION_TRAIN[0]
    end = CALIBRATION_TEST[1]

    print(f"Fetching from {ZARR_URL}")
    print(f"  region={REGION}, months {start}-{end}")

    t0 = time.time()
    df = load_dataset(
        region=REGION,
        start=start,
        end=end,
        features=FACTORY_FEATURES,
        output_format="dataframe",
        data_dir=ZARR_URL,
    )
    elapsed = time.time() - t0
    print(f"  Fetched in {elapsed:.1f}s: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = df.rename(columns=FEATURE_RENAME)

    pgids = df.index.get_level_values("priogrid_gid")
    df["row"] = ((pgids - 1) // NCOL + 1).astype(np.float64)
    df["col"] = ((pgids - 1) % NCOL + 1).astype(np.float64)

    df = df.fillna(0.0)
    df = df.sort_index()
    return df


def audit(factory: pd.DataFrame, viewser: pd.DataFrame) -> bool:
    """Compare two DataFrames with domain-aware tolerance."""
    print("\n" + "=" * 60)
    print("PARITY AUDIT: datafactory (Hetzner) vs viewser (purple_alien)")
    print("=" * 60)

    failures: list[str] = []
    warnings: list[str] = []

    # ── 1. Structural checks ──────────────────────────────────

    print("\n--- Structure ---")

    # Index names
    if factory.index.names != viewser.index.names:
        failures.append(f"Index names: {factory.index.names} vs {viewser.index.names}")
    print(f"  Index names: {factory.index.names} — {'MATCH' if factory.index.names == viewser.index.names else 'FAIL'}")

    # Month range
    f_months = sorted(factory.index.get_level_values(0).unique())
    v_months = sorted(viewser.index.get_level_values(0).unique())
    months_match = f_months == v_months
    print(f"  Months: {f_months[0]}-{f_months[-1]} ({len(f_months)}) — {'MATCH' if months_match else 'FAIL'}")
    if not months_match:
        failures.append(f"Month ranges differ: factory {len(f_months)}, viewser {len(v_months)}")

    # PGID set
    f_pgids = sorted(factory.index.get_level_values(1).unique())
    v_pgids = sorted(viewser.index.get_level_values(1).unique())
    pgids_match = f_pgids == v_pgids
    print(f"  PGIDs: {len(f_pgids):,} cells — {'MATCH' if pgids_match else 'FAIL'}")
    if not pgids_match:
        failures.append(f"PGID sets differ: factory {len(f_pgids)}, viewser {len(v_pgids)}")

    # Shape
    print(f"  Shape: factory={factory.shape}, viewser={viewser.shape} — {'MATCH' if factory.shape == viewser.shape else 'FAIL'}")
    if factory.shape != viewser.shape:
        failures.append(f"Shapes differ: {factory.shape} vs {viewser.shape}")

    # Columns
    fc, vc = sorted(factory.columns), sorted(viewser.columns)
    print(f"  Columns: {fc} — {'MATCH' if fc == vc else 'FAIL'}")
    if fc != vc:
        failures.append(f"Columns differ: {fc} vs {vc}")

    if failures:
        print(f"\n{'=' * 60}")
        print(f"VERDICT: FAIL — {len(failures)} structural failures")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        return False

    # ── 2. Spatial coordinates (row, col) ─────────────────────

    print("\n--- Spatial coordinates (row, col) ---")
    f_sorted = factory.sort_index()
    v_sorted = viewser.sort_index()

    for col in ["row", "col"]:
        match = np.array_equal(f_sorted[col].values, v_sorted[col].values)
        print(f"  {col}: {'EXACT MATCH' if match else 'DIFFER'}")
        if not match:
            failures.append(f"{col} values differ")

    # ── 3. c_id (identity column — different coding expected) ─

    print("\n--- c_id (identity column) ---")
    f_cid = f_sorted["c_id"]
    v_cid = v_sorted["c_id"]
    cid_match = np.array_equal(f_cid.values, v_cid.values)

    f_uniq = f_cid.nunique()
    v_uniq = v_cid.nunique()
    print(f"  Factory: {f_uniq} unique GAUL codes, range {f_cid.min():.0f}-{f_cid.max():.0f}")
    print(f"  Viewser: {v_uniq} unique viewser IDs, range {v_cid.min():.0f}-{v_cid.max():.0f}")

    if cid_match:
        print("  EXACT MATCH")
    else:
        # Expected: different coding systems (GAUL vs viewser country IDs).
        # c_id is an identity column, not a training feature.
        f_consistency = f_sorted.groupby(level=1)["c_id"].nunique().max()
        v_consistency = v_sorted.groupby(level=1)["c_id"].nunique().max()
        print(f"  Values differ (expected: different coding systems)")
        print(f"  Factory: {f_consistency} c_id per pgid (GAUL, time-invariant)")
        print(f"  Viewser: {v_consistency} c_id per pgid (time-varying lookup)")
        print("  ACCEPTABLE — c_id is identity metadata, not a training feature")
        warnings.append("c_id uses different coding (GAUL vs viewser) — expected")

    # ── 4. Event columns (training features) ──────────────────

    print("\n--- Event columns (training features) ---")

    for col in EVENT_COLS:
        fv = f_sorted[col].astype(np.float64).values
        vv = v_sorted[col].values

        diff = np.abs(fv - vv)
        n_diff = np.count_nonzero(diff > 0.01)
        n_total = len(fv)
        pct_diff = 100.0 * n_diff / n_total

        if n_diff == 0:
            print(f"  {col}: EXACT MATCH")
            continue

        max_diff = diff.max()
        f_sum = fv.sum()
        v_sum = vv.sum()

        print(f"  {col}: {n_diff:,}/{n_total:,} cells differ ({pct_diff:.3f}%)")
        print(f"    max_abs_diff: {max_diff:.1f}")
        print(f"    sum: factory={f_sum:,.1f}, viewser={v_sum:,.1f}")

        if pct_diff < 0.5:
            print(f"    ACCEPTABLE — within expected UCDP annual version residual")
            warnings.append(f"{col}: {pct_diff:.3f}% cells differ (annual version)")
        else:
            print(f"    FAIL — exceeds 0.5% threshold")
            failures.append(f"{col}: {pct_diff:.3f}% cells differ")

    # ── 5. Dtype check ────────────────────────────────────────

    print("\n--- Dtypes ---")
    for col in sorted(factory.columns):
        fd, vd = factory[col].dtype, viewser[col].dtype
        if fd != vd:
            warnings.append(f"{col}: factory={fd}, viewser={vd}")
            print(f"  {col}: factory={fd}, viewser={vd} — ACCEPTABLE (zarr stores float32)")
        else:
            print(f"  {col}: {fd} — MATCH")

    # ── Verdict ───────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    if failures:
        print(f"VERDICT: FAIL — {len(failures)} issues")
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("VERDICT: PASS")

    if warnings:
        print(f"\n  {len(warnings)} expected differences:")
        for w in warnings:
            print(f"    - {w}")

    print("=" * 60)
    return len(failures) == 0


def main():
    if not PURPLE_ALIEN_PARQUET.exists():
        print(f"ERROR: purple_alien parquet not found at {PURPLE_ALIEN_PARQUET}")
        print("Run purple_alien calibration first to cache viewser data.")
        sys.exit(1)

    print(f"Loading viewser reference: {PURPLE_ALIEN_PARQUET.name}")
    viewser = pd.read_parquet(PURPLE_ALIEN_PARQUET)
    print(f"  {viewser.shape[0]:,} rows, columns: {list(viewser.columns)}")

    factory = fetch_from_hetzner()

    passed = audit(factory, viewser)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
