#!/usr/bin/env python3
"""Sanity-check plots for HydraNet model and ensemble predictions.

Usage:
    python scripts/plot_sanity_checks.py --model purple_alien --run calibration
    python scripts/plot_sanity_checks.py --ensemble golden_hour --run calibration
    python scripts/plot_sanity_checks.py --model purple_alien --run calibration --origin 0 --types spatial concentration

Produces up to three plot types into the model/ensemble reports/ directory:
  1. spatial   — 3×N heatmap panel (targets × forecast steps 1/mid/last)
  2. timeseries — top-K grid cells + top-K countries by predicted intensity
  3. concentration — Lorenz curve + Gini for historical vs predicted
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style constants (adapted from views-datafactory viz_style.py) ──

DPI = 200
FONT_TITLE = 13
FONT_LABEL = 10
FONT_TICK = 8
FONT_ANNOT = 8

CMAP_SEQ = "rainbow"

COLOR_SB = "#7A8B3C"
COLOR_NS = "#4878A8"
COLOR_OS = "#D4752E"
COLOR_GRAY = "#666666"

# Presentation-only style lookups (color / short label per target). These are
# documented plot constants, NOT load-bearing logic — keyed by name purely for
# display; missing keys fall back to neutral defaults at the call sites (EPIC #154 / S5).
TARGET_COLORS = {
    "lr_sb_best": COLOR_SB,
    "lr_ns_best": COLOR_NS,
    "lr_os_best": COLOR_OS,
}

TARGET_LABELS = {
    "lr_sb_best": "State-based",
    "lr_ns_best": "Non-state",
    "lr_os_best": "One-sided",
}


def _style_ax(ax, *, title=None, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK)
    if title:
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)


def _save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {name}")


# ── Data loading ──


def _find_prediction_dir(base_dir, run_type):
    gen_dir = base_dir / "data" / "generated"
    candidates = sorted(
        [d for d in gen_dir.glob(f"predictions_{run_type}_*") if d.is_dir()],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {run_type} predictions in {gen_dir}")
    return candidates[0]


def _load_predictions(pred_dir, origin, target):
    origin_dir = pred_dir / f"origin_{origin}" / target
    y_pred = np.load(origin_dir / "y_pred.npy")
    ids = np.load(origin_dir / "identifiers.npz")
    return y_pred, ids["time"], ids["unit"]


def _load_raw_data(base_dir, run_type):
    raw_dir = base_dir / "data" / "raw"
    candidates = list(raw_dir.glob(f"{run_type}_viewser_df.parquet"))
    if not candidates:
        candidates = list(raw_dir.glob(f"{run_type}_*.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[0])


def _available_targets(pred_dir, origin):
    origin_dir = pred_dir / f"origin_{origin}"
    return sorted([d.name for d in origin_dir.iterdir() if d.is_dir() and d.name.startswith("lr_")])


def _get_grid_info(raw_df):
    """Extract grid mapping from raw data: priogrid_gid -> (row, col, c_id)."""
    month_id = raw_df.index.get_level_values("month_id").min()
    month_df = raw_df.loc[month_id]
    row_min = int(month_df["row"].min())
    row_max = int(month_df["row"].max())
    col_min = int(month_df["col"].min())
    col_max = int(month_df["col"].max())
    return month_df, row_min, row_max, col_min, col_max


def _pred_to_grid(y_pred, time_ids, unit_ids, month_df, row_min, row_max, col_min, col_max, step):
    """Reshape predictions for a specific step into a 2D grid (median across samples)."""
    unique_times = np.unique(time_ids)
    target_time = unique_times[step - 1] if step <= len(unique_times) else unique_times[-1]
    mask = time_ids == target_time
    cells = unit_ids[mask]
    values = np.median(y_pred[mask], axis=1)

    nrows = row_max - row_min + 1
    ncols = col_max - col_min + 1
    grid = np.full((nrows, ncols), np.nan)

    rows = month_df.loc[cells, "row"].astype(int).values - row_min
    cols = month_df.loc[cells, "col"].astype(int).values - col_min
    grid[rows, cols] = values
    return grid


# ── Plot type 1: Spatial heatmaps ──


def plot_spatial(pred_dir, origin, targets, raw_df, output_dir, label):
    """3×N panel: targets (rows) × forecast steps 1/mid/last (cols)."""
    month_df, row_min, row_max, col_min, col_max = _get_grid_info(raw_df)
    sample_pred, sample_time, _ = _load_predictions(pred_dir, origin, targets[0])
    n_steps = len(np.unique(sample_time))
    mid_step = n_steps // 2
    steps = [1, mid_step, n_steps]

    n_targets = len(targets)
    fig, axes = plt.subplots(n_targets, 3, figsize=(14, 4 * n_targets))
    if n_targets == 1:
        axes = axes[np.newaxis, :]

    for i, target in enumerate(targets):
        y_pred, time_ids, unit_ids = _load_predictions(pred_dir, origin, target)
        for j, step in enumerate(steps):
            grid = _pred_to_grid(y_pred, time_ids, unit_ids, month_df,
                                 row_min, row_max, col_min, col_max, step)
            ax = axes[i, j]
            display = np.log1p(grid)
            display = np.flipud(display)
            vmax = np.nanpercentile(display, 99)
            im = ax.imshow(display, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=max(vmax, 0.01))
            _style_ax(ax, title=f"{TARGET_LABELS.get(target, target)} — step {step}")
            if j == 0:
                ax.set_ylabel(target, fontsize=FONT_LABEL)
            plt.colorbar(im, ax=ax, shrink=0.7, label="log(1+pred)")

    fig.suptitle(f"{label} — origin {origin} — median predictions (log scale)",
                 fontsize=FONT_TITLE + 2, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, f"spatial_origin{origin}.png")


# ── Plot type 2: Time series ──


def plot_timeseries(pred_dir, origin, targets, raw_df, output_dir, label, top_k=5):
    """Top-K cells + top-K countries by predicted intensity, with 95% credible intervals."""
    month_df, *_ = _get_grid_info(raw_df)

    for target in targets:
        y_pred, time_ids, unit_ids = _load_predictions(pred_dir, origin, target)
        unique_times = np.sort(np.unique(time_ids))
        n_steps = len(unique_times)
        unique_cells = np.sort(np.unique(unit_ids))
        n_cells = len(unique_cells)
        n_samples = y_pred.shape[1]

        cell_medians = np.zeros((n_cells, n_steps))
        cell_lo = np.zeros((n_cells, n_steps))
        cell_hi = np.zeros((n_cells, n_steps))
        cell_samples = np.zeros((n_cells, n_steps, n_samples))
        for s, t in enumerate(unique_times):
            mask = time_ids == t
            samples = y_pred[mask]
            cell_medians[:, s] = np.median(samples, axis=1)
            cell_lo[:, s] = np.percentile(samples, 2.5, axis=1)
            cell_hi[:, s] = np.percentile(samples, 97.5, axis=1)
            cell_samples[:, s, :] = samples

        cell_means = cell_medians.mean(axis=1)
        top_cell_idx = np.argsort(cell_means)[-top_k:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        steps_x = range(1, n_steps + 1)

        for rank, idx in enumerate(top_cell_idx):
            gid = unique_cells[idx]
            color = ax1.plot(steps_x, cell_medians[idx],
                             label=f"pgid {gid}", linewidth=1.5)[0].get_color()
            ax1.fill_between(steps_x, cell_lo[idx], cell_hi[idx],
                             color=color, alpha=0.15)
        _style_ax(ax1, title=f"Top {top_k} cells (95% CI)", xlabel="Forecast step",
                  ylabel="Predicted (median)")
        ax1.legend(fontsize=FONT_ANNOT)

        if "c_id" in month_df.columns:
            cell_to_country = {}
            for idx in range(n_cells):
                gid = unique_cells[idx]
                if gid in month_df.index:
                    cell_to_country[idx] = int(month_df.loc[gid, "c_id"])

            country_medians = {}
            country_lo = {}
            country_hi = {}
            country_ids = set(cell_to_country.values())
            for cid in country_ids:
                member_idx = [i for i, c in cell_to_country.items() if c == cid]
                country_sum = cell_samples[member_idx].sum(axis=0)
                country_medians[cid] = np.median(country_sum, axis=1)
                country_lo[cid] = np.percentile(country_sum, 2.5, axis=1)
                country_hi[cid] = np.percentile(country_sum, 97.5, axis=1)

            sorted_countries = sorted(country_medians.items(),
                                      key=lambda x: x[1].mean(), reverse=True)[:top_k]
            for cid, series in sorted_countries:
                color = ax2.plot(steps_x, series, label=f"c_id {cid}", linewidth=1.5)[0].get_color()
                ax2.fill_between(steps_x, country_lo[cid], country_hi[cid],
                                 color=color, alpha=0.15)
            _style_ax(ax2, title=f"Top {top_k} countries (95% CI)", xlabel="Forecast step",
                      ylabel="Country total (median)")
            ax2.legend(fontsize=FONT_ANNOT)
        else:
            ax2.text(0.5, 0.5, "No c_id in raw data", transform=ax2.transAxes, ha="center")

        fig.suptitle(f"{label} — {TARGET_LABELS.get(target, target)} — origin {origin}",
                     fontsize=FONT_TITLE + 1, fontweight="bold")
        fig.tight_layout()
        _save(fig, output_dir, f"timeseries_{target}_origin{origin}.png")


# ── Plot type 3: Concentration / Gini ──


def _gini(values):
    """Gini coefficient from sorted values."""
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    return (2 * np.sum(np.arange(1, n + 1) * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n


def plot_concentration(pred_dir, origin, targets, raw_df, output_dir, label):
    """Lorenz curve: historical vs predicted Gini, active cell fraction."""
    month_df, *_ = _get_grid_info(raw_df)

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        y_pred, time_ids, unit_ids = _load_predictions(pred_dir, origin, target)
        unique_cells = np.sort(np.unique(unit_ids))

        cell_total_pred = np.zeros(len(unique_cells))
        unique_times = np.unique(time_ids)
        for t in unique_times:
            mask = time_ids == t
            cell_total_pred += np.median(y_pred[mask], axis=1)

        # Historical from raw data
        if target in raw_df.columns:
            # Sum across all months in the test partition (from identifiers time range)
            test_months = list(range(int(unique_times.min()), int(unique_times.max()) + 1))
            available_months = [m for m in test_months if m in raw_df.index.get_level_values("month_id")]
            cell_total_hist = np.zeros(len(unique_cells))
            for m in available_months:
                month_data = raw_df.loc[m]
                for idx, gid in enumerate(unique_cells):
                    if gid in month_data.index:
                        cell_total_hist[idx] += month_data.loc[gid, target]
        else:
            cell_total_hist = None

        # Predicted Lorenz
        sorted_pred = np.sort(cell_total_pred)
        cum_pred = np.cumsum(sorted_pred)
        cum_pred_frac = cum_pred / cum_pred[-1] if cum_pred[-1] > 0 else cum_pred
        x = np.linspace(0, 100, len(sorted_pred))

        gini_pred = _gini(cell_total_pred)
        active_frac = np.mean(cell_total_pred > 0.1) * 100

        ax.plot(x, cum_pred_frac * 100, linewidth=2,
                color=TARGET_COLORS.get(target, "k"), label=f"Predicted (Gini={gini_pred:.3f})")

        if cell_total_hist is not None:
            sorted_hist = np.sort(cell_total_hist)
            cum_hist = np.cumsum(sorted_hist)
            cum_hist_frac = cum_hist / cum_hist[-1] if cum_hist[-1] > 0 else cum_hist
            gini_hist = _gini(cell_total_hist)
            ax.plot(x, cum_hist_frac * 100, linewidth=2, linestyle="--",
                    color=COLOR_GRAY, label=f"Historical (Gini={gini_hist:.3f})")

        ax.plot([0, 100], [0, 100], "--", color="#CCCCCC", linewidth=0.5)

        ax.text(15, 80, f"Active cells: {active_frac:.1f}%",
                fontsize=FONT_ANNOT, color=COLOR_GRAY)

        _style_ax(ax, title=TARGET_LABELS.get(target, target),
                  xlabel="Cumulative % of cells", ylabel="Cumulative % of total")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=FONT_ANNOT, loc="lower right")

    fig.suptitle(f"{label} — origin {origin} — concentration",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, f"concentration_origin{origin}.png")


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Sanity-check plots for predictions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Model name (e.g., purple_alien)")
    group.add_argument("--ensemble", help="Ensemble name (e.g., golden_hour)")
    parser.add_argument("--run", required=True, choices=["calibration", "validation", "forecasting"])
    parser.add_argument("--origin", type=int, default=0)
    parser.add_argument("--types", nargs="+", default=["spatial", "timeseries", "concentration"],
                        choices=["spatial", "timeseries", "concentration"])
    parser.add_argument("--raw-from", help="Load raw data from this model instead (for ensembles)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.model:
        base_dir = repo_root / "models" / args.model
        label = args.model
    else:
        base_dir = repo_root / "ensembles" / args.ensemble
        label = args.ensemble

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    pred_dir = _find_prediction_dir(base_dir, args.run)
    targets = _available_targets(pred_dir, args.origin)
    if not targets:
        raise FileNotFoundError(f"No regression targets found in {pred_dir}/origin_{args.origin}")

    raw_from = args.raw_from or (args.model if args.model else None)
    if raw_from:
        raw_base = repo_root / "models" / raw_from
    else:
        # For ensembles without --raw-from, try first constituent model
        import importlib.util
        meta_path = base_dir / "configs" / "config_meta.py"
        spec = importlib.util.spec_from_file_location("config_meta", meta_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        first_model = mod.get_meta_config()["models"][0]
        raw_base = repo_root / "models" / first_model
        print(f"  Using raw data from first constituent: {first_model}")

    raw_df = _load_raw_data(raw_base, args.run)
    if raw_df is None:
        raise FileNotFoundError(f"No raw data found in {raw_base}/data/raw/ for {args.run}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "reports" / "diagnostic_plots" / timestamp
    print(f"Generating sanity checks for {label} ({args.run}, origin {args.origin})")
    print(f"  Targets: {targets}")
    print(f"  Prediction dir: {pred_dir.name}")
    print(f"  Output: {output_dir}")

    if "spatial" in args.types:
        plot_spatial(pred_dir, args.origin, targets, raw_df, output_dir, label)

    if "timeseries" in args.types:
        plot_timeseries(pred_dir, args.origin, targets, raw_df, output_dir, label)

    if "concentration" in args.types:
        plot_concentration(pred_dir, args.origin, targets, raw_df, output_dir, label)

    print("Done.")


if __name__ == "__main__":
    main()
