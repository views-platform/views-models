"""Draw the eyeball panels for a collapsed parquet, so a person can reject it before it ships.

The tests in `tests/test_collapse_predictions.py` prove the arithmetic. They cannot tell you that
the field looks like conflict — that the mass sits in the Sahel and the Horn rather than smeared
uniformly, that the horizon decays rather than jumping, that collapsing 16 draws to a mean did
something other than pick one of them. That is what these panels are for.

    python -m tools.collapse.plot_collapse_audit <parquet> [--draws-dir <origin dir>] --out <png>

`--draws-dir` is the `origin_i` directory the parquet came from; given it, the script adds the
mean-vs-single-draw panel, which is the one that shows what the collapse actually bought.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TARGETS = ("lr_sb_best", "lr_ns_best", "lr_os_best")
PG_SIDE = 720  # PRIO-GRID is 720 columns wide; id 1 is the south-west corner


def _raster(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Scatter one month's cells back onto the PRIO-GRID raster, NaN where there is no land."""
    pgid = frame["priogrid_id"].to_numpy()
    row, col = (pgid - 1) // PG_SIDE, (pgid - 1) % PG_SIDE
    grid = np.full((row.max() - row.min() + 1, col.max() - col.min() + 1), np.nan)
    grid[row - row.min(), col - col.min()] = frame[column].to_numpy()
    return grid


def plot_audit(parquet: Path, out: Path, draws_dir: Path | None = None) -> Path:
    frame = pd.read_parquet(parquet)
    months = np.sort(frame["month_id"].unique())
    last = frame[frame["month_id"] == months[-1]]

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"{parquet.name}   {len(frame):,} rows   months {months[0]}-{months[-1]}   "
        f"{frame['priogrid_id'].nunique():,} cells",
        fontsize=11,
    )
    gs = fig.add_gridspec(3, 3)

    # row 1 — the map, per target, at the far end of the horizon
    for j, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, j])
        grid = _raster(last, f"pred_{target}")
        im = ax.imshow(np.log1p(grid), origin="lower", cmap="inferno", interpolation="nearest")
        ax.set_title(f"{target}  month {months[-1]}\nlog1p(expected fatalities)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.035)

    # row 2 left — the value distribution, which is mostly zero and must be
    ax = fig.add_subplot(gs[1, 0])
    for target in TARGETS:
        v = frame[f"pred_{target}"].to_numpy()
        ax.hist(np.log10(v[v > 0]), bins=80, histtype="step", label=f"{target} (>0)")
    ax.set_xlabel("log10(prediction)")
    ax.set_ylabel("cells")
    ax.legend(fontsize=7)
    ax.set_title("non-zero predictions", fontsize=9)

    # row 2 middle — zero fraction by target
    ax = fig.add_subplot(gs[1, 1])
    fracs = [float((frame[f"pred_{t}"] == 0).mean()) for t in TARGETS]
    ax.bar(range(3), fracs, color="0.4")
    ax.set_xticks(range(3))
    ax.set_xticklabels(TARGETS, fontsize=7, rotation=20)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share of cells exactly zero")
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.02, f"{f:.3f}", ha="center", fontsize=8)
    ax.set_title("exact zeros — a gate at work, not missing data", fontsize=9)

    # row 2 right — the horizon: does the forecast decay or blow up?
    ax = fig.add_subplot(gs[1, 2])
    by_month = frame.groupby("month_id")[[f"pred_{t}" for t in TARGETS]].mean()
    for target in TARGETS:
        ax.plot(by_month.index, by_month[f"pred_{target}"], marker=".", label=target)
    ax.set_xlabel("month_id")
    ax.set_ylabel("mean prediction")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_title("horizon profile", fontsize=9)

    # row 3 — what the collapse did, if we were given the draws
    if draws_dir is not None:
        draws = np.load(draws_dir / "lr_sb_best" / "y_pred.npy")
        mean, single = draws.astype("float64").mean(axis=1), draws[:, 0].astype("float64")

        ax = fig.add_subplot(gs[2, 0])
        nz = (mean > 0) | (single > 0)
        ax.scatter(single[nz] + 1e-3, mean[nz] + 1e-3, s=1, alpha=0.15, edgecolors="none")
        lim = [1e-3, max(mean.max(), single.max()) * 1.5]
        ax.plot(lim, lim, "r-", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("draw 0 alone")
        ax.set_ylabel(f"mean of {draws.shape[1]} draws")
        ax.set_title("collapse vs taking one draw (lr_sb_best)", fontsize=9)

        ax = fig.add_subplot(gs[2, 1])
        ax.hist(np.log10(mean[mean > 0]), bins=80, histtype="step", label="mean of draws")
        ax.hist(np.log10(single[single > 0]), bins=80, histtype="step", label="draw 0")
        ax.set_xlabel("log10(prediction)")
        ax.legend(fontsize=7)
        ax.set_title("the mean is smoother and has fewer hard zeros", fontsize=9)

        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        spread = draws.astype("float64").std(axis=1)
        ax.text(
            0.0, 0.95,
            "\n".join([
                f"draws per cell         {draws.shape[1]}",
                f"cells                  {draws.shape[0]:,}",
                "",
                f"mean of means          {mean.mean():.5f}",
                f"mean of draw 0         {single.mean():.5f}",
                f"ratio                  {mean.mean() / max(single.mean(), 1e-12):.4f}",
                "",
                f"zero cells, mean       {(mean == 0).mean():.4f}",
                f"zero cells, draw 0     {(single == 0).mean():.4f}",
                "",
                f"max, mean              {mean.max():.2f}",
                f"max, draw 0            {single.max():.2f}",
                "",
                f"mean within-cell sd    {spread.mean():.5f}",
                f"cells where sd > mean  {(spread > mean).mean():.4f}",
            ]),
            va="top", family="monospace", fontsize=9,
        )
        ax.set_title("numbers behind the two panels to the left", fontsize=9)

    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--draws-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(plot_audit(args.parquet, args.out, args.draws_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
