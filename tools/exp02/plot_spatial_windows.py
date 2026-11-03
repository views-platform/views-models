"""Windows onto conflict zones, one panel per EXP-02 arm, so the spatial pattern can be seen.

EXP-02 contrasts `UpsampleConv2` against `HydraBNUNet06_LSTM4` at 300 lessons, two seeds each.
The pre-registered readout is gate-side and numeric. **This module answers a different question
that no scalar answers: does the predicted violence cluster more tightly or smear out?** An
upsampling decoder can score well on CRPS while spreading mass across neighbouring cells, and
the average of a map says nothing about whether the map is sharp.

## The three choices that make the comparison honest, and what each prevents

**One colour scale across every panel of a figure.** Per-panel normalisation is the default in
every plotting library and it is fatal here: a diffuse map and a concentrated map are rendered
identically once each is stretched to its own maximum. The scale is computed across all panels
of a figure, including the actuals, and printed in the caption so it cannot be mistaken.

**`log1p`, matching `tools/collapse/plot_collapse_audit.py:56`.** Fatalities are heavy-tailed;
on a linear scale a window shows one bright pixel and 2,000 black ones, which is a picture of
the maximum and not of the pattern.

**Mean and a single draw, side by side.** The posterior mean is smooth *by construction* — it
averages 16 draws — so comparing means partly measures averaging rather than the model. A single
draw shows the texture the model actually generates. These are different questions and the
answer to "does it cluster" can differ between them, so the figure carries both rather than
making the reader pick.

## Geometry, verified rather than assumed

PRIO-GRID is 720 columns of 0.5°, id 1 at the south-west corner, so for `gid`:

    row, col = (gid - 1) // 720, (gid - 1) % 720
    lat, lon = -90 + 0.5 * row + 0.25, -180 + 0.5 * col + 0.25

Checked against the delivered cell set on 2026-11-03: four open-ocean points (mid-Pacific,
mid-Atlantic, South Indian, North Atlantic) are absent, four conflict-zone land points (Kano,
Kabul, Zambia, Donetsk) are present, Antarctica is absent, and the extent is lat -55.75..83.25,
lon -179.75..179.75 over exactly 64,818 cells. A wrong offset here would still produce a
plausible-looking map, of the wrong place, with nothing to signal it.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PG_SIDE = 720

#: Conflict zones as (lat_min, lat_max, lon_min, lon_max). Chosen to be a few hundred km across:
#: wide enough to hold a conflict system and its periphery, narrow enough that a 0.5° cell is a
#: visible block rather than a pixel. Override with --zone.
ZONES: dict[str, tuple[float, float, float, float]] = {
    "sahel": (10.0, 20.0, -12.0, 8.0),
    "horn": (-2.0, 15.0, 33.0, 51.0),
    "great_lakes": (-9.0, 4.0, 12.0, 32.0),
    "nigeria": (5.0, 14.0, 3.0, 15.0),
    "levant": (29.0, 38.0, 35.0, 49.0),
    "ukraine": (44.0, 53.0, 22.0, 41.0),
    "afpak": (24.0, 39.0, 60.0, 78.0),
    "myanmar": (9.0, 29.0, 92.0, 102.0),
}

TARGETS = ("lr_sb_best", "lr_ns_best", "lr_os_best")


@dataclass(frozen=True)
class Window:
    """One zone's bounding box, and the grid indices it selects."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def rows_cols(self) -> tuple[int, int, int, int]:
        r0 = int((self.lat_min + 90) // 0.5)
        r1 = int((self.lat_max + 90) // 0.5)
        c0 = int((self.lon_min + 180) // 0.5)
        c1 = int((self.lon_max + 180) // 0.5)
        return r0, r1, c0, c1


def raster(gid: np.ndarray, value: np.ndarray, win: Window) -> np.ndarray:
    """One month's cells on the window's grid, NaN where the window holds no land cell.

    NaN rather than 0 is deliberate: an absent cell and a cell predicted to see no violence
    are different claims, and rendering both as the colormap's zero would invent land and
    hide the coastline that makes a window recognisable.
    """
    r0, r1, c0, c1 = win.rows_cols()
    row, col = (gid - 1) // PG_SIDE, (gid - 1) % PG_SIDE
    keep = (row >= r0) & (row <= r1) & (col >= c0) & (col <= c1)
    grid = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan, dtype=np.float32)
    grid[row[keep] - r0, col[keep] - c0] = value[keep]
    return grid


def _top_share(values: np.ndarray, frac: float) -> float:
    """Share of total predicted mass held by the top `frac` of cells — concentration, as a number.

    Simon's question is whether the pattern clusters, and he asked to judge it by eye. This is
    the matching scalar, printed beside each panel so the eye and the number can disagree in
    public. It is deliberately NOT the readout: it is window-dependent and has no pre-registered
    threshold, and quoting it as a result would be the "correct value, wrong question" failure.
    """
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    k = max(1, int(round(values.size * frac)))
    return float(np.sort(values)[-k:].sum() / total)


def load_month(parquet: Path, month: int, target: str) -> tuple[np.ndarray, np.ndarray]:
    """The (gid, value) pair for one month of one target, read from a collapsed origin parquet."""
    frame = pd.read_parquet(parquet, columns=["month_id", "priogrid_id", f"pred_{target}"])
    sel = frame[frame["month_id"] == month]
    if sel.empty:
        months = np.sort(frame["month_id"].unique())
        raise SystemExit(
            f"⛔ {parquet.name} holds no month {month}. It covers {months[0]}-{months[-1]}.\n"
            f"   An empty selection would have rendered an all-NaN panel that looks like a "
            f"model predicting nothing, which is why this refuses instead."
        )
    return sel["priogrid_id"].to_numpy(), sel[f"pred_{target}"].to_numpy()


def load_month_draw(
    delivery: Path, origin: int, month: int, target: str, draw: int
) -> tuple[np.ndarray, np.ndarray]:
    """One posterior DRAW for one month, read out of the arm's `lr` archive.

    The parquet beside it is the posterior MEAN — verified on 2026-11-03 against the archive
    for month 504 of origin 12: `max|parquet - mean| = 1.6e-06` (float32 rounding) against
    62.4 for the median and 332.8 for draw 0. So the two paths answer genuinely different
    questions, and the mean is smooth *by construction*: averaging 16 draws spreads mass
    whatever the architecture does. A clustering comparison read only off means partly
    measures that averaging. This reads a single draw so the texture the model actually
    generates can be seen next to it.
    """
    archives = sorted((delivery / "draws").glob("*_lr.tar.zst"))
    if not archives:
        raise SystemExit(
            f"⛔ no *_lr.tar.zst under {delivery / 'draws'} — --draw needs the posterior "
            f"archive, which the parquet path does not require."
        )
    # `tarfile` has no zstd support, and `zstandard` is not in this prefix, so the archive is
    # opened with the same `zstd` binary the pod used to write it (tar's --use-compress-program).
    # Shelling out keeps one decompressor in the lane rather than adding a dependency that the
    # pod side does not have.
    want = f"origin_{origin}/{target}/"
    with tempfile.TemporaryDirectory() as tmp:
        listing = subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-tf", str(archives[-1])],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        members = [m for m in listing if want in m]
        if len(members) != 2:
            raise SystemExit(
                f"⛔ expected y_pred.npy and identifiers.npz for {want} in "
                f"{archives[-1].name}, found {len(members)}: {members}"
            )
        subprocess.run(
            ["tar", "--use-compress-program=unzstd", "-xf", str(archives[-1]), "-C", tmp, *members],
            check=True, capture_output=True,
        )
        root = next(Path(tmp).rglob(f"origin_{origin}/{target}"))
        y = np.load(root / "y_pred.npy", mmap_mode="r")
        ids = np.load(root / "identifiers.npz")
        if draw >= y.shape[1]:
            raise SystemExit(
                f"⛔ draw {draw} requested but this arm holds {y.shape[1]} draws (0-"
                f"{y.shape[1] - 1}). A silently clamped index would have plotted a "
                f"different draw than the title claims."
            )
        sel = ids["time"] == month
        if not sel.any():
            months = np.unique(ids["time"])
            raise SystemExit(
                f"⛔ origin {origin} holds no month {month}; it covers "
                f"{months.min()}-{months.max()}."
            )
        return ids["unit"][sel], np.asarray(y[sel, draw])


def origin_parquet(delivery: Path, origin: int) -> Path:
    """The collapsed parquet for one origin inside an arm's delivery."""
    hits = sorted((delivery / "parquet").glob(f"*_{origin:02d}.parquet"))
    if not hits:
        available = sorted(p.name for p in (delivery / "parquet").glob("*.parquet"))
        raise SystemExit(
            f"⛔ no parquet for origin {origin} under {delivery / 'parquet'}.\n"
            f"   Present: {available or '(none — is this an arm delivery directory?)'}"
        )
    return hits[-1]


def plot_window(
    arms: dict[str, Path],
    win: Window,
    month: int,
    target: str,
    origin: int,
    out: Path,
    draw: int | None = None,
) -> Path:
    """One figure: the same window and month, one panel per arm, on ONE shared colour scale."""
    grids: dict[str, np.ndarray] = {}
    for label, delivery in arms.items():
        if draw is None:
            gid, value = load_month(origin_parquet(delivery, origin), month, target)
        else:
            gid, value = load_month_draw(delivery, origin, month, target, draw)
        grids[label] = raster(gid, value, win)

    stack = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values()])
    if stack.size == 0:
        raise SystemExit(f"⛔ window '{win.name}' selects no land cell in any arm — widen it.")
    vmax = float(np.log1p(stack).max())

    # Grey for "no land cell here". With `inferno`, NaN defaults to the figure background and a
    # zero-fatality land cell is black, so ocean and quiet land were being told apart only by
    # which white the reader happened to be looking at. Grey makes the coastline legible, which
    # is what lets a window be recognised as the place it claims to be.
    cmap = matplotlib.colormaps["inferno"].with_extremes(bad="0.82")

    fig, axes = plt.subplots(
        1, len(grids), figsize=(4.4 * len(grids), 5.0), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    im = None
    for ax, (label, grid) in zip(axes, grids.items()):
        im = ax.imshow(
            np.log1p(grid),
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            extent=(win.lon_min, win.lon_max, win.lat_min, win.lat_max),
        )
        finite = grid[np.isfinite(grid)]
        ax.set_title(
            f"{label}\nmax {finite.max():.2f}   nonzero {(finite > 0).mean() * 100:.1f}%\n"
            f"top-1% cells hold {_top_share(finite, 0.01) * 100:.0f}% of mass",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        ax.grid(color="white", alpha=0.18, linewidth=0.5)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, label="log1p(expected fatalities)")
    # Two lines rather than one: at a single panel the one-line form ran off the canvas and
    # clipped the zone name, which is the one word that says what is being looked at.
    fig.suptitle(
        f"{win.name}   {target}   month {month}   origin {origin}\n"
        f"{'posterior mean' if draw is None else f'single draw {draw}'}   ·   "
        f"all panels share one scale, 0..{vmax:.2f} log1p   ·   grey = no land cell",
        fontsize=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR",
                    help="an arm's delivery directory, repeatable; panels appear in this order")
    ap.add_argument("--zone", action="append", help=f"one of {sorted(ZONES)}, or lat0:lat1:lon0:lon1")
    ap.add_argument("--month", action="append", type=int, required=True)
    ap.add_argument("--target", default="lr_sb_best", choices=TARGETS)
    ap.add_argument("--origin", type=int, default=12)
    ap.add_argument("--draw", type=int, default=None,
                    help="plot this posterior draw instead of the mean (the mean smooths by "
                         "construction, so clustering can read differently in a draw)")
    ap.add_argument("--out", type=Path, default=Path("reports/exp02_spatial"))
    args = ap.parse_args(argv)

    arms: dict[str, Path] = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"⛔ --arm wants LABEL=DIR, got {spec!r}")
        label, _, path = spec.partition("=")
        arms[label] = Path(path)

    names = args.zone or list(ZONES)
    windows = []
    for name in names:
        if name in ZONES:
            windows.append(Window(name, *ZONES[name]))
        elif name.count(":") == 3:
            a, b, c, d = (float(x) for x in name.split(":"))
            windows.append(Window(f"{a}:{b}:{c}:{d}", a, b, c, d))
        else:
            raise SystemExit(f"⛔ unknown zone {name!r}; known: {sorted(ZONES)}")

    written = []
    for win in windows:
        for month in args.month:
            tag = "mean" if args.draw is None else f"draw{args.draw}"
            out = args.out / f"{win.name}_m{month}_{args.target}_o{args.origin}_{tag}.png"
            written.append(
                plot_window(arms, win, month, args.target, args.origin, out, args.draw)
            )
    for p in written:
        print(p)
    print(f"\n{len(written)} figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
