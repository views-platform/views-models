# brisk_sparrow

## Overview

A distributional baseline member of **epic #595** — CRAF'd's own ensembles.

| Information | Details |
|---|---|
| **Model Algorithm** | ConflictologyModel |
| **Level of Analysis** | pgm |
| **Targets** | `lr_sb_best`, `lr_ns_best`, `lr_os_best` |
| **Setting** | `window_months=18` |
| **Draws** | 128 (ADR-015 standard) |
| **Coverage** | `land_gaul` — 64,742 PRIO-GRID land cells with GAUL country coverage |
| **Maturity** | candidate |
| **Data Source** | views-datafactory |
| **Country twin** | [`brisk_condor`](../brisk_condor) |

## Why it has a twin

Epic #595 reconciles the pgm ensemble `chrome_box` to a country-month ensemble draw
by draw. `views_frames_reconcile` uses ALIGNED_DRAWS, which scales grid draw *i* to
country draw *i*; concatenation puts member *k*'s draws at positions
128(k−1)+1 … 128k. So this model and `brisk_condor` carry **identical
settings and identical positions** in their ensembles' member lists. Changing one
without the other silently reconciles this model against an unrelated country
forecast. A test in #598 pins the pairing.

## Usage

Modify configurations in `configs/`.

```
python main.py -r calibration -t -e

or

./run.sh -r calibration -t -e
```
