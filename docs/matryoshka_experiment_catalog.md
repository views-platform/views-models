# The Matryoshka comparison — model catalog and data ledger

**What this file is for.** One place that says which models the comparison needs, and for each one
whether we hold forecast data from it. Nothing else. If you want to know "do we have that model's
results", this is the only file you should have to open.

**Last verified: 2026-11-17.** Every "yes" in the Data column below was checked on this date by
opening the archive and counting what is inside it, not by reading a note.

**How to re-verify, any time:** `python tools/audit/verify_experiment_data.py` prints the table below
straight off the disk. Add `--deep` to also load the arrays and assert the draw count.

---

## 1. The eight nested-doll models — DATA HELD

These are the subject of the study. Each reads all of history with recency weighting, and pools over
a nest of areas. They differ from each other in which nest.

| model | algorithm | data | where | checked |
|---|---|---|---|---|
| `nested_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `lone_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `local_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `civic_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `global_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `mapless_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `stateless_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `distant_burlap` | MatryoshkaConflictology | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |

⚠️ `global_burlap` over-predicts one-sided violence by about 2.2 times, where every other model in
the set lands between 1.03 and 1.21. Taken from the run's own note, **not re-measured here.**

## 2. The fixed-window controls — DATA HELD, but see section 3

Each of these reads a hard 36-month window that ends at the last training month and never moves. One
pools at a single level instead of a nest.

| model | algorithm | pools over | data | where | checked |
|---|---|---|---|---|---|
| `plain_burlap` | ConflictologyModel | the cell alone | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `narrow_canvas` | PooledConflictology | admin-1 area | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `middle_canvas` | PooledConflictology | country | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |
| `broad_canvas` | PooledConflictology | the whole map | ✅ yes | 2026-11-13 run | 13 origins, 3 targets, 64 draws |

`plain_burlap` also has an older copy from the 2026-11-04 run. Use the 2026-11-13 one: it shares the
store, the engine version and the seed with everything else in sections 1 and 2.

## 3. The moving-window controls — BUILT 2026-11-17, NO DATA YET

This is the whole remaining gap. Sections 1 and 2 cannot be compared past the first forecast origin,
because the nested-doll models get to see data up to each origin and the fixed-window models do not.
A score gap between them mixes the method with a data advantage. These four fix that: the same
pooling levels, but the window moves forward with the origin.

| model | algorithm | pools over | folder | data |
|---|---|---|---|---|
| `rolling_burlap` | RollingConflictology | the cell alone | ✅ built | 🔴 none |
| `narrow_sailcloth` | RollingPooledConflictology | admin-1 area | ✅ built | 🔴 none |
| `middle_sailcloth` | RollingPooledConflictology | country | ✅ built | 🔴 none |
| `broad_sailcloth` | RollingPooledConflictology | the whole map | ✅ built | 🔴 none |

**Why they could not exist before 2026-11-17.** The moving-window pooled algorithm shipped in
views-baseline 1.5.0, released 2026-11-16 23:27 (checked on PyPI). Before that it could not be named
in a config. All four floor views-baseline at `>=1.5.0` — including `rolling_burlap`, whose own
algorithm arrived in 1.2.0, because the four share one environment prefix and a prefix resolved to
1.2.0 would leave the three `_sailcloth` models naming a class the registry cannot build.

**Each differs from its twin in exactly two keys, `algorithm` and `name`, and in nothing else.**
Checked by loading both configs and diffing the merged dictionaries. `rolling_burlap` carried 256
draws when it was written on 2026-11-09 and is now at 64 in all three places, matching every model in
sections 1 and 2 — the metrics are quantised to S+1 levels, so a model at 256 cannot be compared with
one at 64 and nothing raises when they are.

**Acceptance check before any result here is believed:** at the first origin a moving-window model
sees exactly what its fixed-window twin sees, so its scores must come out equal. `rolling_burlap`
against `plain_burlap`, and each `_sailcloth` against its `_canvas`.

**What running them needs:** measured from the 2026-11-13 run of twelve, about 22 minutes per model
on one rented machine, so roughly 1.5 hours for the four. Nothing has been run.

## 4. Not part of this comparison — nine other baselines, data held

These answer a different question (how to smooth a climatology, views-models#695). Listed so nobody
has to wonder whether they are missing. All ✅ verified 2026-11-17: 13 origins, 3 targets, 64 draws.

`plain_linen` · `short_linen` · `long_linen` (InterpolatedQuantileConflictology)
`flat_velvet` · `plush_velvet` · `sheer_velvet` (KDEConflictology)
`even_tweed` · `heavy_tweed` · `light_tweed` (EmpiricalKDEMixtureConflictology)

---

## Where the data actually sits

| run | location | models | checksums |
|---|---|---|---|
| 2026-11-13 | `~/Documents/scripts/views_platform/pod_deliverables/2026-11-13_baseline_twelve/` | 12 | ✅ all 12 match `SHA256.pod` |
| 2026-11-04 | `reports/2026-11-04_baseline_phase1/archives_s64/` (forecasts) and `scores_s64/` (scores) | 10 | 🔴 no checksum file was kept |

⚠️ The 2026-11-04 batch has no checksum list, so its contents are verified but it cannot be proved
byte-identical to what left the pod. Nothing can be done about that now; do not treat it as a
provenance-grade copy.

Both runs: S=64 draws, seed 42, 13 origins, targets `lr_sb_best` / `lr_ns_best` / `lr_os_best`,
REGION `land` (64,818 cells), calibration partition, train 121–456, test 457–504. Each forecast array
is 2,333,448 rows by 64 columns, which is 64,818 cells by 36 months — checked by loading the arrays.

The 2026-11-13 run also carries its own note, `00_READ_BEFORE_COMPARING.md`, and a provenance bundle.

## How this file stays honest

`tools/audit/verify_experiment_data.py` holds the roster of sixteen as a literal and re-derives every
verdict from the disk. Three of the sixteen have no folder at all, and that absence is the most
important fact here, so the set cannot be discovered by listing `models/` — it has to be declared.
`tests/test_experiment_ledger_matches_the_catalog.py` fails if the tool and this document stop naming
the same sixteen models, and if any moving-window model loses its origin-0 twin.

The tool reports on the branch you have checked out. On a branch without `rolling_burlap` it says
"no folder"; on `feat/716-rolling-burlap` it says "folder exists, never run". Both are true.

What it checks, per model claiming data: the archive is where the roster says; its sha256 matches the
batch's own list where the batch kept one; the index holds 13 origins with 3 targets each; and with
`--deep`, one forecast array loads to exactly 2,333,448 by 64.

A model with no data prints `NO DATA` and does not fail the run — four of them are known-missing by
design. `--require-all` makes that fatal, for the day the set is supposed to be complete.
