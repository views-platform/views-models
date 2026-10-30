# chrome_box on a rented pod — 2026-10-28

Evidence and instruments from the run that proved what blocks `chrome_box`, pulled off the
pod before teardown. **Untracked.** Memory says pod hand-fixes die with the pod and that
durable != version-controlled; this directory is the former, not yet the latter.

## What it proved

`chrome_box` runs end to end once two defects are fixed, and no third defect lies behind
them. Patched on the pod via `sitecustomize.py` on `PYTHONPATH` — no installed file edited.

```
Reconciled 'lr_sb_best' to country totals from 'chrome_crate' (pgm_cm).
Reconciled 'lr_ns_best' ...
Reconciled 'lr_os_best' ...
Done. Runtime: 4.427 minutes.
```

| defect | repo | issue |
|---|---|---|
| `load_cm_frame` looks for `.parquet` where a PF ensemble writes a directory | views-pipeline-core | #562 |
| the GAUL mapping stops at the store horizon (564), 58 months short | views-models | #647 |

`probe_reconciled.py` then checked the output was *reconciled*, not merely written: grid
draws summed to country totals match the cm forecast draw-for-draw, worst relative error
**1.2e-07**, including at month 597 — 33 months past the store horizon, so the broadcast
fix is validated at the far end of the window.

## Measured, and correcting earlier claims

| quantity | value |
|---|---|
| one member frame | `(2330712, 128)` float32 = 1.111 GiB |
| x 8 members x 3 targets | 26.67 GiB (`model_results`) |
| one pooled frame (1024 draws) | 8.891 GiB |
| x 3 targets | 26.67 GiB (`forecasts`) |
| input volume + cm frame | 0.51 GiB |
| **frame-bytes floor** | **53.86 GiB** |
| **measured peak anon, pool + reconcile** | **68.2 GiB** (`probe.out`, the PATCHED run — see below) |
| per-member peak anon | 4.7-6.4 GiB |
| member wall clock | 21-25 min each; 8 in ~3h05m |
| pool + reconcile wall clock | 4.4 min |

⛔ **Which of the two runs the peak belongs to.** `phase2.out` (15:25) is unpatched: it
pooled and then died three times on `load_cm_frame` (views-pipeline-core#562), and its pool
reached only 27.8 GiB. The 68.2 GiB is `probe.out` (16:53-16:58), where `sitecustomize.py`
patched the loader to prefer the PF directory — what 3.3.5 now does natively — and all three
targets reconciled (`mode=aligned-draws`, 2,330,712 grid rows x 1,024 draws) before
"Done. Runtime: 4.427 minutes". The sampler was `watch_probe.sh` (30s, watches probe.out),
not `anon_sampler.sh` (10s, watches phase 2). **So the figure does cover reconcile.**

⚠️ It is a LOWER bound: ~9 samples over a 4.4-minute stage cannot see a shorter transient,
and only the watcher's final `PEAK anon:` line was transcribed rather than the series. The
pod is terminated, so a re-measure means a new run.

The 14.3 GiB above the frame floor is transient allocation inside `np.concatenate`
and the reconciler — bounded by subtraction, not attributed to a line. The reason the
floor is not the number to size against is that `_forecast_ensemble` holds member frames
and pooled frames live at once: it builds `model_results` for all members x targets
before the aggregation loop and frees none of it (no `del`, no `gc.collect`).

**An earlier "1.9x the floor" in this file was wrong** — that floor counted one pooled
aggregate instead of three. Both operands are now written out above, because a bare ratio
has no units and nothing contradicts it.

⛔ **WITHDRAWN 2026-10-29: `MEM_FLOOR_GB=16` in `pod_run_crafd_chain.sh` is NOT wrong.**
This file claimed it was "wrong by a factor of four", on the grounds that a box between 16
and 68 GiB would clear preflight and die in the pool. That reused `chrome_box`'s **pgm**
peak as the floor for a runner that executes only `chrome_crate` (**cm** — phase 1 of 2,
stated in its own header). Phase 1's measured figures, from `crafd_chain.out`'s manifest:
`pool: 96M`, `members_kept: 176M`, each member ~2.0 MB peak RSS, 7 minutes, completed on
the 87 GiB box without strain. 16 GiB is ample, so the floor was never too low.

⛔ An earlier version of this paragraph said an issue had been filed on that claim
("views-models#648"). **It had not** — 647 is the highest issue number this repository
has, and 648 was a number I invented. Corrected before this file was pushed.

The exposure is the other way round: **phase 2 has no script, so nothing preflights it.**
68.2 GiB belongs to `chrome_box` and is recorded where phase 2 is configured —
`ensembles/chrome_box/configs/config_hyperparameters.py` — pinned by
`tests/test_chrome_box_memory_is_the_measured_peak.py`, which requires all three
statements of the peak to agree (proved against eight mutations).

The lesson is the one pipeline-core gave about ratios, applied to absolutes: a peak with
no workload attached travels to the wrong question just as easily as a ratio with no
units. 68.2 GiB is the pgm pool. It is not a floor for anything else.

Two figures I reported during the session and withdraw: **19.8 GiB** (that was
`memory.current`, mostly page cache) and **27.8 GiB** (a 60s sampler undersample, and below
the arithmetic floor, so impossible as a peak). Use 68.2 GiB.

## What was NOT brought home, and why

- **28.6 GB of `chrome_box` forecasts.** Produced by a monkeypatched pipeline-core, so it is
  a probe result and not a deliverable. Do not publish it.
- **~30 GB of pgm member forecasts.** Reproducible: draws come from
  `np.random.default_rng(seed)` with `seed: 42`, so a rerun from the artifacts is
  bit-identical, not merely similar. The 8 artifacts are at
  `ensembles/chrome_box/data/generated/pod_delivery_20261028_artifacts/` (362 MB) and
  restore to `models/<member>/artifacts/`. Regeneration cost is forecast-only: ~11 min per
  member, measured, so ~1.5h for eight.

## Files

| file | what it is |
|---|---|
| `sitecustomize.py` | the two-defect patch — **the proven fix shape for #562 and #647** |
| `probe_reconciled.py` | the draw-for-draw reconciliation check |
| `probe_mapping.py` | measures mapping coverage against the asked-for window |
| `phase2_pgm.sh` | the pgm member driver, hand-written on the pod — `pod_run_crafd_chain.sh` has no phase 2 |
| `watch_probe.sh` | memory/progress watcher; deliberately avoids `pgrep` (the pattern matches the watcher's own ssh command line — that bit twice) |
| `crafd_chain.out` | phase 1, cm chain (`chrome_crate`, STATUS OK) |
| `phase2.out` | phase 2, 8 pgm members + the unpatched pool failure |
| `probe.out` | the patched run that completed |
