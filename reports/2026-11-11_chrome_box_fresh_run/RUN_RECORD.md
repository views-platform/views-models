# chrome_box fresh run — 2026-11-11/12 (views-models#721)

**Status: COMPLETE and verified. NOT PUBLISHED, deliberately.**
Publishing is parked pending views-datafactory#547. See §5 — the reason is measured, not cautionary.

## 1. What ran

| | |
|---|---|
| pod | RunPod, 128 cores, RTX PRO 4500, cgroup limit 93,999,996,928 B = **87.6 GiB** |
| repo | `development` @ `d8593ab` (NOT the #721 branch — that carries only tests/tooling) |
| phase 1 | `tools/podrun/pod_run_crafd_chain.sh` → `chrome_crate` (cm), **OK in 8 min** |
| phase 2 | `tools/podrun/pod_run_chrome_box.sh` → `chrome_box` (pgm), **OK in 31 min** |
| total | **~40 min**, not the ~208 min the earlier record implied |
| draws | 8 × 128 = **1,024**, not thinned |
| forecast | 3 targets × (2,330,712 rows × 1,024 draws) float32 = 64,742 cells × 36 months |
| published | **nothing.** The runners hold no publish credential (guide rule 5) |

## 2. The measurements worth keeping

**Memory — size a box from `PEAK anon`, nothing else.**

| figure | value | note |
|---|---|---|
| **PEAK anon** | **77,371 MiB = 75.6 GiB** | THE REQUIREMENT |
| PEAK current | 89,642 MiB | inflated by reclaimable page cache — do NOT size from this |
| cgroup `memory.events` | `max 6413`, **`oom_kill 0`** | pressed the ceiling, reclaimed cache, never killed |

So the 87.6 GiB class fits with ~12 GiB of headroom, and this run used LESS than 2026-11-02's
81.0 GiB. An earlier reading of "87.5 of 87.6, nearly died" was page cache and is wrong.

**Reconciliation — the failure that is silent, measured directly.**

| target | mass unplaced | scaling failures |
|---|---|---|
| `lr_sb_best` | 0 of 540,841,924 — **0.000%** | 0 |
| `lr_ns_best` | 0 of 93,572,747 — **0.000%** | 0 |
| `lr_os_best` | 0 of 225,144,189 — **0.000%** | 0 |

`views-frames 3.0.0` (exactly at the floor) and `views-datafactory 1.18.0` (clears the 1.17.0
requirement that makes the floor reachable). Below frames 3.0.0 a country whose grid is all-zero in
a draw loses its whole mass with `STATUS: OK` and `scaling_failures 0`.

## 3. Where it lives, and what was verified

`pod_deliverables/2026-11-11_chrome_box/` — 28 GB, **148 files, every one md5-matched against the
pod before the pod was destroyed.** Compared as a SET of (md5, path) pairs: an ordering diff from
locale collation on `_chrome_box-wandb` initially looked like a mismatch and was not one.

`pod_deliverables/2026-11-11_chrome_crate/` — 100 files, md5-verified. Phase 1's output, i.e. the
INPUT phase 2 reconciled against. Retrieved only after a `/falsify` audit found it had been left
behind; without it §2's 0.000% rests on the runner's self-report rather than on re-derivable data.

`_runlogs/` — both phase transcripts, md5-verified. `_training_pull_provenance/` — the 74 MB pull.

**Regenerable, left on the pod:** ~30 GB of per-member draws (3.5–4.1 GB × 8). Reconstructible from
the local member artefacts without retraining. ⚠️ What that leaves unproven: the pooling step
("8 of 8 pooled") can no longer be audited from local state alone.

## 4. Why the forecast NUMBERS are provisional

The run read the store on 2026-11-11 at 23:26 UTC, `loa=priogrid_month`. datafactory's spatial
allocator writes its chosen cell to `priogrid_gid` and their grid compiler re-derives the cell from
lat/long and never reads it — the allocation is computed and discarded. It has never worked in any
release, so this is **not a regression**; 1.18.0 is the first time anyone read the delivered side.

Measured here against the 2026-11-02 delivered artefact (same 28,615,964 rows, same months):

| target | 2026-11-02 | 2026-11-11 | delta |
|---|---|---|---|
| sb | 1,973,214 | 2,618,062 | +644,848 (+32.7%) |
| ns | 291,911 | 415,235 | +123,324 (+42.2%) |
| os | 1,231,226 | 1,292,446 | +61,220 (+5.0%) |
| **total** | 3,496,351 | 4,325,743 | **+829,392 (+23.7%)** |

Mass **added** 829,461; mass **removed** 69. Purely additive — formerly excluded, not moved.

⛔ **The added mass is centroid-dumped.** 1,051 cells gained (views-postprocessing predicted
"roughly 1,034" — essentially exact). One cell took 28.2% of the addition:

| priogrid | where | was | now |
|---|---|---|---|
| **200585** | **Cherkasy, central Ukraine** (32.25E, 49.25N) | **7** | **234,222** |
| 180437 | Hama / As-Salamiyeh, Syria | 845 | 88,239 |
| 176147 | Ardekan, Yazd, Iran (NEW cell) | 0 | 16,077 |
| 148759 | Tigray, Ethiopia (the #533 artefact) | 273,930 | 273,930 |

⚠️ **Method note, because the first reading was wrong.** The *level* concentration curves are nearly
identical (top-10 share 32.99% → 32.21%) and that looked like it refuted centroid dumping. It did
not: adding 829k to an already top-heavy 3.5M barely moves the level. The signal is in the **delta**.
Measure on the axis the subjects differ on.

⇒ Country-month sums still reconcile (99.89% of displaced mass stays in the same GAUL country), so
**every aggregating check passes.** Nothing in views-models can detect this; placement is
datafactory's authority and their coverage gates assert cell completeness, never correct placement.

## 5. The decision, and what it rests on

**Hold the CRAF'd delivery until views-datafactory#547 lands** (their target: before the
21 November cron). Agreed with the maintainer 2026-11-12.

Not because the run is bad — it is better than 2026-11-02 temporally and identical spatially — but
because this would be CRAF'd's **first ever delivery** and it is a pgm product, so the cell pattern
is the entire reason a partner takes it. An analyst opening it would see a quarter-million deaths in
rural central Ukraine. The re-run then costs one pod and ~40 minutes, both invocations now proven.

What does NOT expire when the data is refetched: §2's memory figure, the 40-minute runtime, the
reconciliation mechanism working end-to-end, and the retrieval discipline in §3.

## 6. Carried findings

- **`queryset_digest` digests the CONFIG, not the data.** Measured: 8 members, 8 different digests,
  byte-identical pulls (md5 `2cb79304c61b`). So it cannot answer "did these two runs read the same
  store?" — the question #547 makes urgent. Corroborates datafactory's own statement.
- `drift_detection_ran: false` in the pull provenance (cf. C-202: the drift check has no
  outlier/range component).
- The pool directory is named `parquet/` and holds `.npy`/`.npz`. This defeated a probe in the
  `/falsify` audit: a `*.parquet` glob returned zero files and would have read as "verified".
- `note()` in both pod runners hardcodes a `MISSING:` prefix, so a present credential reports
  `MISSING: X is set`. Inherited, shared helper, not fixed here.

Stubs: `reports/2026-11-12_falsify_pod_termination/` (outside the suite; `reports/` is in
`norecursedirs`).
