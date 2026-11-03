# EXP-03 run ledger — what was rented, what it ran, what it produced

**Why this file exists.** Simon, 2026-11-03: *"make sure it is logged and documented what we do
so this money is not wasted."* A pod costs $0.75/h and leaves nothing behind but what was
retrieved from it; a run nobody recorded is a run nobody can defend. This is the record of
which arm ran on which machine, what it cost, and whether its output is on disk and verified.

⛔ **It is not the scientific record.** The readout, the pre-registration and σ belong to
views-hydranet (`PREREG_EXP03_SEEDS.md`). This file answers an operator's question — *did we
get what we paid for* — and nothing else.

---

## The purchase, and what it buys

EXP-02 (seeds 42, 43) left σ resting on **one degree of freedom**. At h36 the between-seed sd
was `0.0049` against a within-seed MDE of `0.0047` — the two noise sources the same size, the
effect below both — and the sd's own 90% interval spans a factor of **31**, so the implied
seed count to resolve h36 ranged from **6 to 1,521**.

EXP-03's eight arms take σ to **5 dof** (n=6 seeds, 42–47). ⚠️ They are **not** expected to
settle P-23; views-hydranet pre-registered the prediction that they will not. What is being
bought is a usable noise estimate, not a verdict.

## The twelve scored arms

σ pools seeds 42–47, so the scored set is twelve arms, not eight. All twelve must run, all
twelve carry `diagnostic_visualizations: False`, and all twelve must land on one data pull —
`assert_diagnostics_off.py --expect 12` and the digest gate enforce those respectively.

| seed | control | treatment | status |
|---|---|---|---|
| 42 | `ctlfullzero_fortytwo` | `upctwofullzero_fortytwo` | ✅ complete, 2026-11-03 (EXP-02) |
| 43 | `ctlfullzero_fortythree` | `upctwofullzero_fortythree` | ✅ complete, 2026-11-03 (EXP-02) |
| 44 | `ctlfullzero_fortyfour` | `upctwofullzero_fortyfour` | running |
| 45 | `ctlfullzero_fortyfive` | `upctwofullzero_fortyfive` | running |
| 46 | `ctlfullzero_fortysix` | `upctwofullzero_fortysix` | running |
| 47 | `ctlfullzero_fortyseven` | `upctwofullzero_fortyseven` | not started |

**Arms are launched in seed PAIRS, deliberately.** A treatment differs from the control of its
own seed in exactly `{"model"}`, so a completed pair reads on its own. Capacity has been
intermittent all day; pairing means a dry spell leaves a usable contrast rather than a
treatment arm with nothing to compare against.

## Machines

All pods: `RTX PRO 4500 Blackwell Server Edition` (32,623 MiB), 128 cores, 755 GB RAM,
`$0.75/h` running, `$0.03/h` stopped.

⛔ **The GPU model is not a free choice.** σ pools 42–47 and EXP-02 ran on this card. Different
silicon would be a hardware difference correlated with seed — the same confound as mixing
diagnostics on and off — so every EXP-03 arm runs on the same model.

⚠️ `taskset -c 0-31` on every launch. `pod_run_model.sh:638` archives with `zstd -3 -T0`, one
thread per core; on a 128-core box that died in 2.3 s and left a **truncated 1 MB archive from
182 MB**, and the runner's own guard reported success. The runner is never-modify, so the bound
is enforced on the box. `--max-cores 128` would silence the guard without fixing the truncation.

## Per-arm log

Times are CET. Cost is wall-clock × $0.75/h.

| arm | pod | launched | gate | finished | cost | retrieved + verified |
|---|---|---|---|---|---|---|
| `ctlfullzero_fortytwo` | (EXP-02) | 2026-11-03 00:28 | ✅ 450 s | 06:16 | ~$1.60 | ✅ 108 MB, sha256 OK |
| `ctlfullzero_fortythree` | (EXP-02) | 2026-11-03 00:30 | ✅ 430 s | 03:38 | ~$1.60 | ✅ 108 MB, sha256 OK |
| `upctwofullzero_fortytwo` | (EXP-02) | 2026-11-03 00:28 | ✅ 430 s | 03:38 | ~$1.60 | ✅ 111 MB, sha256 OK |
| `upctwofullzero_fortythree` | (EXP-02) | 2026-11-03 00:28 | ✅ 440 s | 04:12 | ~$1.60 | ✅ 110 MB, sha256 OK |
| `ctlfullzero_fortyfour` | `216.81.151.57:14318` | 15:25 | ✅ | — | — | — |
| `upctwofullzero_fortyfour` | `216.81.151.63:17838` | 15:25 | ✅ | — | — | — |
| `ctlfullzero_fortyfive` | `216.81.151.71:11067` | 15:47 | ✅ | — | — | — |
| `upctwofullzero_fortyfive` | `216.81.151.63:16039` | 15:53 | ✅ | — | — | — |
| `ctlfullzero_fortysix` | `216.81.151.57:14192` | 16:11 | — | — | — | — |
| `upctwofullzero_fortysix` | `216.81.151.55:18132` | 16:11 | — | — | — | — |

## What "verified" means here

Nothing is counted as retrieved until all four hold:

1. `EXP02_STATUS` reads `OK` — the runner refuses to write it unless the gate slice succeeded
2. `sha256sum -c gate_endpoints.sha256` passes on the local copy
3. the local delivery is **byte-identical** to the pod's, compared as a set of
   `(path, size)` pairs under `LC_ALL=C` — not as a file count
4. `S=16` and `n_origins=13`, read off the arm's own cube rather than a manifest field

⚠️ **(3) is not pedantry.** A count check passes when two sides hold the same number of
different files, which is how a plot feed on 2026-10-26 reported success while missing a
whole directory of 17 files.

## Two things that would have wasted this money and were caught before launch

⛔ **The runner refused every EXP-03 arm.** `pod_run_exp02_arm.sh` holds membership as an
explicit list and rejects anything outside it at preflight. The list was still the four EXP-02
arms, so all eight new arms would have been thrown out **after the clone, before the GPU**.
Fixed in PR #693; the refusal now interpolates `${#ARMS[@]}` rather than spelling a number, so
it cannot go stale against its own contents again.

⛔ **The builder emitted 1 of 7 `.gitkeep`.** Six of the seven `REQUIRED_SUBDIRS` would have
been absent after the pods' `git clone --depth 1` and crashed `ModelPathManager` validation
before training started — eight times, after provisioning. Identical to #685, five days later,
in the tool rather than the instructions. Fixed in PR #692.

Both were free to catch by reading, and each would have cost a pod-hour to discover.

## Measured, for sizing future runs

| quantity | value | source |
|---|---|---|
| arm wall clock | ~2 h 09 m | EXP-02, quiet host |
| training | ~96 min, 13.1–19.5 s/lesson | lesson timestamps |
| emit | 1.70–3.44 min/origin | origin timestamps, 227/228-char attribution gap |
| cost per arm | ~$1.60 | wall clock × $0.75/h |
| delivery size | 108–111 MB | the four EXP-02 arms |
| peak RSS | ~21 GB | ⇒ pod spec ≥32 GB; the runner has no memory guard |

⚠️ **Co-tenancy moves wall clock by 4–9×, not the results.** One EXP-02 arm ran at 15.18
min/origin against 1.70–3.44 on quiet boxes; its host sat at load 43–66 with ~10% of CPU in
nice'd processes that were not ours, while its GPU showed no throttling at all. Training is
seeded and the digest gate pins the data, so a slow box costs patience and money, not validity.

## Operational facts, both confirmed 2026-11-03

- **`/root/.netrc` does NOT survive a stop/start.** `/root` is container disk, not the
  `/workspace` volume. Re-place it on every restart — it fails late, because the fetch is the
  first thing that needs it, minutes in.
- **The repo clone and `/workspace/venv` DO survive.** `pod_run_model.sh:158` skips the venv
  build when one exists, so a restarted pod is minutes from ready.
- ⛔ **Stopping preserves setup; it does NOT reserve hardware.** All four pods stopped overnight
  had their GPUs released and reallocated, and migration found no instances.
