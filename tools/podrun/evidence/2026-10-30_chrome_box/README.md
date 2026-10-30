# 2026-10-30 — the CRAF'd chain ran end to end, on one box

**This is the run that met the definition of done.** `chrome_crate` (cm) then `chrome_box`
(pgm, reconciled against it), on one rented 87 GiB RunPod host, nothing published, no Appwrite
credential ever on the box.

| | |
|---|---|
| phase 1 — `chrome_crate` + 8 `*_condor` members | `STATUS: OK` **04:16:43Z** |
| phase 2 — `chrome_box` + 8 `*_sparrow` members | `STATUS: OK` **07:30:32Z**, 193 min |
| git sha | `a32a1ee` (PR #649 as merged) |
| drivers | `tools/podrun/pod_run_crafd_chain.sh`, then `tools/podrun/pod_run_chrome_box.sh` |

The product: 3 targets × `(2330712, 1024)` float32 — **64,742 grid cells × 36 months × 1,024
draws** — reconciled to `chrome_crate`'s country totals. 26.8 GiB raw, retained off-repo as a
1.70 GiB zstd archive with its sha256; **every one of the 6 forecast files was verified
byte-identical to the pod before the host was destroyed.**

## Why these files are in the repository

The 2026-10-28 run's memory series **never existed** — `watch_probe.sh` wrote to no file, so the
"≥ 68.2 GiB" figure that sized this box was a lower bound with no retrievable evidence behind
it. Everything here is the answer to that: the measurements this run made, committed, so the
next reader does not re-derive them on a third paid box.

⚠️ `durable ≠ version-controlled`. These lived in a folder outside any git repository until this
commit.

| file | what it is |
|---|---|
| `MANIFEST.md` | the MANIFEST phase 2 wrote, verbatim |
| `STATUS` | `OK` |
| `run.out` | the full phase-2 log — 8 HEARTBEATs, the 3 `Reconciled '…' (pgm_cm)` lines, `Done. Runtime: 6.719 minutes` |
| `memory.out` | the `memory.current` series, 10 s sampling |
| `anon.out` | **the `anon` series — the one that matters** |
| `RECONCILIATION_RESIDUAL` | what reconciliation could not place |
| `FREEZE` | the venv's 143 packages |

⛔ **Filenames avoid `*.log` and `*.txt` deliberately.** Both are gitignored here, and so is a
file called `MANIFEST` (`.gitignore:82`) — a `git add` of the obvious names drops them silently,
which is how a committed evidence directory ends up holding nothing.

## The three numbers worth reading

**1. Memory: 63.9 GiB, not 87.** `anon` peaked at **65,404 MiB** at 07:28:00 in
`pool_and_reconcile`. `memory.current` peaked at **89,609 MiB** — pinned against the cgroup
limit, because it counts reclaimable page cache that climbs monotonically as members write 9.5 GB
arrays. During the member stage the two read 4.3 GiB and 22.0 GiB: **5.1× apart.**

`MEM_FLOOR_GB=80` is therefore correct and load-bearing. A 64 GiB box fails. ⚠️ Both figures that
previously set that floor — the 68.2 GiB lower bound and the withdrawn 19.8 GiB in
`chrome_box/configs/config_hyperparameters.py` — were `memory.current` readings. They were high
rather than wrong, which is why they survived three sessions.

**2. Members cost 22–23 min each, flat.** 04:40:07, 05:02:18, 05:24:30, 05:47:17, 06:10:57,
06:34:32, 06:58:27, 07:22:10 — intervals of 22.2, 22.2, 22.8, 23.7, 23.6, 23.9, 23.7 min. 128
CPUs did not help; these are CPU-light conflictology baselines. Pool and reconcile: 6.7 min.

**3. Reconciliation lost 2.9%–7.3% of the fatality mass, and nothing said so.** See
`RECONCILIATION_RESIDUAL` and **C-192**. Measured over all 22,339,584 (country, month, draw)
triples with the run's own country mapping:

```
lr_sb_best  unplaced 12,866,485 of 319,789,635  = 4.023%   scaling_failures 0
lr_ns_best  unplaced  5,272,245 of  72,013,473  = 7.321%   scaling_failures 0
lr_os_best  unplaced  6,553,019 of 222,651,525  = 2.943%   scaling_failures 0
```

✅ The reconciler is **exact**: zero triples disagree where the grid sum is non-zero. ⛔ 100% of
the disagreement is countries whose every grid cell drew zero — `0 × anything = 0`, so a
multiplicative scale cannot lift them. Filed upstream with this measurement as
**views-frames#283**; the redistribution policy is a modelling decision needing an ADR there.
The MANIFEST here is the last one that will say `reconciled:` unqualified.

## Not in this directory

The 26.8 GiB pool, the 16 trained member artefacts and the 50 offline W&B runs are retained
off-repo (two zstd archives, both sha256-verified after the host was destroyed). ⛔ The artefacts
were nearly lost: this runner had no `collect_members` stage until the teardown audit found them
about to die with the box — the 2026-09-28 failure (views-models#549) one artefact class along.
The stage, and a guard for it, land in the same PR as this directory.
