# ADR-024: What every pod runner must honour — STATUS, the credential floor, and the disk it measures

**Status:** Accepted
**Date:** 2026-10-06
**Deciders:** Simon, VIEWS platform team
**Related ADRs:** [ADR-022](022_one_postprocessor_launcher.md) (one launcher body),
[ADR-023](023_collapsing_posterior_draws.md) (the collapse, including §6's darts instance)
**Related:** `docs/runpod_run_guide.md` ground rule 5 (which credentials may exist on rented
hardware — **that policy lives there, not here**), `tools/podrun/__init__.py` (the group's
provisional status and the `_common.sh` extraction trigger), epic #532

---

## Context

`tools/podrun/` began as one script for one model family. It is now three — `pod_run_model.sh`,
`pod_run_fao_delivery.sh`, `pod_run_darts_calibration.sh` — and its own `__init__.py` states the
expectation of a fourth, with a named trigger for extracting the shared helpers.

Three scripts that do not import from each other, written months apart, that nonetheless have to
agree. One of them already **reads another's output to make a decision**:
`pod_run_fao_delivery.sh` reads `$ROOT/deliver/<model>/STATUS` and refuses to pool a roster
unless every member reports `OK`.

Each rule below was a real defect in the third script, caught before it ran. None was caught by
the second script's existence, which is the point: a convention that only lives in the scripts
that already follow it is not a convention, it is a coincidence.

---

## Decision

### 1. `STATUS` is a contract between scripts, not a log line.

`$ROOT/deliver/<model>/STATUS` may hold exactly one of:

| value | meaning |
|---|---|
| `OK` | the deliverable exists **and has been verified**. Safe to consume. |
| `FAILED:<stage>` | the run stopped at `<stage>`. `FAILURE` holds the reason. |
| `PREFLIGHT_OK` | checks passed; **nothing was run and no deliverable exists.** |

**A runner writes `OK` from exactly one place, after verification, and never before.** The
reason this needs stating: `pod_run_darts_calibration.sh` originally wrote plain `OK` on
`--preflight`, which on disk is indistinguishable from a completed run that produced thirteen
verified parquets. A consumer applying the rule `pod_run_fao_delivery.sh` already applies —
"`OK` means usable" — would have pooled nothing and reported success. `PREFLIGHT_OK` exists
because the distinction has to be machine-readable, not inferable from a timestamp.

Any new value is an addition to this table, made here, before a consumer can guess at it.

### 2. A resource floor must measure the resource the work actually uses.

A disk check on `$ROOT` is worthless if the workload writes elsewhere. `views-r2darts2`'s
`PredictionScratch()` passes no `base_dir`, so its scratch honours `TMPDIR` and otherwise lands
in `/tmp` — the pod's **container** disk — while the floor measures `/workspace`, the **volume**.
The two are separately sized (100 GB each in the guide's Phase 1.2), so the check and the
consumption were on different filesystems and the check could not fail for the right reason.

**A runner either points the work at the filesystem it measures, or measures the one the work
uses.** `pod_run_darts_calibration.sh` exports `TMPDIR="$ROOT/tmp"` and says why in-line.

This generalises past disk, and the generalisation is the part worth keeping: the in-code memory
guard has the same shape of bug — it reads the *host's* RAM and approves runs that cannot fit in
the container (guide, "Known caveats"). A floor that cannot fire is worse than no floor, because
it is believed.

### 3. A runner installs no credential it does not need, and the next runner is not written by
copying the last one.

The policy — publish credentials never go on rented hardware — is **ground rule 5 in
`docs/runpod_run_guide.md`** and is not restated here; one rule in two places is two places to
drift (`vmo_021`).

What belongs here is the mechanism by which it breaks. `pod_run_fao_delivery.sh`
**legitimately** installs `views-pipeline-core[appwrite]` and reads nine publish variables,
because publishing is its job. "Start from the script that already works" is therefore the
obvious way to write the next runner and the way that puts write credentials on a machine we do
not own. A calibration runner needs only the datafactory **read** credential in `/root/.netrc`.

**Enforced executably, not by comment**: `tests/test_darts_calibration_runner.py` asserts, on
comment-stripped source, that no Appwrite path is installed and no publish variable is named. A
comment would not survive the next edit; the rule has to be able to fail a build.

---

## Consequences

**Positive.** A consumer can read `STATUS` without knowing which script wrote it. A fourth runner
has three concrete things to honour rather than two existing scripts to reverse-engineer, and the
two older scripts can be audited against this table rather than against each other.

**Negative.** `pod_run_model.sh` and `pod_run_fao_delivery.sh` both write plain `OK` on a
`--preflight`-style early exit and therefore **do not satisfy §1 today.** This ADR is written
knowing that: the darts runner is the only one that complies, and the older two are not being
edited for it, because `pod_run_fao_delivery.sh` has completed a real delivery to the UN FAO and
a correctness-neutral edit to it is not free. **This is a declared debt, not an oversight** —
close it when `_common.sh` is extracted (`tools/podrun/__init__.py` holds that trigger), which is
the one change that will touch all three anyway.

**Also negative.** §2 is stated as a principle and enforced in exactly one place. The memory
guard it generalises to lives in views-hydranet, not here, so this ADR describes a defect it
cannot fix.

---

## Validation & Monitoring

- §1: `tests/test_darts_calibration_runner.py::test_only_a_finished_run_writes_OK_to_status` —
  asserts exactly one bare `OK`, that it follows `stage verify_parquet`, and that the preflight
  path distinguishes itself. Mutation-verified: restoring the bare `OK` turns it red.
- §2: `…::test_tmpdir_is_pointed_at_the_volume_before_the_run` — asserts `TMPDIR` is exported
  *before* `main.py` runs, not merely somewhere in the file.
- §3: `…::test_no_appwrite_extra_is_installed` and `…::test_no_publish_variable_is_read`.

**What is not validated:** nothing checks the two older scripts against §1, by the choice
recorded above. A test asserting the whole group complies would be red today, so writing one now
would mean either a red suite or a weakened assertion — and a weakened assertion is how the
guards in this very directory came to be decorative once already (#501).

---

## References

- `tools/podrun/__init__.py` — provisional status, the silent bugs found by review, the
  `_common.sh` trigger
- `docs/runpod_run_guide.md` — ground rule 5; Phase 1.2 (the two separately-sized disks);
  Phase 4c (the darts chain); "Known caveats" (the host-RAM memory guard)
- `reports/postmortem_runpod_first_deployment_2026-09.md` §2.2–2.4 — why RAM, then vCPU, then
  VRAM, and the 25× slowdown that produced the rule
- views-models **#532** (epic), **#534** (the runner), **#537** (the first real darts run)
- views-r2darts2 **#54** — the scratch directories that filled a 2 TB disk; the reason §2 was
  looked at at all
- Register: **C-151** (the toolz override), **C-154** / **#518** (`/workspace` ignores `chmod`)
