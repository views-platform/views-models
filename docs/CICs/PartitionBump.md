# Class Intent Contract: tools.partitions.bump

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-07
**Related ADRs:** ADR-011

---

## 1. Purpose

> `bump.py` is the CLI orchestrator for the annual partition boundary advance. It ties together domain validation (`domain.py`) and file operations (`fileops.py`) into a phased pipeline: load → validate → discover → pre-flight → apply → verify → lockfile. It is the only module in the package with side effects.

Located in: `tools/partitions/bump.py`

Invocation: `python -m tools.partitions.bump [--execute] [--bump N] [--force REASON]`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate partition values against model performance or data quality
- Does **not** modify the forecasting section of any file
- Does **not** retrain models or trigger pipeline runs
- Does **not** push to git or create commits — that's the operator's responsibility

---

## 3. Responsibilities and Guarantees

### Dry-run mode (default, no `--execute`)
- Prints current and bumped partition values with human-readable dates
- Prints partition inventory (production models, research overrides, test fixtures, total)
- Runs pre-flight check: all standard files must match current canonical
- Reports what would change — modifies nothing
- Exits 0 on success, non-zero on validation/pre-flight failure

### Execute mode (`--execute`)
All of the above, plus:
- Rewrites calibration/validation tuples in all standard config_partitions.py files
- Uses atomic writes (`fileops.write_atomic`: tempfile + os.replace) — no partial file corruption, and existing files keep their permission bits (mode is preserved)
- Re-reads and verifies every written file before proceeding
- Skips files with `PARTITION_OVERRIDE = True`
- Updates `meta/partitions.json` with new canonical values (via `write_atomic`, so its mode is preserved too — `_save_canonical` delegates rather than duplicating the tempfile/replace pattern)
- Writes a JSONL lockfile to `meta/partition_bump_YYYYMMDD_HHMMSS.jsonl` recording: before/after values, dates, git state, every file updated, every file skipped, verification status

### Safety checks (both modes)
- 7 structural invariants validated on both old and new values
- Temporal plausibility: validation test end cannot exceed Dec (current_year - 1)
- Coverage check: all production entities must have partition configs
- Pre-flight: all standard files must match current canonical before any writes
- Missing partition files block `--execute`

### CLI flags
- `--bump N` — advance by N month_ids (default 12). Use 0 to sync without advancing.
- `--force REASON` — bypass temporal plausibility check with a recorded reason
- `--execute` — apply changes (without this, dry-run only)

---

## 4. Inputs and Assumptions

- `meta/partitions.json` must exist and contain valid JSON with calibration/validation train/test arrays
- `meta/fixtures.json` must exist (loaded by `fileops.py` at import time)
- All `config_partitions.py` files must contain a `return {` statement with calibration/validation sections
- Git must be available for lockfile git-state capture (gracefully degrades if unavailable)
- The tool must be invoked from the repo root (or with the repo root on `sys.path`)

---

## 5. Outputs and Side Effects

### Files modified (execute mode only)
- `models/*/configs/config_partitions.py` — calibration/validation tuples rewritten
- `ensembles/*/configs/config_partitions.py` — same
- `extractors/*/configs/config_partitions.py` — same
- `postprocessors/*/configs/config_partitions.py` — same
- `meta/partitions.json` — updated with new canonical values

### Files created (execute mode only)
- `meta/partition_bump_YYYYMMDD_HHMMSS.jsonl` — lockfile

### stdout
- Partition values (current and bumped) with human-readable dates
- Partition inventory breakdown
- Pre-flight check results
- Per-file update/verify status (execute mode)
- Summary with file counts, git commit hash, before/after

---

## 6. Failure Modes and Loudness

| Condition | Behavior | Exit code |
|-----------|----------|:---------:|
| `meta/partitions.json` missing | `ERROR: ... not found` | 1 |
| `meta/partitions.json` corrupt JSON | `ERROR: ... invalid JSON` | 1 |
| `meta/partitions.json` missing keys | `ERROR: ... invalid structure` | 1 |
| Current values violate invariants | `ERROR: Current canonical values violate invariants` | 1 |
| Bumped values violate invariants | `ERROR: Bumped values violate structural invariants` | 1 |
| Bumped values fail temporal check | `ERROR: ... exceeds latest UCDP annual data` | 1 |
| Production entity missing partition file | `WARNING` (dry run) / `ERROR` (execute) | 0 / 1 |
| Pre-flight file mismatch | `MISMATCH: ...` then `ERROR: N file(s) do not match` | 1 |
| File write failure | `WRITE ERROR: ...` then abort | 1 |
| Post-write verification failure | `FATAL: ... verification failure(s)` — lockfile NOT written | 1 |
| Negative `--bump` value | `ERROR: --bump must be non-negative` | 1 |
| Non-multiple-of-12 bump | `WARNING` — proceeds | 0 |
| `--force` with temporal failure | `WARNING: Temporal plausibility bypassed` — proceeds | 0 |

---

## 7. Boundaries and Interactions

- Depends on: `tools.partitions.domain` (PartitionBoundaries, month_id_to_date), `tools.partitions.fileops` (discover, extract, rewrite, verify, write_atomic, has_partition_override)
- Depends on: `subprocess` (git state capture), `argparse` (CLI), `json` (partitions.json I/O)
- Called by: operator via CLI (`python -m tools.partitions.bump`)
- Feeds into: `tests/test_config_partitions.py` (verifies the files it modified)

---

## 8. Examples of Correct Usage

```bash
# Preview what would change (safe, default)
python -m tools.partitions.bump

# Apply the annual bump
python -m tools.partitions.bump --execute

# Sync all files to canonical without advancing
python -m tools.partitions.bump --bump 0 --execute

# Override temporal check with documented reason
python -m tools.partitions.bump --execute --force "UCDP pre-release data available"
```

---

## 9. Examples of Incorrect Usage

```bash
# Wrong: running --execute without reviewing dry-run first
python -m tools.partitions.bump --execute  # works but skips human review

# Wrong: assuming --bump 24 will work in a single year
python -m tools.partitions.bump --execute --bump 24  # blocked by temporal check

# Wrong: running from a different directory
cd /tmp && python -m tools.partitions.bump  # ModuleNotFoundError
```

---

## 10. Test Alignment

- `tests/test_bump_partitions.py::TestBumpIntegration` — 9 integration tests exercising `main(repo_root=tmp_path)` end-to-end: dry run, execute, pre-flight reject, missing JSON, temporal block, override skip, sync mode, lockfile content, partitions.json update
- `tests/test_bump_partitions.py` — 34 unit tests covering domain, fileops, discovery, overrides
- `tests/test_bump_partitions.py::TestAdversarialInputs` — 9 red tests for adversarial inputs
- `tests/test_bump_partitions.py::TestStructuralCompliance` — 5 beige tests for structural compliance
- `tests/test_falsify_bump_robustness.py` — 3 tests verifying resolved findings
- `tests/test_falsify_bump_completeness.py` — 2 tests (ADR-011, override mechanism)
- `tests/test_falsify_bump_edge_cases.py` — 3 tests (error handling, comment safety, temp cleanup)
- `tests/test_config_partitions.py` — 934 tests verifying consistency across all 100 files

---

## 11. Evolution Notes

- `main()` accepts an optional `repo_root: Path` parameter for testability. Defaults to the module-level `_DEFAULT_REPO_ROOT`. All internal functions (`_load_canonical`, `_save_canonical`, `_git_state`) accept path parameters.
- `main()` is a 300-line function. If it grows further, extract phases into named functions.
- The lockfile format is append-only JSONL. If the tool needs to read previous lockfiles (e.g., to detect the last bump date), a `read_latest_lockfile()` function would be needed.

---

## End of Contract

This document defines the **intended meaning** of `tools.partitions.bump`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
