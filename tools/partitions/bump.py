"""Annual partition bump for VIEWS models.

Advances calibration and validation partition boundaries forward by 12
month_ids (= 1 year of UCDP data), rewrites all config_partitions.py
files, verifies every file post-write, and produces a JSONL lockfile
recording exactly what happened.

The training start (121 = Jan 1990) never moves. The forecasting
partition is dynamic and untouched.

Safety mechanisms:
    - Dry-run by default (must pass --execute to write)
    - 7 structural invariant checks on both old and new values
    - Temporal plausibility: validation test end cannot exceed Dec (current_year - 1)
    - Pre-flight: all files must match current canonical before bump
    - Post-write verification: every file re-read and compared
    - Override files (PARTITION_OVERRIDE) are skipped and checked for contamination
    - Atomic file writes (tempfile + os.replace)
    - JSONL lockfile with git state for full audit trail

Usage:
    python -m tools.partitions.bump                        # dry run
    python -m tools.partitions.bump --execute              # apply
    python -m tools.partitions.bump --execute --bump 24    # custom
    python -m tools.partitions.bump --bump 0               # sync to canonical without advancing
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.partitions.domain import (
    PartitionBoundaries,
    month_id_to_date,
)
from tools.partitions.fileops import (
    discover_entity_dirs,
    discover_partition_files,
    extract_values,
    has_partition_override,
    rewrite_values,
    verify_file,
    write_atomic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PARTITIONS_FILE = REPO_ROOT / "meta" / "partitions.json"
LOCK_DIR = REPO_ROOT / "meta"


def _load_canonical() -> dict:
    with open(PARTITIONS_FILE) as f:
        return json.load(f)


def _save_canonical(canonical: dict, boundaries: PartitionBoundaries) -> None:
    merged = {**canonical, **boundaries.to_json_dict()}
    dir_name = PARTITIONS_FILE.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(merged, tmp, indent=2)
        tmp.write("\n")
        tmp_path = tmp.name
    os.replace(tmp_path, str(PARTITIONS_FILE))


def _git_state() -> dict:
    """Capture current git commit, branch, and dirty status."""
    state = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT,
        )
        state["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT,
        )
        state["git_branch"] = result.stdout.strip() if result.returncode == 0 else "unknown"

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=REPO_ROOT,
        )
        state["git_dirty"] = bool(result.stdout.strip()) if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        state.setdefault("git_commit", "unavailable")
        state.setdefault("git_branch", "unavailable")
        state.setdefault("git_dirty", None)
    return state


def _print_boundaries(b: PartitionBoundaries) -> None:
    for name, val in [
        ("calibration train", b.cal_train),
        ("calibration test", b.cal_test),
        ("validation train", b.val_train),
        ("validation test", b.val_test),
    ]:
        print(
            f"  {name}: {val}  "
            f"({month_id_to_date(val[0])} – {month_id_to_date(val[1])})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Bump VIEWS partition boundaries forward by N months.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Apply changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--bump", type=int, default=12,
        help="Number of month_ids to advance (default: 12 = 1 year). Use 0 to sync without advancing.",
    )
    parser.add_argument(
        "--force", type=str, default=None, metavar="REASON",
        help="Bypass temporal plausibility check. Requires a reason string.",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    bump = args.bump

    if bump < 0:
        print(f"ERROR: --bump must be non-negative, got {bump}")
        sys.exit(1)
    if bump > 0 and bump % 12 != 0:
        print(f"WARNING: --bump={bump} is not a multiple of 12 (years).")

    # --- Load and validate current canonical ---
    canonical = _load_canonical()
    current = PartitionBoundaries.from_json(canonical)

    print("=== Current partition values ===")
    _print_boundaries(current)

    pre_errors = current.validate_invariants()
    if pre_errors:
        print("\nERROR: Current canonical values violate invariants:")
        for e in pre_errors:
            print(f"  - {e}")
        print("\nFix meta/partitions.json before bumping.")
        sys.exit(1)

    # --- Compute and validate new values ---
    if bump == 0:
        new = current
        print("\n=== Sync mode (--bump 0): no value change ===")
    else:
        new = current.bumped(bump)
        print(f"\n=== Bumped partition values (+{bump} month_ids) ===")
        _print_boundaries(new)

    post_errors = new.validate_invariants()
    if post_errors:
        print("\nERROR: Bumped values violate structural invariants:")
        for e in post_errors:
            print(f"  - {e}")
        sys.exit(1)

    temporal_errors = new.validate_temporal()
    if temporal_errors and args.force is None:
        print("\nERROR: Bumped values fail temporal plausibility check:")
        for e in temporal_errors:
            print(f"  - {e}")
        sys.exit(1)
    elif temporal_errors and args.force is not None:
        print(f"\nWARNING: Temporal plausibility bypassed (--force: {args.force})")
        for e in temporal_errors:
            print(f"  - {e}")

    new_flat = new.to_flat_dict()
    current_flat = current.to_flat_dict()

    # --- Coverage check ---
    entity_dirs = discover_entity_dirs(REPO_ROOT)
    entity_set = set(entity_dirs)
    files = discover_partition_files(REPO_ROOT)
    partition_parents = {f.parent.parent for f in files}
    missing = [d for d in entity_dirs if d not in partition_parents]

    entity_files = [f for f in files if f.parent.parent in entity_set]
    fixture_files = [f for f in files if f.parent.parent not in entity_set]

    # --- Classify files: standard vs override vs fixture ---
    override_files = []
    standard_files = []
    for f in files:
        source = f.read_text()
        if has_partition_override(source):
            override_files.append(f)
        else:
            standard_files.append(f)

    print("\n=== Partition inventory ===")
    if missing:
        print(
            f"  WARNING: {len(entity_dirs) - len(missing)}/{len(entity_dirs)} "
            f"production models have partition configs"
        )
        for d in missing:
            print(
                f"    MISSING: {d.relative_to(REPO_ROOT)} "
                f"— has main.py but no config_partitions.py"
            )
        if not dry_run:
            print(
                "\nERROR: Cannot bump with missing partition files. "
                "Add them first."
            )
            sys.exit(1)
    else:
        print(
            f"  {len(entity_files)}/{len(entity_dirs)} "
            f"production models — all have partition configs"
        )
    if override_files:
        print(
            f"  {len(override_files)} research override(s) "
            f"(custom partitions, not bumped):"
        )
        for f in override_files:
            print(f"    {f.parent.parent.relative_to(REPO_ROOT)}")
    else:
        print("  0 research overrides")
    print(f"  {len(fixture_files)} test fixtures")
    print(f"  {len(files)} total partition files")

    # --- Pre-flight check (standard files only) ---
    print("\n--- Pre-flight: all standard files must match current canonical ---")
    preflight_failures = []
    target_files = []

    for path in standard_files:
        rel = path.relative_to(REPO_ROOT)

        parsed = extract_values(path.read_text())
        if parsed is None:
            preflight_failures.append((path, "could not parse partition values"))
            print(f"  PARSE ERROR: {rel}")
            continue

        mismatches = []
        for key, expected_val in current_flat.items():
            if parsed.get(key) != expected_val:
                mismatches.append(
                    f"{key}: expected {expected_val}, got {parsed.get(key)}"
                )
        if mismatches:
            preflight_failures.append((path, "; ".join(mismatches)))
            print(f"  MISMATCH: {rel}")
            for m in mismatches:
                print(f"    {m}")
        else:
            target_files.append(path)

    if preflight_failures:
        print(
            f"\nERROR: {len(preflight_failures)} file(s) do not match "
            f"current canonical values."
        )
        print("Fix manually or investigate before bumping.")
        sys.exit(1)

    print(f"\n  {len(target_files)} files to update")
    if override_files:
        print(f"  {len(override_files)} override files skipped")

    if dry_run:
        print("\n=== DRY RUN — no files modified ===")
        print(f"Would update {len(target_files)} files.")
        if override_files:
            print(f"Would skip {len(override_files)} research override(s).")
        print("\nRun with --execute to apply.")
        sys.exit(0)

    # --- Apply changes ---
    print("\n--- Applying changes (atomic writes) ---")
    updated = []
    write_errors = []

    for path in target_files:
        rel = path.relative_to(REPO_ROOT)
        try:
            source = path.read_text()
            new_source = rewrite_values(source, new_flat)
            write_atomic(path, new_source)
            updated.append(path)
            print(f"  Updated: {rel}")
        except Exception as e:
            write_errors.append((path, str(e)))
            print(f"  WRITE ERROR: {rel}: {e}")

    if write_errors:
        print(f"\nERROR: {len(write_errors)} file(s) failed to write.")
        print("Some files may be inconsistent. Check git diff.")
        sys.exit(1)

    # --- Post-write verification ---
    print("\n--- Post-write verification ---")
    verify_failures = []

    for path in updated:
        rel = path.relative_to(REPO_ROOT)
        errors = verify_file(path, new_flat)
        if errors:
            verify_failures.extend(errors)
            print(f"  VERIFY FAILED: {rel}")
            for e in errors:
                print(f"    {e}")
        else:
            print(f"  Verified: {rel}")

    if verify_failures:
        print(f"\nFATAL: {len(verify_failures)} verification failure(s).")
        print("Lockfile will NOT be written. Files may be inconsistent.")
        print(
            "Revert with: git checkout -- "
            "models/ ensembles/ extractors/ postprocessors/"
        )
        sys.exit(1)

    # --- Update meta/partitions.json ---
    print("\n--- Updating meta/partitions.json ---")
    _save_canonical(canonical, new)
    print("  Written: meta/partitions.json")

    # --- Write lockfile ---
    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    lock_path = LOCK_DIR / f"partition_bump_{timestamp}.jsonl"

    git = _git_state()
    lock_entries = []

    lock_entries.append({
        "event": "bump_executed",
        "timestamp": now.isoformat(),
        "bump_months": bump,
        "force": args.force,
        "before": {k: list(v) for k, v in current_flat.items()},
        "after": {k: list(v) for k, v in new_flat.items()},
        "before_dates": {
            k: f"{month_id_to_date(v[0])} – {month_id_to_date(v[1])}"
            for k, v in current_flat.items()
        },
        "after_dates": {
            k: f"{month_id_to_date(v[0])} – {month_id_to_date(v[1])}"
            for k, v in new_flat.items()
        },
        **git,
    })

    for path in updated:
        lock_entries.append({
            "event": "file_updated",
            "file": str(path.relative_to(REPO_ROOT)),
            "verified": True,
        })

    for path in override_files:
        lock_entries.append({
            "event": "file_skipped_research_override",
            "file": str(path.relative_to(REPO_ROOT)),
        })

    lock_entries.append({
        "event": "bump_completed",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files_updated": len(updated),
        "files_skipped_override": len(override_files),
        "verification_failures": 0,
    })

    lock_content = "\n".join(json.dumps(entry) for entry in lock_entries) + "\n"
    write_atomic(lock_path, lock_content)

    print(f"\n--- Lockfile: {lock_path.relative_to(REPO_ROOT)} ---")

    # --- Summary ---
    print("\n=== BUMP COMPLETE ===")
    print(f"  Files updated:            {len(updated)}")
    if override_files:
        print(f"  Research overrides:       {len(override_files)} (not bumped)")
    print("  Verification failures:    0")
    print(f"  Lockfile:                 {lock_path.relative_to(REPO_ROOT)}")
    print(f"  Git commit:               {git.get('git_commit', 'unknown')[:12]}")
    print(
        f"\n  Before: val test "
        f"{current.val_test} ({month_id_to_date(current.val_test[1])})"
    )
    print(
        f"  After:  val test "
        f"{new.val_test} ({month_id_to_date(new.val_test[1])})"
    )
    print("\nNext steps:")
    print("  1. git diff -- review the changes")
    print("  2. pytest tests/ -q -- verify nothing broke")
    print("  3. git add && git commit")


if __name__ == "__main__":
    main()
