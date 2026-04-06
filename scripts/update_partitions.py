"""Migration tool: update all config_partitions.py files to canonical values.

Reads canonical partition boundaries from meta/partitions.json and rewrites
all config_partitions.py files to match. Files with a PARTITION_OVERRIDE
comment are skipped with a warning.

Usage:
    python scripts/update_partitions.py [--dry-run]

See ADR-011 for partition semantics and override mechanism.
"""
import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_FILE = REPO_ROOT / "meta" / "partitions.json"
OVERRIDE_MARKER = "# PARTITION_OVERRIDE:"

SEARCH_DIRS = [
    REPO_ROOT / "models",
    REPO_ROOT / "ensembles",
    REPO_ROOT / "extractors",
    REPO_ROOT / "postprocessors",
]


def discover_partition_files() -> list[Path]:
    """Find all config_partitions.py files under known directories."""
    files = []
    for base_dir in SEARCH_DIRS:
        if not base_dir.exists():
            continue
        for config_file in sorted(base_dir.glob("*/configs/config_partitions.py")):
            files.append(config_file)
    return files


def load_canonical() -> dict:
    """Load canonical values from meta/partitions.json."""
    with open(PARTITIONS_FILE) as f:
        return json.load(f)


def update_file(path: Path, canonical: dict, dry_run: bool) -> str:
    """Update a single config_partitions.py file.

    Returns: 'updated', 'skipped_override', 'already_current', or 'error'.
    """
    source = path.read_text()

    if OVERRIDE_MARKER in source:
        return "skipped_override"

    new_source = source

    replacements = {
        "calibration": {
            "train": tuple(canonical["calibration"]["train"]),
            "test": tuple(canonical["calibration"]["test"]),
        },
        "validation": {
            "train": tuple(canonical["validation"]["train"]),
            "test": tuple(canonical["validation"]["test"]),
        },
    }

    for section, keys in replacements.items():
        # Match the entire section block to scope replacements
        section_pattern = rf'("{section}":\s*\{{)(.*?)(\}})'
        section_match = re.search(section_pattern, new_source, re.DOTALL)
        if not section_match:
            continue
        block = section_match.group(2)
        new_block = block
        for key, (start, end) in keys.items():
            key_pattern = rf'("{key}":\s*\()\d+,\s*\d+(\))'
            new_block = re.sub(key_pattern, rf"\g<1>{start}, {end}\2", new_block)
        new_source = (
            new_source[:section_match.start(2)]
            + new_block
            + new_source[section_match.end(2):]
        )

    offset = abs(canonical["forecasting_offset"])
    new_source = re.sub(
        r'(ViewsMonth\.now\(\)\.id\s*-\s*)\d+',
        rf'\g<1>{offset}',
        new_source,
    )

    if new_source == source:
        return "already_current"

    if not dry_run:
        path.write_text(new_source)

    return "updated"


def main():
    parser = argparse.ArgumentParser(
        description="Update all config_partitions.py to canonical values."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing files."
    )
    args = parser.parse_args()

    canonical = load_canonical()
    files = discover_partition_files()

    counts = {"updated": 0, "skipped_override": 0, "already_current": 0, "error": 0}

    for path in files:
        rel_path = path.relative_to(REPO_ROOT)
        result = update_file(path, canonical, args.dry_run)
        counts[result] += 1

        if result == "updated":
            prefix = "[DRY RUN] Would update" if args.dry_run else "Updated"
            print(f"  {prefix}: {rel_path}")
        elif result == "skipped_override":
            print(f"  WARNING: Skipped (PARTITION_OVERRIDE): {rel_path}")
        elif result == "error":
            print(f"  ERROR: {rel_path}")

    print()
    print(f"Summary: {len(files)} files scanned")
    print(f"  {counts['updated']} {'would be updated' if args.dry_run else 'updated'}")
    print(f"  {counts['skipped_override']} skipped (declared overrides)")
    print(f"  {counts['already_current']} already current")

    if counts["updated"] > 0 and not args.dry_run:
        print()
        print("Verify: pytest tests/test_config_partitions.py -v")


if __name__ == "__main__":
    main()
