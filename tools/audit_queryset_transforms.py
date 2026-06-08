"""Audit all config_queryset.py files for active log transformations.

Parses each queryset config using AST-safe line analysis to identify
which models apply .transform.ops.ln() to which columns, and which
have it commented out.

Usage:
    python tools/audit_queryset_transforms.py
    python tools/audit_queryset_transforms.py --json
"""
import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ["models", "ensembles", "extractors", "postprocessors"]


def parse_queryset_file(path: Path) -> dict:
    """Parse a config_queryset.py for Column definitions and their transforms.

    Returns a dict with:
        - columns: list of dicts with name, from_column, from_loa, transforms, commented_transforms
        - active_ln_count: number of active .transform.ops.ln() calls
        - commented_ln_count: number of commented-out .transform.ops.ln() calls
    """
    source = path.read_text()
    lines = source.splitlines()

    columns = []
    current_column = None

    for line in lines:
        stripped = line.strip()

        # Detect Column() definitions
        col_match = re.search(
            r"""Column\(\s*[\"']([^\"']+)[\"']""", stripped
        )
        if col_match:
            if stripped.startswith("#"):
                continue

            current_column = {
                "name": col_match.group(1),
                "from_column": None,
                "from_loa": None,
                "active_transforms": [],
                "commented_transforms": [],
            }
            columns.append(current_column)

            from_col = re.search(r"from_column\s*=\s*[\"']([^\"']+)[\"']", stripped)
            from_loa = re.search(r"from_loa\s*=\s*[\"']([^\"']+)[\"']", stripped)
            if from_col:
                current_column["from_column"] = from_col.group(1)
            if from_loa:
                current_column["from_loa"] = from_loa.group(1)

        # Detect transforms on current column
        if current_column and ".transform." in stripped:
            transform_matches = re.findall(r"\.transform\.(\w+(?:\.\w+)*(?:\([^)]*\))?)", stripped)
            is_comment_line = stripped.startswith("#")

            for t in transform_matches:
                if is_comment_line:
                    current_column["commented_transforms"].append(t)
                else:
                    current_column["active_transforms"].append(t)

    active_ln = sum(
        1 for c in columns
        for t in c["active_transforms"]
        if "ops.ln()" in t
    )
    commented_ln = sum(
        1 for c in columns
        for t in c["commented_transforms"]
        if "ops.ln()" in t
    )

    return {
        "columns": columns,
        "active_ln_count": active_ln,
        "commented_ln_count": commented_ln,
    }


def discover_queryset_files() -> list[Path]:
    files = []
    for subdir in SEARCH_DIRS:
        base = REPO_ROOT / subdir
        if not base.exists():
            continue
        for path in sorted(base.glob("*/configs/config_queryset.py")):
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Audit config_queryset.py files for log transformations."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable table."
    )
    args = parser.parse_args()

    files = discover_queryset_files()
    results = []

    for path in files:
        model_name = path.parent.parent.name
        model_type = path.parent.parent.parent.name
        try:
            parsed = parse_queryset_file(path)
        except Exception as e:
            results.append({
                "model": model_name,
                "type": model_type,
                "error": str(e),
            })
            continue

        ln_columns = []
        commented_ln_columns = []
        for col in parsed["columns"]:
            has_active_ln = any("ops.ln()" in t for t in col["active_transforms"])
            has_commented_ln = any("ops.ln()" in t for t in col["commented_transforms"])
            if has_active_ln:
                ln_columns.append(col["name"])
            if has_commented_ln:
                commented_ln_columns.append(col["name"])

        results.append({
            "model": model_name,
            "type": model_type,
            "total_columns": len(parsed["columns"]),
            "active_ln_count": parsed["active_ln_count"],
            "commented_ln_count": parsed["commented_ln_count"],
            "ln_columns": ln_columns,
            "commented_ln_columns": commented_ln_columns,
        })

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Human-readable output
    models_with_active = [r for r in results if r.get("active_ln_count", 0) > 0]
    models_with_commented = [r for r in results if r.get("commented_ln_count", 0) > 0 and r.get("active_ln_count", 0) == 0]
    models_clean = [r for r in results if r.get("active_ln_count", 0) == 0 and r.get("commented_ln_count", 0) == 0]

    print("=== Queryset Log Transform Audit ===")
    print(f"  Total queryset files: {len(files)}")
    print(f"  Models with ACTIVE .transform.ops.ln(): {len(models_with_active)}")
    print(f"  Models with commented-out ln() only: {len(models_with_commented)}")
    print(f"  Models with no ln() references: {len(models_clean)}")
    print()

    if models_with_active:
        print("=== Models with ACTIVE log transforms ===")
        print(f"{'Model':<30} {'Active':<8} {'Commented':<10} {'Columns with active ln()'}")
        print("-" * 90)
        for r in sorted(models_with_active, key=lambda x: -x["active_ln_count"]):
            cols = ", ".join(r["ln_columns"][:5])
            if len(r["ln_columns"]) > 5:
                cols += f" (+{len(r['ln_columns']) - 5} more)"
            print(f"{r['model']:<30} {r['active_ln_count']:<8} {r['commented_ln_count']:<10} {cols}")
        print()

    if models_with_commented:
        print("=== Models with COMMENTED-OUT ln() only (no active) ===")
        for r in sorted(models_with_commented, key=lambda x: x["model"]):
            cols = ", ".join(r["commented_ln_columns"][:3])
            print(f"  {r['model']}: {r['commented_ln_count']} commented on {cols}")
        print()

    # Summary of which column NAMES get log-transformed
    all_ln_cols = {}
    for r in models_with_active:
        for col in r["ln_columns"]:
            all_ln_cols.setdefault(col, []).append(r["model"])

    if all_ln_cols:
        print("=== Column names receiving active ln() transforms ===")
        for col_name in sorted(all_ln_cols.keys()):
            models = all_ln_cols[col_name]
            print(f"  {col_name}: {len(models)} model(s)")


if __name__ == "__main__":
    main()
