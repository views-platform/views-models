"""File operations for config_partitions.py files.

Handles discovery, parsing, rewriting, and verification of partition
config files. This is the single place that knows the file format —
shared by the bump CLI and the test suite.
"""
import os
import re
import tempfile
from pathlib import Path

OVERRIDE_MARKER = "# PARTITION_OVERRIDE:"

_SEARCH_SUBDIRS = ["models", "ensembles", "extractors", "postprocessors"]

_FIXTURE_NAMES = {
    "fake_model", "test_model", "test_ensemble",
    "diagonal_dream", "horizontal_dream", "lucid_dream",
    "vertical_dream", "vivid_dream", "waking_dream",
    "synthetic_chant", "synthetic_choir", "synthetic_chorus",
}

_Q = r"""[\"']"""


def discover_entity_dirs(repo_root: Path) -> list[Path]:
    """Find all real entity directories, excluding fixtures.

    Models and ensembles require main.py (functional entities).
    Extractors and postprocessors require configs/config_partitions.py.
    """
    dirs = []
    for subdir in ("models", "ensembles"):
        base = repo_root / subdir
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if (
                d.is_dir()
                and d.name not in _FIXTURE_NAMES
                and (d / "main.py").exists()
            ):
                dirs.append(d)
    for subdir in ("extractors", "postprocessors"):
        base = repo_root / subdir
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if (
                d.is_dir()
                and d.name not in _FIXTURE_NAMES
                and (d / "configs" / "config_partitions.py").exists()
            ):
                dirs.append(d)
    return dirs


def discover_partition_files(repo_root: Path) -> list[Path]:
    """Find all config_partitions.py files under known directories."""
    files = []
    for subdir in _SEARCH_SUBDIRS:
        base = repo_root / subdir
        if not base.exists():
            continue
        for path in sorted(base.glob("*/configs/config_partitions.py")):
            files.append(path)
    return files


def has_override(source: str) -> bool:
    return OVERRIDE_MARKER in source


def has_partition_override(source: str) -> bool:
    """Check if a config_partitions.py declares PARTITION_OVERRIDE = True.

    This is a programmatic flag — not a comment. Only a real assignment
    of True counts. False, commented-out, or absent means no override.
    """
    match = re.search(
        r"^PARTITION_OVERRIDE\s*=\s*(True|False)",
        source,
        re.MULTILINE,
    )
    return match is not None and match.group(1) == "True"


def extract_values(source: str) -> dict[str, tuple[int, int]] | None:
    """Extract calibration/validation train/test tuples from source text.

    Accepts both single-quoted and double-quoted Python dict keys.
    Returns None if the expected structure is not found.
    """
    result = {}
    for section in ("calibration", "validation"):
        section_match = re.search(
            rf"{_Q}{section}{_Q}:\s*\{{(.*?)\}}",
            source,
            re.DOTALL,
        )
        if not section_match:
            return None
        block = section_match.group(1)
        for key in ("train", "test"):
            m = re.search(rf"{_Q}{key}{_Q}:\s*\((\d+),\s*(\d+)\)", block)
            if not m:
                return None
            result[f"{section}_{key}"] = (int(m.group(1)), int(m.group(2)))
    return result


def rewrite_values(source: str, new_values: dict[str, tuple[int, int]]) -> str:
    """Replace calibration/validation tuples in source. Never touches forecasting."""
    new_source = source
    for section in ("calibration", "validation"):
        section_pattern = rf"({_Q}{section}{_Q}:\s*\{{)(.*?)(\}})"
        section_match = re.search(section_pattern, new_source, re.DOTALL)
        if not section_match:
            raise ValueError(f"Could not find '{section}' section in source")
        block = section_match.group(2)
        new_block = block
        for key in ("train", "test"):
            start, end = new_values[f"{section}_{key}"]
            key_pattern = rf"({_Q}{key}{_Q}:\s*\()\d+,\s*\d+(\))"
            new_block = re.sub(
                key_pattern, rf"\g<1>{start}, {end}\2", new_block
            )
        new_source = (
            new_source[: section_match.start(2)]
            + new_block
            + new_source[section_match.end(2) :]
        )
    return new_source


def write_atomic(path: Path, content: str) -> None:
    """Write content to path atomically via tempfile + os.replace."""
    dir_name = path.parent
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, str(path))


def verify_file(path: Path, expected: dict[str, tuple[int, int]]) -> list[str]:
    """Re-read a file after writing and verify values match expected."""
    errors = []
    try:
        source = path.read_text()
    except OSError as e:
        return [f"Could not re-read {path}: {e}"]

    actual = extract_values(source)
    if actual is None:
        return [f"Could not parse partition values from {path} after write"]

    for key, expected_val in expected.items():
        actual_val = actual.get(key)
        if actual_val != expected_val:
            errors.append(
                f"{path}: {key} expected {expected_val}, got {actual_val}"
            )
    return errors
