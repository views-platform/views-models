"""File operations for config_partitions.py files.

Handles discovery, parsing, rewriting, and verification of partition
config files. This is the single place that knows the file format —
shared by the bump CLI and the test suite.
"""
import json
import os
import re
import stat
import tempfile
from pathlib import Path

_SEARCH_SUBDIRS = ["models", "ensembles", "extractors", "postprocessors"]

_FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "meta" / "fixtures.json"
with open(_FIXTURES_PATH) as _f:
    _FIXTURE_NAMES: set[str] = set(json.load(_f))

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


def _strip_comments(source: str) -> str:
    """Remove comment-only lines from Python source."""
    return "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )


def extract_values(source: str) -> dict[str, tuple[int, int]] | None:
    """Extract calibration/validation train/test tuples from source text.

    Accepts both single-quoted and double-quoted Python dict keys.
    Ignores comments to avoid matching values in documentation.
    Returns None if the expected structure is not found.
    """
    clean = _strip_comments(source)
    result = {}
    for section in ("calibration", "validation"):
        section_match = re.search(
            rf"{_Q}{section}{_Q}:\s*\{{(.*?)\}}",
            clean,
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
    """Replace calibration/validation tuples in source. Never touches forecasting.

    Anchors to the ``return {`` statement to avoid matching values
    that appear in comments or docstrings.
    """
    return_pos = source.find("return {")
    if return_pos == -1:
        raise ValueError("Could not find 'return {' in source")
    prefix = source[:return_pos]
    body = source[return_pos:]
    new_body = body
    for section in ("calibration", "validation"):
        section_pattern = rf"({_Q}{section}{_Q}:\s*\{{)(.*?)(\}})"
        section_match = re.search(section_pattern, new_body, re.DOTALL)
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
        new_body = (
            new_body[: section_match.start(2)]
            + new_block
            + new_body[section_match.end(2) :]
        )
    return prefix + new_body


def write_atomic(path: Path, content: str) -> None:
    """Write content to path atomically via tempfile + os.replace.

    Preserves the destination file's permission bits when overwriting an
    existing file; for a new file, applies the umask-respecting default
    (typically 0o644). Without this, the replace would leave the file with
    NamedTemporaryFile's restrictive 0o600 and drop any execute bit — which
    is exactly what silently changed 101 config modes during a partition bump.
    Cleans up the temp file if the chmod or replace fails.
    """
    dir_name = path.parent
    try:
        dest_mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        dest_mode = None
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        if dest_mode is not None:
            os.chmod(tmp_path, dest_mode)
        else:
            current_umask = os.umask(0)
            os.umask(current_umask)
            os.chmod(tmp_path, 0o666 & ~current_umask)
        os.replace(tmp_path, str(path))
    except OSError:
        os.unlink(tmp_path)
        raise


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
