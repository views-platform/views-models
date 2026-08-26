# Class Intent Contract: tools.partitions.fileops

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-06-07
**Related ADRs:** ADR-011, ADR-002

---

## 1. Purpose

> `fileops` is the single module that knows the `config_partitions.py` file format. It handles discovery, parsing, rewriting, and verification of partition config files. Shared by the bump CLI and the test suite — one parser, one format, one place.

Located in: `tools/partitions/fileops.py`

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** validate partition values (that's `domain.py`'s job)
- Does **not** decide whether to bump or skip a file (that's `bump.py`'s job)
- Does **not** handle CLI arguments, user interaction, or lockfile writing
- Does **not** modify the forecasting section of any file

---

## 3. Responsibilities and Guarantees

- `discover_partition_files(repo_root)` — finds all `config_partitions.py` files under models/, ensembles/, extractors/, postprocessors/
- `discover_entity_dirs(repo_root)` — finds all real entity directories (main.py for models/ensembles, config_partitions.py for extractors/postprocessors), excluding fixtures from `meta/fixtures.json`
- `extract_values(source)` — parses calibration/validation train/test tuples from Python source text. Accepts both single and double quotes. Strips comments before matching. Returns `None` if unparseable.
- `rewrite_values(source, new_values)` — replaces calibration/validation tuples in source text. Anchors to `return {` to avoid matching comments. Never touches the forecasting section. Raises `ValueError` if the expected structure is not found.
- `write_atomic(path, content)` — writes via tempfile + `os.replace`. **Preserves the destination file's permission bits when overwriting an existing file; applies a umask-respecting default (typically `0o644`) for a new file.** Cleans up temp file on failure.
- `verify_file(path, expected)` — re-reads a file after writing and compares against expected values. Returns list of error strings.
- `has_partition_override(source)` — checks for `PARTITION_OVERRIDE = True` as a real Python variable (not a comment)
- Fixture names loaded from `meta/fixtures.json` at module import time

---

## 4. Inputs and Assumptions

- `extract_values()` assumes the file contains `"calibration": { "train": (N, N), "test": (N, N) }` and similarly for `"validation"` — with either single or double quotes
- `rewrite_values()` assumes the file contains exactly one `return {` statement
- `discover_entity_dirs()` assumes models/ensembles have `main.py` as a marker of functional entities
- `meta/fixtures.json` must exist and contain a JSON array of strings

---

## 5. Outputs and Side Effects

- `extract_values()` — pure, no side effects, returns `dict[str, tuple[int, int]] | None`
- `rewrite_values()` — pure, returns modified source string
- `write_atomic()` — writes to filesystem. Atomic: either the file is fully written or nothing changes. Permission bits of an existing target are preserved across the replace (a new target gets the umask default, not `NamedTemporaryFile`'s `0o600`).
- `verify_file()` — reads from filesystem. No writes.
- `discover_*()` — reads filesystem (directory listing). No writes.

---

## 6. Failure Modes and Loudness

- `extract_values()` returns `None` silently if the file is unparseable — caller must check
- `rewrite_values()` raises `ValueError` loudly if `return {` or a section is not found
- `write_atomic()` raises `OSError` if write fails — temp file is cleaned up before re-raising
- `verify_file()` returns error strings for any mismatch — never raises
- If `meta/fixtures.json` is missing, the module fails to import with `FileNotFoundError`

---

## 7. Boundaries and Interactions

- Depends on: `json`, `os`, `re`, `tempfile`, `pathlib` (stdlib only)
- Loaded by: `tools/partitions/bump.py` (all functions), `tests/test_config_partitions.py` (`extract_values`, `has_partition_override`), `tests/test_bump_partitions.py` (multiple functions), `tests/test_tooling_scripts.py` (multiple functions)
- Does NOT interact with: `domain.py` (no import between them — independent modules)

---

## 8. Examples of Correct Usage

```python
from tools.partitions.fileops import extract_values, rewrite_values

source = Path("models/counting_stars/configs/config_partitions.py").read_text()
values = extract_values(source)  # {'calibration_train': (121, 444), ...}

new_vals = {k: (v[0], v[1] + 12) for k, v in values.items()}
new_source = rewrite_values(source, new_vals)
```

---

## 9. Examples of Incorrect Usage

```python
# Wrong: assuming extract_values always succeeds
values = extract_values(source)
values["calibration_train"]  # KeyError if extract_values returned None

# Wrong: calling rewrite_values on a file without return {
rewrite_values("PARTITION_OVERRIDE = True\n", new_vals)  # ValueError
```

---

## 10. Test Alignment

- `tests/test_bump_partitions.py::TestExtractValues` — double quote, single quote, all repo variants
- `tests/test_bump_partitions.py::TestRewriteRoundTrip` — round-trip for both quote styles, forecasting untouched
- `tests/test_bump_partitions.py::TestDiscoverEntityDirs` — fixture exclusion, main.py detection, coverage check
- `tests/test_bump_partitions.py::TestPartitionOverrideFlag` — True, False, absent, comment
- `tests/test_bump_partitions.py::TestFixtureSetConsistency` — canonical fixture set matches all consumers
- `tests/test_falsify_bump_edge_cases.py::TestP4` — comment does not confuse parser
- `tests/test_falsify_bump_edge_cases.py::TestP5` — write_atomic cleanup on failure
- `tests/test_config_partitions.py` — uses `extract_values` as shared parser across 100 files

---

## 11. Evolution Notes

- The regex parser is the most fragile component. If `config_partitions.py` format changes significantly (e.g., YAML, TOML), this module must be rewritten.
- `_strip_comments()` is a simple line-level filter. It would fail on inline comments after code (`x = 1  # comment with "calibration"`), but this pattern doesn't exist in any config_partitions.py file.

---

## End of Contract

This document defines the **intended meaning** of `tools.partitions.fileops`.

Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
