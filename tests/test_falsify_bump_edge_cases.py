"""Falsification tests for partition bump tool edge cases.

Claim: 'the tool handles everything that isn't the happy path.'

Key findings:
- Missing/corrupt partitions.json produces raw traceback (no user message)
- Regex matches first occurrence — a comment can corrupt the real dict
- write_atomic leaves orphaned temp files on os.replace failure
"""
from pathlib import Path
import pytest


pytestmark = pytest.mark.green
REPO_ROOT = Path(__file__).resolve().parent.parent


class TestP2_MissingPartitionsJson:
    """_load_canonical() has no error handling — missing or corrupt
    meta/partitions.json produces a raw Python traceback."""

    def test_load_canonical_handles_missing_file(self):
        """Should produce a clear error message, not FileNotFoundError."""
        source = (REPO_ROOT / "tools" / "partitions" / "bump.py").read_text()
        load_func_start = source.index("def _load_canonical")
        next_func = source.index("\ndef ", load_func_start + 1)
        func_body = source[load_func_start:next_func]
        assert "try" in func_body and "except" in func_body, (
            "_load_canonical() has no error handling. A missing or corrupt "
            "meta/partitions.json produces a raw Python traceback instead of "
            "a clear error message."
        )


class TestP4_RegexMatchesCommentNotDict:
    """extract_values matches the FIRST 'calibration' in the file.
    If that's in a comment, it reads the wrong values."""

    def test_comment_does_not_confuse_parser(self):
        from tools.partitions.fileops import extract_values

        source_with_comment = '''
# Legacy: "calibration": {"train": (100, 200), "test": (201, 250)}

def generate(steps=36):
    return {
        "calibration": {
            "train": (121, 444),
            "test": (445, 492),
        },
        "validation": {
            "train": (121, 492),
            "test": (493, 540),
        },
    }
'''
        result = extract_values(source_with_comment)
        assert result is not None, "Failed to parse file with comment"
        assert result["calibration_train"] == (121, 444), (
            f"Regex matched the comment value {result['calibration_train']} "
            f"instead of the real dict value (121, 444)"
        )


class TestP5_WriteAtomicCleansUpOnFailure:
    """write_atomic creates a temp file with delete=False but has no
    cleanup if os.replace raises."""

    def test_write_atomic_has_cleanup(self):
        import inspect
        from tools.partitions.fileops import write_atomic

        source = inspect.getsource(write_atomic)
        has_cleanup = (
            "finally" in source
            or ("try" in source and "unlink" in source)
            or ("try" in source and "remove" in source)
        )
        assert has_cleanup, (
            "write_atomic() has no cleanup for the temp file if os.replace() "
            "fails. A permission error or disk-full condition leaves orphaned "
            ".tmp files in config directories."
        )
