"""Falsification tests for the claim: 'partition bump effort is done.'

Key findings:
- ADR-011 references scripts/update_partitions.py which has been deleted
- Override files become silently stale after a bump — no test fails
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestP1_ADR011ReferencesDeletedScript:
    """ADR-011 is the governance document for partition semantics.
    It tells users to run scripts/update_partitions.py, which was deleted."""

    def test_adr_011_does_not_reference_deleted_scripts(self):
        adr = REPO_ROOT / "docs" / "ADRs" / "011_partition_semantics.md"
        if not adr.exists():
            pytest.skip("ADR-011 not found")
        content = adr.read_text()
        deleted_refs = []
        for i, line in enumerate(content.splitlines(), 1):
            if "scripts/update_partitions" in line:
                deleted_refs.append(f"line {i}: {line.strip()}")
        assert len(deleted_refs) == 0, (
            f"ADR-011 references the deleted script 'scripts/update_partitions.py' "
            f"in {len(deleted_refs)} place(s). Update to reference "
            f"'tools/partitions/bump.py' instead.\n"
            + "\n".join(deleted_refs)
        )


class TestP5_OverrideFilesStalenessDetected:
    """After a bump, override files have old partition values while all
    other models have new values. This must produce a visible signal."""

    def test_bump_tool_reports_stale_overrides(self):
        """The bump summary or lockfile should explicitly list override
        files that still use pre-bump values, not just 'skipped'."""
        source = (REPO_ROOT / "tools" / "partitions" / "bump.py").read_text()
        reports_stale = (
            "stale" in source.lower()
            or "old values" in source.lower()
            or "manual update" in source.lower()
        )
        assert reports_stale, (
            "bump.py skips override files but doesn't warn that they now use "
            "stale (pre-bump) partition values. After a bump, 8 HydraNet models "
            "would silently train/evaluate on different partitions than the rest "
            "of the system. Add an explicit warning listing override files that "
            "need manual review."
        )
