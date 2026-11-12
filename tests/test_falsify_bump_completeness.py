"""Verification tests for partition bump completeness.

Confirms that resolved findings remain fixed:
- ADR-011 no longer references deleted scripts
- No PARTITION_OVERRIDE mechanism exists (ingester3 dependency removed)
"""
from pathlib import Path

import pytest


pytestmark = pytest.mark.green
REPO_ROOT = Path(__file__).resolve().parent.parent


class TestADR011NoStaleReferences:
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
            f"in {len(deleted_refs)} place(s).\n"
            + "\n".join(deleted_refs)
        )


class TestNoOverrideMechanism:
    def test_no_partition_override_in_config_files(self):
        """All config_partitions.py files should use _current_month_id(),
        not ingester3. The PARTITION_OVERRIDE mechanism is retired."""
        for pattern in [
            "models/*/configs/config_partitions.py",
            "ensembles/*/configs/config_partitions.py",
        ]:
            for f in REPO_ROOT.glob(pattern):
                source = f.read_text()
                assert "PARTITION_OVERRIDE" not in source, (
                    f"{f.relative_to(REPO_ROOT)} still has PARTITION_OVERRIDE marker"
                )
                assert "ingester3" not in source, (
                    f"{f.relative_to(REPO_ROOT)} still imports ingester3"
                )
