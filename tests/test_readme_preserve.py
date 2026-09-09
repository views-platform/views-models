"""Tests for readme_preserve.py — manual-block survival through README regeneration.

Guards risk register C-78: tools/catalogs/update_readme.py rebuilds READMEs
from a scaffold; before this mechanism existed, the 2026-06-04 regeneration
(243873a) silently destroyed the hand-written synthetic_chant evaluation
semantics (C-77). Content wrapped in <!-- manual --> ... <!-- /manual -->
must survive regeneration byte-for-byte.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.catalogs.readme_preserve import (  # noqa: E402
    MANUAL_END,
    MANUAL_START,
    extract_manual_blocks,
    merge_manual_blocks,
)

UPDATE_README = REPO_ROOT / "tools" / "catalogs" / "update_readme.py"
CHANT_README = REPO_ROOT / "ensembles" / "synthetic_chant" / "README.md"


def _block(body):
    return f"{MANUAL_START}\n{body}\n{MANUAL_END}"


@pytest.mark.green
class TestManualBlockSurvival:
    """A marked manual section survives a simulated regeneration."""

    def test_single_block_survives_byte_for_byte(self):
        block = _block("## Hand-written\n\nPrecious | content – with *markdown*.")
        old = f"# Old Generated Title\n\nstale tables\n\n{block}\n"
        regenerated = "# New Generated Title\n\nfresh tables\n"
        merged = merge_manual_blocks(regenerated, extract_manual_blocks(old))
        assert block in merged
        assert "fresh tables" in merged

    def test_multiple_blocks_preserved_in_order(self):
        first = _block("## First")
        second = _block("## Second")
        old = f"intro\n{first}\nmiddle\n{second}\nend\n"
        merged = merge_manual_blocks("generated\n", extract_manual_blocks(old))
        assert merged.index(first) < merged.index(second)

    def test_no_markers_is_a_no_op(self):
        generated = "# Generated\n\ncontent\n"
        assert merge_manual_blocks(generated, extract_manual_blocks("old, no markers")) == generated

    def test_merge_is_a_stable_fixed_point(self):
        """Blocks extracted from merged output reproduce themselves on re-merge."""
        block = _block("## Survives forever")
        merged_once = merge_manual_blocks("gen v1\n", extract_manual_blocks(f"old\n{block}\n"))
        merged_twice = merge_manual_blocks("gen v2\n", extract_manual_blocks(merged_once))
        assert extract_manual_blocks(merged_twice) == [block]

    def test_unterminated_block_fails_loud(self):
        """A dangling start marker must crash the regeneration, not silently
        drop the content it was meant to protect (the C-78 failure mode)."""
        with pytest.raises(ValueError, match="unterminated"):
            extract_manual_blocks(f"{MANUAL_START}\nno end marker\n")

    def test_terminated_plus_dangling_still_fails_loud(self):
        block = _block("## kept")
        with pytest.raises(ValueError, match="unterminated"):
            extract_manual_blocks(f"{block}\n{MANUAL_START}\ndangling\n")


@pytest.mark.beige
class TestGeneratorWiring:
    """update_readme.py must actually use the preserve mechanism (static check —
    importing the script would execute the full regeneration)."""

    def test_update_readme_imports_and_calls_preserve(self):
        source = UPDATE_README.read_text()
        assert "readme_preserve" in source, "update_readme.py does not import readme_preserve"
        # Once per loop: models and ensembles.
        assert source.count("merge_manual_blocks(") >= 2, (
            "update_readme.py must merge manual blocks in BOTH the models and ensembles loops"
        )

    def test_created_on_capture_runs_on_stripped_text(self):
        """C-82 guard: the duplication tests replicate the script's logic, so
        they cannot detect a revert in the script itself — this pins that both
        Created-on captures actually search strip_manual_blocks(...) output."""
        source = UPDATE_README.read_text()
        assert source.count("strip_manual_blocks(old_readme_content)") >= 2, (
            "both '## Created on' captures in update_readme.py must search "
            "strip_manual_blocks(old_readme_content), or end-of-file manual "
            "blocks get swallowed into the created section and duplicate (C-82)"
        )


@pytest.mark.beige
class TestSyntheticChantRestoration:
    """The C-77 documentation is restored inside a manual block (so the next
    regeneration cannot wipe it again)."""

    def test_chant_readme_has_manual_block_with_semantics(self):
        blocks = extract_manual_blocks(CHANT_README.read_text())
        assert blocks, "synthetic_chant README has no <!-- manual --> block"
        joined = "\n".join(blocks)
        for needle in ("vertical_stripe", "actuals", "cross-pattern disagreement"):
            assert needle in joined, (
                f"synthetic_chant manual block lost the evaluation-semantics docs ({needle!r} missing)"
            )
