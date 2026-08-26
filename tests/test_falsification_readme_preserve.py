"""Falsification finding — manual-block duplication via the Created-section capture.

Found by a falsification audit of PR #133 (2026-06-12), probe P1; register C-82.
RESOLVED same day: update_readme.py now runs its '## Created on' tail-capture
on strip_manual_blocks(old) so blocks at end-of-file are never swallowed into
the captured created-section. These tests pin the fixed behavior.
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.catalogs.readme_preserve import (  # noqa: E402
    extract_manual_blocks,
    merge_manual_blocks,
    strip_manual_blocks,
)

OLD = (
    "# Old\nbody\n"
    "## Created on 2026-01-01\ncreated text\n\n"
    "<!-- manual -->\n## Precious docs\n<!-- /manual -->\n"
)
SCAFFOLD = "# New\nfresh body\n{{CREATED_SECTION}}\n"


def _regenerate(old, scaffold):
    """Replicate update_readme.py's post-C-82 logic verbatim (incl. the
    match-is-None branch: after one regeneration the heading reads
    '## Model Created on', which the regex no longer matches — C-83)."""
    match = re.search(r"(## Created on.*)", strip_manual_blocks(old), re.DOTALL)
    if match is None:
        created = ""
    else:
        created = match.group(1).strip()
        created = created[:2] + " Model" + created[2:]
    content = scaffold.replace("{{CREATED_SECTION}}", created)
    return merge_manual_blocks(content, extract_manual_blocks(old))


@pytest.mark.green
class TestNoDuplicationWithCreatedSection:
    def test_manual_block_emitted_exactly_once(self):
        merged = _regenerate(OLD, SCAFFOLD)
        assert merged.count("## Precious docs") == 1, (
            "manual block emitted twice: once inside the captured "
            "Created-on section, once via merge_manual_blocks (C-82)"
        )

    def test_created_section_still_preserved(self):
        merged = _regenerate(OLD, SCAFFOLD)
        assert "## Model Created on 2026-01-01" in merged
        assert "created text" in merged

    def test_stable_under_repeated_regeneration(self):
        # Manual-block stability is the C-82 guarantee. (The created-section
        # does NOT survive a second regeneration — pre-existing C-83, out of
        # scope here.)
        once = _regenerate(OLD, SCAFFOLD)
        twice = _regenerate(once, SCAFFOLD)
        assert twice.count("## Precious docs") == 1

    def test_strip_is_inverse_of_block_presence(self):
        assert "Precious" not in strip_manual_blocks(OLD)
        assert strip_manual_blocks("no markers here") == "no markers here"
