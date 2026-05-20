"""Falsification test stubs from audit of merge-readiness claim (PR #56).

Generated: 2026-05-20
Source: /falsify audit — "we are ready to accept and merge the PR"
Findings: F4 (soft falsification)

F4: pytestmark overwrite bug — test_darts_reproducibility.py assigns
pytestmark twice (green on line 13, skipif on line 34). The second
assignment overwrites the first, leaving 32 tests uncategorized by
ADR-005.  test_bright_starship_readiness.py has a skipif mark but
no ADR-005 category at all (2 tests uncategorized).
"""
import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


class TestF4_PytestmarkOverwriteBug:
    """F4: ADR-005 markers must not be silently overwritten.

    When a test file needs both a category marker and a skipif condition,
    pytestmark must be a list, not two separate assignments where the
    second overwrites the first.
    """

    @pytest.mark.red
    def test_darts_reproducibility_has_green_marker_effective(self):
        """test_darts_reproducibility.py must have an effective green marker."""
        source = (TESTS_DIR / "test_darts_reproducibility.py").read_text()
        tree = ast.parse(source)

        pytestmark_assignments = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "pytestmark":
                        pytestmark_assignments.append(node)

        assert len(pytestmark_assignments) <= 1 or isinstance(
            pytestmark_assignments[-1].value, (ast.List, ast.Tuple)
        ), (
            f"test_darts_reproducibility.py has {len(pytestmark_assignments)} "
            f"pytestmark assignments — the last one overwrites earlier markers. "
            f"Use a list: pytestmark = [pytest.mark.green, pytest.mark.skipif(...)]"
        )

    @pytest.mark.red
    def test_bright_starship_has_adr005_marker(self):
        """test_bright_starship_readiness.py must have an ADR-005 category marker."""
        source = (TESTS_DIR / "test_bright_starship_readiness.py").read_text()
        has_adr005 = any(
            marker in source
            for marker in ["pytest.mark.red", "pytest.mark.beige", "pytest.mark.green"]
        )
        assert has_adr005, (
            "test_bright_starship_readiness.py has a skipif marker but no "
            "ADR-005 category (red/beige/green). Add one."
        )

    @pytest.mark.red
    def test_no_pytestmark_overwrites_in_any_test_file(self):
        """No test file should have multiple pytestmark assignments that
        silently overwrite each other."""
        violations = []
        for f in TESTS_DIR.glob("test_*.py"):
            source = f.read_text()
            tree = ast.parse(source)

            assignments = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "pytestmark":
                            assignments.append(node)

            if len(assignments) > 1:
                last = assignments[-1]
                if not isinstance(last.value, (ast.List, ast.Tuple)):
                    violations.append(f.name)

        assert violations == [], (
            f"These test files have multiple pytestmark assignments where "
            f"the last one overwrites earlier markers: {violations}"
        )
