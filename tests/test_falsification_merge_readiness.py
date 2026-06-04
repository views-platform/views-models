"""Falsification test stubs from merge-readiness audits.

Round 1 (2026-05-20, PR #56): F4 pytestmark overwrite bug.
Round 2 (2026-06-04, PR #59): F1 uncommitted work, F4 stale docstrings, F6 risk register headers.
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent


# === Round 1: PR #56 — pytestmark overwrite bug ===

class TestF4_PytestmarkOverwriteBug:
    """F4: ADR-005 markers must not be silently overwritten."""

    @pytest.mark.red
    def test_darts_reproducibility_has_green_marker_effective(self):
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


# === Round 2: PR #59 — merge readiness ===

class TestF1_UncommittedWork:
    """F1: Working-tree changes are committed — nothing lost on GitHub merge."""

    @pytest.mark.red
    def test_no_uncommitted_config_changes(self):
        import subprocess

        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        changed = [
            f
            for f in result.stdout.strip().splitlines()
            if f.endswith("config_hyperparameters.py")
            or f == "tests/test_datafactory_parity.py"
        ]
        assert not changed, f"Uncommitted changes will be lost on merge: {changed}"


class TestF4_StaleDocstrings:
    """F4: Test file docstrings do not reference removed loss functions."""

    @pytest.mark.red
    def test_no_stale_loss_references_in_parity_test(self):
        path = REPO / "tests" / "test_datafactory_parity.py"
        text = path.read_text()
        for stale in ["shrinkage", "basu_dpd", "lognormal_nll"]:
            assert stale not in text, (
                f"test_datafactory_parity.py still references '{stale}' — "
                f"datafactory trio now uses tobit"
            )


class TestF6_RiskRegisterHeader:
    """F6: Risk register header counts match actual entry statuses."""

    @pytest.mark.red
    def test_open_count_accurate(self):
        path = REPO / "reports" / "technical_risk_register.md"
        text = path.read_text()
        header_match = re.search(r"\*\*Concerns:\*\* Open (\d+)", text[:500])
        assert header_match, "Could not find Concerns Open count in header"
        header_open = int(header_match.group(1))
        d_start = text.find("### D-")
        concerns_text = text[:d_start] if d_start > 0 else text
        actual_open = len(re.findall(
            r'\| \*\*Status\*\* \| Open(?:\s*\||\s*\()', concerns_text
        ))
        assert header_open == actual_open, (
            f"Header says Open {header_open}, actual count is {actual_open}"
        )
