"""Falsification test stubs from merge-readiness audits.

Round 1 (2026-05-20, PR #56): F4 pytestmark overwrite bug.
Round 2 (2026-06-04, PR #59): F1 uncommitted work, F4 stale docstrings, F6 risk register headers.
Round 3 (2026-06-04, PR #59): F7 stale xfail marker.
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent


def files_at_risk(dirty, incoming):
    """Dirty working-tree files that an incoming merge would also rewrite.

    The seam is extracted so the rule can be pinned against injected state
    rather than only exercised through live git — a check that silently stops
    being able to fail is the C-61 failure mode.
    """
    return sorted(set(dirty) & set(incoming))


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
    """F1: No uncommitted work sits on a file an incoming merge would rewrite.

    Rewritten 2026-07-31. The original asserted ``git diff --name-only`` was
    empty, i.e. that the working tree was clean, on the stated grounds that
    "uncommitted changes will be lost on merge".

    Both halves were wrong:

    * **The premise is false.** Merging a PR on GitHub does not touch a local
      working tree; nothing is lost. A *local* ``git pull``/``merge`` can only
      refuse or clobber when incoming changes land on a file that is dirty
      here — that, and only that, is the real hazard.
    * **The trigger was perpetual.** Any developer with work in progress failed
      it, so a normal local ``pytest`` was red by construction. It also passed
      in CI (fresh clone ⇒ clean tree) and failed on a working machine — a
      verdict that depends on where it runs, the C-75 class inverted. A test
      that is always red teaches people to ignore red, which is the exact harm
      C-80 records.

    The invariant below is the one that matters and is non-perpetual: dirty
    files are fine; dirty files that *overlap the incoming diff* are not.
    """

    @pytest.mark.red
    def test_no_uncommitted_work_on_files_an_incoming_merge_would_rewrite(self):
        import subprocess

        def _git(*args):
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=REPO
            )

        dirty = set(_git("diff", "--name-only").stdout.split())
        if not dirty:
            return  # clean tree — nothing can be clobbered

        upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        if upstream.returncode != 0 or not upstream.stdout.strip():
            pytest.skip(
                "no upstream tracking ref — cannot compute the incoming diff; "
                "truthful skip rather than a guess (C-75 lesson)"
            )

        incoming = set(
            _git("diff", "--name-only", f"HEAD...{upstream.stdout.strip()}").stdout.split()
        )
        overlap = files_at_risk(dirty, incoming)
        assert not overlap, (
            "Uncommitted work sits on files the incoming merge rewrites — a local "
            f"pull will refuse or clobber: {overlap}. Commit, stash, or pull first. "
            f"({len(dirty)} file(s) dirty overall; the rest are unaffected.)"
        )

    # --- guard against the guard going vacuous (the C-61 lesson) -------------
    # A rewritten test that can no longer fail is worse than the red one it
    # replaced. These pin the decision function against injected state, so the
    # suite fails loud if the overlap rule is ever weakened to "always empty".

    @pytest.mark.red
    def test_files_at_risk_detects_overlap(self):
        assert files_at_risk(
            {"a.py", "b.py", "c.py"}, {"b.py", "d.py"}
        ) == ["b.py"], "overlap rule must flag a dirty file the merge rewrites"

    @pytest.mark.red
    def test_files_at_risk_ignores_dirty_files_the_merge_does_not_touch(self):
        assert files_at_risk({"a.py", "b.py"}, {"c.py"}) == [], (
            "dirty files outside the incoming diff are safe and must not fail the gate "
            "— this is the perpetual-trigger defect the rewrite removed"
        )


class TestF4_StaleDocstrings:
    """F4: Test file docstrings do not reference removed loss functions."""

    @pytest.mark.red
    def test_no_stale_loss_references_in_parity_test(self):
        # The datafactory-parity test was superseded by the roster-conformance test
        # (Epic #242 S3): the viewser↔datafactory trio-mirror programme is resolved,
        # so the file now pins each model to its roster family (all mse, no tobit).
        path = REPO / "tests" / "test_roster_conformance.py"
        text = path.read_text()
        # 'shrinkage'/'basu_dpd' are genuinely-removed loss functions and must not
        # reappear in the conformance test.
        for stale in ["shrinkage", "basu_dpd"]:
            assert stale not in text, (
                f"test_roster_conformance.py still references '{stale}' — "
                f"that loss function was removed"
            )


class TestF7_StaleXfailMarkers:
    """F7: xfail markers must be removed once the underlying issue is resolved."""

    @pytest.mark.red
    def test_no_stale_xfail_in_bright_starship_readiness(self):
        source = (TESTS_DIR / "test_bright_starship_readiness.py").read_text()
        for line in source.splitlines():
            if "xfail" in line and "datafactory_query" in line:
                assert False, (
                    "test_bright_starship_readiness.py still has xfail for "
                    "datafactory_query — the package is now installed; "
                    "remove the stale marker"
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
