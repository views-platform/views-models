"""Failing stubs from the 2026-07-19 falsification audit of the CLAIM
"tools.liveness is 100% test covered, green/beige/red all around".

Verdict: FALSIFIED. These meta-tests encode the gaps; they FAIL until the
taxonomy work lands. ADR-005 (pyproject.toml markers): green = correctness/
functional, beige = convention/structural compliance, red = adversarial/
error-path. NOTE: red does NOT mean "live network probe" — the liveness
suites currently misuse it that way.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige  # these ARE structural-compliance tests

_TESTS_DIR = Path(__file__).resolve().parent
_SUITE_FILES = sorted(
    p for p in _TESTS_DIR.glob("test_liveness_*.py")
    if p.name not in {"test_liveness_taxonomy.py"}
)

# Error-path / adversarial tests are recognizable by what they exercise.
_ERROR_PATH_NAME = re.compile(
    r"def (test_[a-z0-9_]*(unreachable|malformed|rejects|crash|breaks|"
    r"error|failure|stale[a-z0-9_]*budget)[a-z0-9_]*)\("
)


def test_every_liveness_suite_has_beige_structural_tests():
    """ADR-005 beige = convention/structural compliance. The claim says
    'beige all around'; today NO liveness test file contains a single
    beige-marked test."""
    missing = [
        f.name for f in _SUITE_FILES if "beige" not in f.read_text()
    ]
    assert not missing, f"suites with zero beige tests: {missing}"


def test_error_path_tests_carry_the_red_marker():
    """ADR-005 red = adversarial/error-path. The liveness suites' actual
    adversarial tests (UNREACHABLE paths, malformed inputs, rejection cases)
    sit under file-level green; the red marker is spent on live network
    probes instead. Error-path tests must be red-marked."""
    offenders = []
    for suite in _SUITE_FILES:
        text = suite.read_text()
        for match in _ERROR_PATH_NAME.finditer(text):
            # Look for a red mark in the decorator block above the def.
            preceding = text[: match.start()].rsplit("\n\n", 1)[-1]
            if "pytest.mark.red" not in preceding:
                offenders.append(f"{suite.name}::{match.group(1)}")
    assert not offenders, (
        f"{len(offenders)} error-path tests not red-marked, e.g. {offenders[:5]}"
    )


def test_liveness_branch_coverage_is_complete():
    """The claim says 100% covered; measured 2026-07-19: 95% branch coverage
    (21 statements + 13 partial branches missed — default network clients'
    error paths, resolve_credentials fallbacks, __main__ guards). Skips when
    coverage tooling is absent; fails at the claimed bar when present."""
    pytest.importorskip("coverage")
    import subprocess
    import sys

    repo_root = _TESTS_DIR.parent
    subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--branch",
         "--source=tools/liveness", "-m", "pytest", "tests/", "-k",
         "liveness and not taxonomy", "-q"],
        cwd=repo_root, capture_output=True, check=False,
    )
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--format=total"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    assert result.stdout.strip() == "100", f"coverage: {result.stdout.strip()}%"
