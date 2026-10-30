"""Regression tests from the 2026-07-19 falsification audit of tools.liveness.

Claim audited: "tools.liveness is air and water tight." Verdict: FALSIFIED
(register C-101, C-102). P1/P2/P4/P7 pin the fixed defects; P5 is the
roster-mirror tripwire (fails when monthly_run.sh drifts from
MONTHLY_ENSEMBLES); P8 is an xfail marking the C-102 coverage gap — it
starts passing (XPASS, strict) the day a viewser surface ships, forcing
this file and the register to be updated together.
"""

from __future__ import annotations

import io
import re
import contextlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.liveness import vpn_store
from tools.liveness.__main__ import run_all
from tools.liveness.datafactory_input import DatafactoryInputCheck
from tools.liveness.report import exit_code_for
from tools.liveness.wandb_execution import MONTHLY_ENSEMBLES, WandbExecutionCheck

pytestmark = pytest.mark.green

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.red
def test_p1_multiline_error_value_stays_one_fact_per_line():
    """P1 (hard): render contract is 'one fact per line, key: value', but a
    multi-line error value (seen live: sqlalchemy OperationalError) emits
    continuation lines that belong to no key — machine-parsing breaks."""
    error = "OperationalError: boom\n\n(Background on this error at: https://sqlalche.me/e/14/e3q8)"
    report = vpn_store.CheckReport(verdict="VPN_REQUIRED", now_month_id=559, error=error)
    for line in vpn_store.render(report).splitlines():
        assert re.match(r"^[a-z_.]+: ", line), f"line breaks key: value contract: {line!r}"


def test_p2_datafactory_input_missing_package_is_truthful_skip():
    """P2 (hard): README exit contract says a missing package is a truthful
    SKIP (exit 0) on every surface; datafactory_input reports UNREACHABLE
    (exit 2) — a false alarm on any machine without datafactory_query."""

    def reader_without_package():
        raise ModuleNotFoundError("No module named 'datafactory_query'")

    report = DatafactoryInputCheck(
        read_last_valid_month_id=reader_without_package,
        netrc_probe=lambda: False,
        required_month_id=552,
    ).run()
    assert report.verdict == "SKIP_NO_PACKAGE"
    assert exit_code_for(report.verdict) == 0


@pytest.mark.red
def test_p4_wandb_malformed_created_at_does_not_crash_the_check():
    """P4 (hard): _judge runs outside the per-ensemble try; one run with a
    malformed created_at crashes the whole check uncaught (standalone module
    dies with a traceback, no report), instead of that ensemble landing in
    the failures fact."""

    def latest_run(project):
        if project == "pink_ponyclub_forecasting":
            return {"run_name": "broken", "created_at": None, "state": "finished",
                    "train_end_month_id": 557}
        return {"run_name": "ok", "created_at": "2026-07-15T15:00:00Z",
                "state": "finished", "train_end_month_id": 558}

    check = WandbExecutionCheck(latest_run=latest_run, netrc_probe=lambda: True)
    report = check.run(now=datetime(2026, 7, 19, tzinfo=timezone.utc))  # must not raise
    assert report.verdict in {"EXECUTION_CURRENT", "EXECUTION_STALE"}
    assert report.error is not None  # the malformed ensemble is a reported fact


@pytest.mark.beige
def test_p5_monthly_ensembles_mirrors_monthly_run_sh():
    """P5 (soft): the docstring says 'update BOTH when the roster changes'
    but nothing enforced it. This IS the tripwire: it passes today and fails
    the moment monthly_run.sh's ensemble roster drifts from the constant."""
    text = (_REPO_ROOT / "monthly_run.sh").read_text()
    roster = tuple(re.findall(r'run_folder\s+"ensembles/([a-z_]+)"', text))
    assert roster == MONTHLY_ENSEMBLES


def test_p7_unknown_verdict_fails_before_printing(capsys):
    """P7 (soft): a verdict missing from EXIT_CODE_BY_VERDICT must fail loud
    BEFORE the report prints — otherwise the runner's containment appends a
    second, contradictory verdict block for the same surface."""

    class StubCheck:
        def run(self, now_month_id=None):
            return vpn_store.CheckReport(verdict="BOGUS_VERDICT")

    with pytest.raises(KeyError):
        vpn_store.main(check=StubCheck())
    assert capsys.readouterr().out == ""  # nothing printed before the failure

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_all(surfaces=(("vpn_store", lambda: vpn_store.main(check=StubCheck())),))
    verdict_lines = [
        line for line in buffer.getvalue().splitlines() if line.startswith("verdict:")
    ]
    assert verdict_lines == ["verdict: UNREACHABLE"], f"blocks: {verdict_lines}"


@pytest.mark.xfail(
    strict=True,
    reason="C-102 (open, scope decision pending): viewser — the actual input "
    "of the four production ensembles — has no liveness surface yet",
)
def test_p8_viewser_input_surface_exists():
    """P8 (soft, adequacy): epic #238's charter is 'every input and output
    destination' — but viewser, the ACTUAL input of the four production
    ensembles, has no surface (the suite watches the datafactory input that
    production does not yet consume)."""
    from tools.liveness.__main__ import SURFACES

    assert any("viewser" in name for name, _ in SURFACES)
