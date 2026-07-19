"""Liveness S5: wandb execution recency — issue #243, epic #238.

TDD suite written BEFORE the implementation. The check answers: "did the
team compute this cycle?" — the question that started the 2026-07-19 trust
crisis, answered then by hand-probing wandb. Ground truth from that probe
(entity views_pipeline, project naming '{name}_forecasting' per pipeline-core
model.py:983):

    pink_ponyclub_forecasting  latest finished 2026-06-29T16:56:27
    skinny_love_forecasting    latest finished 2026-06-29T21:06:00
    rude_boy_forecasting       latest finished 2026-07-15T15:49:03
    first_love_forecasting     latest finished 2026-07-15T15:11:37

Each run's config also records its train-window end (the data-cutoff receipt
that resolved the run-naming ambiguity: June-29 runs trained to month 557).

Unit tests are offline (injected client + clock); the live test is
@pytest.mark.red and skips truthfully.
"""

from datetime import datetime, timezone

import pytest

from tools.liveness.wandb_execution import (
    CYCLE_BUDGET_DAYS,
    MONTHLY_ENSEMBLES,
    CheckReport,
    EnsembleRun,
    WandbExecutionCheck,
    main,
    render,
)

pytestmark = pytest.mark.green

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)

REAL_FACTS = {
    "pink_ponyclub_forecasting": {"run_name": "vivid-flower-127", "created_at": "2026-06-29T16:56:27", "state": "finished", "train_end_month_id": 557},
    "skinny_love_forecasting": {"run_name": "sage-shape-97", "created_at": "2026-06-29T21:06:00", "state": "finished", "train_end_month_id": 557},
    "rude_boy_forecasting": {"run_name": "zesty-energy-67", "created_at": "2026-07-15T15:49:03", "state": "finished", "train_end_month_id": 558},
    "first_love_forecasting": {"run_name": "brisk-cloud-7", "created_at": "2026-07-15T15:11:37", "state": "finished", "train_end_month_id": 558},
}


def _client(facts):
    def latest_run(project):
        value = facts.get(project)
        if isinstance(value, Exception):
            raise value
        return value
    return latest_run


# ── the ensemble list is encoded, with its receipt ────────────────────

def test_monthly_ensembles_mirror_the_bash_list():
    assert MONTHLY_ENSEMBLES == ("pink_ponyclub", "skinny_love", "rude_boy", "first_love")


# ── verdicts ──────────────────────────────────────────────────────────

def test_current_when_all_recent():
    report = WandbExecutionCheck(latest_run=_client(REAL_FACTS), netrc_probe=lambda: True).run(now=NOW)
    assert report.verdict == "EXECUTION_CURRENT"
    by_name = {e.ensemble: e for e in report.ensembles}
    assert by_name["pink_ponyclub"].days_since == 19
    assert by_name["first_love"].days_since == 3
    assert by_name["rude_boy"].train_end_month_id == 558
    assert all(e.verdict == "COMPUTED" for e in report.ensembles)

def test_stale_when_one_ensemble_old():
    facts = dict(REAL_FACTS)
    facts["pink_ponyclub_forecasting"] = {"run_name": "old", "created_at": "2026-03-01T00:00:00", "state": "finished", "train_end_month_id": 553}
    report = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True).run(now=NOW)
    assert report.verdict == "EXECUTION_STALE"
    by_name = {e.ensemble: e for e in report.ensembles}
    assert by_name["pink_ponyclub"].verdict == "NOT_COMPUTED"
    assert by_name["pink_ponyclub"].days_since == 140

def test_missing_project_is_never_run():
    facts = dict(REAL_FACTS)
    facts["rude_boy_forecasting"] = None
    report = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True).run(now=NOW)
    by_name = {e.ensemble: e for e in report.ensembles}
    assert by_name["rude_boy"].verdict == "NEVER_RUN"
    assert report.verdict == "EXECUTION_STALE"

def test_unfinished_latest_run_is_not_computed():
    facts = dict(REAL_FACTS)
    facts["skinny_love_forecasting"] = {"run_name": "crashed-1", "created_at": "2026-07-18T00:00:00", "state": "crashed", "train_end_month_id": None}
    report = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True).run(now=NOW)
    by_name = {e.ensemble: e for e in report.ensembles}
    assert by_name["skinny_love"].verdict == "NOT_COMPUTED"
    assert report.verdict == "EXECUTION_STALE"

def test_unreachable_when_client_fails():
    facts = {p: OSError("api down") for p in REAL_FACTS}
    report = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True).run(now=NOW)
    assert report.verdict == "UNREACHABLE"
    assert "api down" in (report.error or "")

def test_skip_when_no_netrc():
    report = WandbExecutionCheck(latest_run=_client(REAL_FACTS), netrc_probe=lambda: False).run(now=NOW)
    assert report.verdict == "SKIP_NO_CREDENTIALS"


# ── raw facts ─────────────────────────────────────────────────────────

def test_render_per_ensemble_facts():
    text = render(WandbExecutionCheck(latest_run=_client(REAL_FACTS), netrc_probe=lambda: True).run(now=NOW))
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert any("EXECUTION_CURRENT" in line for line in lines)
    assert any("pink_ponyclub" in line and "2026-06-29" in line for line in lines)
    assert any("train_end" in line and "558" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_current(capsys):
    check = WandbExecutionCheck(latest_run=_client(REAL_FACTS), netrc_probe=lambda: True)
    assert main(check=check, now=NOW) == 0

def test_exit_zero_skip(capsys):
    check = WandbExecutionCheck(latest_run=_client(REAL_FACTS), netrc_probe=lambda: False)
    assert main(check=check, now=NOW) == 0

def test_exit_one_stale(capsys):
    facts = dict(REAL_FACTS)
    facts["first_love_forecasting"] = None
    check = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True)
    assert main(check=check, now=NOW) == 1

def test_exit_two_unreachable(capsys):
    facts = {p: OSError("down") for p in REAL_FACTS}
    check = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True)
    assert main(check=check, now=NOW) == 2


# ── live integration (netrc + network; skips truthfully) ──────────────

@pytest.mark.red
def test_live_wandb_execution_invariants():
    check = WandbExecutionCheck()
    try:
        report = check.run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"wandb unreachable: {type(e).__name__}: {e}")
    if report.verdict in ("UNREACHABLE", "SKIP_NO_CREDENTIALS"):
        pytest.skip(f"wandb not checkable here: {report.verdict} {report.error or ''}")
    assert len(report.ensembles) == len(MONTHLY_ENSEMBLES)
    assert any(e.created_at is not None for e in report.ensembles)
