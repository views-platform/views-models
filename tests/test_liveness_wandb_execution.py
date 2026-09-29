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
@pytest.mark.live and skips truthfully.
"""

from datetime import datetime, timezone

import pytest

from tools.liveness.wandb_execution import (
    MONTHLY_ENSEMBLES,
    CheckReport,
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

@pytest.mark.red
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

@pytest.mark.red
def test_exit_two_unreachable(capsys):
    facts = {p: OSError("down") for p in REAL_FACTS}
    check = WandbExecutionCheck(latest_run=_client(facts), netrc_probe=lambda: True)
    assert main(check=check, now=NOW) == 2


# ── live integration (netrc + network; skips truthfully) ──────────────

@pytest.mark.live
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


@pytest.mark.beige
def test_structural_conventions_wandb_execution():
    """ADR-005 beige: surface module conventions — check/render/main exposed,
    and every verdict this surface can emit is registered in the exit map."""
    import tools.liveness.wandb_execution as module
    from tools.liveness.report import EXIT_CODE_BY_VERDICT

    assert callable(module.main) and callable(module.render)
    assert hasattr(module, "CheckReport")
    for verdict in ('EXECUTION_CURRENT', 'EXECUTION_STALE', 'SKIP_NO_CREDENTIALS', 'UNREACHABLE'):
        assert verdict in EXIT_CODE_BY_VERDICT, verdict


@pytest.mark.red
def test_netrc_probe_failure_never_sinks_the_check():
    def exploding_probe():
        raise OSError("~/.netrc unreadable")

    check = WandbExecutionCheck(
        latest_run=lambda project: None, netrc_probe=exploding_probe
    )
    report = check.run(now=datetime(2026, 7, 19, tzinfo=timezone.utc))
    assert report.netrc_present is None  # unknown, reported as such
    assert report.verdict == "EXECUTION_STALE"  # all NEVER_RUN


def test_render_omits_unknown_netrc_hint():
    report = CheckReport(verdict="UNREACHABLE", netrc_present=None, error="boom")
    assert "netrc_present" not in render(report)


def _fake_wandb_module(behavior):
    import types

    class FakeApi:
        def __init__(self, timeout=None):
            pass

        def runs(self, path, order=None, per_page=None):
            return behavior(path)

    module = types.ModuleType("wandb")
    module.Api = FakeApi
    return module


def test_default_client_reads_newest_run(monkeypatch):
    import sys
    import types

    run = types.SimpleNamespace(
        name="vivid-flower-127", created_at="2026-06-29T16:56:27Z",
        state="finished", config={"forecasting": {"train": (121, 557)}},
    )
    monkeypatch.setitem(
        sys.modules, "wandb", _fake_wandb_module(lambda path: iter([run]))
    )
    facts = WandbExecutionCheck._latest_run_via_wandb("pink_ponyclub_forecasting")
    assert facts == {"run_name": "vivid-flower-127",
                     "created_at": "2026-06-29T16:56:27Z",
                     "state": "finished", "train_end_month_id": 557}


def test_default_client_returns_none_for_absent_project(monkeypatch):
    import sys

    def not_found(path):
        raise ValueError(f"Could not find project {path}")

    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb_module(not_found))
    assert WandbExecutionCheck._latest_run_via_wandb("gone_forecasting") is None


def test_default_client_returns_none_for_empty_project(monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules, "wandb", _fake_wandb_module(lambda path: iter([]))
    )
    assert WandbExecutionCheck._latest_run_via_wandb("empty_forecasting") is None


@pytest.mark.red
def test_default_client_reraises_other_errors(monkeypatch):
    import sys

    def exploding(path):
        raise RuntimeError("api down")

    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb_module(exploding))
    with pytest.raises(RuntimeError):
        WandbExecutionCheck._latest_run_via_wandb("pink_ponyclub_forecasting")
