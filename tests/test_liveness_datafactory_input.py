"""Liveness S2: the datafactory input store (remote zarr) — issue #240, epic #238.

TDD suite written BEFORE the implementation. The check answers: does the
datafactory's observed-data coverage (`last_valid_month_id`, read live from
the store's .zattrs) reach what this repo's canonical partitions require
(`meta/partitions.json` — max test-window end)? This automates the register
C-96 tripwire, which re-arms at every partition bump.

Ground truth used below (verified live 2026-07-06/19): the store's
last_valid_month_id was 558; meta/partitions.json currently requires 552.

Unit tests are offline (injected reader + fixed requirements); the single
live test is @pytest.mark.live and skips truthfully.
"""

import json
from pathlib import Path

import pytest

from tools.liveness.datafactory_input import (
    DatafactoryInputCheck,
    main,
    render,
    required_month_id_from_partitions,
)
from tools.partitions.domain import month_id_to_date

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── requirement derivation (never hardcoded) ──────────────────────────

def test_required_month_derives_from_canonical_partitions_file():
    canonical = json.loads((REPO_ROOT / "meta" / "partitions.json").read_text())
    expected = max(
        canonical["calibration"]["test"][1], canonical["validation"]["test"][1]
    )
    assert required_month_id_from_partitions(REPO_ROOT) == expected

def test_required_month_from_synthetic_partitions(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "partitions.json").write_text(json.dumps({
        "calibration": {"train": [1, 2], "test": [3, 400]},
        "validation": {"train": [1, 4], "test": [5, 390]},
        "steps_default": 36,
    }))
    assert required_month_id_from_partitions(tmp_path) == 400


# ── verdicts (injected reader; deterministic) ─────────────────────────

def _check(last_valid, netrc_present=True, required=552):
    def read_last_valid():
        if isinstance(last_valid, Exception):
            raise last_valid
        return last_valid
    return DatafactoryInputCheck(
        read_last_valid_month_id=read_last_valid,
        netrc_probe=lambda: netrc_present,
        required_month_id=required,
    )

def test_fresh_when_coverage_reaches_requirement():
    report = _check(558).run()
    assert report.verdict == "INPUT_FRESH"
    assert report.last_valid_month_id == 558
    assert report.required_month_id == 552
    assert report.margin_months == 6

def test_fresh_at_exact_boundary():
    assert _check(552).run().verdict == "INPUT_FRESH"

def test_stale_when_coverage_short_of_requirement():
    report = _check(540).run()
    assert report.verdict == "INPUT_STALE"
    assert report.margin_months == -12

@pytest.mark.red
def test_unreachable_when_reader_fails():
    report = _check(OSError("connection timed out")).run()
    assert report.verdict == "UNREACHABLE"
    assert "connection timed out" in (report.error or "")

def test_missing_netrc_is_reported_as_fact():
    report = _check(558, netrc_present=False).run()
    assert report.netrc_present is False
    assert report.verdict == "INPUT_FRESH"  # reachable store trumps the hint


# ── the report is raw facts ───────────────────────────────────────────

def test_report_dates_rendered_from_month_ids():
    report = _check(558).run()
    assert report.last_valid_date == month_id_to_date(558) == "2026-06"
    assert report.required_date == month_id_to_date(552) == "2025-12"

def test_render_is_one_fact_per_line():
    text = render(_check(558).run())
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert any("INPUT_FRESH" in line for line in lines)
    assert any("558" in line for line in lines)
    assert any("552" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_when_fresh(capsys):
    assert main(check=_check(558)) == 0
    assert "INPUT_FRESH" in capsys.readouterr().out

def test_exit_one_when_stale(capsys):
    assert main(check=_check(500)) == 1

@pytest.mark.red
def test_exit_two_when_unreachable(capsys):
    assert main(check=_check(OSError("no route"))) == 2


# ── live integration (network + datafactory install; skips truthfully) ─

@pytest.mark.live
def test_live_datafactory_input_invariants():
    try:
        report = DatafactoryInputCheck().run()
    except Exception as e:  # noqa: BLE001 — any env problem skips, never false-red
        pytest.skip(f"datafactory input unreachable: {type(e).__name__}: {e}")
    if report.verdict == "UNREACHABLE":
        pytest.skip(f"datafactory input unreachable: {report.error}")
    # SKIP_NO_PACKAGE is the check reporting truthfully that datafactory_query is not
    # installed -- which is CI's normal state. Falling through asserted on facts the
    # report never carried, so a truthful skip surfaced as a red test (ADR-005).
    if report.verdict == "SKIP_NO_PACKAGE":
        pytest.skip(f"datafactory_query not installed: {report.error}")
    assert report.last_valid_month_id is not None
    assert report.last_valid_month_id > 500  # sanity: post-2021 coverage
    assert report.required_month_id == required_month_id_from_partitions(REPO_ROOT)


@pytest.mark.beige
def test_structural_conventions_datafactory_input():
    """ADR-005 beige: surface module conventions — check/render/main exposed,
    and every verdict this surface can emit is registered in the exit map."""
    import tools.liveness.datafactory_input as module
    from tools.liveness.report import EXIT_CODE_BY_VERDICT

    assert callable(module.main) and callable(module.render)
    assert hasattr(module, "CheckReport")
    for verdict in ('INPUT_FRESH', 'INPUT_STALE', 'SKIP_NO_PACKAGE', 'UNREACHABLE'):
        assert verdict in EXIT_CODE_BY_VERDICT, verdict


@pytest.mark.red
def test_netrc_probe_failure_never_sinks_the_check():
    def exploding_probe():
        raise OSError("~/.netrc unreadable")

    report = DatafactoryInputCheck(
        read_last_valid_month_id=lambda: 558,
        netrc_probe=exploding_probe,
        required_month_id=552,
    ).run()
    assert report.netrc_present is None  # unknown, reported as such
    assert report.verdict == "INPUT_FRESH"
