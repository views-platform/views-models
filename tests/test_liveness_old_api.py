"""Liveness S1: the old public API (api.viewsforecasting.org) — issue #239, epic #238.

TDD suite written BEFORE the implementation. All expectations derive from the
REAL API response captured live on 2026-07-19 (CAPTURED_RUNS below, verbatim,
88 entries) and from the run-naming convention evidenced that day:

    fatalities{generation}_{yyyy}_{mm}_t{seq}, where {yyyy}_{mm} is the
    DATA-CUTOFF month — publication happens ~1 month later. Evidence: the
    wandb run of 2026-06-29 trained to month_id 557 (May 2026) and was
    published as fatalities003_2026_05_t01.

Unit tests are offline (fake fetch, injected clock). The single live test is
@pytest.mark.red and skips truthfully on any network problem (house pattern:
tests/test_reconciliation_viewser_provider.py).
"""

import pytest

from tools.liveness.old_api import (
    BASE_URL,
    FRESHNESS_BUDGET_MONTHS,
    OldApiCheck,
    latest_fatalities_run,
    main,
    parse_run_name,
    render,
)
from tools.partitions.domain import date_to_month_id, month_id_to_date

pytestmark = pytest.mark.green


# ── the real response, frozen (2026-07-19) ────────────────────────────
CAPTURED_RUNS = [
    'd_2021_02_01',
    'escwa_2021_02_01',
    'escwa_2021_03_01',
    'escwa_2021_04_01',
    'escwa_2021_05_01',
    'escwa_2021_06_01',
    'escwa_2021_07_01',
    'escwa_2021_08_01',
    'escwa_2021_09_01',
    'escwa_2021_10_01',
    'escwa_2021_11_01',
    'escwa_2021_12_01',
    'escwa_data_2021_10_01',
    'escwa_features_2021_05_01',
    'f_2021_06_01',
    'fatalities001_2021_12_t01',
    'fatalities001_2022_00_t01',
    'fatalities001_2022_01_t01',
    'fatalities001_2022_02_t01',
    'fatalities001_2022_03_t01',
    'fatalities001_2022_04_t01',
    'fatalities001_2022_05_t01',
    'fatalities001_2022_06_t01',
    'fatalities001_2022_07_t01',
    'fatalities001_2022_08_t01',
    'fatalities001_2022_09_t01',
    'fatalities001_2022_10_t01',
    'fatalities001_2022_11_t01',
    'fatalities001_2022_12_t01',
    'fatalities001_2023_00_t01',
    'fatalities001_2023_01_t01',
    'fatalities001_2023_02_t01',
    'fatalities001_2023_03_t01',
    'fatalities002_2023_04_t01',
    'fatalities002_2023_05_t01',
    'fatalities002_2023_06_t01',
    'fatalities002_2023_07_t01',
    'fatalities002_2023_08_t01',
    'fatalities002_2023_09_t01',
    'fatalities002_2023_09_t02',
    'fatalities002_2023_10_t01',
    'fatalities002_2023_10_t02',
    'fatalities002_2023_11_t01',
    'fatalities002_2023_12_t01',
    'fatalities002_2024_01_t01',
    'fatalities002_2024_02_t01',
    'fatalities002_2024_03_t01',
    'fatalities002_2024_04_t01',
    'fatalities002_2024_05_t01',
    'fatalities002_2024_06_t01',
    'fatalities002_2024_07_t01',
    'fatalities002_2024_08_t01',
    'fatalities002_2024_09_t01',
    'fatalities002_2024_10_t01',
    'fatalities002_2024_11_t01',
    'fatalities002_2024_12_t01',
    'fatalities002_2025_01_t01',
    'fatalities002_2025_02_t01',
    'fatalities002_2025_03_t01',
    'fatalities002_2025_04_t01',
    'fatalities002_2025_05_t01',
    'fatalities002_2025_06_t01',
    'fatalities002_2025_07_t01',
    'fatalities002_2025_08_t01',
    'fatalities002_2025_09_t01',
    'fatalities002_2025_10_t01',
    'fatalities003_2025_10_t01',
    'fatalities003_2025_11_t01',
    'fatalities003_2025_12_t01',
    'fatalities003_2026_01_t01',
    'fatalities003_2026_02_t01',
    'fatalities003_2026_03_t01',
    'fatalities003_2026_04_t01',
    'fatalities003_2026_05_t01',
    'predictors_fatalities002_2025_12',
    'predictors_fatalities003_0000_00',
    'r_2021_01_01',
    'r_2021_02_01',
    'r_2021_03_01',
    'r_2021_04_01',
    'r_2021_05_01',
    'r_2021_06_01',
    'r_2021_07_01',
    'r_2021_08_01',
    'r_2021_09_01',
    'r_2021_10_01',
    'r_2021_11_01',
    'r_2021_12_01',
]


# The month the capture was made (July 2026) — used as the injected "now"
# so verdict tests are deterministic forever.
CAPTURE_NOW = date_to_month_id(2026, 7)  # 559


# ── parse_run_name ────────────────────────────────────────────────────

def test_parses_canonical_run_name():
    assert parse_run_name("fatalities003_2026_05_t01") == (3, 2026, 5, 1)

def test_parses_second_tag_sequence():
    assert parse_run_name("fatalities002_2023_09_t02") == (2, 2023, 9, 2)

def test_rejects_legacy_month_zero_names():
    # fatalities001_2022_00_t01 is real in the listing; month 00 must not
    # reach the month math (S1 review finding).
    assert parse_run_name("fatalities001_2022_00_t01") is None

@pytest.mark.parametrize("name", ["escwa_2021_02_01", "d_2021_02_01", "r_2021_12_01",
                                  "escwa_features_2021_05_01", "f_2021_06_01", ""])
def test_rejects_non_fatalities_names(name):
    assert parse_run_name(name) is None


# ── latest run selection: THE pinned bug ──────────────────────────────
# The API's list is NOT chronologically sorted; its alphabetical tail is
# r_2021_12_01. Naive "take the last element" reports a 2021 run. This test
# pins the chronological selection against the full real capture.

def test_latest_run_on_real_capture_is_2026_05():
    assert CAPTURED_RUNS[-1] == "r_2021_12_01"  # the trap, preserved
    assert latest_fatalities_run(CAPTURED_RUNS) == "fatalities003_2026_05_t01"

def test_latest_run_empty_list_is_none():
    assert latest_fatalities_run([]) is None
    assert latest_fatalities_run(["escwa_2021_02_01"]) is None


# ── month math (reused from tools.partitions.domain) ──────────────────

def test_data_cutoff_month_id_of_latest_capture():
    assert date_to_month_id(2026, 5) == 557
    assert month_id_to_date(557) == "2026-05"


# ── verdicts (fake fetch, injected now) ───────────────────────────────

def _fake_fetch(responses):
    """Dict-driven fetch: url -> parsed JSON, or a raise-marker Exception."""
    def fetch(url):
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected url fetched: {url}")
    return fetch

def _runs_doc(names):
    return {"runs": list(names)}

SERVING_ROWS = {"data": [{"country_id": 1, "month_id": 558}, {"country_id": 2, "month_id": 558}]}
EMPTY_ROWS = {"data": []}


def test_fresh_when_cutoff_within_budget():
    # cutoff May (557), now July (559): 2 months behind == budget -> FRESH
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "LIVE_FRESH"
    assert report.months_behind == FRESHNESS_BUDGET_MONTHS == 2
    assert report.latest_run == "fatalities003_2026_05_t01"

def test_stale_when_cutoff_beyond_budget():
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW + 1)  # Aug: 3 behind
    assert report.verdict == "LIVE_STALE"
    assert report.months_behind == 3

def test_unreachable_when_list_fetch_fails():
    fetch = _fake_fetch({BASE_URL: OSError("connection refused")})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "UNREACHABLE"
    assert "connection refused" in (report.error or "")

def test_not_serving_when_latest_run_returns_no_rows():
    fetch = _fake_fetch({"?month=": EMPTY_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "LIVE_NOT_SERVING"

def test_not_serving_when_no_fatalities_runs_listed():
    fetch = _fake_fetch({BASE_URL: _runs_doc(["escwa_2021_02_01"])})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    assert report.verdict == "LIVE_NOT_SERVING"


# ── the report is raw facts ───────────────────────────────────────────

def test_report_contains_literal_url_and_run_name():
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    assert report.url == BASE_URL
    assert report.run_count == len(CAPTURED_RUNS)
    assert report.data_cutoff_month_id == 557
    assert report.data_cutoff_date == "2026-05"
    assert report.serving_rows_sampled == 2

def test_render_is_one_fact_per_line():
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    report = OldApiCheck(fetch=fetch).run(now_month_id=CAPTURE_NOW)
    text = render(report)
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)          # fact per line
    assert any(BASE_URL in line for line in lines)      # literal url
    assert any("fatalities003_2026_05_t01" in line for line in lines)
    assert any("LIVE_FRESH" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_when_fresh(capsys):
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    assert main(fetch=fetch, now_month_id=CAPTURE_NOW) == 0
    assert "LIVE_FRESH" in capsys.readouterr().out

def test_exit_one_when_stale(capsys):
    fetch = _fake_fetch({"?month=": SERVING_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    assert main(fetch=fetch, now_month_id=CAPTURE_NOW + 6) == 1

def test_exit_one_when_not_serving(capsys):
    fetch = _fake_fetch({"?month=": EMPTY_ROWS, BASE_URL: _runs_doc(CAPTURED_RUNS)})
    assert main(fetch=fetch, now_month_id=CAPTURE_NOW) == 1

def test_exit_two_when_unreachable(capsys):
    fetch = _fake_fetch({BASE_URL: OSError("no route")})
    assert main(fetch=fetch, now_month_id=CAPTURE_NOW) == 2


# ── live integration (network; skips truthfully) ──────────────────────

@pytest.mark.red
def test_live_old_api_invariants():
    try:
        report = OldApiCheck().run()
    except Exception as e:  # noqa: BLE001 — any network/env problem skips, never false-red
        pytest.skip(f"old API unreachable from this environment: {type(e).__name__}: {e}")
    if report.verdict == "UNREACHABLE":
        pytest.skip(f"old API unreachable: {report.error}")
    assert report.run_count and report.run_count > 0
    assert report.latest_run is not None
    assert parse_run_name(report.latest_run) is not None
