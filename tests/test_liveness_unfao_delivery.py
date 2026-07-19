"""Liveness S4: the FAO delivery bucket (unfao_bucket) — issue #242, epic #238.

TDD suite written BEFORE the implementation. Ground truth captured live
2026-07-19: bucket `unfao_bucket`, 18 files, two delivery streams —
``forecast_dataset_*.parquet`` (newest 2026-03-10, ~191MB) and
``historical_dataset_*.parquet`` (newest 2026-03-30, ~20MB). Real monthly
deliveries ran Jan–Mar 2026, then stalled; nobody noticed (register C-99
class). This check makes "when did FAO last receive anything?" a machine
answer with per-stream verdicts.

Unit tests are offline (fake fetch, fake credentials, injected clock).
Credentials resolution is REUSED from tools.liveness.appwrite_store (same
project, same .env — S7 will home it in a shared credentials module).
"""

from datetime import datetime, timezone

import pytest

from tools.liveness.appwrite_store import AppwriteCredentials
from tools.liveness.unfao_delivery import (
    UNFAO_BUCKET_ID,
    CheckReport,
    UnfaoDeliveryCheck,
    main,
    render,
)

pytestmark = pytest.mark.green

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
SENTINEL_KEY = "SECRET-KEY-NEVER-RENDER"
CREDS = AppwriteCredentials("https://fra.cloud.appwrite.io/v1", "proj", SENTINEL_KEY)

REAL_LISTING = {
    "total": 18,
    "files": [
        {"$id": "h1", "$createdAt": "2026-03-30T09:48:45.000+00:00",
         "name": "historical_dataset_20260330_114835.parquet", "sizeOriginal": 20_500_000},
        {"$id": "f1", "$createdAt": "2026-03-10T10:47:48.000+00:00",
         "name": "forecast_dataset_20260310_114703.parquet", "sizeOriginal": 191_400_000},
        {"$id": "f2", "$createdAt": "2026-02-04T07:33:52.000+00:00",
         "name": "forecast_dataset_20260204_083338.parquet", "sizeOriginal": 191_300_000},
        {"$id": "x1", "$createdAt": "2026-01-01T00:00:00.000+00:00",
         "name": "readme.txt", "sizeOriginal": 100},
    ],
}
FRESH_LISTING = {
    "total": 2,
    "files": [
        {"$id": "f", "$createdAt": "2026-07-10T08:00:00.000+00:00",
         "name": "forecast_dataset_20260710_100000.parquet", "sizeOriginal": 190_000_000},
        {"$id": "h", "$createdAt": "2026-07-11T08:00:00.000+00:00",
         "name": "historical_dataset_20260711_100000.parquet", "sizeOriginal": 20_000_000},
    ],
}


def _fake_fetch(responses):
    def fetch(url, headers):
        assert headers["X-Appwrite-Key"] == SENTINEL_KEY
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected url: {url}")
    return fetch


# ── verdicts on the real capture ──────────────────────────────────────

def test_both_streams_stalled_on_real_capture():
    fetch = _fake_fetch({UNFAO_BUCKET_ID: REAL_LISTING})
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "DELIVERY_STALLED"
    assert report.forecast_verdict == "STALLED"
    assert report.historical_verdict == "STALLED"
    assert report.forecast_newest_name == "forecast_dataset_20260310_114703.parquet"
    assert report.forecast_days_since == 131
    assert report.historical_newest_name == "historical_dataset_20260330_114835.parquet"
    assert report.historical_days_since == 111
    assert report.other_files == 1  # readme.txt matches neither stream

def test_delivering_when_both_streams_recent():
    fetch = _fake_fetch({UNFAO_BUCKET_ID: FRESH_LISTING})
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "DELIVERING"
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "DELIVERING"

def test_mixed_streams_yield_overall_stalled():
    mixed = {"total": 2, "files": [
        {"$id": "f", "$createdAt": "2026-07-10T08:00:00.000+00:00",
         "name": "forecast_dataset_x.parquet", "sizeOriginal": 1},
        {"$id": "h", "$createdAt": "2026-03-30T09:48:45.000+00:00",
         "name": "historical_dataset_y.parquet", "sizeOriginal": 1},
    ]}
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=_fake_fetch({UNFAO_BUCKET_ID: mixed})).run(now=NOW)
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "STALLED"
    assert report.verdict == "DELIVERY_STALLED"

def test_missing_stream_reported_as_never_delivered():
    only_hist = {"total": 1, "files": [
        {"$id": "h", "$createdAt": "2026-07-11T08:00:00.000+00:00",
         "name": "historical_dataset_z.parquet", "sizeOriginal": 1},
    ]}
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=_fake_fetch({UNFAO_BUCKET_ID: only_hist})).run(now=NOW)
    assert report.forecast_verdict == "NEVER_DELIVERED"
    assert report.verdict == "DELIVERY_STALLED"

def test_unreachable_when_fetch_fails():
    report = UnfaoDeliveryCheck(credentials=CREDS,
                                fetch=_fake_fetch({UNFAO_BUCKET_ID: OSError("dns fail")})).run(now=NOW)
    assert report.verdict == "UNREACHABLE"
    assert "dns fail" in (report.error or "")

def test_skip_when_no_credentials():
    report = UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})).run(now=NOW)
    assert report.verdict == "SKIP_NO_CREDENTIALS"


# ── raw facts + redaction ─────────────────────────────────────────────

def test_render_facts_and_redaction():
    fetch = _fake_fetch({UNFAO_BUCKET_ID: REAL_LISTING})
    text = render(UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW))
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert SENTINEL_KEY not in text
    assert any("forecast_dataset_20260310_114703.parquet" in line for line in lines)
    assert any("DELIVERY_STALLED" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_delivering(capsys):
    fetch = _fake_fetch({UNFAO_BUCKET_ID: FRESH_LISTING})
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 0

def test_exit_zero_skip(capsys):
    assert main(check=UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})), now=NOW) == 0

def test_exit_one_stalled(capsys):
    fetch = _fake_fetch({UNFAO_BUCKET_ID: REAL_LISTING})
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 1

def test_exit_two_unreachable(capsys):
    fetch = _fake_fetch({UNFAO_BUCKET_ID: OSError("boom")})
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 2


# ── live integration (creds + network; skips truthfully) ──────────────

@pytest.mark.red
def test_live_unfao_delivery_invariants():
    check = UnfaoDeliveryCheck()
    if check.credentials is None:
        pytest.skip("no Appwrite credentials resolvable in this environment")
    try:
        report = check.run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"unfao bucket unreachable: {type(e).__name__}: {e}")
    if report.verdict == "UNREACHABLE":
        pytest.skip(f"unfao bucket unreachable: {report.error}")
    assert report.total_files and report.total_files > 0
    assert report.forecast_newest_name is not None
