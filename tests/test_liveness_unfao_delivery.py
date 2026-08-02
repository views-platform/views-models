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
    UnfaoDeliveryCheck,
    main,
    render,
)

pytestmark = pytest.mark.green

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
SENTINEL_KEY = "SECRET-KEY-NEVER-RENDER"
CREDS = AppwriteCredentials("https://fra.cloud.appwrite.io/v1", "proj", SENTINEL_KEY)

REAL_FORECAST_DOC = {
    "total": 9,
    "files": [{"$id": "f1", "$createdAt": "2026-03-10T10:47:48.000+00:00",
               "name": "forecast_dataset_20260310_114703.parquet", "sizeOriginal": 191_400_000}],
}
REAL_HISTORICAL_DOC = {
    "total": 8,
    "files": [{"$id": "h1", "$createdAt": "2026-03-30T09:48:45.000+00:00",
               "name": "historical_dataset_20260330_114835.parquet", "sizeOriginal": 20_500_000}],
}
REAL_OVERALL_DOC = {"total": 18, "files": []}

FRESH_FORECAST_DOC = {
    "total": 1,
    "files": [{"$id": "f", "$createdAt": "2026-07-10T08:00:00.000+00:00",
               "name": "forecast_dataset_20260710_100000.parquet", "sizeOriginal": 190_000_000}],
}
FRESH_HISTORICAL_DOC = {
    "total": 1,
    "files": [{"$id": "h", "$createdAt": "2026-07-11T08:00:00.000+00:00",
               "name": "historical_dataset_20260711_100000.parquet", "sizeOriginal": 20_000_000}],
}
FRESH_OVERALL_DOC = {"total": 2, "files": []}

def _responses(forecast, historical, overall):
    # order matters: stream keys first, catch-all base last
    return {"forecast_dataset_": forecast, "historical_dataset_": historical,
            "unfao_bucket/files": overall,
            # The bucket GET that proves the key is accepted before any listing is
            # believed. Last, so the `/files` listings above still match first. Its
            # body is never read — only whether it raises.
            "buckets/unfao_bucket": {"$id": "unfao_bucket"}}

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


@pytest.fixture
def no_machine_credentials(monkeypatch, tmp_path):
    """Cut this test off from the machine it runs on.

    ``credential_gap_report()`` reads process env AND this repository's own
    ``.env``, so a test asserting "nothing is configured" was in fact asserting
    something about the laptop. These passed for as long as the developer's
    ``.env`` happened to carry all twelve Appwrite coordinates, and went red the
    hour those were removed (correctly, per ADR-018) — the removal exposed the
    defect, it did not cause it. Nothing uncommitted may decide a verdict here.
    """
    from tools.liveness import appwrite_api

    for key in appwrite_api.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    monkeypatch.setattr(appwrite_api, "REPO_ROOT", empty_root)

# ── verdicts on the real capture ──────────────────────────────────────

def test_both_streams_stalled_on_real_capture():
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC))
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
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, FRESH_HISTORICAL_DOC, FRESH_OVERALL_DOC))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "DELIVERING"
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "DELIVERING"

def test_mixed_streams_yield_overall_stalled():
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, REAL_HISTORICAL_DOC, {"total": 2, "files": []}))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.forecast_verdict == "DELIVERING"
    assert report.historical_verdict == "STALLED"
    assert report.verdict == "DELIVERY_STALLED"

def test_missing_stream_reported_as_never_delivered():
    fetch = _fake_fetch(_responses({"total": 0, "files": []}, FRESH_HISTORICAL_DOC, {"total": 1, "files": []}))
    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.forecast_verdict == "NEVER_DELIVERED"
    assert report.verdict == "DELIVERY_STALLED"

@pytest.mark.red
def test_unreachable_when_fetch_fails():
    report = UnfaoDeliveryCheck(credentials=CREDS,
                                fetch=_fake_fetch({"unfao_bucket": OSError("dns fail")})).run(now=NOW)
    assert report.verdict == "UNREACHABLE"
    assert "dns fail" in (report.error or "")

def test_skip_when_no_credentials(no_machine_credentials):
    report = UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})).run(now=NOW)
    assert report.verdict == "SKIP_NO_CREDENTIALS"


# ── raw facts + redaction ─────────────────────────────────────────────

def test_render_facts_and_redaction():
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC))
    text = render(UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW))
    lines = text.strip().splitlines()
    assert all(":" in line for line in lines)
    assert SENTINEL_KEY not in text
    assert any("forecast_dataset_20260310_114703.parquet" in line for line in lines)
    assert any("DELIVERY_STALLED" in line for line in lines)


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_delivering(capsys):
    fetch = _fake_fetch(_responses(FRESH_FORECAST_DOC, FRESH_HISTORICAL_DOC, FRESH_OVERALL_DOC))
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 0

def test_exit_zero_skip(no_machine_credentials, capsys):
    assert main(check=UnfaoDeliveryCheck(credentials=None, fetch=_fake_fetch({})), now=NOW) == 0

def test_exit_one_stalled(capsys):
    fetch = _fake_fetch(_responses(REAL_FORECAST_DOC, REAL_HISTORICAL_DOC, REAL_OVERALL_DOC))
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 1

@pytest.mark.red
def test_exit_two_unreachable(capsys):
    fetch = _fake_fetch({"unfao_bucket/files": OSError("boom")})
    assert main(check=UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch), now=NOW) == 2


# ── live integration (creds + network; skips truthfully) ──────────────

@pytest.mark.live
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


@pytest.mark.beige
def test_structural_conventions_unfao_delivery():
    """ADR-005 beige: surface module conventions — check/render/main exposed,
    and every verdict this surface can emit is registered in the exit map."""
    import tools.liveness.unfao_delivery as module
    from tools.liveness.report import EXIT_CODE_BY_VERDICT

    assert callable(module.main) and callable(module.render)
    assert hasattr(module, "CheckReport")
    for verdict in ('DELIVERING', 'DELIVERY_STALLED', 'SKIP_NO_CREDENTIALS', 'UNREACHABLE'):
        assert verdict in EXIT_CODE_BY_VERDICT, verdict


# ── the rejected-key blind spot (see the sibling suite for the measurement) ──
# Same mechanism, and worse here: this surface's name for "the listing came back
# empty" is DELIVERY_STALLED — indistinguishable, before the fix, from a dead key
# on the bucket the UN FAO is served from.

@pytest.mark.red
def test_rejected_key_is_unreachable_not_stalled():
    def fetch(url, headers):
        if "/files?" in url:
            return {"total": 0, "files": []}
        raise PermissionError("HTTP Error 401: Unauthorized")

    report = UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "UNREACHABLE", (
        "a rejected key read as a partner bucket that had gone quiet"
    )
    assert "401" in (report.error or "")


def test_key_is_verified_before_any_listing_is_believed():
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return {"total": 0, "files": []}

    UnfaoDeliveryCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert calls and calls[0].endswith("/storage/buckets/unfao_bucket"), (
        f"first call was {calls[0] if calls else '<none>'}"
    )
