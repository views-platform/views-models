"""Liveness S3: the Appwrite production_forecasts store — issue #241, epic #238.

TDD suite written BEFORE the implementation. Ground truth captured live
2026-07-19 with the datastore key:

    database  id='file_metadata'          name='File Metadata'
    collection id='production_forecasts'  name='Production Forecasts'
    collection id='unfao'                 name='UNFAO File Metadata'
    bucket 'production_forecasts': 318 files, newest 2025-11-27 (~34KB stubs)

The June 2026 live failure used collection ID 'forecasts_metadata' — which
does not exist and never did (register C-100's founding incident). This
check encodes the REAL IDs and reports drift.

Unit tests are offline (fake fetch, fake credentials, injected clock);
secrets never appear in any rendered output (pinned test). The live test is
@pytest.mark.red and skips truthfully.
"""

from datetime import datetime, timezone

import pytest

from tools.liveness.appwrite_store import (
    HISTORICAL_WRONG_COLLECTION_ID,
    REAL_METADATA_DATABASE_ID,
    REAL_PROD_FORECASTS_COLLECTION_ID,
    AppwriteCredentials,
    AppwriteStoreCheck,
    CheckReport,
    load_credentials_from_env_file,
    main,
    render,
)

pytestmark = pytest.mark.green

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
SENTINEL_KEY = "SECRET-API-KEY-MUST-NEVER-RENDER"

CREDS = AppwriteCredentials(
    endpoint="https://fra.cloud.appwrite.io/v1",
    project_id="proj123",
    api_key=SENTINEL_KEY,
)

FILES_DOC = {
    "total": 318,
    "files": [
        {"$id": "a", "$createdAt": "2025-11-26T12:07:40.000+00:00",
         "name": "predictions_forecasting_20251126_125446.parquet", "sizeOriginal": 42843},
        {"$id": "b", "$createdAt": "2025-11-27T12:12:56.000+00:00",
         "name": "predictions_forecasting_20251127_125227.parquet", "sizeOriginal": 33721},
    ],
}
COLLECTIONS_DOC = {
    "total": 2,
    "collections": [
        {"$id": "production_forecasts", "name": "Production Forecasts"},
        {"$id": "unfao", "name": "UNFAO File Metadata"},
    ],
}
FRESH_FILES_DOC = {
    "total": 5,
    "files": [
        {"$id": "c", "$createdAt": "2026-07-10T09:00:00.000+00:00",
         "name": "predictions_forecasting_20260710.parquet", "sizeOriginal": 5_000_000},
    ],
}


def _fake_fetch(responses):
    def fetch(url, headers):
        assert headers["X-Appwrite-Key"] == SENTINEL_KEY  # creds reach the wire
        for key, value in responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected url: {url}")
    return fetch


# ── encoded ground truth ──────────────────────────────────────────────

def test_real_ids_are_encoded():
    assert REAL_METADATA_DATABASE_ID == "file_metadata"
    assert REAL_PROD_FORECASTS_COLLECTION_ID == "production_forecasts"
    assert HISTORICAL_WRONG_COLLECTION_ID == "forecasts_metadata"


# ── credentials resolution (presence, never values) ───────────────────

def test_credentials_parse_export_style_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        'export APPWRITE_ENDPOINT="https://x.example/v1"\n'
        "export APPWRITE_DATASTORE_PROJECT_ID=p1\n"
        "export APPWRITE_DATASTORE_API_KEY='k1'\n"
        "export OTHER=ignored\n"
    )
    creds = load_credentials_from_env_file(env)
    assert creds == AppwriteCredentials("https://x.example/v1", "p1", "k1")

def test_credentials_none_when_file_missing(tmp_path):
    assert load_credentials_from_env_file(tmp_path / "absent.env") is None

def test_credentials_none_when_keys_incomplete(tmp_path):
    env = tmp_path / ".env"
    env.write_text("export APPWRITE_ENDPOINT=https://x\n")
    assert load_credentials_from_env_file(env) is None


# ── verdicts ──────────────────────────────────────────────────────────

def test_skip_when_no_credentials():
    report = AppwriteStoreCheck(credentials=None, fetch=_fake_fetch({})).run(now=NOW)
    assert report.verdict == "SKIP_NO_CREDENTIALS"

def test_idle_when_newest_file_old():
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": COLLECTIONS_DOC})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "STORE_IDLE"
    assert report.newest_file_name == "predictions_forecasting_20251127_125227.parquet"
    assert report.days_since_newest == 233  # 23h47m short of day 234

def test_active_when_newest_file_recent():
    fetch = _fake_fetch({"/storage/": FRESH_FILES_DOC, "/databases/": COLLECTIONS_DOC})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "STORE_ACTIVE"
    assert report.days_since_newest == 9

def test_unreachable_when_fetch_fails():
    fetch = _fake_fetch({"/storage/": OSError("tls handshake failed")})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "UNREACHABLE"
    assert "tls handshake" in (report.error or "")

def test_empty_bucket_is_idle_with_no_newest():
    fetch = _fake_fetch({"/storage/": {"total": 0, "files": []},
                         "/databases/": COLLECTIONS_DOC})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "STORE_IDLE"
    assert report.newest_file_name is None


# ── collection discovery facts ────────────────────────────────────────

def test_real_collection_confirmed_and_wrong_id_absent():
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": COLLECTIONS_DOC})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.real_collection_present is True
    assert "production_forecasts" in report.collections_found
    assert HISTORICAL_WRONG_COLLECTION_ID not in report.collections_found

def test_collection_listing_failure_is_a_fact_not_a_crash():
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": OSError("403")})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    assert report.verdict == "STORE_IDLE"           # storage verdict unaffected
    assert report.real_collection_present is None    # unknown, honestly


# ── secrets never render ──────────────────────────────────────────────

def test_render_never_contains_key_material():
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": COLLECTIONS_DOC})
    report = AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW)
    text = render(report)
    assert SENTINEL_KEY not in text
    assert "api_key_chars: " in text  # presence signalled by length only

def test_render_is_one_fact_per_line():
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": COLLECTIONS_DOC})
    text = render(AppwriteStoreCheck(credentials=CREDS, fetch=fetch).run(now=NOW))
    assert all(":" in line for line in text.strip().splitlines())
    assert any("STORE_IDLE" in line for line in text.splitlines())


# ── exit codes ────────────────────────────────────────────────────────

def test_exit_zero_active(capsys):
    fetch = _fake_fetch({"/storage/": FRESH_FILES_DOC, "/databases/": COLLECTIONS_DOC})
    assert main(check=AppwriteStoreCheck(credentials=CREDS, fetch=fetch), now=NOW) == 0

def test_exit_zero_skip_no_creds(capsys):
    assert main(check=AppwriteStoreCheck(credentials=None, fetch=_fake_fetch({})), now=NOW) == 0
    assert "SKIP_NO_CREDENTIALS" in capsys.readouterr().out

def test_exit_one_idle(capsys):
    fetch = _fake_fetch({"/storage/": FILES_DOC, "/databases/": COLLECTIONS_DOC})
    assert main(check=AppwriteStoreCheck(credentials=CREDS, fetch=fetch), now=NOW) == 1

def test_exit_two_unreachable(capsys):
    fetch = _fake_fetch({"/storage/": OSError("boom")})
    assert main(check=AppwriteStoreCheck(credentials=CREDS, fetch=fetch), now=NOW) == 2


# ── live integration (creds + network; skips truthfully) ──────────────

@pytest.mark.red
def test_live_appwrite_store_invariants():
    check = AppwriteStoreCheck()
    if check.credentials is None:
        pytest.skip("no Appwrite credentials resolvable in this environment")
    try:
        report = check.run()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"appwrite unreachable: {type(e).__name__}: {e}")
    if report.verdict == "UNREACHABLE":
        pytest.skip(f"appwrite unreachable: {report.error}")
    assert report.total_files and report.total_files > 0
    assert report.real_collection_present is True
