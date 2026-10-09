"""Liveness check for the Appwrite production_forecasts store (the "new shelf").

Answers, with raw facts: is the store reachable with resolvable credentials,
when did the newest forecast file land, and do the REAL metadata IDs match
what we encode?

Usage:
    python -m tools.liveness.appwrite_store   # exit 0 active/skip / 1 idle / 2 unreachable

THE REAL IDs (discovered live 2026-07-19 via the datastore key; register
C-100's founding incident was a config naming a collection that never
existed):

    database   'file_metadata'          ("File Metadata")
    collection 'production_forecasts'   ("Production Forecasts")  <- the real one
    collection 'unfao'                  ("UNFAO File Metadata")

    HISTORICAL WRONG VALUE: 'forecasts_metadata' — used by the June 2026
    un_fao run config; does not exist; killed the run at store lookup.

Credentials resolution (encoded once, values never rendered): process env
vars first, else **this repository's own** `.env` at the repo root, in either
line style. It no longer reads `views-faoapi/.env` — doing so meant this repo
observed its own shelf under the FAO identity (#298). Reports presence via
character counts only.

Design (house rules, mirrors the S1/S2 checks): injected fetch + credentials
+ clock (DIP), lazy stdlib urllib in the default fetch, no import-time side
effects (C-93), zero new dependencies, truthful SKIP without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

APPWRITE_BUCKET_ID = "production_forecasts"
REAL_METADATA_DATABASE_ID = "file_metadata"
REAL_PROD_FORECASTS_COLLECTION_ID = "production_forecasts"
HISTORICAL_WRONG_COLLECTION_ID = "forecasts_metadata"

# A monthly-cadence store is "active" if something landed within ~1.5 cycles.
ACTIVE_WITHIN_DAYS = 45

# Shared Appwrite plumbing lives in appwrite_api (S7 extraction, #245);
# names are re-exported here so existing imports/tests keep working.
from tools.liveness.appwrite_api import (  # noqa: E402  (kept near use)
    AppwriteCredentials,
    FetchJson,
    assert_bucket_reachable,
    fetch_json,
    load_credentials_from_env_file,
    newest_first_query,
    credential_gap_report,
    resolve_credentials,
)
from tools.liveness.report import exit_code_for, render_facts  # noqa: E402

__all__ = [
    "AppwriteCredentials",
    "AppwriteStoreCheck",
    "CheckReport",
    "load_credentials_from_env_file",
    "main",
    "newest_first_query",
    "render",
    "resolve_credentials",
]


@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the production_forecasts store — no narration."""

    verdict: str  # STORE_ACTIVE | STORE_IDLE | UNREACHABLE | SKIP_NO_CREDENTIALS
    #              | CREDENTIALS_INCOMPLETE
    endpoint: Optional[str] = None
    bucket: str = APPWRITE_BUCKET_ID
    api_key_chars: Optional[int] = None
    total_files: Optional[int] = None
    newest_file_name: Optional[str] = None
    newest_file_created: Optional[str] = None
    newest_file_bytes: Optional[int] = None
    days_since_newest: Optional[int] = None
    collections_found: Optional[List[str]] = None
    real_collection_present: Optional[bool] = None
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value``; key material never appears."""
    facts = [
        ("surface", "appwrite_store"),
        ("verdict", report.verdict),
        ("endpoint", report.endpoint),
        ("bucket", report.bucket),
        ("api_key_chars", report.api_key_chars),
        ("total_files", report.total_files),
        ("newest_file_name", report.newest_file_name),
        ("newest_file_created", report.newest_file_created),
        ("newest_file_bytes", report.newest_file_bytes),
        ("days_since_newest", report.days_since_newest),
        ("active_within_days", ACTIVE_WITHIN_DAYS),
        ("metadata_database", REAL_METADATA_DATABASE_ID),
        ("collections_found", report.collections_found),
        ("real_collection_present", report.real_collection_present),
        ("historical_wrong_collection_id", HISTORICAL_WRONG_COLLECTION_ID),
        ("error", report.error),
    ]
    return render_facts(facts)


class AppwriteStoreCheck:
    """Freshness + schema-truth check for the new shelf (all seams injected)."""

    def __init__(
        self,
        credentials: Optional[AppwriteCredentials] = "RESOLVE",  # type: ignore[assignment]
        fetch: Optional[FetchJson] = None,
    ) -> None:
        self.credentials = (
            resolve_credentials() if credentials == "RESOLVE" else credentials
        )
        self._fetch = fetch or fetch_json

    def run(self, now: Optional[datetime] = None) -> CheckReport:
        if self.credentials is None:
            verdict, error = credential_gap_report()
            return CheckReport(verdict=verdict, error=error)
        now = now or datetime.now(timezone.utc)
        creds = self.credentials
        headers = {
            "X-Appwrite-Project": creds.project_id,
            "X-Appwrite-Key": creds.api_key,
        }

        try:
            # FIRST, and not merged into the listing below: the listing endpoint
            # answers a rejected key with 200/total=0, so an expired credential
            # reads as an empty bucket. See assert_bucket_reachable.
            assert_bucket_reachable(
                creds.endpoint, APPWRITE_BUCKET_ID, headers, self._fetch
            )
            # Server-side newest-first + limit: Appwrite returns 25/page by
            # default, and an unsorted first page of a large bucket made this
            # check report a FALSE newest (the 2026-07-19 pagination bug —
            # pinned by test_storage_request_orders_server_side).
            listing = self._fetch(
                f"{creds.endpoint}/storage/buckets/{APPWRITE_BUCKET_ID}/files"
                f"?{newest_first_query()}",
                headers,
            )
            files = list(listing.get("files", []))  # type: ignore[union-attr]
            total = int(listing.get("total", len(files)))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — any storage failure is the fact
            return CheckReport(
                verdict="UNREACHABLE",
                endpoint=creds.endpoint,
                api_key_chars=len(creds.api_key),
                error=f"{type(exc).__name__}: {exc}",
            )

        collections, real_present = self._discover_collections(creds, headers)

        newest = max(files, key=lambda f: f.get("$createdAt", ""), default=None)
        if newest is None:
            return CheckReport(
                verdict="STORE_IDLE",
                endpoint=creds.endpoint,
                api_key_chars=len(creds.api_key),
                total_files=total,
                collections_found=collections,
                real_collection_present=real_present,
                error="bucket contains no files",
            )

        created_text = str(newest["$createdAt"])
        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        days = (now - created).days
        verdict = "STORE_ACTIVE" if days <= ACTIVE_WITHIN_DAYS else "STORE_IDLE"

        return CheckReport(
            verdict=verdict,
            endpoint=creds.endpoint,
            api_key_chars=len(creds.api_key),
            total_files=total,
            newest_file_name=str(newest.get("name")),
            newest_file_created=created_text[:19],
            newest_file_bytes=newest.get("sizeOriginal"),
            days_since_newest=days,
            collections_found=collections,
            real_collection_present=real_present,
        )

    def _discover_collections(
        self, creds: AppwriteCredentials, headers: Dict[str, str]
    ) -> tuple:
        """List the metadata database's collections; failure is unknown, not fatal."""
        try:
            doc = self._fetch(
                f"{creds.endpoint}/databases/{REAL_METADATA_DATABASE_ID}/collections",
                headers,
            )
            ids = [c["$id"] for c in doc.get("collections", [])]  # type: ignore[union-attr]
            return ids, REAL_PROD_FORECASTS_COLLECTION_ID in ids
        except Exception:  # noqa: BLE001 — discovery is auxiliary; unknown, honestly
            return None, None



def main(check: Optional[AppwriteStoreCheck] = None, now: Optional[datetime] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or AppwriteStoreCheck()).run(now=now)
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover — __main__ guard
