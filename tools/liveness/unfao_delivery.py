"""Liveness check for the FAO delivery bucket (unfao_bucket).

Answers, with raw facts: when did FAO last actually receive anything —
per delivery stream? The bucket carries two streams (verified live
2026-07-19; register context: real monthly deliveries ran Jan–Mar 2026,
then stalled unnoticed):

    forecast_dataset_*.parquet    (~191MB — the forecast delivery)
    historical_dataset_*.parquet  (~20MB  — the historical actuals delivery)

Usage:
    python -m tools.liveness.unfao_delivery   # exit 0 delivering/skip / 1 stalled / 2 unreachable

Credentials are REUSED from tools.liveness.appwrite_store (same Appwrite
project + .env; S7 will home this in a shared credentials module — noted,
deliberate). Design mirrors the sibling checks: injected fetch/credentials/
clock (DIP), lazy stdlib urllib, no import-time side effects (C-93),
secrets never rendered, truthful SKIP without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from tools.liveness.appwrite_api import (
    AppwriteCredentials,
    FetchJson,
    fetch_json,
    newest_first_query,
    resolve_credentials,
    stream_newest_query,
)
from tools.liveness.report import exit_code_for, render_facts

UNFAO_BUCKET_ID = "unfao_bucket"
FORECAST_PREFIX = "forecast_dataset_"
HISTORICAL_PREFIX = "historical_dataset_"

# Monthly delivery cadence: a stream is "delivering" if something landed
# within ~1.5 cycles.
DELIVERING_WITHIN_DAYS = 45

@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the FAO delivery bucket — no narration."""

    verdict: str  # DELIVERING | DELIVERY_STALLED | UNREACHABLE | SKIP_NO_CREDENTIALS
    endpoint: Optional[str] = None
    bucket: str = UNFAO_BUCKET_ID
    total_files: Optional[int] = None
    forecast_verdict: Optional[str] = None  # DELIVERING | STALLED | NEVER_DELIVERED
    forecast_newest_name: Optional[str] = None
    forecast_newest_created: Optional[str] = None
    forecast_newest_bytes: Optional[int] = None
    forecast_days_since: Optional[int] = None
    historical_verdict: Optional[str] = None
    historical_newest_name: Optional[str] = None
    historical_newest_created: Optional[str] = None
    historical_newest_bytes: Optional[int] = None
    historical_days_since: Optional[int] = None
    other_files: Optional[int] = None
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value``; key material never appears."""
    facts = [
        ("surface", "unfao_delivery"),
        ("verdict", report.verdict),
        ("endpoint", report.endpoint),
        ("bucket", report.bucket),
        ("total_files", report.total_files),
        ("delivering_within_days", DELIVERING_WITHIN_DAYS),
        ("forecast_verdict", report.forecast_verdict),
        ("forecast_newest_name", report.forecast_newest_name),
        ("forecast_newest_created", report.forecast_newest_created),
        ("forecast_newest_bytes", report.forecast_newest_bytes),
        ("forecast_days_since", report.forecast_days_since),
        ("historical_verdict", report.historical_verdict),
        ("historical_newest_name", report.historical_newest_name),
        ("historical_newest_created", report.historical_newest_created),
        ("historical_newest_bytes", report.historical_newest_bytes),
        ("historical_days_since", report.historical_days_since),
        ("other_files", report.other_files),
        ("error", report.error),
    ]
    return render_facts(facts)


def _newest(files: list) -> Optional[dict]:
    return max(files, key=lambda f: f.get("$createdAt", ""), default=None)


class UnfaoDeliveryCheck:
    """Per-stream freshness of the FAO delivery bucket (all seams injected)."""

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
            return CheckReport(
                verdict="SKIP_NO_CREDENTIALS",
                error="no Appwrite credentials in env or known .env files",
            )
        now = now or datetime.now(timezone.utc)
        creds = self.credentials
        headers = {
            "X-Appwrite-Project": creds.project_id,
            "X-Appwrite-Key": creds.api_key,
        }

        base = f"{creds.endpoint}/storage/buckets/{UNFAO_BUCKET_ID}/files"
        try:
            # Per-stream server-side newest (startsWith + orderDesc + limit):
            # immune to Appwrite's 25-per-page default (the 2026-07-19
            # pagination bug found in the sibling check).
            overall = self._fetch(f"{base}?{newest_first_query(limit=1)}", headers)
            total = int(overall.get("total", 0))  # type: ignore[union-attr]
            forecast_doc = self._fetch(
                f"{base}?{stream_newest_query(FORECAST_PREFIX)}", headers
            )
            historical_doc = self._fetch(
                f"{base}?{stream_newest_query(HISTORICAL_PREFIX)}", headers
            )
        except Exception as exc:  # noqa: BLE001 — any storage failure is the fact
            return CheckReport(
                verdict="UNREACHABLE",
                endpoint=creds.endpoint,
                error=f"{type(exc).__name__}: {exc}",
            )

        forecast_files = list(forecast_doc.get("files", []))  # type: ignore[union-attr]
        historical_files = list(historical_doc.get("files", []))  # type: ignore[union-attr]
        forecast_total = int(forecast_doc.get("total", 0))  # type: ignore[union-attr]
        historical_total = int(historical_doc.get("total", 0))  # type: ignore[union-attr]
        other = total - forecast_total - historical_total

        f_verdict, f_facts = self._stream_verdict(forecast_files, now)
        h_verdict, h_facts = self._stream_verdict(historical_files, now)

        overall = (
            "DELIVERING"
            if f_verdict == "DELIVERING" and h_verdict == "DELIVERING"
            else "DELIVERY_STALLED"
        )

        return CheckReport(
            verdict=overall,
            endpoint=creds.endpoint,
            total_files=total,
            forecast_verdict=f_verdict,
            forecast_newest_name=f_facts[0],
            forecast_newest_created=f_facts[1],
            forecast_newest_bytes=f_facts[2],
            forecast_days_since=f_facts[3],
            historical_verdict=h_verdict,
            historical_newest_name=h_facts[0],
            historical_newest_created=h_facts[1],
            historical_newest_bytes=h_facts[2],
            historical_days_since=h_facts[3],
            other_files=other,
        )

    @staticmethod
    def _stream_verdict(
        files: list, now: datetime
    ) -> Tuple[str, Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]]:
        newest = _newest(files)
        if newest is None:
            return "NEVER_DELIVERED", (None, None, None, None)
        created_text = str(newest["$createdAt"])
        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        days = (now - created).days
        verdict = "DELIVERING" if days <= DELIVERING_WITHIN_DAYS else "STALLED"
        return verdict, (
            str(newest.get("name")),
            created_text[:19],
            newest.get("sizeOriginal"),
            days,
        )



def main(check: Optional[UnfaoDeliveryCheck] = None, now: Optional[datetime] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or UnfaoDeliveryCheck()).run(now=now)
    print(render(report))
    return exit_code_for(report.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
