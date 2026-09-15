"""Liveness check for the FAO delivery bucket (unfao_bucket).

Answers, with raw facts: when did FAO last actually receive anything —
per delivery stream? The bucket carries two streams:

    <source>_forecasting_<ts>__manifest.json   the ADR-013 commit marker, written LAST
                                               (after shards and sidecar) — so its
                                               presence means the run finished, and its
                                               age is the forecast stream's freshness
    historical_dataset_*.parquet               (~170MB — the historical actuals delivery)

Observed live 2026-08-24: 108 shards + 1 sidecar + 1 manifest + 1 historical = 111 files.

**The forecast stream was judged on the wrong name until 2026-08-24** (C-102, #411). This
module matched `forecast_dataset_*.parquet`, a per-file name from the pre-ADR-013 era that
nothing writes any more, so it reported `NEVER_DELIVERED` over 110 delivered files —
permanently, on every run, and indistinguishably from a real stall. That matters beyond
tidiness: #320 is "FAO forecast delivery has been stalled for 145 days and nothing detected
it", and this surface is the answer to it. It was blind in exactly the stream it exists to
watch, and the 110 shards sat in `other_files` reading as an incidental count.

Usage:
    python -m tools.liveness.unfao_delivery   # exit 0 delivering/skip / 1 stalled / 2 unreachable

Credentials are REUSED from tools.liveness.appwrite_api (same Appwrite project;
resolved from process env, else **this repo's own** `.env` — never another
repo's, per #298). Design mirrors the sibling checks: injected fetch/credentials/
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
    assert_bucket_reachable,
    fetch_json,
    newest_first_query,
    credential_gap_report,
    resolve_credentials,
    count_with_prefix_query,
    stream_newest_query,
    stream_newest_suffix_query,
)
from tools.liveness.report import exit_code_for, render_facts

UNFAO_BUCKET_ID = "unfao_bucket"
#: The ADR-013 commit marker. A forecast delivery writes its shards, then a sidecar, then
#: this — so the manifest's presence is what says the run finished, and its age is the
#: freshness of the forecast stream.
#:
#: Matched by SUFFIX deliberately. The full name is
#: `<source>_forecasting_<timestamp>__manifest.json`, and `<source>` is a model that can be
#: replaced. This surface previously matched `forecast_dataset_` — a legacy per-file name
#: nothing writes any more — and therefore reported NEVER_DELIVERED over 110 delivered
#: files, permanently and on every run (C-102, #411). Hardcoding `rusty_bucket_forecasting_`
#: instead would fix today and break the next time the source model changes;
#: `__manifest.json` is the wire convention rather than the producer.
FORECAST_MANIFEST_SUFFIX = "__manifest.json"
HISTORICAL_PREFIX = "historical_dataset_"

# Monthly delivery cadence: a stream is "delivering" if something landed
# within ~1.5 cycles.
#: Where the freshness bound comes from, reported so an operator can tell it is the
#: declared one rather than a number this tool invented.
BOUND_SOURCE = "deliveries/un_fao.py"


def _load_declared_max_age_days() -> int:
    """The bound this delivery declares. Never a default — see the module docstring."""
    from deliveries.status import declared_max_age_days

    return declared_max_age_days("un_fao")

@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the FAO delivery bucket — no narration."""

    verdict: str  # DELIVERING | DELIVERY_STALLED | UNREACHABLE | SKIP_NO_CREDENTIALS
    #              | CREDENTIALS_INCOMPLETE
    endpoint: Optional[str] = None
    bucket: str = UNFAO_BUCKET_ID
    total_files: Optional[int] = None
    max_age_days: Optional[int] = None      # the DECLARED bound this run classified against
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
        ("max_age_days", report.max_age_days),
        ("max_age_declared_in", BOUND_SOURCE),
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
        max_age_days: Optional[int] = None,
    ) -> None:
        self.credentials = (
            resolve_credentials() if credentials == "RESOLVE" else credentials
        )
        self._fetch = fetch or fetch_json
        # Resolved lazily in run(), not here: a construction that reads the
        # declaration would make an unreadable one fail before the credential
        # skip could report itself, turning a truthful SKIP into a crash.
        self._max_age_days = max_age_days

    def run(self, now: Optional[datetime] = None) -> CheckReport:
        if self.credentials is None:
            verdict, error = credential_gap_report()
            return CheckReport(verdict=verdict, error=error)
        now = now or datetime.now(timezone.utc)
        max_age_days = (
            self._max_age_days if self._max_age_days is not None
            else _load_declared_max_age_days()
        )
        creds = self.credentials
        headers = {
            "X-Appwrite-Project": creds.project_id,
            "X-Appwrite-Key": creds.api_key,
        }

        base = f"{creds.endpoint}/storage/buckets/{UNFAO_BUCKET_ID}/files"
        try:
            # FIRST, and not merged into the listings below: the listing endpoint
            # answers a rejected key with 200/total=0, so an expired credential
            # reads as a partner bucket that has simply gone quiet — which is a
            # verdict this surface already has a name for. See
            # assert_bucket_reachable.
            assert_bucket_reachable(
                creds.endpoint, UNFAO_BUCKET_ID, headers, self._fetch
            )
            # Per-stream server-side newest (startsWith + orderDesc + limit):
            # immune to Appwrite's 25-per-page default (the 2026-07-19
            # pagination bug found in the sibling check).
            overall = self._fetch(f"{base}?{newest_first_query(limit=1)}", headers)
            total = int(overall.get("total", 0))  # type: ignore[union-attr]
            forecast_doc = self._fetch(
                f"{base}?{stream_newest_suffix_query(FORECAST_MANIFEST_SUFFIX)}", headers
            )
            historical_doc = self._fetch(
                f"{base}?{stream_newest_query(HISTORICAL_PREFIX)}", headers
            )
            # The manifest identifies its own run, so the run's files can be counted
            # rather than landing in `other_files`. Without this the residual would read
            # 109 on a healthy bucket — one misleading number swapped for another.
            run_total = 0
            manifest_files = list(forecast_doc.get("files", []))  # type: ignore[union-attr]
            if manifest_files:
                stem = manifest_files[0]["name"][: -len(FORECAST_MANIFEST_SUFFIX)]
                run_doc = self._fetch(f"{base}?{count_with_prefix_query(stem)}", headers)
                run_total = int(run_doc.get("total", 0))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — any storage failure is the fact
            return CheckReport(
                verdict="UNREACHABLE",
                endpoint=creds.endpoint,
                error=f"{type(exc).__name__}: {exc}",
            )

        forecast_files = list(forecast_doc.get("files", []))  # type: ignore[union-attr]
        historical_files = list(historical_doc.get("files", []))  # type: ignore[union-attr]
        historical_total = int(historical_doc.get("total", 0))  # type: ignore[union-attr]
        # `run_total` counts the whole forecast run (shards + sidecar + manifest), not just
        # the manifest, so this stays a real residual: files belonging to neither stream.
        other = total - run_total - historical_total

        f_verdict, f_facts = self._stream_verdict(forecast_files, now, max_age_days)
        h_verdict, h_facts = self._stream_verdict(historical_files, now, max_age_days)

        overall = (
            "DELIVERING"
            if f_verdict == "DELIVERING" and h_verdict == "DELIVERING"
            else "DELIVERY_STALLED"
        )

        return CheckReport(
            verdict=overall,
            endpoint=creds.endpoint,
            total_files=total,
            max_age_days=max_age_days,
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
        files: list, now: datetime, max_age_days: int
    ) -> Tuple[str, Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]]:
        newest = _newest(files)
        if newest is None:
            return "NEVER_DELIVERED", (None, None, None, None)
        created_text = str(newest["$createdAt"])
        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        days = (now - created).days
        verdict = "DELIVERING" if days <= max_age_days else "STALLED"
        return verdict, (
            str(newest.get("name")),
            created_text[:19],
            newest.get("sizeOriginal"),
            days,
        )



def main(check: Optional[UnfaoDeliveryCheck] = None, now: Optional[datetime] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or UnfaoDeliveryCheck()).run(now=now)
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover — __main__ guard
