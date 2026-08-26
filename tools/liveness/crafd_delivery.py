"""Liveness check for the CRAF'd delivery bucket (crafd_bucket).

Answers, with raw facts: when did CRAF'd last actually receive anything — per delivery
stream? Same two-stream shape as the FAO surface, verified live 2026-08-24 rather than
assumed: 108 shards + 1 sidecar + 1 `__manifest.json` + 1 `historical_dataset_*` = 111
files, identical naming conventions, same `rusty_bucket` source.

    <source>_forecasting_<ts>__manifest.json   the ADR-013 commit marker, written LAST
    historical_dataset_*.parquet               the historical actuals delivery

**Why this exists.** views-models#399 armed the CRAF'd delivery on 2026-08-14. Until this
module there was no instrument for it at all: the only way to answer "did CRAF'd get its
forecast?" was to list the bucket by hand, which is exactly what we did on the day. A live
partner delivery with no monitor is the shape of #320 — "FAO forecast delivery has been
stalled for 145 days and nothing detected it" — waiting to happen to the second partner.

**Why it is a near-copy of `unfao_delivery`, deliberately.** Measured: 14 differing lines
in 269 once the consumer name is normalised. That is a clone, and it is a considered one —
`tools/liveness/README.md` records that shared code here was extracted only after SIX
surfaces duplicated it, and this is the second partner surface. The duplication is
registered with a named trigger (C-141) rather than left to be rediscovered.

**It does NOT inherit C-102.** The forecast stream is judged on the manifest suffix from
the start. Had this module been written before #411, it would have copied a matcher that
reported NEVER_DELIVERED over a healthy delivery — which is the concrete reason S5 was
sequenced after S4.

Credentials, injected fetch/clock, truthful SKIP, redaction: as the sibling surfaces.
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

CRAFD_BUCKET_ID = "crafd_bucket"
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
BOUND_SOURCE = "deliveries/un_crafd.py"


def _load_declared_max_age_days() -> int:
    """The bound this delivery declares. Never a default — see the module docstring."""
    from deliveries.status import declared_max_age_days

    return declared_max_age_days("un_crafd")

@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the FAO delivery bucket — no narration."""

    verdict: str  # DELIVERING | DELIVERY_STALLED | UNREACHABLE | SKIP_NO_CREDENTIALS
    #              | CREDENTIALS_INCOMPLETE
    endpoint: Optional[str] = None
    bucket: str = CRAFD_BUCKET_ID
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
        ("surface", "crafd_delivery"),
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


class CrafdDeliveryCheck:
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

        base = f"{creds.endpoint}/storage/buckets/{CRAFD_BUCKET_ID}/files"
        try:
            # FIRST, and not merged into the listings below: the listing endpoint
            # answers a rejected key with 200/total=0, so an expired credential
            # reads as a partner bucket that has simply gone quiet — which is a
            # verdict this surface already has a name for. See
            # assert_bucket_reachable.
            assert_bucket_reachable(
                creds.endpoint, CRAFD_BUCKET_ID, headers, self._fetch
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



def main(check: Optional[CrafdDeliveryCheck] = None, now: Optional[datetime] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or CrafdDeliveryCheck()).run(now=now)
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover — __main__ guard
