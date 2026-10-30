"""Liveness check for the VPN-only legacy prediction store (gjoll).

Answers, with raw facts: does the legacy store hold a fresh fatalities run —
i.e. are computed runs being uploaded, possibly awaiting public promotion?

Usage:
    python -m tools.liveness.vpn_store   # exit 0 fresh/vpn-required/skip / 1 stale / 2 unreachable

The store is Postgres on ``gjoll.muspelheim.local`` — PRIO-internal,
resolvable ONLY on the PRIO VPN. Off-VPN this check reports the truthful
verdict ``VPN_REQUIRED`` (never a false RED): the observation boundary that
caused the 2026-07-19 "who is lying?" episode is now an encoded, named
verdict instead of a trap.

Access is via ``views_forecasts.db_ops.ViewsMetadata`` (constructor connects;
``.get_runs()`` -> name/description/min_month/max_month). Historical receipt:
that store's Postgres schema is literally ``forecasts_metadata`` — the origin
of the phantom Appwrite collection ID that killed the June 2026 un_fao run
(the legacy schema name was copied into the new store's config).

Run-name parsing and the freshness budget are REUSED from
tools.liveness.old_api — one parser, one naming convention, everywhere.
Design (house rules): injected list_runs client + clock (DIP), lazy imports
in the default client only, no import-time side effects (C-93), zero new
dependencies, truthful skips (C-75).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional

from tools.liveness.old_api import (
    FRESHNESS_BUDGET_MONTHS,
    latest_fatalities_run,
    parse_run_name,
)
from tools.partitions.domain import date_to_month_id, month_id_to_date

from tools.liveness.report import exit_code_for, render_facts

STORE_HOST = "gjoll.muspelheim.local"
LEGACY_SCHEMA = "forecasts_metadata"  # the phantom-collection-ID origin (see docstring)

# The injected client: () -> list of run rows, each a dict with at least
# {"name": str, "max_month": int}. The default connects via views_forecasts.
ListRuns = Callable[[], List[dict]]

_HOST_RESOLUTION_MARKERS = (
    "could not translate host name",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
)


@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the legacy VPN store — no narration."""

    verdict: str  # STORE_FRESH | STORE_STALE | VPN_REQUIRED | SKIP_NO_PACKAGE | UNREACHABLE
    host: str = STORE_HOST
    schema: str = LEGACY_SCHEMA
    run_count: Optional[int] = None
    latest_run: Optional[str] = None
    data_cutoff_month_id: Optional[int] = None
    data_cutoff_date: Optional[str] = None
    latest_max_month: Optional[int] = None
    now_month_id: Optional[int] = None
    months_behind: Optional[int] = None
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value``."""
    facts = [
        ("surface", "vpn_store"),
        ("verdict", report.verdict),
        ("host", report.host),
        ("schema", report.schema),
        ("run_count", report.run_count),
        ("latest_run", report.latest_run),
        ("data_cutoff_month_id", report.data_cutoff_month_id),
        ("data_cutoff_date", report.data_cutoff_date),
        ("latest_max_month", report.latest_max_month),
        ("now_month_id", report.now_month_id),
        ("months_behind", report.months_behind),
        ("freshness_budget_months", FRESHNESS_BUDGET_MONTHS),
        ("error", report.error),
    ]
    return render_facts(facts)


class VpnStoreCheck:
    """Freshness of the legacy store's newest fatalities run (seams injected)."""

    def __init__(self, list_runs: Optional[ListRuns] = None) -> None:
        self._list_runs = list_runs or self._list_runs_via_views_forecasts

    def run(self, now_month_id: Optional[int] = None) -> CheckReport:
        if now_month_id is None:
            today = date.today()
            now_month_id = date_to_month_id(today.year, today.month)

        try:
            rows = list(self._list_runs())
        except (ImportError, ModuleNotFoundError) as exc:
            return CheckReport(
                verdict="SKIP_NO_PACKAGE",
                now_month_id=now_month_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — classified below, never a crash
            message = str(exc).lower()
            if any(marker in message for marker in _HOST_RESOLUTION_MARKERS):
                verdict = "VPN_REQUIRED"
            else:
                verdict = "UNREACHABLE"
            return CheckReport(
                verdict=verdict,
                now_month_id=now_month_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        names = [str(row.get("name", "")) for row in rows]
        latest = latest_fatalities_run(names)
        if latest is None:
            return CheckReport(
                verdict="STORE_STALE",
                run_count=len(rows),
                now_month_id=now_month_id,
                error="no fatalities runs in the store listing",
            )

        _, year, month, _ = parse_run_name(latest)  # type: ignore[misc]
        cutoff_id = date_to_month_id(year, month)
        months_behind = now_month_id - cutoff_id
        latest_row = next((r for r in rows if r.get("name") == latest), {})

        verdict = (
            "STORE_FRESH" if months_behind <= FRESHNESS_BUDGET_MONTHS else "STORE_STALE"
        )
        return CheckReport(
            verdict=verdict,
            run_count=len(rows),
            latest_run=latest,
            data_cutoff_month_id=cutoff_id,
            data_cutoff_date=month_id_to_date(cutoff_id),
            latest_max_month=latest_row.get("max_month"),
            now_month_id=now_month_id,
            months_behind=months_behind,
        )

    @staticmethod
    def _list_runs_via_views_forecasts() -> List[dict]:
        """Default client: the legacy store's own metadata API (lazy import;
        the constructor is the connection probe)."""
        from views_forecasts.db_ops import ViewsMetadata

        frame = ViewsMetadata().get_runs().reset_index()
        return frame[["name", "min_month", "max_month"]].to_dict("records")




def main(check: Optional[VpnStoreCheck] = None, now_month_id: Optional[int] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or VpnStoreCheck()).run(now_month_id=now_month_id)
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover — __main__ guard
