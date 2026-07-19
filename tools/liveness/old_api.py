"""Liveness check for the old public API (api.viewsforecasting.org).

Answers, with raw facts: is the API reachable, what is the newest published
fatalities run, how fresh is it, and does it actually serve rows?

Usage:
    python -m tools.liveness.old_api      # exit 0 fresh / 1 stale-or-not-serving / 2 unreachable

THE NAMING CONVENTION (encoded once, colleague-confirmable — issue #239):
    Runs are named ``fatalities{generation}_{yyyy}_{mm}_t{seq}`` where
    ``{yyyy}_{mm}`` is the DATA-CUTOFF month, not the execution month;
    execution/publication happens ~1 month after the cutoff.
    Evidence (2026-07-19 forensics): the wandb pink_ponyclub run executed
    2026-06-29 trained on data through month_id 557 (May 2026) and was
    published as ``fatalities003_2026_05_t01``. Misreading this convention
    as execution-month produced a false "production stalled" alarm; this
    module exists so that mistake can never be repeated.

API facts (captured live 2026-07-19):
    GET /                      -> {"runs": [...]}  (NOT chronologically sorted)
    GET /{run}/cm?month={id}&pagesize=N -> {"data": [rows...]}
    unknown run                -> HTTP 422

Design: injected fetch callable (DIP; mirrors
reconciliation/viewser_country_mapping_provider.py), pure parsing functions,
no import-time side effects (C-93). Month math is reused from
tools.partitions.domain — not reimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, List, Optional, Tuple

from tools.partitions.domain import date_to_month_id, month_id_to_date

BASE_URL = "https://api.viewsforecasting.org"

# Freshness budget: cadence is one run per data-month, published ~1 month
# after cutoff (PUBLICATION_LAG_MONTHS) + 1 month of in-flight grace.
PUBLICATION_LAG_MONTHS = 1
GRACE_MONTHS = 1
FRESHNESS_BUDGET_MONTHS = PUBLICATION_LAG_MONTHS + GRACE_MONTHS

_FETCH_TIMEOUT_SECONDS = 20
_SERVING_SAMPLE_PAGESIZE = 2

_RUN_NAME_PATTERN = re.compile(r"^fatalities(\d+)_(\d{4})_(\d{2})_t(\d+)$")

# (generation, year, month, sequence)
RunId = Tuple[int, int, int, int]

FetchJson = Callable[[str], object]


def parse_run_name(name: str) -> Optional[RunId]:
    """Parse a fatalities run name; None for any other run family."""
    match = _RUN_NAME_PATTERN.match(name)
    if match is None:
        return None
    generation, year, month, seq = (int(g) for g in match.groups())
    return (generation, year, month, seq)


def latest_fatalities_run(run_names: List[str]) -> Optional[str]:
    """Chronologically newest fatalities run (by cutoff, then generation/seq).

    The API's list is NOT sorted chronologically — its alphabetical tail is
    ``r_2021_12_01`` — so naive last-element selection is wrong (the pinned
    bug from the 2026-07-19 forensics).
    """
    best: Optional[Tuple[Tuple[int, int, int, int], str]] = None
    for name in run_names:
        parsed = parse_run_name(name)
        if parsed is None:
            continue
        generation, year, month, seq = parsed
        sort_key = (year, month, generation, seq)
        if best is None or sort_key > best[0]:
            best = (sort_key, name)
    return None if best is None else best[1]


@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the old API — no narration."""

    url: str
    verdict: str  # LIVE_FRESH | LIVE_STALE | LIVE_NOT_SERVING | UNREACHABLE
    run_count: Optional[int] = None
    latest_run: Optional[str] = None
    data_cutoff_month_id: Optional[int] = None
    data_cutoff_date: Optional[str] = None
    now_month_id: Optional[int] = None
    months_behind: Optional[int] = None
    serving_rows_sampled: Optional[int] = None
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value`` — machine- and human-scannable."""
    facts = [
        ("surface", "old_api"),
        ("url", report.url),
        ("verdict", report.verdict),
        ("run_count", report.run_count),
        ("latest_run", report.latest_run),
        ("data_cutoff_month_id", report.data_cutoff_month_id),
        ("data_cutoff_date", report.data_cutoff_date),
        ("now_month_id", report.now_month_id),
        ("months_behind", report.months_behind),
        ("freshness_budget_months", FRESHNESS_BUDGET_MONTHS),
        ("serving_rows_sampled", report.serving_rows_sampled),
        ("error", report.error),
    ]
    return "\n".join(f"{key}: {value}" for key, value in facts if value is not None)


class OldApiCheck:
    """Liveness check for the old public API (DIP: fetch is injectable)."""

    def __init__(self, fetch: Optional[FetchJson] = None) -> None:
        self._fetch = fetch or self._fetch_json

    def run(self, now_month_id: Optional[int] = None) -> CheckReport:
        """Fetch the run list, pick the newest fatalities run, judge freshness,
        and confirm the run serves rows. ``now_month_id`` is injectable for
        deterministic tests; defaults to the current calendar month."""
        if now_month_id is None:
            today = date.today()
            now_month_id = date_to_month_id(today.year, today.month)

        try:
            listing = self._fetch(f"{BASE_URL}/")
            run_names = list(listing["runs"])  # type: ignore[index]
        except Exception as exc:  # noqa: BLE001 — any failure is the UNREACHABLE fact
            return CheckReport(
                url=BASE_URL,
                verdict="UNREACHABLE",
                now_month_id=now_month_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        latest = latest_fatalities_run(run_names)
        if latest is None:
            return CheckReport(
                url=BASE_URL,
                verdict="LIVE_NOT_SERVING",
                run_count=len(run_names),
                now_month_id=now_month_id,
                error="no fatalities runs in listing",
            )

        _, year, month, _ = parse_run_name(latest)  # type: ignore[misc]
        cutoff_id = date_to_month_id(year, month)
        months_behind = now_month_id - cutoff_id

        rows_sampled, serving_error = self._sample_serving_rows(latest, cutoff_id)

        if rows_sampled == 0:
            verdict = "LIVE_NOT_SERVING"
        elif months_behind <= FRESHNESS_BUDGET_MONTHS:
            verdict = "LIVE_FRESH"
        else:
            verdict = "LIVE_STALE"

        return CheckReport(
            url=BASE_URL,
            verdict=verdict,
            run_count=len(run_names),
            latest_run=latest,
            data_cutoff_month_id=cutoff_id,
            data_cutoff_date=month_id_to_date(cutoff_id),
            now_month_id=now_month_id,
            months_behind=months_behind,
            serving_rows_sampled=rows_sampled,
            error=serving_error,
        )

    def _sample_serving_rows(
        self, run_name: str, cutoff_id: int
    ) -> Tuple[int, Optional[str]]:
        """Sample rows from the run's first forecast month (cutoff + 1)."""
        url = (
            f"{BASE_URL}/{run_name}/cm"
            f"?month={cutoff_id + 1}&pagesize={_SERVING_SAMPLE_PAGESIZE}"
        )
        try:
            payload = self._fetch(url)
            rows = payload.get("data", [])  # type: ignore[union-attr]
            return len(rows), None
        except Exception as exc:  # noqa: BLE001 — a serving failure is a fact, not a crash
            return 0, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _fetch_json(url: str) -> object:
        """Default fetch: stdlib urllib, lazy import, explicit timeout."""
        import json
        import urllib.request

        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            return json.load(response)


_EXIT_CODE_BY_VERDICT = {
    "LIVE_FRESH": 0,
    "LIVE_STALE": 1,
    "LIVE_NOT_SERVING": 1,
    "UNREACHABLE": 2,
}


def main(
    fetch: Optional[FetchJson] = None, now_month_id: Optional[int] = None
) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = OldApiCheck(fetch=fetch).run(now_month_id=now_month_id)
    print(render(report))
    return _EXIT_CODE_BY_VERDICT[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
