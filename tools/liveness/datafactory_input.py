"""Liveness check for the datafactory input store (the remote zarr).

Answers, with raw facts: is the input store reachable, how far does its
OBSERVED data coverage reach (`last_valid_month_id`, read from the live
store's .zattrs), and does that cover what this repo's canonical partitions
require (`meta/partitions.json`, max test-window end)?

Usage:
    python -m tools.liveness.datafactory_input   # exit 0 fresh / 1 stale / 2 unreachable

This automates the register C-96 tripwire: partition windows that outrun
observed coverage mean models are evaluated against zero-fill, warned about
only in run logs. The requirement is DERIVED from meta/partitions.json at
run time — never hardcoded — so every partition bump re-arms the check
automatically (ADR-013 spirit).

Context receipts: live value 558 (2026-07-06/19); current requirement 552;
store host carries HTTP basic auth via ~/.netrc (presence reported as a
fact, value never read).

Design (house rules, mirrors tools/liveness/old_api.py): injected reader
callables (DIP), lazy datafactory import inside the default reader only,
no import-time side effects (C-93), month math reused from
tools.partitions.domain, zero new dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tools.partitions.domain import month_id_to_date

from tools.liveness.report import exit_code_for, render_facts

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def required_month_id_from_partitions(repo_root: Path = _DEFAULT_REPO_ROOT) -> int:
    """Max test-window end across the canonical partitions (the coverage bar)."""
    canonical = json.loads((repo_root / "meta" / "partitions.json").read_text())
    return max(
        canonical["calibration"]["test"][1],
        canonical["validation"]["test"][1],
    )


@dataclass(frozen=True)
class CheckReport:
    """Raw facts about the datafactory input store — no narration."""

    verdict: str  # INPUT_FRESH | INPUT_STALE | SKIP_NO_PACKAGE | UNREACHABLE
    netrc_present: Optional[bool] = None
    last_valid_month_id: Optional[int] = None
    last_valid_date: Optional[str] = None
    required_month_id: Optional[int] = None
    required_date: Optional[str] = None
    margin_months: Optional[int] = None
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value``."""
    facts = [
        ("surface", "datafactory_input"),
        ("verdict", report.verdict),
        ("netrc_present", report.netrc_present),
        ("last_valid_month_id", report.last_valid_month_id),
        ("last_valid_date", report.last_valid_date),
        ("required_month_id", report.required_month_id),
        ("required_date", report.required_date),
        ("margin_months", report.margin_months),
        ("error", report.error),
    ]
    return render_facts(facts)


class DatafactoryInputCheck:
    """Coverage-vs-requirement check for the input store (DIP seams throughout)."""

    def __init__(
        self,
        read_last_valid_month_id: Optional[Callable[[], int]] = None,
        netrc_probe: Optional[Callable[[], bool]] = None,
        required_month_id: Optional[int] = None,
    ) -> None:
        self._read_last_valid = read_last_valid_month_id or self._read_live_last_valid
        self._netrc_probe = netrc_probe or self._netrc_has_store_host
        self._required_month_id = required_month_id

    def run(self) -> CheckReport:
        required = (
            self._required_month_id
            if self._required_month_id is not None
            else required_month_id_from_partitions()
        )
        netrc_present = self._safe_netrc_probe()

        try:
            last_valid = int(self._read_last_valid())
        except (ImportError, ModuleNotFoundError) as exc:
            # Truthful skip, mirroring vpn_store (C-75/C-101): a machine
            # without datafactory_query is an environment fact, not an alarm.
            return CheckReport(
                verdict="SKIP_NO_PACKAGE",
                netrc_present=netrc_present,
                required_month_id=required,
                required_date=month_id_to_date(required),
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — any failure is the UNREACHABLE fact
            return CheckReport(
                verdict="UNREACHABLE",
                netrc_present=netrc_present,
                required_month_id=required,
                required_date=month_id_to_date(required),
                error=f"{type(exc).__name__}: {exc}",
            )

        margin = last_valid - required
        verdict = "INPUT_FRESH" if margin >= 0 else "INPUT_STALE"
        return CheckReport(
            verdict=verdict,
            netrc_present=netrc_present,
            last_valid_month_id=last_valid,
            last_valid_date=month_id_to_date(last_valid),
            required_month_id=required,
            required_date=month_id_to_date(required),
            margin_months=margin,
        )

    def _safe_netrc_probe(self) -> Optional[bool]:
        try:
            return bool(self._netrc_probe())
        except Exception:  # noqa: BLE001 — the hint must never sink the check
            return None

    @staticmethod
    def _read_live_last_valid() -> int:
        """Default reader: the datafactory's own live .zattrs accessor."""
        from datafactory_query.defaults import get_last_valid_month_id

        return int(get_last_valid_month_id())

    @staticmethod
    def _netrc_has_store_host() -> bool:
        """Does ~/.netrc mention the store host? (presence only; never values)."""
        import netrc
        from urllib.parse import urlparse

        from datafactory_query.defaults import DEFAULT_REMOTE

        host = urlparse(DEFAULT_REMOTE.zarr_url).hostname
        credentials = netrc.netrc()
        return host is not None and credentials.authenticators(host) is not None




def main(check: Optional[DatafactoryInputCheck] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or DatafactoryInputCheck()).run()
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
