"""Liveness check for monthly execution, via wandb run history.

Answers, with raw facts: "did the team compute this cycle?" — per monthly
ensemble, when did its latest finished forecasting run happen, and what
data-cutoff month did it train to?

Usage:
    python -m tools.liveness.wandb_execution   # exit 0 current/skip / 1 stale / 2 unreachable

Receipts encoded here (2026-07-19 forensics): wandb entity is
``views_pipeline``; project naming is ``{name}_{run_type}`` (pipeline-core
``model.py:983``) — monthly forecasting runs live in ``{name}_forecasting``.
Each run's config records its forecasting train window; ``train[1]`` is the
data-cutoff month_id (the receipt that resolved the run-naming ambiguity:
the 2026-06-29 runs trained to month 557 = May, published as ``2026_05``).

The monthly ensemble list mirrors ``monthly_run.sh`` — hand-encoded rather
than parsed from bash (fragile); update BOTH when the roster changes.

Design (house rules): injected latest-run client + netrc probe + clock
(DIP), lazy wandb import inside the default client only, no import-time
side effects (C-93), zero new dependencies (wandb is already installed),
truthful SKIP without credentials (C-75).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from tools.liveness.report import exit_code_for, one_line

WANDB_ENTITY = "views_pipeline"

# Mirrors monthly_run.sh (the hand-run production list). Keep in sync by hand.
MONTHLY_ENSEMBLES = ("pink_ponyclub", "skinny_love", "rude_boy", "first_love")

# Monthly cadence + slack: an ensemble is "computed this cycle" if its latest
# FINISHED forecasting run is at most this many days old.
CYCLE_BUDGET_DAYS = 40

# The injected client: project name -> facts dict or None (project/run absent).
# Facts keys: run_name, created_at (ISO), state, train_end_month_id.
LatestRunClient = Callable[[str], Optional[dict]]


@dataclass(frozen=True)
class EnsembleRun:
    """Raw facts for one monthly ensemble's latest forecasting run."""

    ensemble: str
    project: str
    verdict: str  # COMPUTED | NOT_COMPUTED | NEVER_RUN
    run_name: Optional[str] = None
    created_at: Optional[str] = None
    state: Optional[str] = None
    train_end_month_id: Optional[int] = None
    days_since: Optional[int] = None


@dataclass(frozen=True)
class CheckReport:
    """Raw facts about monthly execution — no narration."""

    verdict: str  # EXECUTION_CURRENT | EXECUTION_STALE | UNREACHABLE | SKIP_NO_CREDENTIALS
    entity: str = WANDB_ENTITY
    netrc_present: Optional[bool] = None
    ensembles: Tuple[EnsembleRun, ...] = ()
    error: Optional[str] = None


def render(report: CheckReport) -> str:
    """One fact per line, ``key: value``; per-ensemble blocks are prefixed."""
    lines = [
        "surface: wandb_execution",
        f"verdict: {report.verdict}",
        f"entity: {report.entity}",
        f"cycle_budget_days: {CYCLE_BUDGET_DAYS}",
    ]
    if report.netrc_present is not None:
        lines.append(f"netrc_present: {report.netrc_present}")
    for run in report.ensembles:
        prefix = run.ensemble
        lines.append(f"{prefix}.verdict: {run.verdict}")
        if run.run_name is not None:
            lines.append(f"{prefix}.run: {run.run_name}")
        if run.created_at is not None:
            lines.append(f"{prefix}.created_at: {run.created_at}")
        if run.state is not None:
            lines.append(f"{prefix}.state: {run.state}")
        if run.train_end_month_id is not None:
            lines.append(f"{prefix}.train_end_month_id: {run.train_end_month_id}")
        if run.days_since is not None:
            lines.append(f"{prefix}.days_since: {run.days_since}")
    if report.error is not None:
        lines.append(f"error: {one_line(report.error)}")
    return "\n".join(lines)


class WandbExecutionCheck:
    """Per-ensemble execution recency via wandb (all seams injected)."""

    def __init__(
        self,
        latest_run: Optional[LatestRunClient] = None,
        netrc_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._latest_run = latest_run or self._latest_run_via_wandb
        self._netrc_probe = netrc_probe or self._netrc_has_wandb_host

    def run(self, now: Optional[datetime] = None) -> CheckReport:
        netrc_present = self._safe_netrc_probe()
        if netrc_present is False:
            return CheckReport(
                verdict="SKIP_NO_CREDENTIALS",
                netrc_present=False,
                error="no api.wandb.ai entry in ~/.netrc",
            )
        now = now or datetime.now(timezone.utc)

        runs = []
        failures = []
        for ensemble in MONTHLY_ENSEMBLES:
            project = f"{ensemble}_forecasting"
            try:
                facts = self._latest_run(project)
                # _judge inside the try: a malformed run (e.g. created_at=None)
                # is a per-ensemble failure fact, never a check crash (C-101/P4).
                runs.append(self._judge(ensemble, project, facts, now))
            except Exception as exc:  # noqa: BLE001 — collected; all-fail => UNREACHABLE
                failures.append(f"{project}: {type(exc).__name__}: {exc}")
                continue

        if not runs:
            return CheckReport(
                verdict="UNREACHABLE",
                netrc_present=netrc_present,
                error="; ".join(failures) or "no projects reachable",
            )

        overall = (
            "EXECUTION_CURRENT"
            if len(runs) == len(MONTHLY_ENSEMBLES)
            and all(r.verdict == "COMPUTED" for r in runs)
            else "EXECUTION_STALE"
        )
        return CheckReport(
            verdict=overall,
            netrc_present=netrc_present,
            ensembles=tuple(runs),
            error="; ".join(failures) if failures else None,
        )

    @staticmethod
    def _judge(
        ensemble: str, project: str, facts: Optional[dict], now: datetime
    ) -> EnsembleRun:
        if facts is None:
            return EnsembleRun(ensemble=ensemble, project=project, verdict="NEVER_RUN")
        created_text = str(facts.get("created_at"))
        created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days = (now - created).days
        finished = facts.get("state") == "finished"
        verdict = "COMPUTED" if finished and days <= CYCLE_BUDGET_DAYS else "NOT_COMPUTED"
        return EnsembleRun(
            ensemble=ensemble,
            project=project,
            verdict=verdict,
            run_name=facts.get("run_name"),
            created_at=created_text[:19],
            state=facts.get("state"),
            train_end_month_id=facts.get("train_end_month_id"),
            days_since=days,
        )

    def _safe_netrc_probe(self) -> Optional[bool]:
        try:
            return bool(self._netrc_probe())
        except Exception:  # noqa: BLE001 — the hint must never sink the check
            return None

    @staticmethod
    def _netrc_has_wandb_host() -> bool:
        import netrc

        return netrc.netrc().authenticators("api.wandb.ai") is not None

    @staticmethod
    def _latest_run_via_wandb(project: str) -> Optional[dict]:
        """Default client: wandb public API, lazy import, newest run only."""
        import wandb

        api = wandb.Api(timeout=25)
        try:
            runs = api.runs(f"{WANDB_ENTITY}/{project}", order="-created_at", per_page=1)
            run = next(iter(runs), None)
        except Exception as exc:
            if "Could not find project" in str(exc):
                return None
            raise
        if run is None:
            return None
        train = (run.config.get("forecasting") or {}).get("train") or (None, None)
        return {
            "run_name": run.name,
            "created_at": run.created_at,
            "state": run.state,
            "train_end_month_id": train[1],
        }




def main(check: Optional[WandbExecutionCheck] = None, now: Optional[datetime] = None) -> int:
    """Run the check, print raw facts, return the exit code."""
    report = (check or WandbExecutionCheck()).run(now=now)
    # Classify BEFORE printing: an unregistered verdict must fail loud
    # without emitting a half-block the runner would then contradict (C-101/P7).
    code = exit_code_for(report.verdict)
    print(render(report))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
