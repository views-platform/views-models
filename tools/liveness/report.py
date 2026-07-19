"""Shared report utilities for the liveness checks (S7 extraction, issue #245).

Extracted ONLY after six checks demonstrably duplicated these pieces
(WET-before-DRY, epic #238): the one-fact-per-line renderer and the
verdict -> exit-code classification.

Exit-code contract (uniform across every surface):
    0 — healthy, or a truthful SKIP (missing creds/package/VPN is a fact
        about the environment, not a failure of the surface)
    1 — reachable but stale/idle/not-serving (attention needed)
    2 — unreachable (the surface cannot be observed at all)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

EXIT_CODE_BY_VERDICT = {
    # healthy
    "LIVE_FRESH": 0,
    "INPUT_FRESH": 0,
    "STORE_ACTIVE": 0,
    "STORE_FRESH": 0,
    "EXECUTION_CURRENT": 0,
    "DELIVERING": 0,
    # truthful skips
    "SKIP_NO_CREDENTIALS": 0,
    "SKIP_NO_PACKAGE": 0,
    "VPN_REQUIRED": 0,
    # attention
    "LIVE_STALE": 1,
    "LIVE_NOT_SERVING": 1,
    "INPUT_STALE": 1,
    "STORE_IDLE": 1,
    "STORE_STALE": 1,
    "DELIVERY_STALLED": 1,
    "EXECUTION_STALE": 1,
    # unobservable
    "UNREACHABLE": 2,
}


def one_line(value: object) -> str:
    """Collapse a fact value to one line — embedded newlines become ``\\n``.

    The one-fact-per-line contract must hold even for error strings that
    arrive multi-line (e.g. sqlalchemy OperationalError, C-101/P1)."""
    return str(value).replace("\r", "").replace("\n", "\\n")


def render_facts(facts: Iterable[Tuple[str, object]]) -> str:
    """One ``key: value`` line per fact; None values are omitted."""
    return "\n".join(
        f"{key}: {one_line(value)}" for key, value in facts if value is not None
    )


def exit_code_for(verdict: str) -> int:
    """Map a check verdict to the uniform exit code; unknown verdicts raise."""
    return EXIT_CODE_BY_VERDICT[verdict]


def worst_exit(codes: List[int]) -> int:
    """The aggregate exit code: the worst of the parts (empty -> 0)."""
    return max(codes, default=0)
