"""The liveness dashboard: every forecast surface, one command, raw facts.

Usage:
    python -m tools.liveness      # exit = worst verdict across all surfaces
                                  # (0 healthy/skips, 1 attention, 2 unreachable)

Runs each surface's check in sequence and prints its raw-facts block. A
crashing check is contained and reported as an UNREACHABLE-class fact —
one broken surface must never hide the others (epic #238, S7).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from tools.liveness import (
    appwrite_store,
    datafactory_input,
    old_api,
    unfao_delivery,
    vpn_store,
    wandb_execution,
)
from tools.liveness.report import worst_exit

SURFACES: Tuple[Tuple[str, Callable[[], int]], ...] = (
    ("old_api", old_api.main),
    ("datafactory_input", datafactory_input.main),
    ("appwrite_store", appwrite_store.main),
    ("unfao_delivery", unfao_delivery.main),
    ("wandb_execution", wandb_execution.main),
    ("vpn_store", vpn_store.main),
)

_RULE = "─" * 60


def run_all(surfaces: Optional[Tuple[Tuple[str, Callable[[], int]], ...]] = None) -> int:
    """Run every surface check; print each block; return the worst exit code."""
    codes: List[int] = []
    for name, run_main in surfaces if surfaces is not None else SURFACES:
        print(f"{_RULE}\n{name}\n{_RULE}")
        try:
            codes.append(int(run_main()))
        except Exception as exc:  # noqa: BLE001 — contain: one crash must not hide the rest
            print(f"surface: {name}\nverdict: UNREACHABLE\nerror: {type(exc).__name__}: {exc}")
            codes.append(2)
        print()
    print(f"{_RULE}\nworst_exit: {worst_exit(codes)}")
    return worst_exit(codes)


if __name__ == "__main__":
    raise SystemExit(run_all())  # pragma: no cover — __main__ guard
