"""Bound a third-party call that has no timeout of its own, so a test cannot hang.

WHY THIS EXISTS
---------------
Four tests in this suite call `viewser`, which **cannot be bounded from the call site**:

- `viewser/commands/queryset/operations.py:255-311` polls with
  `requests.get(url, stream=True)` — no `timeout=` — inside a `while` loop;
- `operations.py:32` declares `max_retries: int = sys.maxsize`, and the module-level
  singleton the tests reach (`models/queryset.py:10-12`) does not override it.

So any *persistent* failure retries forever at 5s intervals. Measured 2026-08-23: the
backend was returning **502 Bad Gateway** and a single `.build()` was still going after
150 seconds. Bare `pytest` therefore never returns, which silently defeats the repo's own
ship-it gate — lint, test, commit — because the test step neither passes nor fails.

The original diagnosis (views-models#409) said "off-VPN". That was too narrow: off-VPN is
one trigger, a 502 is another, and the tests cannot tell them apart. **Anything persistent
hangs**, which is why a reachability preflight does not work — a TCP connect to the
configured host succeeds in 0.02s while the fetch never returns.

WHY `signal`
------------
`socket.setdefaulttimeout` bounds each request but not the loop around it, so it converts
an infinite hang into `sys.maxsize * 5s`. `pytest-timeout` is not installed, and this repo
has nowhere to declare a test dependency — `pyproject.toml` holds only pytest markers
(ADR-019). A SIGALRM deadline is stdlib, needs no dependency, and interrupts the call
whatever it is blocked on — provided it re-arms, see the interval below. Verified against the real `viewser` call before this was
written: the deadline fired at 20.0s on a 502-retry loop that had not returned in 150s.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not decide the test's verdict. A call that exceeds its deadline is **not a
failure** — the backend being down is not this repository's defect — so callers pair this
with `pytest.skip`, matching the house idiom for live tests
(`tests/test_liveness_appwrite_store.py:337-350`). What it guarantees is only that the
suite finishes and says why.

POSIX main thread only. That is asserted rather than silently degraded: falling back to an
unbounded call would reinstate the hang this exists to prevent, and do it invisibly.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager


#: Default bound for a live `viewser` call. One home, because three call sites using the
#: same number with three different rationales is a number that drifts. Long enough that a
#: healthy fetch finishes; short enough that a suite run does not stall. It is provisional:
#: the backend was returning 502 when this was chosen, so a *healthy* fetch time could not
#: be measured. Raise it deliberately if a real integration run needs longer.
VIEWSER_DEADLINE_SECONDS = 90


class DeadlineExceeded(Exception):
    """A bounded call did not return in time. Carries the bound, for the skip message."""

    def __init__(self, seconds: float, what: str):
        self.seconds = seconds
        self.what = what
        super().__init__(f"{what} did not return within {seconds:g}s")


@contextmanager
def deadline(seconds: float, what: str):
    """Raise `DeadlineExceeded` if the block has not finished within `seconds`.

    Restores the previous SIGALRM handler and cancels the timer on every exit path,
    including when the block raises something else — a leaked timer would fire during an
    unrelated later test and be attributed to it.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "live_deadline.deadline() needs the main thread — signal handlers cannot be "
            "installed elsewhere. Bounding was requested and cannot be provided, and "
            "running unbounded is the hang this module exists to prevent."
        )
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover — POSIX only, and CI is Linux
        raise RuntimeError(
            "live_deadline.deadline() needs SIGALRM (POSIX). Refusing to run the call "
            "unbounded rather than silently reinstating an unbounded network wait."
        )

    def _fire(signum, frame):
        raise DeadlineExceeded(seconds, what)

    previous = signal.signal(signal.SIGALRM, _fire)
    # REPEATING, not one-shot. viewser's fetch loop wraps `pd.read_parquet` in a bare
    # `except:` (operations.py, inside `while not (succeeded or failed)`), which
    # catches DeadlineExceeded, bumps `retries`, sleeps, and continues. A one-shot
    # alarm swallowed there never fires again and the bound silently evaporates —
    # leaving a hang that now *looks* protected. Re-arming every `seconds` means a
    # swallowed alarm is retried until it escapes. The `finally` cancels it.
    signal.setitimer(signal.ITIMER_REAL, seconds, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
