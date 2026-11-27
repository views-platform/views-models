"""Guards on the bound that keeps a hanging network call from taking the suite with it.

`tests/live_deadline.py` exists because four tests call `viewser`, which retries a
persistent failure `sys.maxsize` times at 5s intervals and therefore never returns
(#409). The bound is a SIGALRM deadline — the only mechanism that works when the call
being bounded is opaque.

**Why this file exists rather than trusting the four tests to exercise it.** Those four
skip when the backend is unavailable, which is most of the time, so they prove nothing
about the bound. Worse, the two failure modes here are silent: a handler that is not
restored changes how a *later* unrelated test reacts to SIGALRM, and a timer that is not
cancelled fires during a *later* unrelated test and is attributed to it. Neither shows up
as a failure in the file that caused it.

Every test here is offline and finishes in under a second.
"""

import signal
import threading
import time

import pytest

from tests.live_deadline import DeadlineExceeded, deadline

pytestmark = [pytest.mark.green]


class TestTheBoundActuallyBounds:
    def test_a_call_that_overruns_is_interrupted(self):
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded) as caught:
            with deadline(0.25, "a call that never returns"):
                time.sleep(30)  # stands in for viewser's sys.maxsize retry loop
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"the deadline did not interrupt: took {elapsed:.1f}s"
        assert caught.value.seconds == 0.25
        assert "a call that never returns" in str(caught.value)

    def test_a_call_that_finishes_in_time_is_untouched(self):
        with deadline(5, "a fast call"):
            result = 2 + 2
        assert result == 4

    def test_the_timer_re_arms_so_a_swallowed_alarm_is_not_lost(self):
        """viewser's fetch loop wraps `pd.read_parquet` in a bare `except:` and then
        continues retrying, so it will catch `DeadlineExceeded` and drop it on the floor.

        With a one-shot timer that alarm is gone for good and the bound evaporates —
        leaving the original hang, now wearing the appearance of protection. A repeating
        interval means a swallowed alarm is re-raised until one escapes.

        Asserted on the interval rather than by driving a swallowing loop, because the
        behavioural version *hangs* under the mutation instead of failing, and a guard
        whose failure mode is a hang is the defect this whole module is about.
        """
        with deadline(30, "x"):
            _value, interval = signal.getitimer(signal.ITIMER_REAL)
        assert interval == 30, (
            "the timer is one-shot; an alarm swallowed by a bare `except:` in the code "
            "being bounded would never fire again and the call would run unbounded"
        )

    def test_the_message_names_the_call_and_the_bound(self):
        """The skip text a developer reads has to say what timed out, and after how long."""
        with pytest.raises(DeadlineExceeded) as caught:
            with deadline(0.1, "viewser geography fetch"):
                time.sleep(30)
        assert str(caught.value) == "viewser geography fetch did not return within 0.1s"


class TestItLeavesNothingBehind:
    """Both failure modes here are silent and land on an unrelated later test."""

    def test_the_previous_handler_is_restored_after_a_clean_run(self):
        sentinel = signal.getsignal(signal.SIGALRM)
        with deadline(5, "fast"):
            pass
        assert signal.getsignal(signal.SIGALRM) is sentinel

    def test_the_previous_handler_is_restored_after_the_deadline_fires(self):
        sentinel = signal.getsignal(signal.SIGALRM)
        with pytest.raises(DeadlineExceeded):
            with deadline(0.1, "slow"):
                time.sleep(30)
        assert signal.getsignal(signal.SIGALRM) is sentinel

    def test_the_previous_handler_is_restored_when_the_block_raises_something_else(self):
        sentinel = signal.getsignal(signal.SIGALRM)
        with pytest.raises(ValueError):
            with deadline(5, "raises"):
                raise ValueError("not a timeout")
        assert signal.getsignal(signal.SIGALRM) is sentinel

    def test_no_timer_survives_to_fire_during_a_later_test(self):
        """A leaked timer fires under the *restored* handler — which is SIG_DFL, and the
        default action for SIGALRM is to kill the process.

        So the real consequence of forgetting to cancel is not a failing test: it is
        pytest dying part-way through with no report. Measured — removing the cancel ran
        6 of 9 tests and then the process vanished.

        The bound here is deliberately long and the check immediate: a short bound would
        already have fired by the time we look, `getitimer` would read 0 either way, and
        this test would pass while the defect shipped. The first version of this test did
        exactly that (armed 0.2s, slept 0.5s, asserted 0) and was found vacuous by
        mutation.
        """
        with deadline(30, "fast enough"):
            pass
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert remaining == 0, (
            f"a timer is still armed with {remaining:.1f}s to run — it will fire during "
            f"an unrelated later test and kill the process"
        )

    def test_no_timer_survives_an_exception_in_the_block(self):
        with pytest.raises(ValueError):
            with deadline(10, "raises"):
                raise ValueError("boom")
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert remaining == 0, f"a timer is still armed with {remaining}s to run"


class TestItRefusesRatherThanRunningUnbounded:
    def test_off_the_main_thread_it_raises_instead_of_silently_not_bounding(self):
        """Falling back to an unbounded call would reinstate the hang, invisibly."""
        captured = []

        def run():
            try:
                with deadline(1, "anything"):
                    pass
            except Exception as exc:  # noqa: BLE001 — the type is the assertion
                captured.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=10)

        assert captured, "off the main thread it silently proceeded — that is the bug"
        assert isinstance(captured[0], RuntimeError)
        assert "main thread" in str(captured[0])
