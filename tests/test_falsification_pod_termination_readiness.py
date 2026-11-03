"""Nothing in this repository can answer "is this pod safe to terminate?"

Six pods were terminated across 2026-11-03 and the decision was made by hand each time:
ssh in, read `EXP02_STATUS`, compare a file listing, judge. That worked five times and failed
once — `216.81.151.55:18132` was terminated on a reading of "stage=collapse, nearly done" when
the arm had in fact died 1h35m earlier, taking with it the one artifact that could have settled
whether its predictions were in count or log1p space (views-models#697).

⛔ The failure mode is specific and it is not carelessness: a pod is safe to terminate iff its
arm's delivery is COMPLETE AND VERIFIED ON LOCAL DISK, and every cheap proxy for that is wrong
in a different direction.

    GPU utilisation      0% during the whole draw phase, which is CPU-bound by design
    `STAGE`              names the stage REACHED, not whether it finished or died in it
    a running process    absent on a pod whose arm died cleanly and wrote a FAILURE
    `EXP02_STATUS` on    the pod can read OK while nothing has been pulled home
    file COUNT parity    passes when two sides hold the same number of different files

⚠️ And the asymmetry that makes it matter: STOPPING a pod is reversible — `/workspace` and its
venv survive, so a restart resumes in minutes. TERMINATING is not. The cost of a wrong "safe"
is an arm, ~3.5 h and ~$2.60; the cost of a wrong "unsafe" is pennies of idle time.

These stubs describe the check that should exist. They are xfail because it does not.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.red


@pytest.mark.xfail(reason="no such tool exists; written 2026-11-03", strict=False)
def test_a_pod_is_not_safe_until_its_delivery_is_verified_on_local_disk():
    """The subject of the check is the LOCAL delivery, never the pod's own status file.

    An arm can report `EXP02_STATUS=OK` on the pod and have been pulled nowhere. That happened
    to `upctwofullzero_fortyseven`: it reported OK at ~20:39 and the harvester never collected
    it, because the arm it relaunched was appended to a list the loop had already expanded.
    Reading the pod would have said "done"; reading local disk said "absent".
    """
    raise NotImplementedError(
        "the check must assert, for the named arm: a local delivery directory exists, its "
        "EXP02_STATUS reads OK, `sha256sum -c gate_endpoints.sha256` passes, and the local "
        "(path, size) set equals the pod's under LC_ALL=C — and must REFUSE if the pod is "
        "unreachable, because an unreachable pod cannot be compared against."
    )


@pytest.mark.xfail(reason="no such tool exists; written 2026-11-03", strict=False)
def test_a_stage_that_has_not_moved_is_a_failure_until_proven_otherwise():
    """`collapse` takes under 30 seconds on a healthy arm. Sitting in it is not progress.

    Measured on the arms that completed: `collapse 19:23:37 -> verify_parquet 19:23:59`, 22
    seconds. `upctwofullzero_fortysix` was reported as "at collapse, nearly done" three times
    across 40 minutes while its log had been silent for 2,542 s and no python process was
    alive. The stage name was true and the inference from it was false.
    """
    raise NotImplementedError(
        "the check must compare the age of exp02_arm.log against the measured duration of the "
        "stage the arm claims to be in, and must treat a log older than that duration as a "
        "FAILURE to investigate rather than as work in progress."
    )


@pytest.mark.xfail(reason="no such tool exists; written 2026-11-03", strict=False)
def test_two_pods_on_one_ip_must_be_distinguished_before_either_is_named_safe():
    """`216.81.151.57` hosted two pods at once; one was idle, one was training.

    `.57:14192` held a finished arm and `.57:14318` was training `upctwofullzero_fortyseven` at
    ~235/300. They are separate containers — hostnames `5a6a77a15d30` and `d7cefa045fbd`, same
    `boot_id` — but a console row shows neither. Terminating by IP would have destroyed ~2 h.
    """
    raise NotImplementedError(
        "the check must report the container hostname and the arm name together, so a pod is "
        "identified by what it is RUNNING and not by an address two pods can share."
    )


@pytest.mark.xfail(reason="no such tool exists; written 2026-11-03", strict=False)
def test_an_arm_still_running_is_never_safe_even_when_its_siblings_are_done():
    """A whole-campaign verdict hides the one arm that is still working.

    At the moment this was written, 10 of 12 arms were verified on disk, one had failed, and
    `upctwofullzero_fortyfive` was mid-`evaluate` with a 4-second-old log and its trained
    artifact on `/workspace`. "We have everything we need" was true of ten arms and false of
    the campaign, and a per-pod answer is the only one that cannot average them together.
    """
    raise NotImplementedError(
        "the check must answer per pod, never per campaign, and must name which arm each pod "
        "holds so a 'mostly done' reading cannot be applied to the exception."
    )
