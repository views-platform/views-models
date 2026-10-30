"""Failing test stubs from the 2026-10-30 falsification of "the next step is a pod".

Claim under test: *nothing in this repository blocks renting an 87 GiB pod and running
`pod_run_crafd_chain.sh` then `pod_run_chrome_box.sh` to completion; every remaining obstacle
is on the pod or beyond the definition of done.*

**The claim SURVIVES.** Eight probes, six passed, one unverifiable without credentials. But two
statements I had made *about* it did not survive, and one of them is a hard falsification with a
silent-wrong-output consequence.

What passed, recorded so it is not re-probed: the runpod guide clones `-b development`
(`:147`), so a pod gets the merged runners; the pod runs as root and `~/.netrc` is
`/root/.netrc` (`:123`, `:244`), so the preflight checks the file the fetch will read; the
eight `*_sparrow` requirements are exactly the three packages the inline install set provides;
`chrome_box/main.py` wires `build_reconciler_for_run` and passes `reconciler=` while
`chrome_crate/main.py` deliberately does not; and retrieval is the guide's Phase 5 with an
explicit rsync and a local verification step.

Unverifiable here: whether `land_gaul` resolves on the remote datafactory store for months
457..622. That needs the credential and the network, so it is recorded as UNVERIFIED rather
than as a pass — the distinction this repository's ADR-025 exists for.
"""

import pytest


# ---------------------------------------------------------------------------
# HARD (probe P6) -- `expected_models` is enforced by nothing at run time
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no runtime check on the constituent COUNT; see docstring", strict=False)
def test_the_pooled_constituent_count_is_verified_against_the_declared_one() -> None:
    """Nothing at run time checks that the number of members pooled is `expected_models`.

    ⛔ This falsifies a claim I made three times — in #649's commit message, in register
    entry C-188, and in a test docstring — that a seven-of-eight pool would be caught as
    "best case an ADR-015 `expected_models` failure after three paid hours". **There is no
    such failure.** Measured in the installed pipeline-core 3.3.4:

        grep -rn "expected_models" <views_pipeline_core>/  ->  0 matches

    `expected_samples_per_model` IS enforced — `_assert_expected_sample_count`
    (`prediction_frame_ensemble.py:153`) raises when a constituent's produced
    `pf.sample_count` differs from the declared 128, and its docstring cites C-85 as the
    reason. And the #160 balance guard in `_aggregate_prediction_frames` rejects
    constituents whose sample counts differ **from each other**.

    Neither sees a MISSING constituent. Seven members that each produced 128 draws are
    mutually balanced and individually correct, so a short pool passes both guards and
    yields 7 x 128 = **896 draws where the config, the epic and the delivery contract all
    say 1,024** — then reconciles, writes a MANIFEST, and reports OK. `expected_models: 8`
    is compared only against `len(config_modelset["models"])` by
    `tests/test_ensemble_configs.py`, which is a config-against-config check at CI time and
    says nothing about what a run pooled.

    **Why this matters more than it reads:** it means the member-skip defect fixed in #649
    had no backstop at all. I had recorded that bug as bounded by an ADR-015 failure; it was
    not bounded by anything. The fix is load-bearing rather than belt-and-braces.

    The rule: a declared contract that only CI checks is not a run-time guarantee. Either
    the runner counts what it is about to pool, or pipeline-core asserts
    `len(model_results) == expected_models` beside the sample-count assert that is already
    there. The first is one `if` in a shell script and is done in this branch; the second is
    pipeline-core's and is the general fix.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT (probe P5) -- I reported a cost as unmeasured while the measurement sat
#                   in a file I had already read
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="a reporting discipline, not a code defect", strict=False)
def test_a_cost_is_called_unmeasured_only_after_the_evidence_was_searched() -> None:
    """The eight `*_condor` members' training time WAS measured, and I said twice it was not.

    In the plan and again in a status report I wrote that phase 1's member training was
    "not measured. cm, ~2 MB RSS each; I will not invent a number" — presenting the refusal
    to estimate as a discipline. The figures were in
    `tools/podrun/evidence/2026-10-28_chrome_box/crafd_chain.out`, a file in the evidence
    directory I had read twice that night, as eight HEARTBEAT lines:

        ashen 0min, brisk 0min, candid 1min, dusky 0min,
        eager 0min, faint 0min, gentle 0min, humble 0min

    So phase 1 is roughly **ten minutes end to end**, not an unknown. Declining to estimate
    is right; declining to LOOK and calling the result unmeasured is the same error as
    reporting a number I had not taken — it just fails safe instead of optimistic, which is
    why it survived.

    The rule, which `feedback_never_state_an_unmeasured_cost` has the other half of:
    "unmeasured" is a claim about the evidence, and it needs the same search before it is
    asserted as a figure needs before it is quoted.
    """
    raise NotImplementedError
