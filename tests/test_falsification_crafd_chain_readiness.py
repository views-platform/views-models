"""Failing test stubs from the 2026-10-30 falsification of "we are ready to run the chain".

Audit subject: the claim *"we are ready to run chrome_box and chrome_crate with
`tools/podrun/pod_run_crafd_chain.sh` on a pod to get it all on the CRAF'd API"*, against
`development` at `45468336`. Ten probes, categories A/B/C/D/E/F.

**Verdict: FALSIFIED.** Four hard falsifications, two soft. The runner is sound for what it
actually does — phase 1, chrome_crate, nothing published — and the claim is about three
things it does not do.

⛔ The reason this file exists rather than a comment on the issue: each stub below is the
guard that would have made the claim checkable instead of arguable. They are `xfail` because
none of them passes today, and because a red file in `tests/` turns CI red by construction
for everyone (ADR-005). Flip a stub to a real assertion when its subject exists.

Related: epic #595, #640 (the runner), #599 (the delivery), views-crafdapi#180, C-186.
"""

import pytest


# ---------------------------------------------------------------------------
# HARD-1 (probe P1) -- the runner does not run chrome_box, and says so itself
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="phase 2 is unscripted by design; no guard ties the two", strict=False)
def test_a_runner_claiming_the_chain_covers_both_ensembles() -> None:
    """Nothing connects "the chain" as spoken to "the chain" as scripted.

    `pod_run_crafd_chain.sh` hardcodes `ENSEMBLE=chrome_crate` (:90) and the eight
    `*_condor` members. Its header (:14-15) states that phase 2 — the eight `*_sparrow`
    models and `chrome_box` — is **deliberately not here**, it prints
    `phase: 1 of 2 — chrome_box (pgm) is NOT in this script` (:389), and it closes with
    `### phase 2 (chrome_box, pgm) is a separate decision` (:396).

    So the script is honest and the sentence "run chrome_box and chrome_crate with the
    runpod script" is false. The gap is not in the code; it is between the code and what
    a reader believes the name `pod_run_crafd_chain.sh` covers.

    The rule: a runner named for a chain must either cover the chain or refuse to be
    described as covering it from outside — which means a guard asserting that every
    ensemble in `reconcile_with` reachability is either executed by the script or named in
    a declared NOT-COVERED list that something reads.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# HARD-2 (probes P2, P6, P9) -- nothing reaches the CRAF'd API, two ways over
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no scripted path from the pod's output to the wire", strict=False)
def test_the_pods_deliverable_reaches_the_crafd_wire() -> None:
    """The runner's output is files on disk; the wire is a separate, unscripted run.

    Deliverable (:57-62): `/workspace/deliver/chrome_crate/{parquet,members,MANIFEST,
    STATUS,FAILURE,run.log}`. There is no postprocessor invocation, no shard, no sidecar,
    no `__manifest.json`. The script states it (:50) — *"NO publish credential,
    deliberately. This chain produces a forecast on disk and uploads nothing"* — and
    prints `published: nothing` (:388). It also **refuses** to run if a publish variable
    is set (:172-174), which is correct per RunPod guide rule 5.

    So "get it all on the CRAF'd API" needs a second act: retrieve the artifacts, then run
    the `un_crafd` postprocessor from a machine that legitimately holds the nine Appwrite
    publish variables. No script performs that, and the handoff is where the 2026-09-29
    FAO delivery lost its record (`pod_run_fao_delivery.sh` names that incident).
    """
    raise NotImplementedError


@pytest.mark.xfail(reason="intent is paused; upload_armed() derives False", strict=False)
def test_the_crafd_delivery_is_armed_when_a_run_is_called_ready() -> None:
    """Even run by hand with credentials, the postprocessor makes zero store calls.

    `deliveries/un_crafd.py:127` is `intent = paused(...)`, and
    `postprocessors/un_crafd/configs/config_meta.py:58` derives
    `"wire_upload_enabled": upload_armed(CONSUMER)` from it. So the upload is disarmed at
    the declaration, which is the correct state today and is also the thing that makes
    "ready to get it on the API" untrue.

    ⛔ The asymmetry worth a guard: `paused` is checked by the launcher, but nothing checks
    the SENTENCE. A run can be declared ready in prose while the declaration says
    otherwise, which is exactly what happened in reverse on 2026-08-14 — the declaration
    said `live` while the prose said disarmed, and 111 files went to the partner (C-186).
    The rule: a statement of readiness must be derived from `upload_armed()`, never typed.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# HARD-3 (probe P4) -- the preflight would approve a box chrome_box dies on
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="MEM_FLOOR_GB is 16, correct for phase 1 only", strict=False)
def test_a_memory_floor_refuses_a_box_that_cannot_hold_what_is_being_run() -> None:
    """`MEM_FLOOR_GB=16` (:209) against chrome_box's measured peak of >=68.2 GiB.

    The floor is **right** for what the script runs: chrome_crate is country-month, its
    pool measured 96 MB, members ~2.0 MB peak RSS, 7 minutes, and 16 GiB exists to refuse
    a box too small for the venv and the datafactory cache. The script warns against
    confusing the two itself (:195) — *"DO NOT RAISE THIS TO chrome_box's FIGURE"*.

    The hazard is the claim, not the number: if phase 2 is ever added to this script
    without moving the floor, the preflight APPROVES a 16 GiB box and the run dies in the
    pool after the member training is paid for. The rule: the floor must be derived from
    the ensemble being run, not from a constant — i.e. read
    `ensembles/<ENSEMBLE>/configs/config_hyperparameters.py`, or refuse when the ensemble
    named is not the one the floor was measured for.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# HARD-4 (probe P8) -- the partner has not verified it can serve the result
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="views-crafdapi#180 is open", strict=False)
def test_the_consumer_has_verified_it_can_serve_the_draw_count_being_sent() -> None:
    """views-crafdapi#180, OPEN: *"Verify 1,024 draws per cell serve inside the 9 GB cap"*.

    chrome_box declares `expected_models: 8` x `expected_samples_per_model: 128` = 1,024
    draws (`config_hyperparameters.py:109`). At 1,024 draws a month is ~0.80 GB raw, which
    is inside their cap by arithmetic and unverified in the container they run.

    Sending a wire a partner cannot serve is worse than not sending it, because the failure
    surfaces on their side with our data in it. The rule: a delivery whose shape changed
    (128 -> 1,024 draws) may not be armed while the consumer's own acceptance issue is
    open — and that is a cross-repository state nothing here reads.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT-1 (probe P5) -- phase 2's members do not exist and nothing trains them
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no sparrow artifacts; no script trains them", strict=False)
def test_every_member_a_run_needs_either_exists_or_is_trained_by_the_script() -> None:
    """Zero PredictionFrames exist for `ashen_condor` or `ashen_sparrow` on this machine.

    Phase 1 handles it: the runner trains the eight condors itself with `-t -f` (:337) and
    then runs the ensemble `-f -sa` (:353) so `_train_model_artifact` is never reached and
    no `models/*/run.sh` is invoked — which is why the pod lane is immune to the conda
    lane's problems.

    Phase 2 has no equivalent. The eight `*_sparrow` pgm models are global-land and
    untrained, and `ensembles/chrome_box/configs/config_hyperparameters.py` puts member
    training at roughly three hours of paid compute before the pool even starts. So
    "ready to run chrome_box" silently includes "train eight pgm models first".
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT-2 (probe P3) -- the pod lane is immune to a gap the conda lane is not
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="chrome_box declares only views-pipeline-core", strict=False)
def test_an_ensemble_declares_every_package_its_reconciler_imports() -> None:
    """`chrome_box` imports `datafactory_query` and does not declare it.

    `reconciliation/datafactory_country_mapping_provider.py::_fetch_from_datafactory` does
    `from datafactory_query import load_dataset`, inside the method — so it is reached
    during reconciliation, after pooling. `ensembles/chrome_box/requirements.txt` declares
    `views-pipeline-core>=3.3.5,<4.0.0` and nothing else, and `datafactory_query` is absent
    from `envs/views_ensemble` (measured 2026-10-30).

    ⛔ The pod lane does not have this problem, which is why it is SOFT for the claim as
    stated and HARD for the conda lane. `pod_run_crafd_chain.sh:269-286` declares its
    install set **in the script**, including `views-datafactory>=1.13.0,<2.0.0`, with
    `toolz>=0.12.1` as a separate LAST install (:293) — deliberately so the pod can be
    immune to a `requirements.txt` that cannot resolve (#644). That is a fourth place these
    dependencies are stated, and the script says so.

    The rule: a declared dependency set must cover the imports the declaring source
    actually reaches, including lazily-imported ones inside injected collaborators.
    """
    raise NotImplementedError
