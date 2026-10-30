"""Findings from the 2026-10-30 falsification of "I can terminate the pod".

Claim audited: *everything on `216.81.151.57:13663` that cannot be remade is off the box and
verified, so destroying it loses nothing.*

⛔ **FALSIFIED as stated.** Two classes of output existed only on the pod when the claim was
made, and one of them cannot be regenerated without a new paid run. Both were retrieved
during the audit, so **the claim is true now** — but it was not true when it was asserted,
and the reason it was asserted is the subject of the last stub below.

The trigger index was consulted first, which is what it is for. **C-168's trigger fired
verbatim** — *"Declaring a pod ready to destroy on the strength of a bundle-vs-manifest
comparison"* — and C-170's class ("output written outside `$OUT`") supplied both findings.

What SURVIVED, recorded so it is not re-probed:

* **P3 — integrity, per file.** Every file in the local 1.70 GiB archive was hashed in one
  streaming pass and compared against sha256 computed on the pod. **6 of 6 forecast files
  byte-identical** (3 x `y_pred.npy` at 9,104 MiB, 3 x `identifiers.npz`), 13 of 14 files
  identical overall. The two that differed were explained and then closed: `anon.log` was
  still being appended by a sampler this session had left running, and
  `RECONCILIATION_RESIDUAL` post-dated the archive; both were re-fetched and now match
  (`57ce28b3…`, `6e2f0c11…`). This is the comparison **C-180** says to check is not empty —
  the digests are real 64-hex values over 27 GB of streamed payload, not filenames.
* **P5 — credential hygiene.** No publish variable in the environment; the only Appwrite
  string on the box is pipeline-core's own `prediction_store.py` in the uv cache, not a
  secret. `/root/.netrc` (3 lines, the datafactory read credential) is present and **stays
  there to die with the pod**, which is the rule. The four probe scripts this session left in
  `/root` contain no secrets.
* **Deliberately NOT retrieved:** ~30 GB of member forecasts under
  `models/*_sparrow/data/generated/`. The pool is the product, and with the trained models
  now retrieved these are regenerable by a forecast pass without retraining. Stated as a
  choice rather than an omission, which is the distinction C-168 exists to force.
"""

import pytest


# ---------------------------------------------------------------------------
# HARD -- the runner collects the product and not the thing that cannot be remade
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="pod_run_chrome_box.sh has no collect_artifact stage", strict=False)
def test_the_runner_collects_the_trained_artefacts_and_not_only_the_pool() -> None:
    """⛔ Eight trained models were going to die with the pod, and the guide says so in bold.

    `tools/podrun/pod_run_chrome_box.sh` copies exactly one thing into `deliver/`:

        mkdir -p "$OUT/parquet"
        cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/." "$OUT/parquet/"

    There is **no `collect_artifact` stage**. `tools/podrun/pod_run_model.sh` has had one
    since before this runner was written (`:395` `stage collect_artifact`, `:412` `cp "$PT"*
    "$OUT/artifact/"`), and `docs/runpod_run_guide.md:429` states the reason in bold: *"⛔
    `artifact/` is the one that cannot be remade"*, citing 2026-09-28, when eight models
    trained, delivered their parquets, reported `OK`, **and their weights died with the
    pods**.

    Measured on this pod: `models/*_sparrow/artifacts/forecasting_model_*.pkl`, **16 files,
    363.7 MiB** (8 `*_sparrow` + 8 `*_condor`), none of them in `deliver/`, none of them in
    the 1.70 GiB archive that had been verified and declared complete. One was extracted from
    the rescue archive and `pickle.load`s to a `MixtureBaseline`, so they are real and intact.

    **Why this is the hard finding.** The 2026-10-28 reasoning for copying the pool and not
    the members was sound and is recorded: 26.7 GiB of member *forecasts* is pure duplication
    of data the pool already contains. ⚠️ But that argument was about the forecasts, and it
    silently took `artifacts/` and `wandb/` with it — **542 MB, not 30 GB**, and the one
    class the guide names as irreplaceable. A size-based exclusion inherited a
    correctness-based one.

    Retrieved during this audit (`chrome_box_20261030_rescue.tar.zst`, 3.5 MiB compressed,
    sha256 verified, 366 MiB of payload), so nothing is lost. The runner is unchanged and
    will do it again. Fix: a `collect_artifact` stage mirroring `pod_run_model.sh`'s, copying
    each member's `artifacts/` and `wandb/` — and a guard that the stage exists, since this
    runner's tests pin the pool copy and never asked what else the run produced.

    Cross-refs: **C-161** (the runner archives half the posterior and the guard counts only
    the half it archives — the same shape, one artefact class along), **C-170** (four classes
    of run output written outside `$OUT`), **C-168** (ROOT).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT -- the run's own logged record was outside the collected scope
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="50 offline W&B runs outside $OUT; C-170's class", strict=False)
def test_the_runs_own_logged_record_is_collected() -> None:
    """50 `offline-run-*` directories, 164 MB, sat outside `deliver/` and were not retrieved.

    `WANDB_MODE=offline` is forced by the runner — correctly, because nothing may be
    published from rented hardware. The consequence nobody followed through is that the run's
    metrics, system stats and config snapshots are then written to
    `models/<member>/wandb/offline-run-*/` and `ensembles/<ensemble>/wandb/`, which is **not
    `$OUT`**, and the runner's collection step never looks there.

    This is exactly **C-170**'s second class ("the W&B tables"), recorded against
    `pod_run_model.sh` and now true of this runner too. It is cheap to fix and cheap to lose,
    which is why it keeps being lost: 164 MB against a 27 GB product.

    Retrieved during this audit inside the rescue archive. Same fix as the stub above.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT -- nothing recorded what the run actually ran on
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no frozen package set is retained; MANIFEST names two of 143", strict=False)
def test_the_exact_package_set_the_run_used_is_recorded() -> None:
    """The MANIFEST pins 2 versions. The venv had **143** packages.

    `MANIFEST` records `pipeline_core: 3.3.5` and `toolz: 1.2.0 (C-151 override)` — the two
    the runner asserts a floor on. Nothing recorded the rest, and
    `/workspace/venv-crafd/bin/python -m pip freeze` returns **zero lines**, because the venv
    was built with `uv` and has no `pip`. So the obvious way to capture it silently produces
    nothing — absence reading as success, one more time.

    Captured during this audit with `uv pip freeze` (143 packages) and retained. The ones a
    future reader would want and could not otherwise recover:

        views-frames==1.11.0        <- the reconciler; the subject of C-192 and views-frames#283
        views-datafactory==1.16.0   <- the geography the country mapping came from
        views-baseline==1.0.2       views-evaluation==2.0.0   views-schema==2.3.3
        numpy==1.26.4               pandas==1.5.3             scipy==1.17.1

    ⚠️ `views-frames==1.11.0` matters specifically: the 2.9%-7.3% mass deficit in **C-192**
    is a property of that reconciler's `aligned-draws` implementation, and a reproduction
    attempt that re-resolves to a different views-frames is not a reproduction. The MANIFEST
    asserted a floor on the two packages the runner was worried about, which is not the same
    as recording what ran.

    Fix: the runner writes `uv pip freeze` into `deliver/<ensemble>/FREEZE` beside the
    MANIFEST. One line of shell, and it is the difference between a reproducible run and a
    similar one.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# SOFT -- the verification's scope was the producer's declared scope
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="C-168 fired verbatim; the entry predicted this exactly", strict=False)
def test_a_teardown_readiness_claim_is_scoped_by_looking_at_the_host() -> None:
    """⛔ "Everything is off it and verified" was checked against the set the runner chose.

    The sequence: the runner copied the pool into `deliver/chrome_box/`; the pod tarred
    `deliver/chrome_box/` and computed a sha256 **of its own archive**; the archive was
    fetched and that sha256 verified locally; and the conclusion drawn was *"The pod can be
    stopped — everything is off it and verified."*

    Every step of that is correct and the conclusion does not follow. The sha256 establishes
    that the bytes arrived intact. It establishes nothing about whether `deliver/chrome_box/`
    was the right set, because the thing that decided that set is the same script whose
    output was being verified.

    **C-168 is the cluster root and its Trigger is this sentence:** *"Declaring a pod,
    container or any rented host ready to destroy on the strength of a bundle-vs-manifest
    comparison — or reporting 'we have everything' after a file-by-file diff against the
    collection script's own output directory."* It has been open since 2026-10-25. The
    register already held the answer, and C-161's third instance was logged for exactly this
    reason: the entry was *"correct, specific, and unreachable"*.

    ⚠️ What actually worked was the **trigger index**, which was built after that incident and
    was consulted this time before any probe was designed — and it named C-168, C-170 and
    C-169 in one line. So the mechanism the register grew in response to this failure
    functioned. The failure here is that it was consulted at `/falsify` time rather than at
    "the pod can be stopped" time, which was four hours and one confident sentence earlier.

    The procedural fix is not another entry. It is that **the enumeration must come from the
    host** — `du -sh /workspace/*`, `find` for large and recently-modified files, rooted at
    `/workspace` and **without `-xdev`** (C-169: `-xdev` silently excludes `/workspace` on a
    RunPod host, so the sweep that exists to catch what was left behind returns nothing and
    reads clean) — before any claim about completeness, not after.
    """
    raise NotImplementedError
