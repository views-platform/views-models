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

**Three of the four findings were closed in the same branch** and are recorded here rather
than kept as stubs, because an xfail whose `reason` names a defect the diff fixes reads as an
open defect forever — it can never flip to XPASS to signal its own staleness. Flagged by
`/code-review high` on PR #657:

* **The trained artefacts (HARD).** `pod_run_chrome_box.sh` copied only the pool and had no
  `collect_artifact` stage, while `pod_run_model.sh:395` has had one for weeks and
  `docs/runpod_run_guide.md:429` names `artifact/` in bold as the thing that cannot be
  remade. 16 `.pkl` files, 363.7 MiB, on a host declared safe to destroy. CLOSED by
  `stage collect_members`, which pairs each artefact to its forecast by the timestamp both
  filenames carry — and whose first version was worse than none, failing the resume path
  after deleting the copies it had made last time (see
  `test_the_collect_stage_survives_a_resume`).
* **The offline W&B record (SOFT).** 50 `offline-run-*` directories, 164 MB, outside `$OUT`
  because `WANDB_MODE=offline` is forced. C-170's second class. CLOSED by the same stage.
* **The package set (SOFT).** The MANIFEST pinned 2 versions; the venv had 143, and
  `pip freeze` returns nothing in a uv-built venv — an empty file, exit 0. CLOSED by a
  `FREEZE` step asking `importlib.metadata`, with an empty result fatal.
  `views-frames==1.11.0` is the entry that matters: C-192's deficit is a property of that
  reconciler, so a reproduction that re-resolves it is not one.

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
