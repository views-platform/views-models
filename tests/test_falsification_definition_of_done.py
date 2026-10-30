"""Open findings from the 2026-10-30 falsification of "we have reached our definition of done".

Claim audited, Simon's words and the only target:

    "Without breaking anything and without making anything more complicated than it already
    is, I must be able to run chrome_box and chrome_crate."

**The claim SURVIVES.** Both ensembles ran end to end on one 87 GiB box (phase 1
`STATUS: OK` 04:16:43Z, phase 2 `STATUS: OK` 07:30:32Z, 193 min), nothing was published, no
Appwrite credential went near the pod, and the three suite failures that remain are in an
untracked file about the hydranet pin which fails identically with this branch's edits
stashed out.

Six probes: 1 hard, 3 soft, 1 observation, 1 UNVERIFIED. **Four were closed the same day**
and are recorded here rather than kept as stubs, because a stub for a property that now has
a real guard is clutter that reads as an open defect:

* **P1 reporting half (HARD)** — the delivered pool lost 2.9%-7.3% of the country forecast's
  fatality mass to countries whose grid cells were all zero, and the MANIFEST said
  `reconciled: against chrome_crate` unqualified. CLOSED by `reconciliation/residual.py`
  (8 tests, 7/8 mutations caught and the 8th was dead code it exposed), wired into
  `pod_run_chrome_box.sh` as a fatal-on-failure stage, and pinned by
  `test_the_residual_check_runs_unconditionally_and_its_failure_is_fatal` and
  `test_the_manifest_cannot_describe_the_pool_as_simply_reconciled`. Verified by running the
  new module on the pod against the real artefacts: it reproduced the audit's figures
  exactly, with `scaling_failures 0` across 22,339,584 triples.
* **P2 (SOFT)** — a `parquet/` -> `pool/` rename made one of five runners inconsistent while
  `pod_run_model.sh` may not be edited, so the family could never be made consistent again.
  CLOSED by reverting it; the inaccuracy is now stated in a comment instead.
* **P3 (SOFT)** — the fixed runner had never executed. CLOSED far enough: the revert removed
  the only unexecuted line with a runtime failure mode, the peak parsers were executed
  against the real pod log, and the residual module was executed on the pod.
* **P5 (OBSERVATION)** — complexity. The operator's path got *shorter*: a 67-line provisional
  driver in an evidence folder with no preflight, which had to be hand-ported, became two
  tracked commands each with a `--preflight` that refuses. Lines up, steps and judgement
  calls down. Not a complication.

**P6 remains UNVERIFIED, which is not a pass.** `views_postprocessing` is a namespace package
here with no `crafd` module installed, so whether the CRAF'd publisher consumes `chrome_box`'s
`.npy` PredictionFrame directories could not be executed. Confirmed instead:
`deliveries/un_crafd.py:121` declares `send = [pgm("chrome_box")]`, `reconciled = True`, and
`intent = paused(...)` — the delivery points at chrome_box and cannot fire by accident.
ADR-025's distinction, kept rather than rounded up.

What is left below is the one finding nobody in this repository can close.
"""

import pytest


# ---------------------------------------------------------------------------
# OPEN -- upstream: where should a zero-grid country's fatalities go?
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="upstream modelling decision; views-frames#283", strict=False)
def test_the_reconciled_pool_preserves_the_country_forecast_mass() -> None:
    """Reconciliation still loses 2.9%-7.3% of the fatality mass. It is now *reported*.

    Measured on the real 2026-10-30 artefacts with the run's own mapping (202 countries,
    0 cells without a country), over all 22,339,584 (country, month, draw) triples:

        target        triples >0.1% off    cm mass unplaced              scaling failures
        lr_sb_best    164,429  (2.208%)    12,866,485 / 319,789,635  4.023%     0
        lr_ns_best    188,481  (2.531%)     5,272,245 /  72,013,473  7.321%     0
        lr_os_best    193,903  (2.604%)     6,553,019 / 222,651,525  2.943%     0

    ✅ The reconciler is exact: zero triples disagree while the grid sum is non-zero.
    ⛔ 100.00% of the disagreement is the zero-grid case. `0 x anything = 0`, so a country
    whose every cell drew zero cannot be lifted to a positive country total, and its
    fatalities are absent from the grid product.

    **Why this stays open while the reporting half is closed.** Placing that mass requires
    deciding *which cells* receive it — population, historical conflict, the mean of the
    same country's other draws, or an explicit residual term carried beside the frame.
    Each changes the spatial distribution of a delivered forecast, so it needs an ADR in
    views-frames, not a patch here. Filed with this measurement as **views-frames#283**,
    including the offer of the retained 1,024-draw archive so a candidate policy can be
    scored against a real forecast.

    ⚠️ Not views-frames#145: pipeline-core 3.3.5's `reconcile_frames.py:19-22` still points
    readers there for the `aligned-draws` approximation, and **#145 is CLOSED**. Also
    distinct from **C-200b**, which asks whether pairing grid-draw `s` to country-draw `s`
    is meaningful; this is mass the pairing cannot place at all, either way.

    NOT MEASURED, and deliberately not guessed: whether the lost mass concentrates in a few
    small countries or spreads across all 202. That determines how a recipient experiences
    the error. It is one more pass over the same arrays.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# OPEN -- needs a commit, which is the operator's
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="needs Simon's word to commit; artefacts are outside git", strict=False)
def test_the_evidence_of_a_paid_pod_run_is_version_controlled() -> None:
    """Nothing in this repository records that the run happened.

        $ git -C ../pod_deliveries rev-parse --show-toplevel
        fatal: not a git repository

    The MANIFEST, `run.log`, `memory.log`, `anon.log` and now `RECONCILIATION_RESIDUAL` —
    including the **63.9 GiB `anon` peak** that is the only honest basis `MEM_FLOOR_GB` has
    ever had, and the 1.70 GiB archive that is the only copy of a 193-minute paid run — sit
    in `views_platform/pod_deliveries/chrome_box_20261030/`, outside any repository. A grep
    across the tree for the run's own timestamp finds only
    `reports/technical_risk_register.md`, and that entry is from the previous night's
    review, not from this run.

    This is the pattern `feedback_commit_in_scratchpad_worktrees` records after it nearly
    cost us three times — worktree code (#529), pod artefacts (2026-10-20), six days of
    risk-register curation (2026-10-27): **durable is not the same as version-controlled.**

    The 27 GB of arrays cannot go in git and should not. The MANIFEST and the four logs are
    a few hundred KB and belong in `tools/podrun/evidence/2026-10-30_chrome_box/`, beside
    the 2026-10-28 run's evidence, which is the precedent.
    """
    raise NotImplementedError
