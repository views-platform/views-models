"""Failing-test stubs from the /falsify audit of 2026-11-05 (second audit of the session).

CLAIM AUDITED: "Nothing is outstanding on my side. Sessions baseline, hydranet and datafactory have
all the information they need from me to move forward."

VERDICT: FALSIFIED — 5 hard, 2 soft, plus one unpredicted finding that dominates them.

These tests FAIL as written. Each encodes an obligation that was stated in writing and not performed.
They pass when the obligation is discharged.

Deliberately under reports/ and NOT tests/ — an untracked stub in tests/ reddened CI once this week.

    conda run -n views_pipeline python -m pytest \
        reports/2026-11-04_baseline_phase1/test_falsification_nothing_outstanding.py -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTER = REPO / "reports" / "technical_risk_register.md"
WRITEUP = HERE / "sparse_support_question.md"


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        pytest.fail(f"cannot see subject: git {' '.join(args)} exited {out.returncode}")
    return out.stdout


# ------------------------------------------------------------------ HARD

def test_the_campaign_output_is_version_controlled() -> None:
    """HARD (P2). 454 MB of analysis — a 704-line write-up, eleven scripts, 31 stubs and every
    md5-verified archive — exists only as untracked files on one disk. `reports/` is not even
    gitignored; it was simply never committed. `feedback_commit_in_scratchpad_worktrees` records
    three prior near-losses of exactly this shape: durable is not the same as version-controlled.
    """
    tracked = [p for p in _git("ls-files", "reports/2026-11-04_baseline_phase1/").splitlines() if p.strip()]
    assert tracked, (
        "zero files tracked under reports/2026-11-04_baseline_phase1/. "
        "The write-up, scripts, stubs and archives are recoverable only from this disk."
    )
    assert any(p.endswith("sparse_support_question.md") for p in tracked), (
        "the 704-line write-up carrying all six retractions is not tracked"
    )


def test_the_drift_finding_is_in_the_register() -> None:
    """HARD (P1). Told the views-datafactory session verbatim: "'drift_detection_ran: false means
    nothing even looked' is going into my register verbatim, with this cell as the named test case."
    It was not. A drift check with no outlier or range component cannot see the single largest value
    in the dataset — a guard that cannot fire, which is the register's own recurring theme.
    """
    if not REGISTER.exists():
        pytest.fail(f"cannot see subject: {REGISTER} does not exist")
    text = REGISTER.read_text(encoding="utf-8")
    assert "148759" in text, (
        "the register does not name cell 148759 — the promised drift entry was never written. "
        "Note a pre-existing entry mentions drift_detection_ran for an unrelated concern; that is "
        "not this finding."
    )
    assert "nothing even looked" in text, (
        "the sentence promised to views-datafactory is absent from the register"
    )


def test_the_pipeline_core_items_were_raised() -> None:
    """HARD (P1). Told both views-baseline and views-datafactory: "I will raise both there rather
    than leaving them floating" — the queryset_digest name-dependency (C-194) and the sink layout
    that puts predictions and metricframes in sibling directories under data/generated/. Both were
    correctly bounced to pipeline-core as not views-baseline's. Zero messages were sent to the
    pipeline-core session.

    There is no programmatic way to assert a message was sent, so this test asserts the durable
    proxy: an issue or a register entry naming the handoff.
    """
    if not REGISTER.exists():
        pytest.fail(f"cannot see subject: {REGISTER} does not exist")
    text = REGISTER.read_text(encoding="utf-8")
    assert "pipeline-core" in text and "sink layout" in text.lower(), (
        "no durable record that the sink-layout item was handed to pipeline-core; "
        "it exists only in a peer message thread"
    )


def test_prior_session_obligations_are_closed() -> None:
    """HARD (P3). Session-start state carried two open items that were never returned to:
    PR #715 (the runpod disk chart) still open, and branch docs/p22-oracle-ceiling-results pushed
    with no PR — the P-22 oracle ceiling results, preserved specifically because the pod was about
    to be destroyed.
    """
    branches = _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    assert "origin/docs/p22-oracle-ceiling-results" not in branches, (
        "docs/p22-oracle-ceiling-results is still pushed with no PR; the P-22 results it was "
        "created to preserve are still unmerged"
    )


def test_tonights_parked_findings_are_recorded() -> None:
    """HARD (P7). project_parked_findings was last modified 2026-11-04 01:34 — before the campaign —
    and contains none of tonight's set-aside items. Per feedback_park_findings_dont_hand_them_over
    the memory is where side findings go INSTEAD of being handed over, so an unupdated file means
    they are in neither place.
    """
    mem = Path.home() / ".claude/projects/-home-simon-Documents-scripts-views-platform-views-models/memory/project_parked_findings.md"
    if not mem.exists():
        pytest.fail(f"cannot see subject: {mem} does not exist")
    text = mem.read_text(encoding="utf-8")
    assert "2026-11-05" in text or "baseline_phase1" in text, (
        "none of tonight's parked items are recorded: PR #715 still open, the p22 branch with no PR, "
        "the uncommitted 454 MB, the unraised pipeline-core items, the unrun alpha=0.75 arm"
    )


# ------------------------------------------------------------------ SOFT

def test_self_declared_cheap_gaps_are_closed() -> None:
    """SOFT (P4). The write-up names two gaps and calls each cheap: short_linen's stratified row
    (lost to output truncation) and extending the battery beyond one target/origin. Calling a gap
    cheap in writing and leaving it is worse than not naming it, because the reader assumes it was
    judged and declined rather than forgotten.
    """
    if not WRITEUP.exists():
        pytest.fail(f"cannot see subject: {WRITEUP} does not exist")
    text = WRITEUP.read_text(encoding="utf-8")
    assert "lost to output truncation" not in text, (
        "short_linen's stratified row is still missing and still described as the one gap"
    )


def test_the_alpha_monotonicity_prediction_is_resolved() -> None:
    """SOFT (P5). A risky prediction was recorded before measurement: arm E's gap-filling must rise
    monotonically with alpha, light_tweed(0.25) < even_tweed(0.50) < heavy_tweed(0.75). Confirmed on
    the lower half only; alpha=0.75 was never measured on the support diagnostic, and the hydranet
    session holds the prediction as logged. A half-tested pre-registered prediction left in a peer's
    log is worse than an untested one.
    """
    if not WRITEUP.exists():
        pytest.fail(f"cannot see subject: {WRITEUP} does not exist")
    text = WRITEUP.read_text(encoding="utf-8")
    assert "NOT TESTED at a=0.75" not in text and "not yet measured" not in text, (
        "the alpha=0.75 half of the pre-registered prediction is still unmeasured"
    )


# ------------------------------------------------------------------ THE UNPREDICTED FINDING

def test_no_published_claim_rests_on_a_density_read_as_a_mass() -> None:
    """THE DOMINANT FINDING — which no probe in this audit was designed to catch.

    While the audit ran, the hydranet session showed that the campaign's headline sentence was
    wrong: CRPS = integral of BS(t) dt, so a table of raw BS(t) at thresholds 0,1,2,5,10,25,100,1000
    reports DENSITIES, and comparing their heights is not comparing contributions. The integral's
    mass is ~59% in [100, 1000) — the opposite of the claim. The sentence had already been quoted
    verbatim onto views-datafactory#533 under attribution.

    Every probe in this audit assumed delivered work was correct and asked only whether it was
    COMPLETE. The audit's blind spot is the finding: it checked promises, not claims.
    """
    if not WRITEUP.exists():
        pytest.fail(f"cannot see subject: {WRITEUP} does not exist")
    text = WRITEUP.read_text(encoding="utf-8")
    bad = "puts its mass where there is no information"
    if bad in text:
        # the claim may remain only if explicitly marked retracted next to it
        idx = text.index(bad)
        window = text[max(0, idx - 600): idx + 600]
        assert "RETRACTED" in window or "SUPERSEDED" in window, (
            "the write-up still asserts the density-read-as-mass claim without a retraction marker"
        )
