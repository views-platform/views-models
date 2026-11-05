"""Failing-test stubs from the /falsify audit of 2026-11-05.

CLAIM AUDITED: "when heavy_tweed lands we can terminate all pods without losing anything."
VERDICT: FALSIFIED (1 hard, 2 soft). All findings were remediated before any pod was destroyed,
so these tests pass NOW — they are kept because the *procedure* that produced the defect is
unchanged and will reproduce it on the next pod campaign.

Deliberately placed under reports/ and NOT under tests/: an untracked stub in tests/ reddened CI
once already this week. Run explicitly:

    conda run -n views_pipeline python -m pytest \
        reports/2026-11-04_baseline_phase1/test_falsification_pod_bundle_completeness.py -v
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PREDICTIONS = HERE / "archives_s64"
SCORES = HERE / "scores_s64"
INPUT = HERE / "input_s64"

TEN = [
    "plain_burlap", "short_linen", "plain_linen", "long_linen", "sheer_velvet",
    "plush_velvet", "flat_velvet", "light_tweed", "even_tweed", "heavy_tweed",
]


def _entries(path: Path) -> list[str]:
    """List a .tar.zst without extracting. Refuses rather than returning [] (ADR-025)."""
    if not path.exists():
        pytest.fail(f"cannot see subject: {path} does not exist")
    out = subprocess.run(
        ["tar", "-I", "zstd -d", "-tf", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        pytest.fail(f"cannot read {path.name}: tar exited {out.returncode}: {out.stderr[:200]}")
    names = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not names:
        pytest.fail(f"cannot see subject: {path.name} listed zero entries")
    return names


# ---------------------------------------------------------------- HARD falsification

@pytest.mark.parametrize("model", TEN)
def test_every_predictions_bundle_has_a_scores_bundle(model: str) -> None:
    """HARD. The pod-run archive procedure captured only `predictions_*`.

    The computed evaluation metrics live in `data/generated/<model>/calibration/
    metricframe_lr_{sb,ns,os}_best/` and were NOT archived, so terminating the pods would have
    destroyed every score while leaving the draws intact. The scores were the stated purpose of
    the exercise ("at least we can get them scored").
    """
    preds = PREDICTIONS / f"{model}_s64.tar.zst"
    scores = SCORES / f"{model}_scores.tar.zst"
    assert preds.exists(), f"{model}: predictions bundle missing"
    assert scores.exists(), (
        f"{model}: predictions bundle exists but NO scores bundle. "
        "The metricframes are pod-only and die with the pod."
    )
    frames = [n for n in _entries(scores) if "metricframe_" in n and n.endswith("values.npy")]
    assert len(frames) == 3, (
        f"{model}: expected 3 metricframes (lr_sb/lr_ns/lr_os), found {len(frames)}: {frames}"
    )


# ---------------------------------------------------------------- SOFT falsifications

@pytest.mark.parametrize("model", TEN)
def test_the_exact_artifact_that_produced_the_draws_is_retained(model: str) -> None:
    """SOFT. Training is seeded (seed=42) but consumes an unpinned datafactory pull, so a
    retrained artifact is not guaranteed identical. Without the artifact, an n=256 re-measurement
    would measure a different model and could not be compared to tonight's numbers.
    """
    scores = SCORES / f"{model}_scores.tar.zst"
    pkls = [n for n in _entries(scores) if n.startswith("artifacts/") and n.endswith(".pkl")]
    assert len(pkls) == 1, f"{model}: expected exactly 1 trained artifact, found {pkls}"

    # the artifact must be the one whose timestamp the predictions bundle is named for
    run_id = _entries(PREDICTIONS / f"{model}_s64.tar.zst")[0]
    run_id = run_id.replace("predictions_calibration_", "").strip("/")
    assert run_id in pkls[0], (
        f"{model}: retained artifact {pkls[0]} is not the one that produced run {run_id}. "
        "A stale artifact from an abandoned batch reads as valid provenance."
    )


def test_the_input_the_arms_were_scored_against_is_retained_once() -> None:
    """SOFT. The datafactory pull is pod-only and `drift_detection_ran` is false, so a re-pull is
    not guaranteed to reproduce it. One copy suffices only because the pull proved byte-identical
    across all three pods despite three different queryset_digests (C-194: the descriptor name is
    an input to the digest, so the digest cannot be used to prove data identity).
    """
    pq = INPUT / "calibration_datafactory_df.parquet"
    prov = INPUT / "calibration_datafactory_df.parquet.provenance.json"
    assert pq.exists(), "the evaluated input pull was not retained"
    assert prov.exists(), "the pull's provenance sidecar was not retained"
    assert pq.stat().st_size == 64_720_626, (
        f"input pull is {pq.stat().st_size} bytes, expected 64,720,626 — not the evaluated pull"
    )
    meta = json.loads(prov.read_text())
    assert (meta["month_first"], meta["month_last"]) == (121, 504), (
        f"retained pull covers months {meta['month_first']}-{meta['month_last']}, not 121-504"
    )


# ---------------------------------------------------------------- the guard that was missing

@pytest.mark.parametrize("model", TEN)
def test_a_bundle_that_cannot_be_read_fails_rather_than_passes(model: str) -> None:
    """ADR-025. Each bundle must be listable and non-empty. `_entries` fails rather than
    returning [], so 'nothing matched' can never be mistaken for 'I could not see'.
    This is the check whose absence let a truncated archive read as a successful one earlier
    today: 9 of 10 bundles held 4-21 entries against a complete 117 and were reported as fine.
    """
    n_pred = len(_entries(PREDICTIONS / f"{model}_s64.tar.zst"))
    assert n_pred == 131, f"{model}: predictions bundle has {n_pred} entries, expected 131"
