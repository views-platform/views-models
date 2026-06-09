"""
Falsification stubs — "chunky_bunny is ready to run" (2026-06-09).

FALSIFIED. The ensemble (`EnsembleManager._train_model_artifact`) trains each
constituent by subprocessing that model's own `run.sh`, which activates a PER-MODEL
conda env (`envs/views_stepshifter` for plain/Hurdle, `envs/views_r2darts2` for the
4 DL models). Those envs:
  (a) do NOT exist on this box (only views-baseline / views_ensemble / views-hydranet
      are present), so run.sh would create them from `requirements.txt`, and
  (b) pin the PUBLISHED `views-stepshifter>=1.0.0,<2.0.0`, which has NO
      `target_transform` mechanism (origin/main: 0 occurrences) — so the `log1p`
      fix we put in the configs would be SILENTLY IGNORED and the constituents
      would train RAW again (the exact bug we are fixing), or error.

Our entire validation (car_radio/bittersweet_symphony/counting_stars MSLE ~0.41)
was run in `views_pipeline` (editable install of the FIXED branch) — which is NOT
the env the ensemble uses. The green test gave false confidence.

These tests encode the readiness preconditions. They FAIL today; they pass once the
fix is released (merged to main + published) OR the per-model envs are provisioned
with the fixed code.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STEPSHIFTER_REPO = REPO.parent / "views-stepshifter"


def _origin_main_gate_has_target_transform() -> bool:
    out = subprocess.run(
        ["git", "show", "origin/main:views_stepshifter/infrastructure/reproducibility_gate.py"],
        cwd=STEPSHIFTER_REPO, capture_output=True, text=True,
    )
    return "target_transform" in out.stdout


@pytest.mark.xfail(reason="FALSIFIED: target_transform fix not merged to main / not released", strict=True)
def test_target_transform_fix_is_released():
    """The published views-stepshifter that constituents pip-install must support
    target_transform — otherwise the log1p fix is silently ignored at ensemble run."""
    assert _origin_main_gate_has_target_transform(), (
        "origin/main reproducibility_gate has no target_transform — a fresh "
        "envs/views_stepshifter (pip install views-stepshifter>=1.0.0,<2.0.0) "
        "would train RAW, silently discarding the log1p fix."
    )


@pytest.mark.xfail(reason="FALSIFIED: per-model envs not provisioned", strict=True)
def test_per_model_envs_exist():
    """The ensemble runs each constituent via run.sh in its own env; those envs must
    exist (and carry the fixed code), not be created on-the-fly from published reqs."""
    envs = REPO / "envs"
    assert (envs / "views_stepshifter").is_dir(), "envs/views_stepshifter missing"
    assert (envs / "views_r2darts2").is_dir(), "envs/views_r2darts2 missing (4 DL constituents)"


@pytest.mark.xfail(reason="FALSIFIED: chunky_bunny constituents not all artifact-ready in a consistent env", strict=True)
def test_ensemble_uses_the_fixed_code_path():
    """Guard against the views_pipeline (tested) vs run.sh/views_stepshifter (executed)
    mismatch: the env the ensemble actually invokes must be the one validated."""
    # Placeholder for an integration assertion once the release/env path is decided.
    assert False, "ensemble execution env != validation env (views_pipeline editable)"
