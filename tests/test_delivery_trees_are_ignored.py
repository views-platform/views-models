"""Pod delivery trees are gitignored, and `data/generated/.gitkeep` is not.

⛔ WHY THIS EXISTS. On 2026-11-04 a single `git add -A models/` staged **1.26 GB** across 222
files — pod delivery artefacts that were untracked but *not ignored*, and untracked is not
ignored. Among them was a 115.87 MB draw archive named `INCOMPLETE_9pct_DO_NOT_USE_...`, which
is over GitHub's 100 MB hard limit, belonging to a model that change did not touch.

`*.parquet` already covered the prediction frames, which is exactly why the hazard looked small:
the heavy files are the `.tar.zst` archives (`gate_endpoints.tar.zst` is ~85 MB per arm) and
nothing covered those.

⚠️ The rule is scoped to the delivery SUBDIRECTORY rather than to `*.zst` or to
`data/generated/`, and both halves of that matter:

  - an extension rule leaves `STATUS`, `MANIFEST`, `PROVENANCE` and the wandb staging trees
    stageable, so the hazard stays half-open;
  - a rule on `data/generated/` itself would ignore `data/generated/.gitkeep`, which
    `tests/test_model_structure.py` requires to be TRACKED for all 162 models.

So this file asserts both directions. Paths are synthetic — `git check-ignore` matches patterns
and does not need the file to exist, which keeps this test independent of what happens to be on
this disk and makes it meaningful in CI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]


def _is_ignored(rel_path: str) -> bool:
    """True when git would ignore `rel_path`. Raises if git cannot answer at all."""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", rel_path],
        cwd=REPO, capture_output=True, text=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error. Anything else is not an answer.
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"`git check-ignore` could not answer for {rel_path!r} "
            f"(exit {proc.returncode}): {proc.stderr.strip()!r}. "
            "⛔ Failing rather than reporting a clean result (ADR-025)."
        )
    return proc.returncode == 0


def test_git_check_ignore_is_usable_here():
    """⛔ ADR-025: if the instrument cannot run, every assertion below is vacuous."""
    assert _is_ignored(".gitignore") is False, (
        "`git check-ignore` reports .gitignore as ignored, which cannot be right — the "
        "instrument is misconfigured and the results below mean nothing."
    )


@pytest.mark.parametrize(
    "path",
    [
        "models/calm_orbiter/data/generated/calibration_delivery_20261103/STATUS",
        "models/calm_orbiter/data/generated/calibration_delivery_20261103/gate_endpoints.tar.zst",
        "models/calm_orbiter/data/generated/calibration_delivery_20261103/artifact/PROVENANCE",
        "models/amber_kestrel/data/generated/calibration_delivery_20261013/draws/x.tar.zst",
        "ensembles/chrome_box/data/generated/pod_delivery_20261102/FREEZE",
        "ensembles/chrome_box/data/generated/pod_delivery_20261028_artifacts/ashen_sparrow/.gitkeep",
    ],
)
def test_a_delivery_artefact_is_ignored(path):
    """Everything inside a delivery tree, whatever its extension."""
    assert _is_ignored(path), (
        f"{path} is NOT ignored, so one `git add -A` stages it. The delivery trees ran to "
        "1.26 GB on 2026-11-04 including a file over GitHub's 100 MB limit."
    )


@pytest.mark.parametrize(
    "path",
    [
        "models/calm_orbiter/data/generated/.gitkeep",
        "models/amber_kestrel/data/generated/.gitkeep",
        "ensembles/chrome_box/data/generated/.gitkeep",
    ],
)
def test_the_required_gitkeep_is_not_ignored(path):
    """The other direction, which an over-broad rule breaks for all 162 models at once."""
    assert not _is_ignored(path), (
        f"{path} is ignored, but `tests/test_model_structure.py` requires all seven "
        "REQUIRED_SUBDIRS to be tracked via `git ls-files`. The ignore rule is too broad — "
        "scope it to the delivery subdirectory, not to data/generated itself."
    )
