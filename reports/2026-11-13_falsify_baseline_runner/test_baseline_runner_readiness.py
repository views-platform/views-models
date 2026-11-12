"""Failing stubs from /falsify on "the 21 baseline sources AND RUNNERS are ready for the pod".

2026-11-13. Verdict FALSIFIED: 2 hard, 1 soft. Self-audit of code written twenty minutes earlier,
so every verdict here is the author auditing himself — weaker than an independent pass.

NOT part of the suite (`reports/` is in norecursedirs). Run explicitly:
    python -m pytest reports/2026-11-13_falsify_baseline_runner -v
"""

import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools" / "podrun" / "pod_run_baseline_arms.sh"


def _git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout


def test_p1_hard_the_runner_is_committed_and_pushed():
    """HARD. A pod clones a REMOTE branch. An untracked runner does not exist there.

    At audit time: `git ls-files` returned 0, status was `?? tools/podrun/pod_run_baseline_arms.sh`,
    and it was absent from origin. The configs were pushed; the thing that runs them was not.
    """
    assert _git("ls-files", str(RUNNER.relative_to(REPO))).strip(), "runner is untracked"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    on_remote = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"origin/{branch}:{RUNNER.relative_to(REPO)}"],
        capture_output=True,
    ).returncode == 0
    assert on_remote, f"runner is not on origin/{branch}; a pod cannot clone it"


def test_p7_hard_the_runner_asserts_the_output_SHAPE_not_merely_non_emptiness():
    """HARD — and the worst finding, because it fails SILENTLY and then destroys the evidence.

    The runner's only assertion about a source's output is:

        if [ -d "$GEN" ] && [ -n "$(ls -A "$GEN" 2>/dev/null)" ]; then

    A correct calibration run writes 13 origins x 3 targets x 2 arrays (measured from the #695
    campaign archive `plain_burlap_s64.tar.zst`: 13 distinct `origin_N`, 3 distinct `lr_*_best`,
    6 `.np*` files under origin_0). A run that produced ONE origin, or two targets, is non-empty
    and therefore passes.

    ⛔ Compounded by archive-and-clear: the runner then archives that wrong output AND clears
    data/generated. So a run with 1 of 13 origins ends with STATUS OK, a MANIFEST claiming 21
    archives, and the source data gone. Afterwards nothing distinguishes it from a good run.

    Remedy: count `origin_N` directories and `lr_*_best` subdirectories before archiving, and die
    if either is short. The numbers are known, not guessed.
    """
    src = RUNNER.read_text()
    assert "origin_" in src, (
        "the runner never mentions origins, so it cannot be checking how many were produced; "
        "it asserts only that data/generated is non-empty"
    )


def test_p5_soft_clearing_does_not_destroy_the_tracked_gitkeep():
    """SOFT. `find "${GEN:?}" -mindepth 1 -delete` removes `data/generated/.gitkeep`, which is a
    TRACKED file (`git ls-files models/nested_burlap/data/generated` -> .gitkeep).

    Harmless on a disposable pod clone, but it makes the checkout structurally invalid mid-run:
    `tests/test_model_structure.py` requires 7 tracked .gitkeep per source, and anything that
    re-validates structure between sources would fail on a source already run. Cheap to avoid:
    `-mindepth 1 ! -name .gitkeep`.
    """
    src = RUNNER.read_text()
    assert ".gitkeep" in src, (
        "the clear step does not exclude .gitkeep, so it deletes a tracked file"
    )
