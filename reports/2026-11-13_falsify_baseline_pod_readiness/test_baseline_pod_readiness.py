"""Failing stubs from /falsify on "the 21 baseline sources are ready for the pod" (2026-11-13).

NOT part of the suite — `reports/` is in norecursedirs. Run explicitly:
    python -m pytest reports/2026-11-13_falsify_baseline_pod_readiness -v

Verdict was FALSIFIED: 2 hard, 1 soft. The CONFIG CONTENT passed every probe (21/21 on gate,
S-consistency, naming, dependency floor, region). What failed is everything around it.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASELINE_21 = (
    "plain_burlap short_linen plain_linen long_linen sheer_velvet plush_velvet flat_velvet "
    "light_tweed even_tweed heavy_tweed nested_burlap lone_burlap local_burlap civic_burlap "
    "global_burlap mapless_burlap stateless_burlap distant_burlap narrow_canvas middle_canvas "
    "broad_canvas"
).split()


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout


def test_p1_hard_every_source_is_committed_and_the_branch_is_pushed():
    """HARD. A pod does a fresh clone of a REMOTE branch. Uncommitted edits and an unpushed
    branch do not exist there, however correct they are on this disk.

    At audit time: all 21 sources had 2 modified-but-unstaged files each (the S=64 edit),
    62 tracked files were uncommitted, and the branch had 0 remote refs.
    """
    dirty = [n for n in BASELINE_21
             if any(l[:2] in (" M", "MM", "??") for l in
                    _git("status", "--porcelain", f"models/{n}").splitlines())]
    assert not dirty, f"{len(dirty)} source(s) have uncommitted changes: {dirty}"

    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    remote = _git("ls-remote", "--heads", "origin", branch).strip()
    assert remote, f"branch {branch!r} is not pushed; a pod cannot clone it"


def test_p5_hard_a_runner_exists_that_can_execute_a_views_baseline_source(tmp_path):
    """HARD, remediated by tools/podrun/pod_run_baseline_arms.sh (ac76c9ec).

    ⚠️ THIS PROBE WAS WRONG BEFORE IT WAS RIGHT. Its first version grepped for
    `MODEL="${1` — the variable name pod_run_model.sh happens to use — and so returned a FALSE
    NEGATIVE against a runner that takes `SOURCES="${*:-...}"` instead. It was matching a token,
    not the behaviour. It now EXECUTES the runner's preflight against a named source and reads
    what it says, which cannot pass on a comment or fail on a naming choice.

    What it must establish: some runner accepts an arbitrary views-baseline source AND installs
    views-baseline. pod_run_model.sh does the first and not the second — it installs
    views-hydranet and verifies by importing views_hydranet, so its verify_env reports a HEALTHY
    environment for a run that then dies on `import views_baseline`.
    """
    runners = sorted((REPO / "tools" / "podrun").glob("pod_run_*.sh"))
    assert runners, "no podrun scripts at all"

    capable = []
    for r in runners:
        src = r.read_text()
        # necessary: it must actually install the engine a baseline source imports
        if "views-baseline" not in src or "uv pip install" not in src:
            continue
        # and it must ACCEPT a source we name — established by running its preflight
        done = subprocess.run(
            ["bash", str(r), "--preflight", "nested_burlap"],
            capture_output=True, text=True, timeout=180,
            env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path),
                 "PODRUN_REPO": str(REPO), "PODRUN_DISK_FLOOR_GB": "1"},
        )
        out = done.stdout + done.stderr
        if "no such source" in out:
            continue                      # it refused the name
        if "sources to run: 1" in out or "nested_burlap" in out:
            capable.append(r.name)

    assert capable, (
        "no runner both accepts an arbitrary views-baseline source AND installs views-baseline. "
        f"checked: {[r.name for r in runners]}"
    )


@pytest.mark.xfail(reason="per-source cleanup now exists but is unverified end to end", strict=False)
def test_p7_soft_twenty_one_sources_fit_the_pod_disk_quota():
    """SOFT, partially remediated. One source at S=64 fits 100 GB; twenty-one sequentially does
    not without archive-and-clear between them.

    views-baseline C-54: ~180 GB written per model at S=256, 7.17 GB per origin (13 origins =
    93 GB at S=256). Scaled to the declared S=64 that is ~23-45 GB per source. The pod quota
    measured 2026-11-12 was 100 GB on /. So 21 x 23..45 GB = 480..945 GB.

    pod_run_baseline_arms.sh now archives each source, reads the archive back, clears
    data/generated (preserving the tracked .gitkeep) and re-checks free space before each source.
    ⚠️ Still xfail because that path has NEVER RUN END TO END: the preflight checks /root/.netrc
    and uv, neither fakeable off a pod without defeating a real guard, so the shape gate and the
    clear are verified at LINE level only. The first pod run must be ONE source.
    """
    runner = (REPO / "tools" / "podrun" / "pod_run_baseline_arms.sh").read_text()
    assert "tar -I" in runner and "! -name .gitkeep" in runner, "no archive-and-clear step"
    raise AssertionError(
        "archive-and-clear exists but is unverified end to end; run one source on a pod first"
    )
