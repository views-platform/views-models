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


def test_p5_hard_a_runner_exists_that_can_execute_a_views_baseline_source():
    """HARD. No runner in tools/podrun/ can run an arbitrary views-baseline source.

    - pod_run_crafd_chain.sh  : MODELS hardcoded to the eight *_condor, ENSEMBLE=chrome_crate
    - pod_run_chrome_box.sh   : MODELS hardcoded to the eight *_sparrow, ENSEMBLE=chrome_box
    - pod_run_model.sh        : takes any MODEL, but installs views-hydranet and NEVER
                                views-baseline, and its verify step imports views_hydranet.

    ⛔ The last one is worse than absence: verify_env would report a HEALTHY environment, then
    main.py dies on `from views_baseline.manager.baseline_manager import ...`. A green
    precondition check in front of a run that cannot work.
    """
    runners = sorted((REPO / "tools" / "podrun").glob("pod_run_*.sh"))
    assert runners, "no podrun scripts at all"
    capable = []
    for r in runners:
        src = r.read_text()
        takes_arg = 'MODEL="${1' in src or "MODEL=\"${1" in src
        installs_baseline = "views-baseline" in src and "uv pip install" in src
        hardcoded = "MODELS=" in src and "ashen_" in src
        if takes_arg and installs_baseline and not hardcoded:
            capable.append(r.name)
    assert capable, (
        "no runner both accepts an arbitrary model AND installs views-baseline. "
        f"checked: {[r.name for r in runners]}"
    )


@pytest.mark.xfail(reason="no runner exists, so cleanup behaviour is undefined", strict=False)
def test_p7_soft_twenty_one_sources_fit_the_pod_disk_quota():
    """SOFT. One source at S=64 fits 100 GB. Twenty-one sequentially, without archiving and
    deleting between them, does not.

    views-baseline C-54: ~180 GB written per model at S=256, and 7.17 GB per origin
    (13 origins = 93 GB at S=256). Scaled to S=64 that is ~23-45 GB per source depending on
    which figure applies. The pod quota measured on 2026-11-12 was 100 GB on /.

    21 x 23..45 GB = 480..945 GB. So a driver MUST archive-and-delete between sources, and
    since no driver exists that requirement is currently written down nowhere.
    """
    raise AssertionError(
        "unverifiable until a runner exists: 21 sources x ~23-45 GB at S=64 against a 100 GB "
        "quota requires per-source cleanup, which nothing specifies"
    )
