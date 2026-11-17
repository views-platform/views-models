"""Failing stubs from the 2026-11-17 falsification audit of this claim:

    "We are ready to run the four moving-window models, and when that run finishes we will
     hold all the data the experiment requires to compare the eight Matryoshka nested-doll
     models against every relevant simplified conflictology analog."

Verdict: FALSIFIED — one hard falsification (H1), two soft (S1, S3).

S2 was designed and REFUTED, and is recorded here because the refutation is the useful
part: views-baseline DOES assert the origin-0 identity the sailcloth acceptance check
rests on — `tests/test_rolling_pooled_conflictology.py:47`,
`test_origin_zero_is_pooled_conflictology_byte_for_byte`, alongside
`test_one_cell_areas_are_rolling_conflictology_at_every_origin`,
`test_a_gap_after_train_end_inside_the_window_raises_naming_the_cell` and
`test_the_catalog_requires_pool_by_and_forwards_every_key`. Read upstream, not executed
here: this laptop has no environment holding views-baseline 1.5.0.

These are stubs, in reports/, NOT in the suite. Each fails today and names what would make it
pass. Promote one to tests/ only if it should hold forever; H1's does not (it is true again the
moment the branch is pushed).
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PLATFORM = REPO.parent
BASELINE = PLATFORM / "views-baseline"
BRANCH = "feat/728-729-matryoshka-pooled-sorter-models"
MOVING_WINDOW = ("rolling_burlap", "narrow_sailcloth", "middle_sailcloth", "broad_sailcloth")


def test_h1_the_four_sources_exist_in_the_commit_a_pod_would_clone():
    """H1, HARD. The pod does not read this laptop. `pod_run_baseline_arms.sh` runs against a
    CLONE of views-models (`[ -d "$REPO/.git" ] || note "clone views-models first"`), and its
    preflight requires `models/<source>/main.py` for every source named. The four moving-window
    models exist only in local commits, so the preflight stops before the paid install.

    Passes when the branch carrying them is pushed. ADR-025: this refuses rather than skips when
    it cannot reach the remote, because a check that cannot see its subject must not report clean.
    """
    ls = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", BRANCH],
        cwd=REPO, capture_output=True, text=True,
    )
    assert ls.returncode == 0 and ls.stdout.strip(), (
        "cannot see the remote branch, so this check cannot be answered (ADR-025)"
    )
    remote_sha = ls.stdout.split()[0]
    missing = [
        m for m in MOVING_WINDOW
        if subprocess.run(
            ["git", "cat-file", "-e", f"{remote_sha}:models/{m}/main.py"],
            cwd=REPO, capture_output=True,
        ).returncode != 0
    ]
    assert not missing, (
        f"the pod would clone {remote_sha[:8]}, which has no "
        f"{', '.join(f'models/{m}/main.py' for m in missing)}. The preflight would stop with "
        f"'no such source'. Push {BRANCH}."
    )


def test_s1_every_doll_level_has_a_simplified_single_level_control():
    """S1, SOFT (adequacy). The eight nested-doll models draw their levels from
    {cell, near, country, map}. `near` is a DISTANCE-WEIGHTED NEIGHBOURHOOD (ADR-024: "nearby
    cells, the closest ones counting most"), not an administrative unit.

    The simplified controls pool by a categorical column: plain_burlap (cell only),
    middle_canvas (`c_id`, country), broad_canvas (`map`). narrow_canvas pools by `gaul1_code`
    (admin-1), a level NO nested-doll model uses. So `near` has no single-level control, and
    four of the eight models contain it: nested, local, mapless, stateless.

    `PooledConflictology` can only pool by a column, and the queryset carries no neighbourhood
    or spatial-block column, so this control needs either a new algorithm in views-baseline or
    a new column in the queryset. It is not fixable by copying a config.
    """
    dolls: dict[str, list[str]] = {}
    for d in sorted((REPO / "models").glob("*_burlap")):
        hp = (d / "configs" / "config_hyperparameters.py").read_text()
        m = re.search(r'"dolls"\s*:\s*\[([^\]]*)\]', hp, re.S)
        if m:
            dolls[d.name] = re.findall(r'"([a-z_]+)"', m.group(1))

    pool_by = {}
    for d in sorted((REPO / "models").glob("*_canvas")):
        meta = (d / "configs" / "config_hyperparameters.py").read_text()
        m = re.search(r'"pool_by"\s*:\s*"([^"]+)"', meta)
        if m:
            pool_by[d.name] = m.group(1)

    #: Which pooling column stands in for which doll level. `near` has no entry, and that is
    #: the finding — not a gap in this map.
    CONTROL_FOR_LEVEL = {"cell": "plain_burlap (no pooling)", "country": "c_id", "map": "map"}

    levels = {lv for ls in dolls.values() for lv in ls}
    uncontrolled = sorted(lv for lv in levels if lv not in CONTROL_FOR_LEVEL)
    affected = sorted(n for n, ls in dolls.items() if set(ls) & set(uncontrolled))
    assert not uncontrolled, (
        f"doll level(s) {uncontrolled} have no simplified single-level control; "
        f"{len(affected)} of {len(dolls)} nested-doll models use one: {affected}. "
        f"Controls available: {pool_by}."
    )


def test_s3_the_runner_pins_the_engine_for_a_comparison_run():
    """S3, SOFT (comparability). `pod_run_baseline_arms.sh` installs
    `views-baseline>=1.4.0,<2.0.0` — a floor, not a pin. On 2026-11-13 that resolved to 1.4.0,
    which is what produced the twelve archives we must compare against; today it resolves to
    1.5.0. The 1.4.0 -> 1.5.0 diff was read and is non-behavioural for those twelve (model_name
    became an attribute, pooling.py is purely additive, the Matryoshka file is untouched), and
    pipeline-core 3.5.0 / datafactory 1.18.1 / evaluation 2.0.0 are unchanged — so the skew is
    benign TODAY. Nothing in the runner keeps it that way tomorrow.

    This stub documents the exposure. The runner is production infrastructure and the remedy is
    an operator one: read `engine:` in the MANIFEST after the run and refuse the data if it is
    not the version the comparison assumes.
    """
    runner = (REPO / "tools" / "podrun" / "pod_run_baseline_arms.sh").read_text()
    floors = re.findall(r'"(views-[a-z-]+)([<>=0-9.,]+)"', runner)
    unpinned = [f"{p}{s}" for p, s in floors if "==" not in s]
    pytest.fail(
        "the engine version of a comparison run is resolved at run time, not pinned: "
        f"{unpinned}. The twelve held archives were made with views-baseline 1.4.0."
    )
