"""Every tracked directory under `models/` and `ensembles/` carries a `main.py`.

⛔ WHY. On 2026-11-04 a rename left twelve directories in git holding **exactly one file** —
`data/generated/.gitkeep` — and nothing else. No `main.py`, no `configs/`. They were created
when a staging pathspec of the form `':(exclude)models/*/data/generated/*'` excluded the
*deletion* of the old `.gitkeep` alongside the artefacts it was aimed at, so the old paths were
never removed from the index while the new ones were added.

⚠️ THE WHOLE SUITE WAS BLIND TO IT. `tests/conftest.py:17-28` builds `ALL_MODEL_DIRS` from
directories that CONTAIN `main.py`, so a ghost directory is skipped by every parametrized model
test — 11,000+ of them stayed green. The catalog generator does not filter that way:
`tools/catalogs/update_readme.py:131` iterates `target_dir.iterdir()` over every directory, and
`ModelPathManager(configs_dir)` on a missing `configs/` died with
`TypeError: expected string or bytes-like object, got 'PosixPath'`.

So the failure surfaced only AFTER the merge, in `update_catalogs.yml`, which runs on push and
not on a pull request. CI was green on the PR and `development` was left with its catalogs
unregenerated.

⭐ The lesson this encodes: a test roster derived from "directories that look like models" cannot
see a directory that stopped looking like one. This check derives its roster from *git* instead.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]

#: Same single source of truth the generator uses (`meta/fixtures.json`, C-61), so a scaffold
#: entry that is deliberately not a model does not read as a ghost.
FIXTURES: set[str] = set(json.loads((REPO / "meta" / "fixtures.json").read_text()))


def _tracked_source_dirs() -> list[str]:
    """Top-level `models/*` and `ensembles/*` directories **as git sees them**.

    Deliberately not `Path.iterdir()`: an untracked directory on one machine is not a repository
    defect, and a tracked directory that is missing locally still breaks CI. git is the subject.
    """
    out = subprocess.run(
        ["git", "ls-files", "--", "models", "ensembles"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return sorted({
        "/".join(p.split("/")[:2]) for p in out if p.count("/") >= 2
    })


SOURCE_DIRS = _tracked_source_dirs()


def test_the_roster_is_not_empty():
    """⛔ ADR-025: if `git ls-files` returned nothing, every assertion below is vacuous."""
    assert len(SOURCE_DIRS) >= 150, (
        f"only {len(SOURCE_DIRS)} tracked source directories found, expected 150+. "
        "`git ls-files` is the subject here — if it cannot answer, this check must fail rather "
        "than pass over an empty list."
    )


@pytest.mark.parametrize("source_dir", SOURCE_DIRS)
def test_a_tracked_source_directory_has_a_main_py(source_dir):
    """A directory in git without `main.py` is a ghost: invisible to conftest, fatal to the
    catalog generator."""
    if Path(source_dir).name in FIXTURES:
        pytest.skip(f"{source_dir} is a declared fixture in meta/fixtures.json")

    tracked = subprocess.run(
        ["git", "ls-files", "--", f"{source_dir}/main.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert tracked, (
        f"{source_dir} is tracked in git but has no main.py. `tools/catalogs/update_readme.py` "
        f"iterates every directory and will die on its missing configs/ — AFTER the merge, in "
        f"update_catalogs.yml, which does not run on pull requests. Either complete the "
        f"directory or remove it from git; files it does hold: "
        f"{subprocess.run(['git', 'ls-files', '--', source_dir], cwd=REPO, capture_output=True, text=True).stdout.split()}"
    )
