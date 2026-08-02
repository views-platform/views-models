"""Every tracked shell script declares an interpreter that exists on Linux (C-39).

**Why this test exists rather than just the fix.** C-39 was fixed on 2026-04-21 —
`83fb3a2e`, "replace #!/bin/zsh with #!/usr/bin/env bash in all 79 scripts" — and
marked Resolved. It then regressed, silently, 24 times:

    2026-05-04  first_love, bad_romance, smol_cat, and others
    2026-05-19  fake_model
    2026-06-28  the 12 r2darts models (ravaging_*, roaming_*, warring_*)

Every one of those postdates the fix. They are not stragglers the sweep missed.
The `run.sh` template lives in **views-pipeline-core**
(`views_pipeline_core/templates/model/template_run_sh.py`), it still emits
`#!/bin/zsh`, and it was last touched on 2026-04-03 — eighteen days *before* the
fix landed here. So the fix was applied to the output and never to the generator,
and every model scaffolded since has been born with the defect.

That is the thing this test catches. A fix applied to 131 copies cannot hold when
copy 132 comes from somewhere else; a test can say so on the day it happens
instead of eight weeks later.

**What breaks in practice.** `models/execute_all.sh:10` invokes `"$script"`
directly, and every ensemble README documents `./run.sh` — including
`ensembles/first_love/README.md:52`, which is one of the four ensembles
`monthly_run.sh` runs in production. On a Linux server, where zsh is usually not
installed, that is `bad interpreter: /bin/zsh: No such file or directory`.
`monthly_run.sh` itself calls `bash run.sh`, which ignores the shebang — which is
precisely why this went unnoticed for so long.

**What this test does NOT cover, deliberately.** 18 tracked `run.sh` are committed
non-executable (mode 100644). Those fail the same two entry points with
"Permission denied" instead, and 13 of them overlap the set fixed here. That is a
real gap in the same family, left uncovered rather than quietly folded in, because
changing file modes on production launchers is the maintainer's call and had not
been made when this was written.
"""

from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.green

REPO_ROOT = Path(__file__).resolve().parents[1]

# `env bash` is the portable form; a bare `/bin/bash` is acceptable but discouraged.
# `/bin/zsh` is the one that is actually absent on the machines this platform runs on.
PORTABLE_SHEBANGS = ("#!/usr/bin/env bash", "#!/bin/bash", "#!/usr/bin/env sh", "#!/bin/sh")
NON_PORTABLE = ("zsh",)


def _tracked_shell_scripts():
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.sh"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for name in out.split("\0"):
        if not name:
            continue
        path = REPO_ROOT / name
        if path.is_file():
            yield name, path


def _first_line(path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.readline().rstrip("\n")


def test_no_tracked_shell_script_declares_zsh():
    """The regression itself: an interpreter most of our machines do not have."""
    offenders = [
        (name, _first_line(path))
        for name, path in _tracked_shell_scripts()
        if any(bad in _first_line(path) for bad in NON_PORTABLE)
    ]
    assert not offenders, (
        "C-39 has regressed — these declare zsh, which is absent on Linux servers "
        "and CI runners. If a newly scaffolded model appears here, fix the template "
        "in views-pipeline-core, not the copy:\n"
        + "\n".join(f"  {name}: {line}" for name, line in offenders)
    )


def test_every_tracked_shell_script_has_a_shebang():
    """A script with no shebang is executed by whatever shell happens to call it."""
    missing = [
        name for name, path in _tracked_shell_scripts()
        if not _first_line(path).startswith("#!")
    ]
    assert not missing, "no shebang:\n" + "\n".join(f"  {n}" for n in missing)


def test_shebangs_are_from_the_known_set():
    """Fail loud on an interpreter nobody has considered, rather than allow-by-default."""
    unknown = [
        (name, _first_line(path))
        for name, path in _tracked_shell_scripts()
        if _first_line(path).startswith("#!")
        and _first_line(path) not in PORTABLE_SHEBANGS
    ]
    assert not unknown, (
        "unrecognised interpreter — add it to PORTABLE_SHEBANGS if it is genuinely "
        "portable, and say why:\n"
        + "\n".join(f"  {name}: {line}" for name, line in unknown)
    )
