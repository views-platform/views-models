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

**The executable bit is the second half of the same failure.** A `run.sh` committed
non-executable (mode 100644) fails those same two entry points with "Permission
denied" — a different error from the same cause, and 13 of the 18 that had it
overlapped the zsh set, so those went from one Linux failure straight to another.
Fixed and covered here, on the maintainer's decision (2026-08-02).

`tools/credentials/platform_env.sh` is the one deliberate exception: it is a library,
`source`d and never executed (ADR-018), and marking it executable would advertise an
entry point it does not have.
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


def test_every_run_sh_is_executable():
    """`./run.sh` is the documented entry point, so the bit that permits it is required.

    Every ensemble README says `./run.sh -r calibration ...`, and
    `models/execute_all.sh:10` invokes `"$script"` directly. Without the executable
    bit both give "Permission denied" — the same breakage as the zsh shebang, wearing
    a different error message. 18 files carried this until 2026-08-02.
    """
    offenders = [
        name for name, path in _tracked_shell_scripts()
        if name.endswith("run.sh") and not path.stat().st_mode & 0o111
    ]
    assert not offenders, (
        "run.sh committed non-executable — `./run.sh` and models/execute_all.sh "
        "will fail with Permission denied:\n" + "\n".join(f"  {n}" for n in offenders)
    )


#: Shell files that are SOURCED, never executed. Each defines functions and does nothing
#: useful when run. An executable bit on one would claim an entry point it does not have,
#: which is the inverse of the defect above.
SOURCED_LIBRARIES = (
    ("tools/credentials/platform_env.sh", "ADR-018"),
    ("tools/launcher/postprocessor.sh", "ADR-022"),
)


@pytest.mark.parametrize("relative,adr", SOURCED_LIBRARIES)
def test_the_sourced_libraries_stay_non_executable(relative, adr):
    """The exception, pinned so it is a decision and not an oversight."""
    library = REPO_ROOT / relative
    assert library.is_file(), f"{relative} moved — update this test"
    assert not library.stat().st_mode & 0o111, (
        f"{relative} is sourced, never executed ({adr}); it should not be marked executable"
    )


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
