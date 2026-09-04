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


#: Launchers that are NOT scaffold output. Each is hand-written and owns its own body,
#: so neither the clone header nor the template's macOS block applies. Shrink-only: a
#: new entry here means someone hand-wrote a launcher, which is a decision, not a sweep.
HAND_WRITTEN_LAUNCHERS = frozenset({
    "apis/seldon_api/run.sh",
    "apis/un_fao/run.sh",
    "postprocessors/un_crafd/run.sh",   # ADR-022 wrapper
    "postprocessors/un_fao/run.sh",     # ADR-022 wrapper
})

CLONE_HEADER = "# GENERATED — do not edit by hand."

#: The ONE script permitted to write the user's shell profile. `bootstrap.sh` is
#: one-time setup that the operator runs knowingly and once; persisting the macOS
#: libomp flags is its job, and #310 says so explicitly ("it belongs in one-time
#: setup — which is what bootstrap.sh (S8) is for"). A launcher is the opposite: it
#: runs on every model, every month, and its user did not ask for a profile edit.
#: Shrink-only. Adding a second entry means something other than setup is mutating
#: global user state, which is the defect this file exists to prevent.
PROFILE_WRITERS = frozenset({"bootstrap.sh"})


def _generated_run_scripts():
    """Scaffold-produced launchers only.

    Matched on the exact basename, not `endswith("run.sh")`: the repo root holds
    `monthly_run.sh`, the production orchestrator, which is hand-written and would be
    swept up by a suffix match. A test that quietly demands a clone header from the
    monthly run is worse than no test.
    """
    for name, path in _tracked_shell_scripts():
        if path.name == "run.sh" and name not in HAND_WRITTEN_LAUNCHERS:
            yield name, path


def test_no_run_sh_rewrites_the_user_shell_profile():
    """A script named "run this model" must not mutate ~/.zshrc (views-models#310).

    Until this sweep, 129 launchers appended `LDFLAGS`/`CPPFLAGS`/`DYLD_LIBRARY_PATH`
    to the user's profile and then `source`d it. The block is macOS-gated, so Linux
    never saw it — which is exactly why it survived: the machines that run production
    could not observe the defect, and the machines that could are not the ones anyone
    audits.

    The fix is a REPLACEMENT, not a deletion. libomp really is off the default search
    paths on a Mac; upstream's template exports the three variables for the duration of
    the run instead (views-pipeline-core#384, their `31054cf`). Deleting the block
    outright would break Mac runs, so this test must not be "satisfied" that way — see
    the companion assertion below.
    """
    offenders = [
        name for name, path in _tracked_shell_scripts()
        if name not in PROFILE_WRITERS
        and "zshrc" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, (
        "a launcher writes to the user's shell profile — export for the run instead, "
        "as views_pipeline_core.templates.model.template_run_sh does:\n"
        + "\n".join(f"  {n}" for n in offenders)
    )


def test_the_macos_libomp_support_was_replaced_not_deleted():
    """The other half of the rule above: Mac support must still be there.

    Without this, deleting the whole `darwin` block would turn the previous test green
    while silently breaking every Mac run — a fix that passes by removing the feature.
    """
    missing = [
        name for name, path in _generated_run_scripts()
        if "libomp" not in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not missing, (
        "the macOS libomp block is gone, not replaced — Mac runs will fail to link:\n"
        + "\n".join(f"  {n}" for n in missing)
    )


def test_every_generated_run_sh_is_stamped_as_a_clone():
    """131 clones existed and nobody typed anything (views-models#310).

    The header is only truthful as of views-pipeline-core#384: before it, the template
    emitted `#!/bin/zsh` while these files carried bash after the `83fb3a2e` hand-fix,
    so "GENERATED — do not edit by hand" would have contradicted the very edit that
    made them correct. Now the template emits what they contain, and the claim holds.
    """
    unstamped = [
        name for name, path in _generated_run_scripts()
        if CLONE_HEADER not in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not unstamped, (
        "scaffolded run.sh with no clone header. If this is newly generated, the "
        "generator in views-pipeline-core should emit it — do not hand-stamp copy "
        "132 and call it fixed (that is how C-39 regressed 24 times):\n"
        + "\n".join(f"  {n}" for n in unstamped)
    )


def test_the_hand_written_launcher_pin_only_shrinks():
    """A hand-written launcher is a decision; the list must not grow by accident."""
    actual = {
        name for name, path in _tracked_shell_scripts()
        if path.name == "run.sh"
        and CLONE_HEADER not in path.read_text(encoding="utf-8", errors="replace")
    }
    unexpected = actual - HAND_WRITTEN_LAUNCHERS
    assert not unexpected, (
        "unstamped launcher not on the hand-written list — either it is scaffold "
        "output (stamp it) or it is genuinely hand-written (say so, and why):\n"
        + "\n".join(f"  {n}" for n in sorted(unexpected))
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
