"""A launcher's conda prefix must be an environment this repository owns, and it must work.

`tests/test_environment_sharing.py` maps which launcher declares which prefix, and who it
shares that prefix with. It never looks at the prefix. For five months that blindness hid
the following, found 2026-10-28:

    envs/views-baseline  -> /home/simon/anaconda3/envs/views_pipeline      (symlink, 2026-05-23)
    envs/views-hydranet  -> /home/simon/anaconda3/envs/views-hydranet-env  (symlink, 2026-05-24)

`run.sh` tests its prefix with `[ -d "$env_path" ]`, which is true for a symlink to a
directory. So the `conda create --prefix` branch never fires, and the launcher silently
adopts a foreign environment instead of building an isolated one. Two consequences, both
measured:

    the test env is written   45 models declare `envs/views-baseline`, so each of their runs
                              pip-installs into the conda env this repository's suite runs in
                              (`project_test_env_views_pipeline`). A green suite taken after a
                              model run says nothing about the committed tree.
    C-151 becomes unfollowable  C-151's rule for that env is "the integration runner only;
                              never a model's `run.sh` in `views_pipeline`". Through the
                              symlink, every one of the 45 runs there by construction. The
                              rule cannot be obeyed by choosing differently.

Why the defect reads as a misconfiguration rather than a symlink: `python` reports the local
prefix, because it is invoked through the symlink path and finds the `lib/python3.11`
landmark beside it, while `pip` reports the resolved one. Both are correct and they are the
same environment under two names — so comparing them concludes, wrongly, that pip is broken.

## Why this asks the interpreter instead of reading the filesystem

Two independent clean-context guard-mode audits were run against earlier versions of this
file before it was relied on. The first found 18 states it called acceptable; the second,
against the repaired version, found 22. The lesson of both, and the reason the current shape
is what it is:

**Inspecting path names cannot establish that an environment is usable.** The version that
asked "does `realpath` stay inside the repository", then the version that added
`conda-meta/history` plus `bin/python` and `lib` resolving in-repo, both passed:

  - a prefix whose `lib/python3.11/site-packages` is a symlink into a foreign environment —
    the prefix directory is innocently inside `envs/`, every landmark checks out, and a
    `pip install` there writes the foreign env. That is the #644 harm itself, reported green.
  - a `cp -al` hardlinked clone of a foreign env, inode-shared, writes propagating out.
  - `bin/python` symlinked to *another* prefix's interpreter, or dangling.
  - a half-built prefix: zero-byte `conda-meta/history`, zero-byte `bin/python`, empty `lib`.
  - `toolz-0.12.1.dist-info` beside an empty or foreign-symlinked `toolz/`, so the recorded
    version is compliant and `import toolz` fails.

And it *failed* on healthy environments: `.egg-info` metadata, a stale second `lib/python*`
tree, a stale `dist-info` from an interrupted uninstall, and `toolz-0.12.1-py3.11.dist-info`
were all reported broken. A signal that is green when the environment is broken and red when
it is fine cannot carry a go/no-go decision.

So the check now **runs the prefix's own interpreter once** and asks it where it is and what
it imported. That answers the question the decision actually asks — can a model run here —
and it subsumes every survivor above while removing every false positive above.

**The toolz floor is applied to every owned prefix, not only to "datafactory tenants".**
Detecting tenancy from `requirements.txt` text missed the three most likely real spellings
(a `git+https` pin, an `-e path`/local wheel, and the common case where datafactory arrives
*transitively* through `views_pipeline_core`), and made two states permanently red with no
green reachable (a launcher using `-r`, and a launcher with no `requirements.txt`). Since
satisfying the floor costs nothing in a prefix that does not need it, the tenancy question
is gone. Less code, no survivors, no false positives.

## Scope, stated honestly: these are workstation guards, not CI coverage

`envs/` is gitignored (`.gitignore:192`, `git ls-files envs/` is empty), so on a fresh CI
checkout no declared prefix exists and the prefix tests have nothing to inspect — measured:
they pass vacuously there. Only the census and the `env_path` sweep are load-bearing in CI.
That is a real limit on where this invariant can be checked. **Do not read a green CI run as
evidence about any environment.**

**ADR-025 applies to the shape of these checks, not only their subject.** An absent `envs/`
means zero environments exist, which is a real observation. An `envs/` that exists but is not
a directory, a prefix nothing declares, a launcher the census cannot parse, and an unreadable
file are all cases of *cannot see the subject*, and each fails here rather than passing
quietly or raising a bare traceback.
"""

from pathlib import Path
import os
import re
import subprocess

import pytest

from packaging.version import InvalidVersion, Version

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_DIR = REPO_ROOT / "envs"

# Every shell spelling of the inline conda-prefix declaration that bash would honour:
# optional `export`, leading whitespace, spaces around `=`, single/double/no quotes, and
# `${project_path}` as well as `$project_path`. A guard that only matches today's spelling
# stops guarding the moment the template is regenerated, and five legal spellings were each
# shown to drop a launcher out of the census silently.
_ENV_PATH = re.compile(
    r"""^[ \t]*(?:export[ \t]+)?env_path[ \t]*=[ \t]*["']?\$\{?project_path\}?/envs/(?P<env>[^"'\s#]+)""",
    re.M,
)
# Any assignment to env_path at all, however spelled. Compared against the count above so a
# launcher cannot carry one parseable declaration plus a second, effective one the census
# never sees — a dead `if false` branch, a heredoc, or simply the next line.
_ENV_PATH_ANY = re.compile(r"""^[ \t]*(?:export[ \t]+)?env_path[ \t]*=""", re.M)
# The postprocessors name their prefix and let the shared delivery body build the path
# (ADR-022). It still lands at `envs/<name>`: `tools/launcher/postprocessor.sh:32` is
# `env_path="$project_path/envs/$POSTPROCESSOR_ENV_NAME"`. An earlier version of this file
# excluded them on the written claim that it "is not where it lives", which was false and
# left the prefix producing the FAO deliverable outside the only guard that would notice.
_POSTPROCESSOR_ENV = re.compile(
    r"""^[ \t]*(?:export[ \t]+)?POSTPROCESSOR_ENV_NAME[ \t]*=[ \t]*["']?(?P<env>[^"'\s#]+)""",
    re.M,
)

# The toolz floor that makes `import tlz.curried` work on Python 3.11. Below it, every
# datafactory fetch dies on `AttributeError: _uninitialized_submodules` several libraries from
# the cause, because run.sh does not check pip's exit status (views-pipeline-core#560).
TOOLZ_FLOOR = Version("0.12.1")

# One import and three answers. `-I` so neither the user's site-packages nor PYTHONPATH can
# supply the modules — without it a foreign toolz on the path would answer for the prefix.
_PROBE = (
    "import sys, toolz, tlz.curried; "
    "print(sys.prefix); print(toolz.__file__); print(toolz.__version__)"
)
_PROBE_TIMEOUT_SECONDS = 180


def _launchers() -> list[Path]:
    return (
        sorted(REPO_ROOT.glob("models/*/run.sh"))
        + sorted(REPO_ROOT.glob("ensembles/*/run.sh"))
        + sorted(REPO_ROOT.glob("postprocessors/*/run.sh"))
    )


def _read(path: Path) -> str | None:
    """File contents, or None when the subject cannot be read (never a silent empty string)."""
    try:
        return path.read_text()
    except OSError:
        return None


def _declared_prefixes() -> dict[str, list[Path]]:
    """Map each declared prefix name to the launcher directories declaring it.

    `finditer`, not `search`: a launcher declaring two prefixes had only its first censused.
    """
    found: dict[str, list[Path]] = {}
    for run_sh in _launchers():
        text = _read(run_sh)
        if text is None:
            continue
        for pattern in (_ENV_PATH, _POSTPROCESSOR_ENV):
            for match in pattern.finditer(text):
                found.setdefault(match.group("env"), []).append(run_sh.parent)
    return found


def _probe(prefix: Path) -> tuple[bool, str]:
    """Run the prefix's interpreter and ask it where it is. (ok, detail-for-the-message).

    This is the whole check. A prefix passes only if its own python starts, imports
    `tlz.curried`, reports `sys.prefix` inside the prefix, loads `toolz` from inside the
    prefix, and reports a toolz at or above the floor.
    """
    interpreter = prefix / "bin" / "python"
    if not interpreter.is_file():
        return False, "no bin/python — not a usable environment"
    try:
        done = subprocess.run(
            [str(interpreter), "-I", "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except OSError as error:
        return False, f"bin/python could not be executed: {error}"
    except subprocess.TimeoutExpired:
        return False, f"bin/python did not answer within {_PROBE_TIMEOUT_SECONDS}s"

    if done.returncode != 0:
        last = (done.stderr or "").strip().splitlines()
        detail = last[-1] if last else f"exit {done.returncode}"
        # The overwhelmingly likely cause here, and the one worth naming: C-151.
        return False, f"its own python cannot `import tlz.curried` — {detail}"

    lines = done.stdout.strip().splitlines()
    if len(lines) != 3:
        return False, f"probe returned {len(lines)} lines, expected 3"
    sys_prefix, toolz_file, toolz_version = (line.strip() for line in lines)

    resolved = prefix.resolve()
    for label, value in (("sys.prefix", sys_prefix), ("toolz", toolz_file)):
        try:
            inside = Path(value).resolve().is_relative_to(resolved)
        except OSError:
            inside = False
        if not inside:
            return False, f"{label} is {value}, outside the prefix — it is using another environment"

    try:
        installed = Version(toolz_version)
    except InvalidVersion:
        return False, f"toolz reports an unparseable version {toolz_version!r}"
    if installed < TOOLZ_FLOOR:
        return False, f"toolz {installed} < {TOOLZ_FLOOR} (C-151)"
    return True, f"toolz {installed}"


def _is_owned_prefix(path: Path) -> bool:
    """Structural ownership: a real, in-repo conda prefix rather than a pointer at one.

    Deliberately shallow and kept separate from `_probe`, because it must give a verdict for
    a prefix whose interpreter does not run at all — which is exactly the 2026-05-23 state.
    An alias to another owned prefix inside `envs/` is allowed: that is a deliberate way to
    let two launchers share one environment, and rejecting it was a false positive.
    """
    if path.is_symlink():
        target = path.resolve()
        return target.parent == ENVS_DIR.resolve() and _is_owned_prefix(target)
    if not path.is_dir():
        return False
    history = path / "conda-meta" / "history"
    if history.is_symlink() or not history.is_file():
        return False
    return path.resolve().is_relative_to(REPO_ROOT)


def test_the_prefix_census_can_still_read_its_subject():
    """Every launcher must declare exactly one prefix, and this file must be able to read it.

    Two properties, because each failed independently under mutation. A launcher the regex
    cannot parse is a launcher whose environment nothing guards; and a launcher with a second
    `env_path` assignment can satisfy the census with a decoy while a later, effective line
    points somewhere else.
    """
    launchers = _launchers()
    assert launchers, (
        "No models/*/run.sh, ensembles/*/run.sh or postprocessors/*/run.sh found at all. "
        "Either the launchers are gone or this file's idea of where they live is wrong; in "
        "both cases the prefix tests are vacuous, so this refuses (ADR-025)."
    )
    unreadable, unmatched, ambiguous = [], [], []
    for run_sh in launchers:
        name = run_sh.relative_to(REPO_ROOT)
        text = _read(run_sh)
        if text is None:
            unreadable.append(f"{name}: cannot be read")
            continue
        parseable = len(_ENV_PATH.findall(text))
        if parseable == 0 and not _POSTPROCESSOR_ENV.search(text):
            unmatched.append(f"{name}: declares no prefix this file can parse")
        elif len(_ENV_PATH_ANY.findall(text)) != parseable:
            ambiguous.append(
                f"{name}: {len(_ENV_PATH_ANY.findall(text))} assignments to env_path but only "
                f"{parseable} parseable — the effective one may not be the one censused"
            )
    problems = unreadable + unmatched + ambiguous
    assert not problems, "\n".join(
        [
            "The prefix census cannot account for every launcher, so some launcher's "
            "environment is guarded by nothing:",
            *(f"  {line}" for line in problems),
            "",
            "Widen the declaration patterns at the top of this file rather than accepting "
            "the gap — a launcher invisible to the census is exactly how a prefix stayed "
            "symlinked to the suite's own conda env for five months (#644).",
        ]
    )


def test_no_script_points_a_launcher_environment_outside_envs():
    """`env_path` must always land under `envs/`, in every shell script in the repository.

    The census only reads the three launcher globs. This sweeps every `.sh` for an `env_path`
    assignment that escapes `envs/`, which is the shape that would reintroduce #644 from a
    script the census does not look at. Scripts that build a venv somewhere else entirely
    (`tools/podrun/*` use `uv venv /workspace/venv`) never assign `env_path`, so they are
    unaffected rather than exempted.
    """
    offenders = []
    for script in sorted(REPO_ROOT.glob("**/*.sh")):
        if ENVS_DIR in script.parents:
            continue
        text = _read(script)
        if text is None or not _ENV_PATH_ANY.search(text):
            continue
        if len(_ENV_PATH_ANY.findall(text)) != len(_ENV_PATH.findall(text)):
            offenders.append(
                f"{script.relative_to(REPO_ROOT)}: assigns env_path to something other than "
                f"$project_path/envs/<name>"
            )
    assert not offenders, "\n".join(
        [
            "A script points a launcher environment outside envs/:",
            *(f"  {line}" for line in offenders),
            "",
            "Every launcher environment belongs at envs/<name> so it is isolated and "
            "guarded. A prefix elsewhere — above all one in ~/anaconda3 — means model runs "
            "install into an environment this repository does not own (#644).",
        ]
    )


def test_envs_holds_no_prefix_that_no_launcher_declares():
    """An environment nothing declares means the census is blind, not that the env is unused."""
    if not os.path.lexists(ENVS_DIR):
        return
    assert ENVS_DIR.is_dir(), (
        f"{ENVS_DIR} exists but is not a directory "
        f"({'dangling symlink' if ENVS_DIR.is_symlink() else 'regular file'}). Every prefix "
        "test below would read it as 'no environments exist', which is indistinguishable "
        "from a fresh checkout — so this refuses instead (ADR-025)."
    )
    declared = set(_declared_prefixes())
    undeclared = sorted(
        entry.name
        for entry in ENVS_DIR.iterdir()
        # Only things that could be a prefix. A `.gitkeep` or a README in a gitignored
        # directory is not a stale environment, and reporting it as one told the reader to
        # delete a file that belongs there.
        if entry.name not in declared
        and not entry.name.startswith(".")
        and (entry.is_dir() or entry.is_symlink())
    )
    assert not undeclared, "\n".join(
        [
            "envs/ holds entries that no launcher declares:",
            *(f"  envs/{name}" for name in undeclared),
            "",
            "Either a launcher declares a prefix in a spelling the census cannot parse — in "
            "which case that environment is unguarded — or this is a stale prefix. Park a "
            "prefix you are retiring OUTSIDE envs/ (e.g. `mv envs/x ~/views-envs-parked/x`), "
            "not under a suffix here: a name-based exemption in this directory is a hole a "
            "live, mis-rooted prefix can hide in.",
        ]
    )


def test_every_declared_prefix_is_an_environment_this_repository_owns():
    """Structural: a declared prefix that exists must be an in-repo conda prefix.

    This is the check that was missing on 2026-05-23, and it is kept separate from the probe
    below so that a prefix whose interpreter cannot run still gets a precise verdict. It goes
    green when the prefix is a real environment, which `run.sh` builds by itself via its
    `conda create --prefix` branch once the symlink is out of the way.
    """
    declared = _declared_prefixes()
    # `lexists`, not `exists`: a dangling symlink into anaconda3 is a mis-rooted prefix that
    # `exists()` reports as absent, so it was skipped rather than examined.
    present = {n: ENVS_DIR / n for n in declared if os.path.lexists(ENVS_DIR / n)}
    if not present:
        # Looked, and there is nothing to be wrong: no declared prefix is built here. The
        # normal state in CI. The three tests above cover every "cannot see" case, so this
        # branch can no longer be reached while a mis-rooted prefix sits on disk.
        return

    def why(path: Path) -> str:
        if path.is_symlink():
            return f"symlink -> {os.readlink(path)}"
        if not path.is_dir():
            return "not a directory"
        history = path / "conda-meta" / "history"
        if history.is_symlink():
            return "conda-meta/history is a symlink — its provenance is another environment's"
        if not history.is_file():
            return "no conda-meta/history — not a conda prefix"
        return f"resolves to {path.resolve()}, outside the repository"

    disowned = {n: p for n, p in present.items() if not _is_owned_prefix(p)}
    assert not disowned, "\n".join(
        [
            "A launcher's conda prefix is not an environment this repository owns, so every "
            "launcher declaring it installs into something else:",
            *(
                f"  envs/{name}: {why(path)}   ({len(declared[name])} launchers)"
                for name, path in sorted(disowned.items())
            ),
            "",
            "`run.sh` uses `[ -d \"$env_path\" ]`, which is true for a symlink, so it adopts "
            "the target instead of building an isolated prefix. Where the target is the "
            "suite's own conda env, a model run rewrites the environment that validates this "
            "repository. See #644; remedy: move it outside envs/ and let run.sh rebuild.",
        ]
    )


def test_every_owned_prefix_can_import_tlz_from_inside_itself():
    """Functional: run each prefix's interpreter and make it prove it is self-contained.

    One execution answers what no amount of filesystem inspection could: that the prefix has
    a working interpreter, that it imports from its own site-packages rather than a foreign
    environment's, and that `import tlz.curried` — the import C-151 breaks and every
    datafactory fetch depends on — actually succeeds.

    The floor applies to every owned prefix, not only to prefixes whose `requirements.txt`
    mentions datafactory. See the module docstring: tenancy read off that file missed VCS
    pins, path installs and the transitive case, and made two states permanently red.
    """
    declared = _declared_prefixes()
    owned = {
        name: ENVS_DIR / name
        for name in declared
        if os.path.lexists(ENVS_DIR / name) and _is_owned_prefix(ENVS_DIR / name)
    }
    if not owned:
        # Nothing owned to probe. Unowned prefixes are the test above's finding, not a second
        # failure here, and an absent envs/ is CI.
        return

    findings = []
    for name, prefix in sorted(owned.items()):
        ok, detail = _probe(prefix)
        if not ok:
            findings.append(f"  envs/{name}: {detail}   ({len(declared[name])} launchers)")

    assert not findings, "\n".join(
        [
            "A prefix this repository owns cannot run a model: its own interpreter could not "
            "import `tlz.curried` from inside the prefix (C-151):",
            *findings,
            "",
            'Remedy, after the requirements install: pip install "toolz>=0.12.1" using THAT '
            "prefix's own pip. It must be the LAST install, or the next resolve reverts it — "
            "`run.sh` re-resolves requirements.txt on every invocation.",
        ]
    )
