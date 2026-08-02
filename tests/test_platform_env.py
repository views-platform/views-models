"""`tools/credentials/platform_env.sh` — the one writer of the Appwrite environment (#308, #309).

These are **behavioural** tests: each one sources the real shell file against a fixture
registry and asserts on exit codes and stderr. The tests they replace grepped `run.sh` for
strings, which was the best available when the logic was inline and unreachable — a string
test cannot tell "the code does this" from "a comment mentions this", which is C-57 and
which those tests actually tripped over once.

Two rules under test:

* **#308** — an unresolvable or unreadable registry is fatal. Warning and continuing does
  not save the run; it moves the failure to the datastore boundary minutes later, where it
  describes a symptom instead of a cause.
* **#309** — coordinates come from the registry and the secret from the operator. `.env`
  declaring a coordinate the registry owns is a data race decided by line order, so it is
  reported as an error rather than resolved by precedence.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.green]

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_ENV = REPO_ROOT / "tools" / "credentials" / "platform_env.sh"

pytest.importorskip("tomllib", reason="registry_to_env needs Python 3.11+ for tomllib")

REGISTRY = """
[connection.APPWRITE_ENDPOINT]
class = "connection"
value = "https://fixture.test/v1"

[target.APPWRITE_UNFAO_BUCKET_ID]
class = "target"
value = "unfao_bucket"
"""


@pytest.fixture
def repo(tmp_path):
    """A minimal fake checkout carrying only what platform_env.sh needs."""
    (tmp_path / "tools" / "credentials").mkdir(parents=True)
    shutil.copy(PLATFORM_ENV, tmp_path / "tools" / "credentials" / "platform_env.sh")
    shutil.copy(
        REPO_ROOT / "tools" / "credentials" / "registry_to_env.py",
        tmp_path / "tools" / "credentials" / "registry_to_env.py",
    )
    (tmp_path / "registry.toml").write_text(REGISTRY, encoding="utf-8")
    return tmp_path


def call(repo, snippet, registry=None, env_extra=None):
    """Source platform_env.sh in the fake repo and run `snippet`."""
    env = dict(os.environ)
    # The interpreter running these tests can read the registry; make it the one on PATH.
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    env["APPWRITE_REGISTRY"] = str(registry if registry is not None else repo / "registry.toml")
    for key in ("APPWRITE_ENDPOINT", "APPWRITE_UNFAO_BUCKET_ID", "APPWRITE_DATASTORE_API_KEY"):
        env.pop(key, None)
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", "-c", f'. "{repo}/tools/credentials/platform_env.sh"\n{snippet}'],
        capture_output=True, text=True, env=env, timeout=60,
    )


# ── #308: absence and unreadability are both fatal ────────────────────────────────────

def test_missing_registry_is_fatal_and_names_the_path_and_the_override(repo):
    result = call(repo, "platform_env_require_registry", registry="/nonexistent/registry.toml")
    assert result.returncode == 1
    assert "/nonexistent/registry.toml" in result.stderr, "must name the path it tried"
    assert "APPWRITE_REGISTRY=" in result.stderr, (
        "must name the override — the person hitting this has a different layout and "
        "needs the escape hatch, not a diagnosis"
    )


def test_unreadable_registry_is_fatal(repo):
    (repo / "broken.toml").write_text("this is not = valid = toml [[[", encoding="utf-8")
    result = call(repo, "platform_env_coordinates", registry=repo / "broken.toml")
    assert result.returncode == 1, "a registry that exists but will not parse is not a source"
    assert "could not be read" in result.stderr


def test_registry_declaring_no_coordinates_is_fatal(repo):
    """Silence here would export nothing and let validation pass on an empty set."""
    (repo / "empty.toml").write_text("[meta]\nnote = 'nothing here'\n", encoding="utf-8")
    result = call(repo, "platform_env_coordinates", registry=repo / "empty.toml")
    assert result.returncode == 1
    assert "declared no coordinates" in result.stderr


@pytest.mark.red
def test_a_failed_read_does_not_report_success(repo):
    """The bug this file shipped with, pinned so it cannot return.

    The first draft captured the exit status inside `if ! cmd; then status=$?; fi`, where
    `$?` is the status of the NEGATION — 0, because the negation succeeded. Every caller
    then reported success while exporting nothing: the same shape as a write path that
    logs "uploaded successfully" and uploads nothing.
    """
    result = call(
        repo,
        'platform_env_load; echo "LOAD=$?"\nplatform_env_validate; echo "VALIDATE=$?"',
        registry="/nonexistent/registry.toml",
    )
    assert "LOAD=1" in result.stdout, "a failed load must not report success"
    assert "VALIDATE=1" in result.stdout, "validation must not pass on an unloaded environment"


# ── #309: one writer ──────────────────────────────────────────────────────────────────

def test_env_declaring_a_registry_owned_coordinate_is_fatal(repo):
    (repo / ".env").write_text(
        "APPWRITE_DATASTORE_API_KEY=fake\nAPPWRITE_ENDPOINT=https://wrong.example\n",
        encoding="utf-8",
    )
    result = call(repo, "platform_env_assert_no_env_conflicts")
    assert result.returncode == 1
    assert "APPWRITE_ENDPOINT" in result.stderr, "must name the variable"
    assert str(repo / ".env") in result.stderr, "must name .env as one source"
    assert "registry" in result.stderr.lower(), "must name the registry as the other"


def test_env_carrying_only_the_secret_is_fine(repo):
    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=fake\n", encoding="utf-8")
    assert call(repo, "platform_env_assert_no_env_conflicts").returncode == 0


def test_coordinates_are_exported_from_the_registry(repo):
    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=fake\n", encoding="utf-8")
    result = call(repo, 'platform_env_load && echo "E=$APPWRITE_ENDPOINT B=$APPWRITE_UNFAO_BUCKET_ID"')
    assert result.returncode == 0, result.stderr
    assert "E=https://fixture.test/v1 B=unfao_bucket" in result.stdout


def test_an_absent_secret_is_fatal_and_names_the_variable(repo):
    """No `.env`, so the secret is absent. The contract is that the NAME is reported.

    Asserting on the word "missing" would pin the wording rather than the contract: after
    the C-112 fix the gap is caught earlier, by `platform_env_export_secret`, whose message
    is more specific ("is not set and <path> does not exist"). Naming the variable is what
    a person needs; the phrasing is not the promise.
    """
    result = call(repo, "platform_env_load; echo STATUS=$?")
    assert "STATUS=0" not in result.stdout, "an absent secret must be fatal"
    assert "APPWRITE_DATASTORE_API_KEY" in result.stderr, (
        "the failure must name the variable, not just report a count"
    )


def test_the_secret_value_is_never_rendered(repo):
    """A check that prints a credential to prove it found one has published it."""
    sentinel = "SENTINEL-SECRET-MUST-NOT-APPEAR"
    (repo / ".env").write_text(f"APPWRITE_DATASTORE_API_KEY={sentinel}\n", encoding="utf-8")
    result = call(repo, "platform_env_load && platform_env_validate")
    assert result.returncode == 0, result.stderr
    assert sentinel not in result.stdout and sentinel not in result.stderr


# ── sourcing must be inert ────────────────────────────────────────────────────────────

def test_sourcing_has_no_side_effects(repo):
    """A library that mutates the environment on source cannot be reasoned about.

    Both halves are measured **inside one bash process**, so the comparison cannot be
    perturbed by the ambient environment. The first version built the two sides through
    different code paths — one via `call()` (which pops three Appwrite variables), one via
    raw `os.environ` — and so failed spuriously whenever the developer's shell already had
    those variables set, which is the normal state after sourcing the file by hand.
    """
    result = call(repo, f'''
        before="$(env | sort)"
        . "{repo}/tools/credentials/platform_env.sh"
        after="$(env | sort)"
        if [ "$before" = "$after" ]; then echo UNCHANGED; else
          echo CHANGED; diff <(echo "$before") <(echo "$after") | head -20
        fi
    ''')
    assert "UNCHANGED" in result.stdout, (
        f"sourcing platform_env.sh changed the environment — it must only define "
        f"functions.\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.red
def test_a_set_but_unexported_secret_is_promoted_not_skipped(repo):
    """C-112, and the bug this PR shipped with before review caught it.

    `un_fao/run.sh` sources `.env` early for GITHUB_TOKEN, which leaves the secret as a
    SHELL variable. A guard that tests `[ -n "$VAR" ]` sees a value and skips the export,
    so the child process receives nothing while every check reports success. Exported
    scope is the only scope that answers "will the child see this?".
    """
    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=fake-secret\n", encoding="utf-8")
    result = call(repo, f'''
        . "{repo}/.env"                 # set, NOT exported — the run.sh sequence
        platform_env_export_secret || echo "EXPORT_FAILED"
        python -c 'import os; print("CHILD=" + ("yes" if os.environ.get("APPWRITE_DATASTORE_API_KEY") else "no"))'
    ''')
    assert "CHILD=yes" in result.stdout, (
        f"a set-but-unexported secret was skipped rather than promoted, so the child got "
        f"nothing — #293 reintroduced (C-112).\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.red
def test_an_unsourceable_env_is_fatal_not_swallowed(repo):
    """`|| true` on the source would report success with the secret unset."""
    (repo / ".env").write_text('APPWRITE_DATASTORE_API_KEY="unterminated\n', encoding="utf-8")
    result = call(repo, 'platform_env_export_secret; echo "STATUS=$?"')
    assert "STATUS=0" not in result.stdout, (
        f"a .env that fails to source reported success.\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.red
def test_is_exported_handles_readonly_and_empty_and_prefixes(repo):
    """`export -p | grep '^declare -x NAME='` was wrong three ways; `compgen -e` is not.

    Bash prints `declare -rx NAME=` for a readonly export, so the original pattern reported
    a variable the child demonstrably receives as unavailable — the same category of error
    as the bug it was written to fix.
    """
    result = call(repo, '''
        export RO=x; readonly RO
        export EMPTYV=""
        platform_env_is_exported RO      && echo "RO=yes"      || echo "RO=no"
        platform_env_is_exported EMPTYV  && echo "EMPTY=yes"   || echo "EMPTY=no"
        platform_env_is_exported ROX     && echo "PREFIX=yes"  || echo "PREFIX=no"
        NOTEXPORTED=1
        platform_env_is_exported NOTEXPORTED && echo "LOCAL=yes" || echo "LOCAL=no"
    ''')
    assert "RO=yes" in result.stdout, "readonly+exported is still exported"
    assert "EMPTY=yes" in result.stdout, "exported-but-empty is still exported"
    assert "PREFIX=no" in result.stdout, "must not match a name that merely shares a prefix"
    assert "LOCAL=no" in result.stdout, "a shell-local variable is not exported"


def test_the_registry_is_read_once_per_load_not_once_per_step(repo, tmp_path):
    """`platform_env_load` runs three functions that each need the registry.

    Without a cache that is three subprocess spawns and three TOML parses per launcher
    invocation, across ~130 launchers. The first memoisation attempt assigned the cache
    inside a command substitution — a subshell — so it silently saved nothing.
    """
    counter = tmp_path / "pycalls"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    real = Path(sys.executable)
    (shim_dir / "python").write_text(
        f'#!/bin/bash\necho x >> "{counter}"\nexec "{real}" "$@"\n', encoding="utf-8"
    )
    (shim_dir / "python").chmod(0o755)

    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=fake\n", encoding="utf-8")
    env_extra = {"PATH": f"{shim_dir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    result = call(repo, "platform_env_load", env_extra=env_extra)
    assert result.returncode == 0, result.stderr

    spawns = counter.read_text().count("x") if counter.exists() else 0
    assert spawns == 1, (
        f"the registry was read {spawns} times in one platform_env_load; the cache is not "
        f"reaching the caller's shell (assigning it inside $(...) does nothing)"
    )


def test_the_probe_is_silent_and_non_fatal_when_there_is_no_env(repo):
    """`bootstrap.sh` must be able to ask "is the secret there?" on a virgin machine.

    The fatal `platform_env_export_secret` cannot answer it: with no `.env` it printed
    "FATAL: ... does not exist. Run ./bootstrap.sh" — at the person running
    ./bootstrap.sh. A setup script whose first output is a false FATAL and circular
    advice has failed at the one job #311 gave it.
    """
    result = call(repo, 'platform_env_secret_available; echo "STATUS=$?"')
    assert "STATUS=1" in result.stdout, "no .env means the secret is not available"
    assert "FATAL" not in result.stderr, (
        f"the probe must be silent — it is called before the machine is set up.\n"
        f"{result.stderr}"
    )
    assert result.stderr.strip() == "", f"the probe printed: {result.stderr!r}"


def test_the_probe_finds_a_secret_in_env_and_in_the_file(repo):
    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=fake\n", encoding="utf-8")
    assert "STATUS=0" in call(repo, 'platform_env_secret_available; echo "STATUS=$?"').stdout

    (repo / ".env").write_text("APPWRITE_DATASTORE_API_KEY=\n", encoding="utf-8")
    assert "STATUS=1" in call(repo, 'platform_env_secret_available; echo "STATUS=$?"').stdout, (
        "a declared-but-empty secret is not an available secret"
    )


def test_load_validates_and_uses_the_documented_order(repo):
    """`platform_env_load` claims to be the whole contract; it must include validation."""
    (repo / ".env").write_text("SOMETHING_ELSE=1\n", encoding="utf-8")   # no secret
    result = call(repo, 'platform_env_load; echo "STATUS=$?"')
    assert "STATUS=0" not in result.stdout, (
        "platform_env_load returned success with the secret missing — it must validate"
    )
