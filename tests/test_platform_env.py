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


def test_validate_names_what_is_missing(repo):
    """No `.env`, so the secret is absent — and validation must say which name."""
    result = call(repo, "platform_env_load; platform_env_validate")
    assert result.returncode == 1
    assert "APPWRITE_DATASTORE_API_KEY" in result.stderr
    assert "missing" in result.stderr.lower()


def test_the_secret_value_is_never_rendered(repo):
    """A check that prints a credential to prove it found one has published it."""
    sentinel = "SENTINEL-SECRET-MUST-NOT-APPEAR"
    (repo / ".env").write_text(f"APPWRITE_DATASTORE_API_KEY={sentinel}\n", encoding="utf-8")
    result = call(repo, "platform_env_load && platform_env_validate")
    assert result.returncode == 0, result.stderr
    assert sentinel not in result.stdout and sentinel not in result.stderr


# ── sourcing must be inert ────────────────────────────────────────────────────────────

def test_sourcing_has_no_side_effects(repo):
    """A library that mutates the environment on source cannot be reasoned about."""
    result = call(repo, 'echo "N=$(env | wc -l)"')
    before = subprocess.run(
        ["bash", "-c", 'echo "N=$(env | wc -l)"'],
        capture_output=True, text=True, env=dict(os.environ, APPWRITE_REGISTRY=str(repo / "registry.toml")),
    )
    assert result.stdout.strip() == before.stdout.strip(), (
        "sourcing platform_env.sh changed the environment — it must only define functions"
    )
