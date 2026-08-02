"""The un_fao launcher delegates its environment to the one writer (#309).

The *behaviour* — registry fatal, one writer, secret by name, nothing rendered — is tested
against the real shell functions in `tests/test_platform_env.py`. These are the remaining
assertions that can only be made about the launcher itself: that it uses that file rather
than reimplementing it, and that it does so in the right order.

Ordering is the part a behavioural test cannot reach. A fatal registry check placed after
the conda/pip block still works, and still wastes several minutes building an environment
the run is about to throw away — on a fresh machine, that is the difference between a
useful failure and an infuriating one.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

RUN_SH = Path(__file__).resolve().parent.parent / "postprocessors" / "un_fao" / "run.sh"


def _run_sh() -> str:
    return RUN_SH.read_text(encoding="utf-8")


def test_launcher_sources_the_shared_environment_writer():
    assert "tools/credentials/platform_env.sh" in _run_sh(), (
        "the launcher must source tools/credentials/platform_env.sh rather than reimplementing the "
        "environment contract — two implementations is how the two writers happened (#309)"
    )


def _first_code_line(text: str, needle: str):
    """Line number of the first NON-COMMENT line containing `needle`, else None.

    Comments must be excluded or this fails on prose. It did: the launcher's own comment
    reads "macOS notes, conda lifecycle, pip install", and a naive search placed `pip
    install` before the guard. That is C-57 — a regex cannot tell a commented mention from
    a live statement — committed inside the test whose docstring warns about C-57.
    """
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line and not line.strip().startswith("#"):
            return number
    return None


def test_registry_check_happens_before_conda_and_pip():
    text = _run_sh()
    guard = _first_code_line(text, "platform_env_require_registry")
    assert guard is not None, "the launcher must assert the registry resolves (#308)"
    for later in ("conda shell.bash hook", "conda activate", "pip install"):
        position = _first_code_line(text, later)
        assert position is None or guard < position, (
            f"the registry guard (line {guard}) must precede {later!r} (line {position}) — "
            f"otherwise a run with no registry builds a conda environment it is about to "
            f"discard (#308)"
        )


def test_the_full_contract_is_asserted_before_main_runs():
    """Every step, and each one fatal. A partial environment is not a starting state."""
    text = _run_sh()
    # `_first_code_line`, not `str.find` — the same C-57 trap that broke the ordering test
    # above. These happen not to have a commented mention today; relying on that is how it
    # comes back.
    main_at = _first_code_line(text, "main.py")
    assert main_at is not None
    for call in (
        "platform_env_assert_no_env_conflicts",
        "platform_env_export_secret",
        "platform_env_export_coordinates",
        "platform_env_validate",
    ):
        at = _first_code_line(text, call)
        assert at is not None, f"{call} must be called by the launcher"
        assert at < main_at, f"{call} must run before main.py"
        line = next(ln for ln in text.splitlines() if call in ln and not ln.strip().startswith("#"))
        assert "|| exit 1" in line, f"{call} must be fatal, not advisory: {line.strip()!r}"


def test_the_launcher_does_not_export_the_secret_itself():
    """One writer. The launcher sources .env for GITHUB_TOKEN and nothing else."""
    text = _run_sh()
    live = [
        ln for ln in text.splitlines()
        if "export APPWRITE_DATASTORE_API_KEY" in ln and not ln.strip().startswith("#")
    ]
    assert not live, (
        f"the launcher exports the secret directly: {live}. platform_env_export_secret "
        f"owns it; doing both reinstates the second writer #309 removes"
    )


def test_no_live_set_a():
    """`set -a` exports unquoted *_NAME values truncated at the first space (#293).

    Checks for USE, not mention: the file explains in a comment why `set -a` is wrong, and
    a substring search flags the explanation as the offence. That is C-57, and the first
    draft of this test made exactly that mistake.
    """
    live = [
        ln for ln in _run_sh().splitlines()
        if "set -a" in ln and not ln.strip().startswith("#")
    ]
    assert not live, f"`set -a` is used, not merely explained: {live}"


def test_the_293_correction_survived_the_extraction():
    """A removal must name what it was carrying (þing-02, S25).

    #293 records that `source` without `export` left no carrier for the secret at all
    between two dates — a real production failure. Extractions lose that kind of note
    silently, so it is pinned in both places it could live.
    """
    from_launcher = "#293" in _run_sh()
    shared = Path(__file__).resolve().parent.parent / "tools" / "credentials" / "platform_env.sh"
    from_shared = "#293" in shared.read_text(encoding="utf-8")
    assert from_launcher or from_shared, (
        "the #293 correction must survive wherever the secret handling now lives"
    )
