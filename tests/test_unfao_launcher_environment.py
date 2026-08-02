"""The un_fao launcher's environment contract: one source, and absence is fatal (#308, #309).

Two rules, both learned the hard way:

**#308 — an unresolvable registry is fatal, and fatal early.** The default registry path is
a relative hop to a sibling checkout of views-appwrite. On a different layout, in CI, or in
a container it is simply absent. Warning and continuing does not save the run; it moves the
failure to the datastore boundary minutes later, where it describes a symptom instead of a
cause, in front of whoever is least equipped to trace it back to a checkout layout.

**#309 — the registry is the only source of coordinates.** `.env` and the registry writing
the same names is a data race whose winner is decided by line order: reverse the blocks and
the semantics invert silently, with no test failing.

These tests pin the rules against the launcher, which is production infrastructure and not
otherwise covered by anything executable.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

RUN_SH = (
    Path(__file__).resolve().parent.parent / "postprocessors" / "un_fao" / "run.sh"
)


def _run_sh() -> str:
    return RUN_SH.read_text(encoding="utf-8")


def test_missing_registry_is_fatal_before_any_conda_or_pip_work():
    """#308's acceptance criterion: exit non-zero *before any work begins*.

    Position matters as much as behaviour. A fatal check placed after the conda/pip block
    still wastes an environment build, and on a fresh machine that is minutes.
    """
    text = _run_sh()
    fatal_at = text.find('if [ ! -f "$APPWRITE_REGISTRY" ]')
    assert fatal_at != -1, "a missing registry must be fatal (#308)"

    for later in ("conda activate", "pip install", "conda shell.bash hook"):
        position = text.find(later)
        assert position == -1 or fatal_at < position, (
            f"the fatal registry check must come BEFORE {later!r} — otherwise the run "
            f"builds an environment it is about to throw away (#308)"
        )


def test_the_fatal_message_names_the_path_and_the_override():
    text = _run_sh()
    assert "APPWRITE_REGISTRY=/path/to" in text, (
        "the failure must name the override variable — the person hitting this is on a "
        "machine with a different layout and needs the escape hatch, not a diagnosis"
    )
    assert "looked for:" in text, "the failure must name the path actually tried"


def test_an_unreadable_registry_is_also_fatal():
    """Existing-but-unparseable is the same outcome as absent: no coordinates."""
    text = _run_sh()
    assert "could not be read" in text, (
        "a registry that exists but fails to parse must be fatal too — half a source is "
        "not a source (#308)"
    )


def test_env_declaring_a_registry_owned_coordinate_is_fatal():
    """#309: two writers to one name is an error, not a precedence question."""
    text = _run_sh()
    assert "both declare:" in text, (
        "a coordinate declared in BOTH .env and the registry must fail loud naming the "
        "variable and both sources (#309)"
    )
    assert "_conflicts" in text, "the conflict detection must exist, not just be described"


def test_the_secret_is_still_exported_by_name_and_never_via_set_a():
    """The one value .env may still carry. `set -a` would corrupt it (#293)."""
    text = _run_sh()
    assert "export APPWRITE_DATASTORE_API_KEY" in text, (
        "the secret must still be exported by name — #293 exists because it was not"
    )
    # Check for USE, not mention: the file deliberately explains in a comment why `set -a`
    # is wrong, and a naive substring search flags that explanation as the offence. That is
    # C-57 — to a regex a commented line is indistinguishable from a live one — and this
    # test made the mistake on its first draft.
    live_set_a = [
        line for line in text.splitlines()
        if "set -a" in line and not line.strip().startswith("#")
    ]
    assert not live_set_a, (
        f"`set -a` is used, not merely explained: {live_set_a}. It exports unquoted "
        f"*_NAME values truncated at the first space (#293)"
    )


def test_the_293_correction_survived_the_rewrite():
    """A removal must name what it was carrying (þing-02, S25).

    The #293 comment records that `source` without `export` meant there was no carrier
    for the secret at all between two dates — a real production failure, honestly
    documented. Rewrites lose that kind of thing silently.
    """
    text = _run_sh()
    assert "#293" in text, "the #293 correction must survive edits to this file"


def test_the_false_coordinate_state_claim_is_gone():
    """The helper it replaced asserted something untrue; it must not come back.

    `_platform001_coordinate_state()` announced "Coordinates ARE present in the
    environment (exported outside this script)" on the strength of a SHELL variable that
    `source .env` sets and nothing exports — so the python child never saw it. It was
    written to avoid asserting an unverified environment fact and did exactly that by
    checking the wrong scope.
    """
    text = _run_sh()
    assert "_platform001_coordinate_state" not in text or "REMOVED HERE" in text, (
        "the coordinate-state helper checked shell variables rather than exported ones "
        "and reported a false 'coordinates ARE present' — it must not be reinstated "
        "without fixing that (#308/#309)"
    )
    assert "Coordinates ARE present in the environment (exported outside this script)" not in text
