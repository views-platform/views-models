"""The un_fao launcher must not deliver an artifact it cannot produce (#294).

`postprocessors/un_fao/run.sh` installs views-postprocessing from `@main` — a MOVING
branch pointer. Whether that build can produce the artifact `config_meta` declares is a
fact about another repository on the day you run, and nothing checked it.

On 2026-07-31 `@main` was 208 commits behind and carried zero wire modules. A run would
have completed successfully, ignored `wire_contract: True`, delivered the legacy parquet
instead of the ADR-013 contract dialect, and left faoapi serving the previous month —
green run, wrong artifact, no signal. views-postprocessing#178 merged on 2026-08-01 and
`@main` now carries the wire, so the pin is correct today. **Correct by timing is not
verified**, and the next drift is silent in exactly the same way.

These tests pin the assertion, not the pin. Whatever pin is eventually chosen, the
launcher must still refuse to run a build that cannot honour its own declaration.
"""
from pathlib import Path

import pytest

pytestmark = [pytest.mark.beige]

RUN_SH = (
    Path(__file__).resolve().parent.parent
    / "postprocessors" / "un_fao" / "run.sh"
)


def _run_sh() -> str:
    assert RUN_SH.exists(), "the un_fao launcher must exist"
    return RUN_SH.read_text(encoding="utf-8")


def test_launcher_asserts_the_declared_capability_before_running():
    text = _run_sh()
    assert "views_postprocessing.contract.wire" in text, (
        "the launcher must verify the installed build can import the wire module before "
        "running a delivery that declares wire_contract — otherwise a stale @main "
        "silently ships the legacy artifact (#294)"
    )
    assert "wire_contract" in text, (
        "the check must key on what config_meta DECLARES, not on a hardcoded assumption "
        "— a postprocessor that legitimately wants the legacy artifact must not be "
        "blocked by it"
    )


def test_capability_check_reads_config_meta_by_import_not_by_grep():
    """C-57: to a regex, a commented-out key is indistinguishable from a live one.

    The register records this exact failure twice — once in the partition tooling and
    once in a model-cloning script that patched a commented template line. A shell
    `grep '"wire_contract": True'` over config_meta would report the declaration as
    present even when it is commented out.
    """
    text = _run_sh()
    assert "importlib.util" in text, (
        "the wire_contract declaration must be read by importing config_meta, never by "
        "grepping it (C-57)"
    )


def test_unreadable_config_meta_does_not_invent_a_failure():
    """A check that adds a failure mode must not add one it cannot justify."""
    text = _run_sh()
    assert "unknown" in text and "SKIPPED" in text, (
        "if config_meta cannot be read, the capability check must skip truthfully rather "
        "than abort — it only ever ADDS a failure mode, and an unreadable config is a "
        "different problem that other checks own"
    )


def test_the_abort_names_what_went_wrong_and_what_to_do():
    """A launcher that aborts at 3am must say why, and what would fix it."""
    text = _run_sh()
    for expected in ("#294", "wire_contract", "legacy artifact"):
        assert expected in text, f"the abort message must mention {expected!r}"
    assert "exit 1" in text, "declaring the capability and lacking it must be fatal"
