"""No launcher may deliver an artifact it cannot produce (#294).

A launcher installs views-postprocessing from a pin. Whether that build can produce the
artifact `config_meta` declares is a fact about another repository on the day you run,
and nothing checked it.

**Parametrised over every postprocessor**, and reading the launcher's *effective* text —
its own `run.sh` plus the shared delivery body it sources (`tools/launcher/postprocessor.sh`,
ADR-022). Before that extraction these assertions covered `un_fao` only, which is precisely
the shape views-postprocessing's #211 got wrong one repo away: "every partner-scoped guard
was scoped to ONE partner".

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

REPO_ROOT = Path(__file__).resolve().parent.parent
POSTPROCESSORS = REPO_ROOT / "postprocessors"
SHARED_BODY = REPO_ROOT / "tools" / "launcher" / "postprocessor.sh"

LAUNCHERS = sorted(p.name for p in POSTPROCESSORS.iterdir() if (p / "run.sh").exists())


def _effective_text(name: str) -> str:
    """A launcher's own text plus the delivery body it sources.

    The guarantees below are about what the launcher *does*, not about which file the
    lines live in. Reading only `run.sh` would make every one of them pass vacuously the
    moment the body moved — a green test measuring the wrong file.
    """
    run_sh = POSTPROCESSORS / name / "run.sh"
    assert run_sh.exists(), f"the {name} launcher must exist"
    text = run_sh.read_text(encoding="utf-8")
    assert "postprocessor_launch" in text, (
        f"{name}/run.sh does not call the shared delivery body. If it has grown its own "
        f"copy of the protocol, that is the duplication ADR-022 exists to prevent."
    )
    return text + "\n" + SHARED_BODY.read_text(encoding="utf-8")


def test_there_is_at_least_one_launcher_to_check():
    """Guard the parametrisation: an empty list would pass every test below."""
    assert LAUNCHERS, "no postprocessor launchers discovered — the checks below assert nothing"


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_asserts_the_declared_capability_before_running(launcher):
    text = _effective_text(launcher)
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


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_capability_check_reads_config_meta_by_import_not_by_grep(launcher):
    """C-57: to a regex, a commented-out key is indistinguishable from a live one.

    The register records this exact failure twice — once in the partition tooling and
    once in a model-cloning script that patched a commented template line. A shell
    `grep '"wire_contract": True'` over config_meta would report the declaration as
    present even when it is commented out.
    """
    text = _effective_text(launcher)
    assert "importlib.util" in text, (
        "the wire_contract declaration must be read by importing config_meta, never by "
        "grepping it (C-57)"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_unreadable_config_meta_does_not_invent_a_failure(launcher):
    """A check that adds a failure mode must not add one it cannot justify."""
    text = _effective_text(launcher)
    assert "unknown" in text and "SKIPPED" in text, (
        "if config_meta cannot be read, the capability check must skip truthfully rather "
        "than abort — it only ever ADDS a failure mode, and an unreadable config is a "
        "different problem that other checks own"
    )


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_the_abort_names_what_went_wrong_and_what_to_do(launcher):
    """A launcher that aborts at 3am must say why, and what would fix it."""
    text = _effective_text(launcher)
    for expected in ("#294", "wire_contract", "legacy artifact"):
        assert expected in text, f"the abort message must mention {expected!r}"
    assert "return 1" in text or "exit 1" in text, (
        "declaring the capability and lacking it must be fatal"
    )
