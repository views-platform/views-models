"""The darts operator documentation must say what it claims to say (#535).

Documentation assertions are where decorative guards are most likely, and this repository has
already shipped one: `TestTheGuideDocumentsTheChmodTrap` in
`tests/test_falsification_40_lesson_run_readiness.py` exists because an earlier version claimed
the guide documented the `/workspace` chmod trap and passed on `/workspace` at line 104 and
`chmod` at line 180 joined by `.*` under `re.S`. The property did not exist. A guard that
certifies an absent safety property is worse than no guard, because it stops the next reader
looking.

So every assertion here is made on a **bounded window** — the darts section, or a few hundred
characters around a keyword — never on the whole file. And the strongest test in this file is not
about wording at all: `test_the_adr_status_table_matches_what_the_script_can_write` compares
ADR-024's contract against the script's actual `STATUS` writes, so the two cannot drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "runpod_run_guide.md"
ADR = REPO / "docs" / "ADRs" / "024_pod_runner_contract.md"
RUNNER = REPO / "tools" / "podrun" / "pod_run_darts_calibration.sh"
PODRUN_INIT = REPO / "tools" / "podrun" / "__init__.py"

PUBLISH_VARS = ("DATASTORE_API_KEY", "DATASTORE_PROJECT_ID", "PREDICTION_STORE", "APPWRITE")


@pytest.fixture(scope="module")
def darts_section() -> str:
    """Phase 4c only — not the whole guide.

    An assertion about the darts chain that reads the whole file would be satisfied by the
    HydraNet phases, which legitimately mention credentials, draws archives and a different
    install. The window IS the test.
    """
    text = GUIDE.read_text()
    start = re.search(r"^## Phase 4c\b.*$", text, re.M)
    assert start, "the guide has no Phase 4c — the darts chain is undocumented"
    rest = text[start.end():]
    end = re.search(r"^## ", rest, re.M)
    return rest[: end.start()] if end else rest


# ── the guide's darts section ─────────────────────────────────────────────────────


def test_it_gives_the_command_and_the_preflight_first(darts_section):
    assert "pod_run_darts_calibration.sh --preflight" in darts_section
    pre = darts_section.index("--preflight")
    run = darts_section.index("nohup")
    assert pre < run, "the guide shows the real run before the free check"


def test_the_engine_pin_comes_with_the_reason_it_exists(darts_section):
    """A pin without its reason is a line someone will 'simplify' back to PyPI.

    The instruction must carry the version, not merely mention it somewhere in the section. A
    mutation that softened "installed from the git tag `0.2.4`" to "a recent version" survived
    an earlier version of this test, because `0.2.4` still appeared two sentences later in the
    explanation of what it fixes.
    """
    pin = re.search(r"git tag[^\n]{0,20}0\.2\.4", darts_section)
    assert pin, (
        "the guide does not tie the install instruction to the git tag 0.2.4 — a reader is left "
        "to infer which version, and 'whatever is newest' is PyPI's 0.2.3"
    )
    window = darts_section[max(0, pin.start() - 400): pin.start() + 900]
    assert re.search(r"scratch|53 GB|2 TB|#54", window), (
        "the guide pins 0.2.4 without saying that 0.2.3 never frees its scratch directory, "
        "which is the only reason not to take PyPI's newest"
    )
    assert re.search(r"not published|deliberately", window, re.I), (
        "the guide does not explain why 0.2.4 is installed from a tag rather than released"
    )


def test_tmpdir_is_documented_with_the_two_disks(darts_section):
    assert "TMPDIR" in darts_section, "the guide never mentions TMPDIR"
    windows = [darts_section[m.start() - 200: m.start() + 700]
               for m in re.finditer(r"TMPDIR", darts_section)]
    assert any(
        re.search(r"container", w) and re.search(r"volume", w)
        for w in windows
    ), (
        "the guide must say, near TMPDIR, that the scratch lands on the CONTAINER disk while "
        "the floor measures the VOLUME. 'Set TMPDIR' without that is a cargo-cult instruction."
    )


def test_the_two_refused_models_are_named_with_the_issue_that_decides_them(darts_section):
    assert "little_talks" in darts_section and "mister_bluesky" in darts_section
    assert "536" in darts_section, "the operator is not told where that decision lives"
    assert re.search(r"303 GB|~303", darts_section), (
        "the guide says they are refused but not that it is a measured 303 GB — so the next "
        "operator cannot tell whether a bigger pod would fix it"
    )


def test_the_determinism_caveat_is_stated_not_left_to_be_inferred(darts_section):
    """Nine of eleven are deterministic, so the HydraNets' q95 fix has no analogue here."""
    assert re.search(r"no `?draws/?`? archive|There is no `draws", darts_section, re.I), (
        "the guide does not say that no draws archive is produced, so its absence reads as a bug"
    )
    assert "q95" in darts_section, (
        "the guide does not say the q95 correction is undefinable for deterministic models"
    )


def test_the_darts_section_names_no_publish_variable(darts_section):
    for var in PUBLISH_VARS:
        assert var not in darts_section, (
            f"the darts section mentions {var}; this chain publishes nothing and an operator "
            f"following it must not be prompted to place a write credential"
        )


def test_the_guide_does_not_send_a_darts_operator_through_the_hydranet_install(darts_section):
    assert re.search(r"do not run Phase 2\.2", darts_section, re.I), (
        "Phase 2.2 installs views-hydranet; a darts operator following the guide top to bottom "
        "would build the wrong environment and the guide must say so"
    )


# ── ground rule 5 and the status line ─────────────────────────────────────────────


def test_ground_rule_five_records_how_it_actually_gets_broken():
    """The policy existed; the mechanism did not. Copying the publishing sibling is the trap."""
    text = GUIDE.read_text()
    m = re.search(r"\*\*Publish credentials never go on rented hardware\.\*\*", text)
    assert m, "ground rule 5 is gone"
    window = text[m.start(): m.start() + 2200]
    assert "pod_run_fao_delivery.sh" in window, (
        "ground rule 5 does not name the script that legitimately needs publish variables, so "
        "'copy the one that works' still looks safe"
    )
    assert re.search(r"test", window), "the rule cites nothing that would fail a build"


def test_the_tooling_status_line_admits_the_second_family():
    head = GUIDE.read_text()[:1200]
    assert "r2darts2" in head, (
        "the guide's Tooling status still claims one model family, which is now false"
    )


# ── the ADR ───────────────────────────────────────────────────────────────────────


def test_the_adr_exists_in_the_house_format():
    assert ADR.is_file(), "ADR-024 is missing"
    text = ADR.read_text()
    for field in ("**Status:**", "**Date:**", "**Deciders:**"):
        assert field in text, f"ADR-024 has no {field} line"
    assert re.search(r"^## (Decision|Consequences)", text, re.M)


def test_the_adr_status_table_matches_what_the_script_can_write():
    """The strongest test here, because it is the one that can rot silently.

    ADR-024 §1 makes STATUS a contract and lists its permitted values. If the script learns a
    fourth value and the ADR does not, a consumer applying the ADR is wrong about the platform.
    """
    writes = set(re.findall(r'echo (\S+) > "\$OUT/STATUS"', RUNNER.read_text()))
    # die() writes FAILED:<stage> through a different expression.
    assert "FAILED:" in RUNNER.read_text()
    writes.add("FAILED:<stage>")

    # Three places now name these values: the script writes them, ADR-024 defines them as a
    # contract, and the guide's Phase 4c restates them for the operator. The ADR is the
    # authority, but a stale value in the GUIDE misleads the person at the terminal — so both
    # are checked here, in one place, rather than becoming two independent drift surfaces.
    adr = ADR.read_text()
    guide = GUIDE.read_text()
    for value in writes:
        token = value.split(":")[0]
        assert re.search(rf"`{re.escape(token)}", adr), (
            f"the script writes {value} to STATUS but ADR-024's table does not list {token}"
        )
        assert re.search(rf"`{re.escape(token)}", guide), (
            f"the script writes {value} to STATUS but the operator guide never mentions {token}"
        )


def test_the_adr_declares_the_debt_rather_than_implying_compliance():
    """The two older scripts do NOT satisfy §1. An ADR that read as if they did would be worse
    than no ADR — a future auditor would trust the group and find two exceptions.

    Asserted on a window that must name BOTH scripts next to the non-compliance, not on an
    `A or B` phrase match. The first version of this test used
    `re.search(r"do not satisfy|declared debt")`, and a mutation that deleted the "do not
    satisfy §1 today" claim survived it, because "declared debt" sat in the next sentence.
    """
    adr = ADR.read_text()
    windows = [
        adr[max(0, m.start() - 600): m.start() + 600]
        for m in re.finditer(r"do not satisfy|does not comply|do not comply", adr, re.I)
    ]
    assert any(
        "pod_run_model.sh" in w and "pod_run_fao_delivery.sh" in w for w in windows
    ), (
        "ADR-024 must name pod_run_model.sh AND pod_run_fao_delivery.sh next to the statement "
        "that they do not satisfy §1 — otherwise an auditor reads the group as compliant"
    )
    assert "_common.sh" in adr, "the debt is declared with no trigger for closing it"


def test_the_adr_does_not_restate_the_credential_policy_as_its_own():
    """One rule in two places is two places to drift (vmo_021). The policy is ground rule 5's."""
    adr = ADR.read_text()
    m = re.search(r"^### 3\.", adr, re.M)
    assert m, "ADR-024 has no section 3"
    section = adr[m.start(): m.start() + 1500]
    assert "ground rule 5" in section, (
        "ADR-024 §3 does not point at the guide as the policy's home, so the two can diverge "
        "with neither looking wrong"
    )


# ── the group's own status declaration ────────────────────────────────────────────


def test_podrun_records_the_second_family_and_the_extraction_trigger():
    text = PODRUN_INIT.read_text()
    assert "r2darts2" in text, "the group still claims one model family"
    assert "_common.sh" in text, "the duplication trigger is not recorded"
    assert re.search(r"fourth script|all three", text), (
        "the trigger is recorded as 'later' rather than as a named condition"
    )


def test_podrun_does_not_claim_promotion_it_has_not_earned():
    """A second model family is one of three promotion criteria, not all of them."""
    text = PODRUN_INIT.read_text()
    assert '__version__ = "0.1.0"' in text, "the version was bumped on documentation alone"
    assert re.search(r"never completed a real run|unproven", text), (
        "the group does not admit that the darts runner's training leg has never run"
    )
