"""The ledger tool and the catalog document must name the same models.

⛔ THE FAILURE THIS GUARDS. `docs/matryoshka_experiment_catalog.md` is what a person reads to answer
"do we have that model's data". `tools/audit/verify_experiment_data.py` is what re-derives the answer
from the disk. If a model is added to one and not the other, the document starts lying while every
check stays green — which is the exact shape of the 2026-11-17 error that caused both of them to be
written (eleven models reported as never run, from looking in one directory).

The roster is declared as a literal in the tool, deliberately: three of the sixteen have no folder on
disk, and their absence is the most important fact in the ledger, so the set cannot be discovered by
listing `models/`. That choice is what makes this test necessary.

⚠️ This test compares MEMBERSHIP only — which names are in the set. It does not check that the
document's verdict for a model matches the tool's, because the tool's verdict is a measurement taken
when it runs and the document's is a measurement taken when it was written. Those are allowed to
differ; a stale document is what the "Last verified" date is for.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "docs/matryoshka_experiment_catalog.md"

#: Section 4 lists nine baselines that answer a DIFFERENT question and are not in the comparison.
#: Everything above that heading is the set; everything below it is not.
OUT_OF_SET_HEADING = "## 4."


def _roster_names() -> set[str]:
    from tools.audit.verify_experiment_data import ROSTER

    return {member.name for member in ROSTER}


def _catalog_names() -> set[str]:
    text = CATALOG.read_text(encoding="utf-8")
    in_set, _, _ = text.partition(OUT_OF_SET_HEADING)
    assert OUT_OF_SET_HEADING in text, (
        f"{CATALOG.name} no longer has a section 4 heading, so this test cannot tell which names are "
        "in the comparison and which are the other baselines listed for completeness. Restore the "
        "heading or rewrite this test — do not leave it passing on the whole document."
    )
    # Model names in this repo are two lowercase words joined by an underscore, in backticks.
    return set(re.findall(r"`([a-z]+_[a-z]+)`", in_set))


def test_the_tool_and_the_catalog_name_the_same_models():
    roster, catalog = _roster_names(), _catalog_names()

    only_in_tool = sorted(roster - catalog)
    only_in_catalog = sorted(catalog - roster)

    assert not only_in_tool and not only_in_catalog, (
        "the experiment ledger and its catalog document disagree about which models are in the "
        "comparison:\n"
        + (f"  in tools/audit/verify_experiment_data.py only: {only_in_tool}\n" if only_in_tool else "")
        + (f"  in docs/matryoshka_experiment_catalog.md only: {only_in_catalog}\n" if only_in_catalog else "")
        + "Add the model to both, or remove it from both."
    )


def test_every_model_with_no_run_has_a_named_origin_zero_twin():
    """A moving-window model is accepted by equalling its fixed-window twin at origin 0.

    ⛔ Without a declared twin there is no acceptance condition, so the model could be run, scored,
    and compared with nothing checking that its implementation is right.
    """
    from tools.audit.verify_experiment_data import ORIGIN_ZERO_TWIN, ROSTER

    moving = {m.name for m in ROSTER if m.role == "moving-window"}
    assert moving, (
        "no model carries the 'moving-window' role, so this test has nothing to check. Either the "
        "role was renamed or the comparators were dropped from the roster."
    )

    fixed = {m.name for m in ROSTER if m.role == "fixed-window"}
    assert fixed, "no model carries the 'fixed-window' role, so no twin can exist to compare against."

    assert set(ORIGIN_ZERO_TWIN) == moving, (
        f"every moving-window model needs a declared origin-0 twin. Declared: {sorted(ORIGIN_ZERO_TWIN)}; "
        f"moving-window members: {sorted(moving)}."
    )
    wrong = {k: v for k, v in ORIGIN_ZERO_TWIN.items() if v not in fixed}
    assert not wrong, (
        f"these twins are not fixed-window members of the roster, so the equality check could not be "
        f"run against them: {wrong}"
    )


def test_every_model_with_no_data_can_be_reported_without_crashing():
    """The report explains, per model that holds no data, what it still needs. Every such model
    must therefore be resolvable in one of the two twin maps.

    ⛔ THE FAILURE THIS GUARDS, which already happened. The report did
    `ORIGIN_ZERO_TWIN[name]` for every model with no data. That is correct only while every such
    model is a moving-window arm. On 2026-11-18 `tight_canvas` — a FIXED-window reference — was
    declared in the roster before it was built, and the tool printed its whole table and then
    died with KeyError. The one tool whose purpose is to be trusted about what data exists
    crashed while saying so, and the membership test above stayed green because membership was
    not what broke.

    This asserts the invariant the report depends on, rather than the text the report prints.
    """
    from tools.audit.verify_experiment_data import ORIGIN_ZERO_TWIN, ROSTER, TWIN_OF

    declared_absent = [m.name for m in ROSTER if m.batch is None]
    unexplainable = [n for n in declared_absent if n not in ORIGIN_ZERO_TWIN and n not in TWIN_OF]
    assert not unexplainable, (
        f"these models are declared in the roster with no data, and the report cannot say what "
        f"they need because they appear in neither ORIGIN_ZERO_TWIN nor its reverse: "
        f"{unexplainable}. Declare the twin relationship, or the report will crash on them."
    )
    assert TWIN_OF == {ref: mov for mov, ref in ORIGIN_ZERO_TWIN.items()}, (
        "TWIN_OF must be derived from ORIGIN_ZERO_TWIN, not written by hand; it has drifted."
    )
