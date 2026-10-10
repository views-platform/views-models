"""Guards on the HydraNet delivery note's load-bearing statements.

These assert on the LaTeX sources rather than the PDF, because the sources
are what a later edit touches. Each one holds a claim that the note would be
wrong without, and that a well-meaning edit could remove without the build
failing:

  - two of the three frames cannot be reproduced, and the note says so;
  - a figure the note prints is one that was measured, not inherited;
  - the recipient can turn a priogrid_id into coordinates;
  - the section that is in our remit reads as decisions, not questions.

The last is the register rule the delivery mail was rewritten four times to
find (reports/delivery_notes/README.md, "Writing for this recipient"). It is
the one a future edit is most likely to break by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

NOTE = (
    Path(__file__).resolve().parents[1]
    / "reports/delivery_notes/hydranet-pgm-global-calibration-2026-09"
)
SECTIONS = NOTE / "sections"

pytestmark = pytest.mark.skipif(
    not SECTIONS.exists(), reason="the HydraNet delivery note is not present"
)


def _all_prose() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SECTIONS.glob("*.tex")))


def test_the_note_says_the_two_experimental_frames_cannot_be_reproduced():
    """The one asymmetry in the delivery.

    Frames II and III have no implementation in this repository and the draws
    they were computed from were never transferred off the rented machines.
    A note that describes all three estimators evenhandedly would imply all
    three can be checked.
    """
    text = (SECTIONS / "the_frames.tex").read_text(encoding="utf-8")
    assert "not reproducible" in text, "the note does not say the frames cannot be reproduced"
    assert "cannot be recomputed" in text
    assert "never transferred off" in text, "the note does not say where the draws went"


def test_the_note_corrects_the_retained_posterior_claim():
    """A memo of 28 September told the recipient the draws had come home.

    They had not. The note carries the correction, because the recipient holds
    the memo and the two cannot both stand.
    """
    prose = _all_prose()
    assert "new GPU run" in prose or "new run" in prose, "the consequence is not stated"
    assert "cost memo" in prose, "the note does not name the document it corrects"


def test_the_note_does_not_cite_an_authority_that_does_not_exist():
    """The folder's own README cited "ADR-023 Amendment 1", which has never existed.

    The note may name it in order to retract it. It may not rest on it.
    """
    prose = _all_prose()
    if "Amendment 1" in prose:
        assert "No such document exists" in prose, (
            "the note names Amendment 1 without saying it does not exist"
        )


def test_the_horizon_figures_are_the_measured_ones():
    """The 'flat after ~5 steps' claim was inherited from a different run.

    These numbers were measured from the delivered parquets for this note. If
    they are edited they must be re-measured, not adjusted.
    """
    text = (SECTIONS / "magnitude_and_horizon.tex").read_text(encoding="utf-8")
    assert "16,404" in text, "the step-1 total is missing"
    assert "8,265" in text and "8,907" in text, "the plateau range is missing"
    assert "thirty-one of the" in text, "the note does not say how much of the horizon is flat"
    assert "misuse" in text, "the note does not say what the consequence is"


def test_the_magnitude_section_states_it_is_not_an_evaluation():
    text = (SECTIONS / "magnitude_and_horizon.tex").read_text(encoding="utf-8")
    assert "ranks nothing and recommends nothing" in text


def test_the_note_distinguishes_live_rows_from_distinct_cell_months():
    """1,469,004 is a row count, and the folder's README called it cell-months.

    The same cell-month is counted once per model and once per origin. A reader
    who takes it as a count of distinct cell-months is out by a factor of 15.
    """
    text = (SECTIONS / "magnitude_and_horizon.tex").read_text(encoding="utf-8")
    assert "1,469,004" in text
    assert "98,995" in text, "the distinct cell-month count is missing"
    assert "not a count of distinct cell-months" in text


def test_the_note_says_how_to_turn_a_priogrid_id_into_coordinates():
    """A pgm delivery carries no coordinates.

    Without this the recipient cannot place a single row on a map.
    """
    text = (SECTIONS / "the_data.tex").read_text(encoding="utf-8")
    assert "720" in text, "the grid width is missing"
    assert "-90 + (" in text or "$-90 +" in text, "the latitude formula is missing"
    assert "-180 + (" in text or "$-180 +" in text, "the longitude formula is missing"
    assert "centre" in text, "it does not say the coordinates are cell centres"
    # The worked example, so the recipient can check their own arithmetic.
    assert "181875" in text and "36.25" in text and "37.25" in text


def test_the_note_warns_against_stacking_the_origins():
    text = (SECTIONS / "the_data.tex").read_text(encoding="utf-8")
    assert "overlap" in text
    assert "panel" in text, "the note does not name the mistake it is warning about"


def test_the_next_steps_section_decides_rather_than_asks():
    """The register rule: anything inside our remit reads as a decision.

    Section 9 is where this goes wrong most easily — a question there asks the
    recipient to adjudicate an engineering choice, and he will.
    """
    text = (SECTIONS / "conclusion.tex").read_text(encoding="utf-8")
    assert "?" not in text, "Next Steps asks the recipient a question"
    assert "None of them needs a decision from the reader" in text


def test_the_note_names_the_column_convention_difference():
    """Two deliveries now use different column names for the same quantity.

    A tool that infers the target column from the schema reads one of the two
    wrongly and silently.
    """
    text = (SECTIONS / "conclusion.tex").read_text(encoding="utf-8")
    assert "pred\\_lr\\_sb\\_best" in text
    assert "pred\\_lr\\_ged\\_sb" in text
    assert "set explicitly" in text, "the note does not say what the consumer must do"


def test_the_authorship_is_stated_as_settled_not_claimed():
    """Conceded in writing on 2026-10-01. A hedge invites it to be reopened."""
    text = (SECTIONS / "the_model.tex").read_text(encoding="utf-8")
    assert "It is my own work" in text
    for hedge in ("I believe", "arguably", "I would argue", "in my view"):
        assert hedge not in text, f"the authorship statement is hedged with '{hedge}'"


def test_the_confounded_comparison_is_flagged_where_it_is_invited():
    """heavy_freighter vs bright_starship looks like a controlled contrast.

    The seeds differ too. Both sections that invite the reading carry the
    caveat, because a reader may read either one alone.
    """
    for name in ("the_eight_models.tex", "additional_notes.tex"):
        text = (SECTIONS / name).read_text(encoding="utf-8")
        assert "seed" in text, f"{name} does not mention the seed confound"
        assert "confounded" in text or "cannot be separated" in text, (
            f"{name} invites the controlled-contrast reading without flagging it"
        )


def test_the_version_gap_is_stated_rather_than_filled_in():
    """Provenance pins the code to a commit but not to a release.

    The title page says so; so must the body, because a reader who notices the
    title page's hedge needs to know whether it is an omission or a fact.
    """
    main = (NOTE / "main.tex").read_text(encoding="utf-8")
    assert "not recorded" in main, "the title page claims versions it does not have"
    text = (SECTIONS / "additional_notes.tex").read_text(encoding="utf-8")
    assert "pinned as ranges" in text
    assert "not identified to a release" in text or "not to a release" in text


def test_the_document_has_a_closing_section():
    body = (NOTE / "main.tex").read_text(encoding="utf-8")
    body = body[body.index("\\section{Context}") :]
    last = [line for line in body.splitlines() if line.startswith("\\section")][-1]
    assert "Next Steps" in last or "Conclusion" in last, (
        f"the note ends on {last!r} rather than a closing section"
    )


def test_the_file_format_is_explained():
    prose = _all_prose()
    assert "Parquet" not in prose or "read\\_parquet" in prose, (
        "the note names Parquet without showing how to open one"
    )
