"""Guards on the HydraNet delivery note's load-bearing statements.

These assert on the LaTeX sources rather than the PDF, because the sources are
what a later edit touches.

THEY ASSERT WHOLE CLAIMS, NOT KEYWORDS. A first version of this file checked
that a word was present --- "not reproducible", "confounded", "overlap" --- and
a mutation audit broke thirteen of fifteen guards by keeping the word and
inverting the sentence around it. "The contrast is confounded: not really ---
the two share a seed" passed. So each guard below pins the sentence that
carries the claim, verbatim and whitespace-normalised, including the \\textbf{}
that makes it a claim rather than an aside.

WHAT THIS STILL CANNOT DO. A string assertion cannot prove a document's
meaning. Someone who keeps a guarded sentence and adds "that is no longer
true" beside it passes every test here. What these guards catch is drift ---
a claim reworded, softened, renumbered or deleted, which is the way a document
actually decays --- and not sabotage. Said plainly rather than left implied,
because a guard whose limits are unstated gets trusted past them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

NOTE = (
    Path(__file__).resolve().parents[1]
    / "reports/delivery_notes/hydranet-pgm-global-calibration-2026-09"
)
SECTIONS = NOTE / "sections"
CHECKSUM = NOTE.parent / "checksum.py"

pytestmark = pytest.mark.skipif(
    not SECTIONS.exists(), reason="the HydraNet delivery note is not present"
)


def _flat(name: str) -> str:
    """One section, whitespace-normalised, so a claim can be matched across the
    line wrapping that LaTeX sources carry."""
    return " ".join((SECTIONS / name).read_text(encoding="utf-8").split())


def _macro(name: str) -> str:
    text = (NOTE / "main.tex").read_text(encoding="utf-8")
    match = re.search(rf"^\\newcommand\{{\\{name}\}}\{{(.*)\}}$", text, re.M)
    assert match, f"\\{name} is not defined in main.tex"
    return match.group(1)


# --- the claims the delivery would be misleading without --------------------


def test_the_note_says_the_two_experimental_frames_cannot_be_reproduced():
    """The one asymmetry in the delivery.

    Frames II and III have no implementation in this repository and the draws
    they were computed from were never transferred off the rented machines. A
    note that describes all three estimators evenhandedly implies all three
    can be checked.
    """
    text = _flat("the_frames.tex")
    assert (
        "\\textbf{Frames II and III are not reproducible, and no code for them exists.}" in text
    ), "the note no longer states the asymmetry as a claim"
    assert "cannot be recomputed, re-derived or checked against the posterior." in text
    assert "never transferred off, and the machines are gone" in text


def test_the_note_corrects_the_retained_posterior_claim():
    """A memo of 28 September told the recipient the draws had come home.

    They had not. The recipient holds that memo, so the note carries the
    correction --- in Caveats, where a reader looking for what is wrong with
    the delivery will find it, and not only in passing.
    """
    caveats = _flat("additional_notes.tex")
    assert "\\textbf{A statistic other than these three needs a new run.}" in caveats
    assert "cost memo of 28 September" in caveats, "the note does not name the document it corrects"
    assert "That was wrong: the draws are not on disk." in caveats
    # And in the section about the frames themselves, for a reader who stops there.
    assert "This also corrects a claim made in my own cost memo" in _flat("the_frames.tex")


def test_the_note_retracts_the_authority_that_does_not_exist():
    """The folder's own README cited "ADR-023 Amendment 1", which never existed.

    The note may name it in order to retract it. It may not rest on it, and it
    may not mention it anywhere else.
    """
    assert "No such document exists." in _flat("the_frames.tex")
    for path in sorted(SECTIONS.glob("*.tex")):
        if path.name == "the_frames.tex":
            continue
        assert "Amendment 1" not in path.read_text(encoding="utf-8"), (
            f"{path.name} refers to Amendment 1 outside the retraction"
        )


def test_the_horizon_figures_are_the_measured_ones():
    """The 'flat after ~5 steps' claim was inherited from a different run.

    These were measured from the delivered parquets. Every number in the
    paragraph is pinned, not just the ones in the table: the prose figures are
    the ones a reader quotes.
    """
    text = _flat("magnitude_and_horizon.tex")
    assert "16,404" in text, "the step-1 total is missing"
    assert "8,265" in text and "8,907" in text, "the plateau range is missing"
    assert "falls by 45\\%" in text, "the step-1-to-5 fall is missing or changed"
    assert "a spread of 7.8\\%" in text, "the plateau spread is missing or changed"
    assert "thirty-one of the thirty-six months delivered" in text
    assert "reading month 30 as a forecast is a misuse of the file" in text


def test_the_magnitude_section_states_it_is_not_an_evaluation():
    text = _flat("magnitude_and_horizon.tex")
    assert "\\textbf{This section ranks nothing and recommends nothing.}" in text
    # A section that ranks nothing does not then recommend one of the eight.
    assert "I recommend" not in text, "the section recommends a model"
    assert "I rank" not in text, "the section ranks the models"


def test_the_note_distinguishes_live_rows_from_distinct_cell_months():
    """1,469,004 is a row count, and the folder's README called it cell-months.

    The same cell-month is counted once per model and once per origin. A reader
    who takes it as distinct cell-months is out by a factor of fifteen.
    """
    text = _flat("magnitude_and_horizon.tex")
    assert "1,469,004" in text
    assert "98,995 distinct cell-months" in text, "the distinct count is missing"
    assert (
        "counts \\emph{rows}, which means the same cell-month is counted once per model "
        "and once per origin; it is not a count of distinct cell-months." in text
    ), "the note no longer says which quantity 1,469,004 is"


def test_the_note_gives_the_coordinate_conversion_correctly():
    """A pgm delivery carries no coordinates.

    The arithmetic is pinned digit by digit. A wrong divisor or a wrong cell
    size puts every row the recipient maps in the wrong place, and nothing
    about the output looks wrong.
    """
    text = _flat("the_data.tex")
    assert "$\\texttt{row} = \\lfloor (\\texttt{priogrid\\_id} - 1) / 720 \\rfloor$" in text
    assert "$\\texttt{col} = (\\texttt{priogrid\\_id} - 1) \\bmod 720$" in text
    assert "$-90 + (\\texttt{row} + 0.5) \\times 0.5$" in text
    assert "$-180 + (\\texttt{col} + 0.5) \\times 0.5$" in text
    assert "cell centre" in text, "it does not say the coordinates are cell centres"
    # The worked example, so the recipient can check their own arithmetic.
    assert "181875" in text and "36.25" in text and "37.25" in text


def test_the_note_warns_against_stacking_the_origins():
    text = _flat("the_data.tex")
    assert (
        "Concatenating the thirteen files and treating the result as a panel produces a "
        "table with up to thirteen different values for the same cell and month, which is "
        "not a forecast of anything." in text
    ), "the note no longer warns against stacking the origins"
    assert "Pick one origin, or carry the origin as an explicit dimension." in text


def test_the_next_steps_section_decides_rather_than_asks():
    """The register rule: anything inside our remit reads as a decision.

    Section 9 is where this goes wrong most easily --- a question there asks
    the recipient to adjudicate an engineering choice, and he will. The
    question-mark check is a floor, not a ceiling: a question can be phrased
    without one, and no test here catches that.
    """
    text = _flat("conclusion.tex")
    assert "?" not in text, "Next Steps asks the recipient a question"
    assert "None of them needs a decision from the reader." in text


def test_the_note_names_the_column_convention_difference():
    """Two deliveries now use different column names for the same quantity.

    A tool that infers the target column from the schema reads one of the two
    wrongly and silently, so the note says what the consumer must do.
    """
    text = _flat("conclusion.tex")
    assert "pred\\_lr\\_sb\\_best" in text and "pred\\_lr\\_ged\\_sb" in text
    assert (
        "the target column has to be set explicitly wherever these files are read, not "
        "inferred from the schema" in text
    ), "the note no longer tells the consumer to set the column explicitly"
    assert "a tool that guesses will silently read the wrong delivery's convention" in text


def test_the_authorship_is_stated_as_settled_not_claimed():
    """Conceded in writing on 2026-10-01. A hedge invites it to be reopened."""
    text = _flat("the_model.tex")
    assert (
        "It is my own work: the design, the implementation and the paper describing it "
        "\\citep{maase_hydranet} are mine." in text
    ), "the authorship statement has been reworded"
    assert "It is not a library model and has no upstream." in text


def test_the_confounded_comparison_is_flagged_where_it_is_invited():
    """heavy_freighter vs bright_starship looks like a controlled contrast.

    The seeds differ too. Both sections that invite the reading carry the
    caveat, because a reader may read either one alone.
    """
    models = _flat("the_eight_models.tex")
    assert (
        "\\textbf{The contrast is confounded:} the two also differ in seed, so the gap "
        "between them cannot be attributed to scheduled sampling alone." in models
    )
    caveats = _flat("additional_notes.tex")
    assert "They also differ in random seed." in caveats
    assert "the two explanations cannot be separated." in caveats


def test_the_version_gap_is_stated_rather_than_filled_in():
    """Provenance pins the code to a commit but not to a release.

    Each of the three version macros is checked on its own: one of them saying
    "not recorded" does not excuse another carrying a version nobody captured.
    """
    for macro in ("deliverylibrary", "deliveryframework", "deliverybackend"):
        assert "not recorded" in _macro(macro), (
            f"\\{macro} claims a version that was never captured"
        )
    caveats = _flat("additional_notes.tex")
    assert "\\textbf{The software is pinned as ranges, not versions.}" in caveats
    assert "identified to a repository state but not to a release" in caveats


# --- the note's own machinery -----------------------------------------------


def test_the_declared_checksum_matches_the_specification_fields():
    """Nothing in the build runs this, so the only thing that does is a test.

    `checksum.py --check` recomputes from the fourteen specification fields and
    compares. Without it a field can be edited and the note ships declaring a
    checksum for a specification it no longer has.
    """
    result = subprocess.run(
        [sys.executable, str(CHECKSUM), str(NOTE), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_file_section_label_is_on_the_section_it_names():
    """`\\label{sec:files}` is what `assemble.sh` counts sections up to.

    Attached to the wrong section it sends the recipient to the wrong page,
    and every `\\ref{sec:files}` in the prose follows it there.
    """
    for label, title in (
        ("sec:files", "The Delivered Files"),
        ("sec:frames", "The Three Frames"),
        ("sec:caveats", "Caveats and Notes"),
    ):
        line = next(
            (
                line
                for line in (NOTE / "main.tex").read_text(encoding="utf-8").splitlines()
                if f"\\label{{{label}}}" in line
            ),
            None,
        )
        assert line, f"\\label{{{label}}} is not in main.tex"
        assert title in line, f"\\label{{{label}}} is on {line!r}, not on {title!r}"


def test_the_document_has_a_closing_section():
    body = (NOTE / "main.tex").read_text(encoding="utf-8")
    body = body[body.index("\\section{Context}") :]
    last = [line for line in body.splitlines() if line.startswith("\\section")][-1]
    assert "Next Steps" in last or "Conclusion" in last, (
        f"the note ends on {last!r} rather than a closing section"
    )


def test_the_file_format_is_explained():
    prose = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SECTIONS.glob("*.tex")))
    assert "Parquet" not in prose or "read\\_parquet" in prose, (
        "the note names Parquet without showing how to open one"
    )
