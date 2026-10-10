"""The HydraNet delivery bundle's paperwork is generated, not hand-written.

The sibling of `test_delivery_bundle_assembly.py`, against the second data
bundle. The shape it guards is different in the way that matters: this
delivery nests a frame dimension under each model, and the frames are the
thing a reader is most likely to misread, so an assembler that cannot tell a
complete frame set from an incomplete one is worse than no assembler.

The case that matters here, as there, is the refusal: writing a README that
describes data the bundle does not contain is the failure mode, not crashing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTE = REPO / "reports/delivery_notes/hydranet-pgm-global-calibration-2026-09"
ASSEMBLE = NOTE / "bundle/assemble.sh"

COLUMNS = ["month_id", "priogrid_id", "pred_lr_sb_best", "pred_lr_ns_best", "pred_lr_os_best"]
MODELS = ("purple_alien", "violet_visitor")
FRAMES = ("I_mean", "II_q95", "III_conditional_mean")
# Over a thousand, so the thousands separator is actually exercised. At 7
# rows the grouping guard below could not fail whatever the locale did.
ROWS = 1234
# The real delivery has exactly thirteen origins per frame and the assembler
# refuses anything else, so the fixture has to carry thirteen too.
ORIGINS = 13

pytestmark = pytest.mark.skipif(
    not ASSEMBLE.exists(), reason="the delivery note's bundle script is not present"
)


def _write_bundle(
    root: Path,
    columns: list[str] = COLUMNS,
    frames: tuple[str, ...] = FRAMES,
    origins: int = ORIGINS,
    notebook: bool = True,
) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for model in MODELS:
        for frame in frames:
            out = root / model / frame
            out.mkdir(parents=True)
            for i in range(origins):
                table = pa.table({c: list(range(ROWS)) for c in columns})
                pq.write_table(
                    table, out / f"predictions_calibration_20260928_201214_{i:02d}.parquet"
                )
    if notebook:
        (root / "audit_selector_frames.ipynb").write_text("{}", encoding="utf-8")
    return root


def _assemble(bundle: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(ASSEMBLE), str(bundle)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "BUNDLE_PYTHON": sys.executable},
    )


@pytest.fixture(scope="module")
def assembled(tmp_path_factory) -> Path:
    bundle = _write_bundle(tmp_path_factory.mktemp("bundle"))
    result = _assemble(bundle)
    assert result.returncode == 0, f"assemble failed:\n{result.stdout}\n{result.stderr}"
    return bundle


def test_the_readme_has_no_unsubstituted_placeholders(assembled: Path):
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "@" not in text, "a placeholder survived into the README"


def test_the_readme_reports_what_the_bundle_actually_holds(assembled: Path):
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "78 Parquet files" in text, "the measured file count is missing"
    assert "1,234" in text, "the measured row count is missing"
    # The real delivery's figures must not appear in a README generated from a
    # fixture: that would mean the script restates rather than measures.
    assert "312 Parquet files" not in text
    assert "2,333,448" not in text
    # The counts that were prose until a mutation audit found them: a two-model
    # fixture produced a packing list that said "eight in all".
    assert f"One folder per model, {len(MODELS)} in all:" in text
    assert f"holds {ORIGINS} Parquet files" in text
    assert "eight in all" not in text, "the model count is hard-coded in the template"
    assert "thirteen Parquet files" not in text, "the origin count is hard-coded"


def test_the_readme_quotes_a_file_that_is_actually_in_the_bundle(assembled: Path):
    """The worked example has to be openable.

    A hard-coded sample filename reads correctly and fails the one thing the
    section exists for, which is being pasted into a session.
    """
    import re

    text = (assembled / "README.txt").read_text(encoding="utf-8")
    quoted = set(re.findall(r'"([^"]+\.parquet)"', text))
    assert quoted, "the README quotes no parquet file"
    for ref in quoted:
        assert (assembled / ref).is_file(), f"the README points at {ref}, which is not there"


def test_the_template_reads_the_section_numbers_rather_than_restating_them(assembled: Path):
    """Section numbers shift when a section is inserted.

    Restating them as literals is what this forbids: the generated README is
    checked against the note's own ordering, and the template is checked for
    still carrying the tokens, because a test that derives the number the same
    way the script does would agree with a hard-coded one.
    """
    template = (NOTE / "bundle/README.txt.in").read_text(encoding="utf-8")
    assert "@FILESECTION@" in template, "the file-section number is hard-coded in the template"
    assert "@FRAMESECTION@" in template, "the frame-section number is hard-coded in the template"


def test_the_readme_agrees_with_the_note_on_identity(assembled: Path):
    """Both sides read main.tex, so agreement between them proves little.

    The checksum is therefore recomputed from the specification fields rather
    than copied: a note and a README that agree on a checksum neither of them
    derived is the failure this looks for.
    """
    import re

    note = (NOTE / "main.tex").read_text(encoding="utf-8")
    text = (assembled / "README.txt").read_text(encoding="utf-8")

    for macro in ("deliveryid", "deliverychecksum", "deliverydate", "deliveryauthor"):
        match = re.search(rf"^\\newcommand\{{\\{macro}\}}\{{(.*)\}}$", note, re.M)
        assert match, f"\\{macro} is not in main.tex"
        assert match.group(1) in text, f"the README does not carry the note's {macro}"

    computed = subprocess.run(
        [sys.executable, str(NOTE.parent / "checksum.py"), str(NOTE)],
        capture_output=True,
        text=True,
    )
    assert computed.returncode == 0, computed.stderr
    assert computed.stdout.strip() in text, (
        "the README carries a checksum that is not the one the specification computes to"
    )


def test_numbers_are_grouped_the_way_the_note_groups_them(assembled: Path):
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "1,234" in text
    assert "1 234" not in text, "a non-C locale grouped with spaces"
    assert "1.234" not in text, "a non-C locale grouped with full stops"


def test_the_readme_names_the_three_frames_and_which_one_is_declared(assembled: Path):
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    for frame in FRAMES:
        assert frame in text, f"{frame} is not named in the packing list"
    assert "not three models and not three runs" in text
    # Against the right frame. "This is the estimator the models declare" under
    # II_q95 would be a packing list that declares the wrong one.
    declared = text[text.index("I_mean/") : text.index("II_q95/")]
    assert "the estimator the models declare" in declared, (
        "the declared estimator is not named against I_mean"
    )


def test_the_readme_sends_the_reader_to_the_right_sections(assembled: Path):
    """The section numbers are read from the note, not restated.

    If a section is inserted ahead of them the README must follow, or it sends
    the recipient to the wrong page of a 16-page document.
    """
    note = (NOTE / "main.tex").read_text(encoding="utf-8")
    sections = [
        line for line in note.splitlines() if line.startswith("\\section")
    ]
    frames = next(i for i, s in enumerate(sections, 1) if "sec:frames" in s)
    files = next(i for i, s in enumerate(sections, 1) if "sec:files" in s)
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert f"Section {frames} of the delivery note" in text
    assert f"Section {files} of the delivery note" in text


def test_the_note_pdf_travels_with_the_data(assembled: Path):
    shipped = (assembled / "DELIVERY_NOTE.pdf").read_bytes()
    assert shipped == (NOTE / "main.pdf").read_bytes()


def test_no_checksum_manifest_is_shipped(assembled: Path):
    assert not (assembled / "SHA256SUMS.txt").exists()
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "sha256" not in text.lower(), "the README still talks about checksums"


def test_a_stale_checksum_manifest_is_removed(tmp_path: Path):
    """The `rm -f SHA256SUMS.txt` exists for a re-assembled folder.

    Without planting one the cleanup cannot be tested, and an earlier run's
    manifest would ship with nothing noticing.
    """
    bundle = _write_bundle(tmp_path / "stale")
    (bundle / "SHA256SUMS.txt").write_text("deadbeef  purple_alien/I_mean/x.parquet\n", encoding="utf-8")
    result = _assemble(bundle)
    assert result.returncode == 0, result.stderr
    assert not (bundle / "SHA256SUMS.txt").exists(), "a stale checksum manifest shipped"


def test_the_superseded_readme_is_removed_from_the_bundle(tmp_path: Path):
    """Two readmes disagreeing is the drift the generator exists to prevent."""
    bundle = _write_bundle(tmp_path / "with_md")
    (bundle / "README.md").write_text("# the hand-written one\n", encoding="utf-8")
    result = _assemble(bundle)
    assert result.returncode == 0, result.stderr
    assert not (bundle / "README.md").exists(), "the superseded README.md was left behind"
    assert (bundle / "README.txt").exists()


def test_it_refuses_rather_than_describing_data_it_did_not_find(tmp_path: Path):
    """The guard that matters. A README is a claim about the data."""
    bundle = _write_bundle(tmp_path / "wrong", columns=COLUMNS[:-1] + ["pred_something_else"])
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted a bundle with the wrong columns"
    assert not (bundle / "README.txt").exists(), "it wrote a README anyway"


def test_it_refuses_a_model_missing_a_frame(tmp_path: Path):
    """The failure this delivery's shape makes possible.

    Two frames out of three is a folder that looks complete model by model and
    is not, and the README would announce three.
    """
    bundle = _write_bundle(tmp_path / "two_frames", frames=FRAMES[:2])
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted a model with two frames"
    assert "frame directories" in result.stderr
    assert not (bundle / "README.txt").exists()


def test_it_refuses_a_frame_missing_an_origin(tmp_path: Path):
    bundle = _write_bundle(tmp_path / "short", origins=ORIGINS - 1)
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted twelve origins"
    assert "expected 13" in result.stderr
    assert not (bundle / "README.txt").exists()


def test_it_refuses_a_bundle_whose_files_disagree_on_row_count(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    bundle = _write_bundle(tmp_path / "ragged")
    odd = next((bundle / MODELS[0] / FRAMES[0]).glob("*_01.parquet"))
    pq.write_table(pa.table({c: [0] for c in COLUMNS}), odd)
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted files with different row counts"
    assert not (bundle / "README.txt").exists()


def test_it_refuses_when_the_audit_notebook_is_not_there(tmp_path: Path):
    """The README tells the recipient to open the notebook.

    This delivery ships no checksum manifest, so the notebook is the only
    thing a recipient can use to check the data. A README that points at a
    file that is not in the folder is worse than one that says nothing.
    """
    bundle = _write_bundle(tmp_path / "no_notebook", notebook=False)
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted a bundle with no audit notebook"
    assert "audit_selector_frames.ipynb" in result.stderr
    assert not (bundle / "README.txt").exists()


def test_it_refuses_when_the_note_has_not_been_built(tmp_path: Path):
    stub = tmp_path / "note"
    (stub / "bundle").mkdir(parents=True)
    shutil.copy(ASSEMBLE, stub / "bundle/assemble.sh")
    shutil.copy(NOTE / "bundle/README.txt.in", stub / "bundle/README.txt.in")
    shutil.copy(NOTE / "main.tex", stub / "main.tex")
    bundle = _write_bundle(tmp_path / "unbuilt")
    result = subprocess.run(
        ["bash", str(stub / "bundle/assemble.sh"), str(bundle)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "BUNDLE_PYTHON": sys.executable},
    )
    assert result.returncode != 0
    assert "main.pdf" in result.stderr
    assert not (bundle / "README.txt").exists()
