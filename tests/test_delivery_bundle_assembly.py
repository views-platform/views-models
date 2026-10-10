"""The delivery bundle's paperwork is generated, not hand-written.

A README that restates the note's figures can drift from it silently; a
README generated from the note and measured against the data cannot. These
guards hold `assemble.sh` to that, and — the case that matters — hold it to
refusing rather than writing a README that describes data the bundle does
not contain.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTE = REPO / "reports/delivery_notes/r2darts2-pgm-global-calibration-2026-10"
ASSEMBLE = NOTE / "bundle/assemble.sh"

COLUMNS = ["month_id", "priogrid_id", "pred_lr_ged_sb", "pred_lr_ged_ns", "pred_lr_ged_os"]
MODELS = ("dark_river", "silent_fox")
ROWS = 7
ORIGINS = 2

pytestmark = pytest.mark.skipif(
    not ASSEMBLE.exists(), reason="the delivery note's bundle script is not present"
)


def _write_bundle(root: Path, columns: list[str] = COLUMNS) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for model in MODELS:
        out = root / model / "parquet"
        out.mkdir(parents=True)
        (root / model / "MANIFEST").write_text(f"model: {model}\n", encoding="utf-8")
        for i in range(ORIGINS):
            table = pa.table({c: list(range(ROWS)) for c in columns})
            pq.write_table(table, out / f"predictions_calibration_20260101_0000_{i:02d}.parquet")
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
    """The template's own guard, held from outside it."""
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "@" not in text, "a placeholder survived substitution"


def test_the_readme_reports_what_the_bundle_actually_holds(assembled: Path):
    """Every figure measured, none recited."""
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert f"{len(MODELS) * ORIGINS} Parquet files" in text
    assert str(ROWS) in text
    # Not the real delivery's figures, which a hand-written README would carry.
    assert "117 Parquet files" not in text
    assert "2,333,448" not in text


def test_the_readme_agrees_with_the_note_on_identity(assembled: Path):
    """The identifier and checksum tie the folder to the document."""
    readme = (assembled / "README.txt").read_text(encoding="utf-8")
    main = (NOTE / "main.tex").read_text(encoding="utf-8")
    for macro in ("deliveryid", "deliverychecksum", "deliverydate"):
        line = next(ln for ln in main.splitlines() if ln.startswith(f"\\newcommand{{\\{macro}}}"))
        value = line.split("{", 2)[2].rsplit("}", 1)[0]
        assert value in readme, f"{macro} = {value!r} is not in the README"


def test_numbers_are_grouped_the_way_the_note_groups_them(assembled: Path):
    """A Danish locale writes 2 333 448 and 1,9 GB. The note writes 2,333,448."""
    text = (assembled / "README.txt").read_text(encoding="utf-8")
    assert "1,9 GB" not in text and "0,9 GB" not in text
    assert "2 333 448" not in text


def test_the_manifest_is_renamed_so_a_browser_will_preview_it(assembled: Path):
    for model in MODELS:
        assert (assembled / model / "MANIFEST.txt").exists()
        assert not (assembled / model / "MANIFEST").exists()


def test_every_delivered_file_is_checksummed_and_the_sums_are_right(assembled: Path):
    sums = assembled / "SHA256SUMS.txt"
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest

    on_disk = {
        str(p.relative_to(assembled))
        for p in assembled.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.txt"
    }
    assert set(listed) == on_disk, "SHA256SUMS.txt does not cover exactly the delivered files"

    for name, digest in listed.items():
        actual = hashlib.sha256((assembled / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name}: recorded {digest}, actual {actual}"


def test_the_note_pdf_travels_with_the_data(assembled: Path):
    shipped = (assembled / "DELIVERY_NOTE.pdf").read_bytes()
    assert shipped == (NOTE / "main.pdf").read_bytes()


def test_it_refuses_rather_than_describing_data_it_did_not_find(tmp_path: Path):
    """The guard that matters.

    A bundle whose columns are not the delivered columns is not this
    delivery. Writing a README over it would hand the recipient a packing
    list for something else.
    """
    bundle = _write_bundle(tmp_path / "wrong", columns=COLUMNS[:-1] + ["pred_something_else"])
    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted a bundle with the wrong columns"
    assert not (bundle / "README.txt").exists(), "it wrote a README anyway"
    assert not (bundle / "SHA256SUMS.txt").exists()


def test_it_refuses_a_bundle_whose_files_disagree_on_row_count(tmp_path: Path):
    """One short file is the shape a lost or truncated origin takes."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    bundle = _write_bundle(tmp_path / "ragged")
    victim = next((bundle / MODELS[0] / "parquet").glob("*_01.parquet"))
    pq.write_table(pa.table({c: [0] for c in COLUMNS}), victim)

    result = _assemble(bundle)
    assert result.returncode != 0, "assemble accepted files with differing row counts"
    assert not (bundle / "README.txt").exists()


def test_it_refuses_when_the_note_has_not_been_built(tmp_path: Path):
    """Without main.pdf there is nothing to deliver, and no page count to read."""
    stub = tmp_path / "note"
    (stub / "bundle").mkdir(parents=True)
    shutil.copy(ASSEMBLE, stub / "bundle/assemble.sh")
    shutil.copy(NOTE / "bundle/README.txt.in", stub / "bundle/README.txt.in")
    shutil.copy(NOTE / "main.tex", stub / "main.tex")

    bundle = _write_bundle(tmp_path / "nopdf")
    result = subprocess.run(
        ["bash", str(stub / "bundle/assemble.sh"), str(bundle)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "BUNDLE_PYTHON": sys.executable},
    )
    assert result.returncode != 0
    assert "main.pdf" in result.stderr
    assert not (bundle / "README.txt").exists()
