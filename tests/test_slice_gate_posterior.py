"""`tools/podrun/slice_gate_posterior.py` must refuse, not slice half the gate.

The script exists because `pod_run_model.sh` archives only `lr_*` and reports `OK`, so the
gate posterior dies with the pod (views-models#558, C-161). A slicer that itself returned a
partial result while reporting success would reproduce that defect one layer up, so every
refusal below is tested against a fixture tree rather than asserted in a docstring.

The fixture is small — 4 land cells, 3 steps, 2 draws — and the land-cell count is injected
through `--expected-rows`. The real default is pinned here instead, so shrinking the fixture
cannot quietly shrink the production guard.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "podrun" / "slice_gate_posterior.py"

ROWS = 4  # "land cells" in the fixture
STEPS = 3
DRAWS = 2
FIRST_MONTH = 457
TARGETS = ("by_sb_best", "by_ns_best", "by_os_best")


def _module():
    spec = importlib.util.spec_from_file_location("slice_gate_posterior", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_target(target_dir: Path, *, rows: int = ROWS, draws: int = DRAWS, value=None):
    target_dir.mkdir(parents=True, exist_ok=True)
    n = rows * STEPS
    rng = np.random.default_rng(0)
    y = rng.random((n, draws), dtype=np.float32) if value is None else np.full((n, draws), value, np.float32)
    time = np.repeat(np.arange(FIRST_MONTH, FIRST_MONTH + STEPS, dtype=np.int32), rows)
    unit = np.tile(np.arange(1, rows + 1, dtype=np.int32), STEPS)
    np.save(target_dir / "y_pred.npy", y)
    np.savez(target_dir / "identifiers.npz", time=time, unit=unit)


@pytest.fixture
def pod(tmp_path):
    """A fixture pod: a repo with one run's posterior, and a deliver directory."""
    repo = tmp_path / "views-models"
    source = repo / "models" / "copper_kestrel" / "data" / "generated" / "predictions_calibration_20261016_000000"
    for origin in ("origin_0", "origin_1"):
        for target in TARGETS:
            _write_target(source / origin / target)
        # the magnitude body is present too, and must be ignored
        _write_target(source / origin / "lr_sb_best")
    out = tmp_path / "deliver" / "copper_kestrel"
    out.mkdir(parents=True)
    return repo, out, source


def _run(repo, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "copper_kestrel",
         "--repo", str(repo), "--out", str(out), "--expected-rows", str(ROWS), *extra],
        capture_output=True, text=True,
    )


# ── the happy path, and what it is required to produce ────────────────────


def test_it_slices_h1_for_every_origin_and_gate_target(pod):
    repo, out, _ = pod
    result = _run(repo, out)
    assert result.returncode == 0, result.stderr

    manifest = json.loads((out / "gate_h1" / "SLICE_MANIFEST.json").read_text())
    assert sorted(manifest) == ["origin_0", "origin_1"]
    for per_origin in manifest.values():
        assert sorted(per_origin) == sorted(TARGETS)
        for stats in per_origin.values():
            assert stats["h1_month"] == FIRST_MONTH
            assert stats["rows"] == ROWS
            assert stats["shape"] == [ROWS, DRAWS]
            assert stats["n_nonfinite"] == 0

    for origin in ("origin_0", "origin_1"):
        for target in TARGETS:
            assert (out / "gate_h1" / origin / target / "y_pred_h1.npy").is_file()
            assert (out / "gate_h1" / origin / target / "identifiers_h1.npz").is_file()


def test_it_slices_the_first_month_and_nothing_else(pod):
    """The point of the slice: 1/36 of the rows, and the right 1/36."""
    repo, out, source = pod
    assert _run(repo, out).returncode == 0
    sliced = np.load(out / "gate_h1" / "origin_0" / "by_sb_best" / "y_pred_h1.npy")
    full = np.load(source / "origin_0" / "by_sb_best" / "y_pred.npy")
    assert sliced.shape == (ROWS, DRAWS)
    np.testing.assert_array_equal(sliced, full[:ROWS])
    ids = np.load(out / "gate_h1" / "origin_0" / "by_sb_best" / "identifiers_h1.npz")
    assert set(ids["time"].tolist()) == {FIRST_MONTH}


def test_it_lands_inside_out_so_the_existing_rsync_carries_it(pod):
    """No runner change: the archive and its digest must be under $OUT."""
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    archive = out / "gate_h1.tar.zst"
    assert archive.is_file() and archive.stat().st_size > 0
    digest_line = (out / "gate_h1.sha256").read_text().split()
    assert digest_line[1] == "gate_h1.tar.zst"
    import hashlib
    assert digest_line[0] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_the_archive_holds_every_file_that_was_written(pod):
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    listing = subprocess.run(
        ["tar", "-I", "zstd", "-tf", str(out / "gate_h1.tar.zst")],
        capture_output=True, text=True, check=True,
    )
    members = [line for line in listing.stdout.splitlines() if not line.endswith("/")]
    # 2 origins x 3 targets x 2 files, plus the manifest
    assert len(members) == 2 * 3 * 2 + 1


def test_the_archive_count_check_actually_refuses(pod):
    """⛔ Found by mutation: the test above did not exercise the SCRIPT's check.

    `test_the_archive_holds_every_file_that_was_written` computes the expected member count
    itself and compares it to the archive, so it passes whether or not the script checks
    anything — deleting `if len(members) != expected_files:` left the whole file green. That
    is the C-161 blindness reproduced inside the fix for C-161: a guard that counts the set
    it selected rather than the set that exists.

    So the check is exercised directly, with a count the archive cannot match.
    """
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    module = _module()
    tree = out / "gate_h1"
    with pytest.raises(module.Refuse) as refusal:
        module._archive(tree, out / "probe.tar.zst", expected_files=10_000)
    assert "the archive holds" in refusal.value.args[0]

    # And it passes on the truth, so the refusal above is not simply "always raises".
    written = sum(1 for path in tree.rglob("*") if path.is_file())
    module._archive(tree, out / "probe_ok.tar.zst", expected_files=written)


def test_the_magnitude_body_is_not_in_the_slice(pod):
    """`lr_*` already comes home in the runner's own archive; duplicating it wastes the link."""
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    assert not list((out / "gate_h1").rglob("lr_*"))


# ── the refusals ──────────────────────────────────────────────────────────


def test_it_refuses_when_there_is_no_gate_posterior_at_all(pod):
    """The whole point. If HydraNet wrote no `by_*`, there is no readout to bring home."""
    repo, out, source = pod
    for origin in source.iterdir():
        for target in TARGETS:
            shutil.rmtree(origin / target)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "no by_* directories" in result.stderr
    assert not (out / "gate_h1").exists()


def test_it_refuses_a_row_count_that_is_not_the_land_cell_count(pod):
    repo, out, source = pod
    _write_target(source / "origin_1" / "by_ns_best", rows=ROWS + 1)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "rows, expected" in result.stderr


def test_it_refuses_a_non_finite_gate_probability(pod):
    repo, out, source = pod
    _write_target(source / "origin_0" / "by_os_best", value=np.nan)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "non-finite" in result.stderr


def test_it_refuses_a_target_missing_from_one_origin(pod):
    repo, out, source = pod
    shutil.rmtree(source / "origin_1" / "by_os_best")
    result = _run(repo, out)
    assert result.returncode == 1
    assert "missing" in result.stderr


def test_it_refuses_targets_that_disagree_on_the_draw_count(pod):
    repo, out, source = pod
    _write_target(source / "origin_1" / "by_sb_best", draws=DRAWS + 1)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "draw count" in result.stderr


def test_it_refuses_rather_than_merging_into_an_existing_slice(pod):
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    second = _run(repo, out)
    assert second.returncode == 1
    assert "already exists" in second.stderr


def test_it_refuses_off_pod_when_the_deliver_directory_is_absent(pod):
    repo, out, _ = pod
    result = _run(repo, out / "does-not-exist")
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_it_refuses_when_no_run_has_happened(tmp_path):
    repo = tmp_path / "views-models"
    (repo / "models" / "copper_kestrel" / "data" / "generated").mkdir(parents=True)
    out = tmp_path / "deliver" / "copper_kestrel"
    out.mkdir(parents=True)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "no predictions_calibration_*" in result.stderr


# ── the production defaults, pinned ───────────────────────────────────────


def test_the_production_defaults_are_what_the_pod_needs():
    """The fixture shrinks the row count through a flag; it must not shrink the default.

    And the zstd thread count is pinned at a bounded value on purpose: `-T0` is one thread
    per core, and on the 128-core pod it died in 2.3 s leaving a truncated 1 MB archive from
    182 MB of input (#558 addendum). A guard that produced a truncated archive and reported
    success is precisely what this file is here to prevent.
    """
    module = _module()
    assert module.EXPECTED_ROWS == 64818, (
        "the global-land cell count is the shape check that catches slicing an array this "
        "script does not understand"
    )
    assert module.ZSTD_THREADS == "-T4", (
        "-T0 OOM-killed the archive on a 128-core pod and left a truncated file behind; "
        "bounded threads are deliberate, not arbitrary"
    )
    assert module.GATE_PREFIX == "by_"
