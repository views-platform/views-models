"""`tools/podrun/slice_gate_posterior.py` must refuse, not slice half the gate.

The script exists because `pod_run_model.sh` archives only `lr_*` and reports `OK`, so the
gate posterior dies with the pod (views-models#558, C-161). A slicer that itself returned a
partial result while reporting success would reproduce that defect one layer up, so every
refusal below is tested against a fixture tree rather than asserted in a docstring.

The fixture is small — 4 land cells, 3 steps, 2 draws — and both the land-cell count and the
step count are injected through flags. The real defaults are pinned in
`test_the_production_defaults_are_what_the_pod_needs`, so shrinking the fixture cannot quietly
shrink the production guard.

⚠️ The fixture's window is 3 months, so its far endpoint is labelled `h3`. On the pod it is
`h36`. The label is derived from `--steps` in both cases, which is the property
`test_the_far_endpoint_is_derived_from_the_window_not_read_off_the_data` exists to pin.
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
LAST_MONTH = FIRST_MONTH + STEPS - 1
NEAR, FAR = "h1", f"h{STEPS}"
TARGETS = ("by_sb_best", "by_ns_best", "by_os_best")
TREE = "gate_endpoints"


def _module():
    spec = importlib.util.spec_from_file_location("slice_gate_posterior", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_target(
    target_dir: Path, *, rows: int = ROWS, draws: int = DRAWS, value=None, months=None
):
    target_dir.mkdir(parents=True, exist_ok=True)
    months = list(range(FIRST_MONTH, FIRST_MONTH + STEPS)) if months is None else list(months)
    n = rows * len(months)
    rng = np.random.default_rng(0)
    y = rng.random((n, draws), dtype=np.float32) if value is None else np.full((n, draws), value, np.float32)
    time = np.repeat(np.array(months, dtype=np.int32), rows)
    unit = np.tile(np.arange(1, rows + 1, dtype=np.int32), len(months))
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
         "--repo", str(repo), "--out", str(out),
         "--expected-rows", str(ROWS), "--steps", str(STEPS), *extra],
        capture_output=True, text=True,
    )


# ── the happy path, and what it is required to produce ────────────────────


def test_it_slices_both_endpoints_for_every_origin_and_gate_target(pod):
    repo, out, _ = pod
    result = _run(repo, out)
    assert result.returncode == 0, result.stderr

    manifest = json.loads((out / TREE / "SLICE_MANIFEST.json").read_text())
    assert sorted(manifest) == ["origin_0", "origin_1"]
    for per_origin in manifest.values():
        assert sorted(per_origin) == sorted(TARGETS)
        for per_target in per_origin.values():
            assert sorted(per_target) == sorted([NEAR, FAR])
            assert per_target[NEAR]["month"] == FIRST_MONTH
            assert per_target[FAR]["month"] == LAST_MONTH
            for stats in per_target.values():
                assert stats["rows"] == ROWS
                assert stats["shape"] == [ROWS, DRAWS]
                assert stats["n_nonfinite"] == 0

    for origin in ("origin_0", "origin_1"):
        for target in TARGETS:
            for label in (NEAR, FAR):
                assert (out / TREE / origin / target / f"y_pred_{label}.npy").is_file()
                assert (out / TREE / origin / target / f"identifiers_{label}.npz").is_file()


def test_each_endpoint_holds_its_own_month_and_nothing_else(pod):
    """The point of the slice: 1/36 of the rows per endpoint, and the right 1/36."""
    repo, out, source = pod
    assert _run(repo, out).returncode == 0
    full = np.load(source / "origin_0" / "by_sb_best" / "y_pred.npy")

    near = np.load(out / TREE / "origin_0" / "by_sb_best" / f"y_pred_{NEAR}.npy")
    assert near.shape == (ROWS, DRAWS)
    np.testing.assert_array_equal(near, full[:ROWS])

    far = np.load(out / TREE / "origin_0" / "by_sb_best" / f"y_pred_{FAR}.npy")
    assert far.shape == (ROWS, DRAWS)
    np.testing.assert_array_equal(far, full[(STEPS - 1) * ROWS: STEPS * ROWS])

    for label, month in ((NEAR, FIRST_MONTH), (FAR, LAST_MONTH)):
        ids = np.load(out / TREE / "origin_0" / "by_sb_best" / f"identifiers_{label}.npz")
        assert set(ids["time"].tolist()) == {month}


def test_the_far_endpoint_is_derived_from_the_window_not_read_off_the_data(pod):
    """⛔ `first + steps - 1`, never `time.max()` — the #686 correctness point.

    On a window SHORTER than the declared one the two differ, and `time.max()` would label
    the last month present as the far endpoint. A readout pre-registered at step 36 would then
    be computed at step 30 and reported under the pre-registered name, with nothing downstream
    able to tell. So a short window must REFUSE, not relabel.

    Driven through `endpoint_months` directly as well as through the script, because the
    refusal is the behaviour under test and the script's exit code alone cannot say which
    check produced it.
    """
    module = _module()

    # The declared window: both endpoints, far one derived.
    full = np.repeat(np.arange(FIRST_MONTH, FIRST_MONTH + STEPS), ROWS)
    assert module.endpoint_months(full, STEPS, where="probe") == {
        NEAR: FIRST_MONTH, FAR: LAST_MONTH
    }

    # One month short. time.max() would be FIRST_MONTH + STEPS - 2 and would be labelled
    # the far endpoint; instead this refuses and says so.
    short = np.repeat(np.arange(FIRST_MONTH, FIRST_MONTH + STEPS - 1), ROWS)
    with pytest.raises(module.Refuse) as refusal:
        module.endpoint_months(short, STEPS, where="probe")
    assert "contiguous months" in refusal.value.args[0]
    assert "time.max()" in refusal.value.args[0]

    # And end to end, so the refusal is reachable from the command the operator runs.
    repo, out, source = pod
    _write_target(
        source / "origin_1" / "by_ns_best",
        months=range(FIRST_MONTH, FIRST_MONTH + STEPS - 1),
    )
    result = _run(repo, out)
    assert result.returncode == 1
    assert "contiguous months" in result.stderr
    assert not (out / TREE).exists()


def test_it_refuses_a_window_with_a_hole_in_it(pod):
    """A gappy window is not a window: h1..h36 must be present, in order, exactly once each."""
    repo, out, source = pod
    _write_target(
        source / "origin_0" / "by_os_best",
        months=[FIRST_MONTH, FIRST_MONTH + 1, FIRST_MONTH + 7],
    )
    result = _run(repo, out)
    assert result.returncode == 1
    assert "contiguous months" in result.stderr


def test_it_lands_inside_out_so_the_existing_rsync_carries_it(pod):
    """No runner change: the archive and its digest must be under $OUT."""
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    archive = out / f"{TREE}.tar.zst"
    assert archive.is_file() and archive.stat().st_size > 0
    digest_line = (out / f"{TREE}.sha256").read_text().split()
    assert digest_line[1] == f"{TREE}.tar.zst"
    import hashlib
    assert digest_line[0] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_the_archive_holds_every_file_that_was_written(pod):
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    listing = subprocess.run(
        ["tar", "-I", "zstd", "-tf", str(out / f"{TREE}.tar.zst")],
        capture_output=True, text=True, check=True,
    )
    members = [line for line in listing.stdout.splitlines() if not line.endswith("/")]
    # 2 origins x 3 targets x 2 endpoints x 2 files, plus the manifest
    assert len(members) == 2 * 3 * 2 * 2 + 1


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
    tree = out / TREE
    with pytest.raises(module.Refuse) as refusal:
        module._archive(tree, out / "probe.tar.zst", expected_files=10_000)
    assert "the archive holds" in refusal.value.args[0]

    # And it passes on the truth, so the refusal above is not simply "always raises".
    written = sum(1 for path in tree.rglob("*") if path.is_file())
    module._archive(tree, out / "probe_ok.tar.zst", expected_files=written)


def test_the_member_count_refuses_a_tar_that_does_not_mark_directories(pod, monkeypatch):
    """The trailing-slash filter is a CONVENTION of the listing tar, so it is checked.

    If a tar build listed directories without a trailing slash, the filter would be a no-op,
    the count would compare files-plus-directories against files, and the mismatch would be
    reported as a partial archive — blaming the data for a property of the instrument
    (ADR-025). The views-hydranet seat read the filter as counting directories on
    2026-11-01, which is the reading this check makes impossible to be quietly wrong about.
    """
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    module = _module()
    real_run = module.subprocess.run

    def listing_without_directories(cmd, *args, **kwargs):
        result = real_run(cmd, *args, **kwargs)
        if "-tf" in cmd:
            result.stdout = "\n".join(
                line.rstrip("/") for line in result.stdout.splitlines()
            )
        return result

    monkeypatch.setattr(module.subprocess, "run", listing_without_directories)
    with pytest.raises(module.Refuse) as refusal:
        module._archive(out / TREE, out / "probe_nodirs.tar.zst", expected_files=25)
    assert "no directory entry" in refusal.value.args[0]


def test_the_magnitude_body_is_not_in_the_slice(pod):
    """`lr_*` already comes home in the runner's own archive; duplicating it wastes the link."""
    repo, out, _ = pod
    assert _run(repo, out).returncode == 0
    assert not list((out / TREE).rglob("lr_*"))


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
    assert not (out / TREE).exists()


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


def test_it_refuses_a_non_finite_value_that_is_only_at_the_far_endpoint(pod):
    """⛔ A check that looks only at h=1 passes a gate that has gone NaN by h=36.

    This is the whole reason the far endpoint is sliced: the near one carries no information
    about it. Before #686 the slice was h=1-only, so this array would have come home, been
    archived, been digest-verified, and reported OK.
    """
    repo, out, source = pod
    target = source / "origin_0" / "by_sb_best"
    _write_target(target)
    y = np.load(target / "y_pred.npy")
    y[-ROWS:, :] = np.nan  # the last month only
    np.save(target / "y_pred.npy", y)
    result = _run(repo, out)
    assert result.returncode == 1
    assert "non-finite" in result.stderr
    assert FAR in result.stderr


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
    """The fixture shrinks the row and step counts through flags; it must not shrink the defaults.

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
    assert module.EXPECTED_STEPS == 36, (
        "EXP-02 pre-registers an endpoint at h=36; the fixture's 3-step window is injected "
        "through --steps and must not become the production default"
    )
    assert module.ZSTD_THREADS == "-T4", (
        "-T0 OOM-killed the archive on a 128-core pod and left a truncated file behind; "
        "bounded threads are deliberate, not arbitrary"
    )
    assert module.GATE_PREFIX == "by_"
    assert module.TREE_NAME == "gate_endpoints", (
        "the tree holds both endpoints; `gate_h1` would be a name that lies about its "
        "contents, which is the defect class this script exists to prevent"
    )


def test_a_refusal_part_way_through_leaves_no_partial_tree(pod):
    """⛔ The docstring's promise, enforced by code rather than by the sentence.

    `origin_0` slices cleanly; the refusal lands on `origin_1`. Before 2026-11-02 the
    half-written `gate_endpoints/` stayed on disk and the operator's rsync carried it home —
    no archive and no `.sha256`, so nothing downstream would be fooled, but a partial slice
    on disk is still a partial slice, and the file claims that is not one of its outcomes.

    Named separately from the far-endpoint test that found it, so a mutation removing the
    cleanup fails a test about cleanup rather than a test about endpoint derivation.
    """
    repo, out, source = pod
    _write_target(source / "origin_1" / "by_os_best", rows=ROWS + 1)

    result = _run(repo, out)
    assert result.returncode == 1
    assert not (out / TREE).exists(), (
        "the first origin had already been written when the second refused"
    )
    # And nothing else was left behind either — no archive, no digest.
    assert not (out / f"{TREE}.tar.zst").exists()
    assert not (out / f"{TREE}.sha256").exists()
