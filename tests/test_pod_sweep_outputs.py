"""Guards for `tools/podrun/pod_sweep_outputs.sh`.

The script exists because on 2026-10-25 a pod was declared ready to destroy on an
omission sweep that returned EMPTY — `find / -xdev` on a host where /workspace is a
network mount on another device. The empty result read as a clean bill of health
(risk register C-168, C-169, C-170; Cluster G).

So the property under test is not "the sweep finds files". It is **the sweep refuses
to answer when it cannot see**. `test_an_unreadable_mount_refuses_instead_of_passing`
is the one that matters: the same mount reports a clean 0 in the happy path, and the
test makes it unreadable and asserts the verdict flips to UNVERIFIED rather than
staying clean. The others pin the exit-code vocabulary that distinguishes the three
answers — safe, collect-first, could-not-look — because a script that returns 0 for
all three is the defect it was written to remove.
"""

import os
import stat
import subprocess
import time

import pytest

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "podrun", "pod_sweep_outputs.sh",
)


def _fixture(tmp_path, *, outside: bool):
    """Two fake mounts: mount A holds $OUT, mount B is everything else."""
    a, b = tmp_path / "mntA", tmp_path / "mntB"
    out = a / "deliver" / "testmodel"
    (out / "parquet").mkdir(parents=True)
    (out / "parquet" / "p1.parquet").touch()
    (out / "run.log").touch()

    stale = b / "models" / "testmodel"
    stale.mkdir(parents=True)
    pre_run = stale / "OLD_FILE"
    pre_run.touch()
    old = time.time() - 7200
    os.utime(pre_run, (old, old))

    if outside:
        (b / "wandb").mkdir()
        (b / "wandb" / "table.json").touch()
        (stale / "metricframe_lr_sb_best").touch()

    mounts = tmp_path / "mounts"
    mounts.write_text(
        f"/dev/sda1 {a} ext4 rw 0 0\n"
        f"/dev/sdb1 {b} ext4 rw 0 0\n"
        "proc /proc proc rw 0 0\n"
        "tmpfs /dev/shm tmpfs rw 0 0\n"
    )
    return a, b, mounts


def _run(root, mounts, since):
    return subprocess.run(
        [SCRIPT, "--since", f"@{since}", "--mounts", str(mounts), "testmodel"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PODRUN_ROOT": str(root)},
    )


@pytest.fixture
def since():
    return int(time.time()) - 3600


@pytest.mark.green
def test_everything_inside_out_is_safe_to_destroy(tmp_path, since):
    a, _b, mounts = _fixture(tmp_path, outside=False)
    r = _run(a, mounts, since)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all inside" in r.stdout
    # the pre-run file on mount B must not be counted as this run's output
    assert "OLD_FILE" not in r.stdout


@pytest.mark.green
def test_output_outside_out_exits_two_and_names_the_directories(tmp_path, since):
    a, _b, mounts = _fixture(tmp_path, outside=True)
    r = _run(a, mounts, since)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "OUTSIDE $OUT" in r.stdout
    assert "wandb" in r.stdout
    assert "data" in r.stdout or "models" in r.stdout


@pytest.mark.red
def test_an_unreadable_mount_refuses_instead_of_passing(tmp_path, since):
    """The guard this file exists for.

    Mount B reports a clean `0 touched` in the happy path. Make the sweep unable to
    plant its sentinel there and the verdict must become UNVERIFIED with a non-zero
    exit — never a clean pass. "Nothing matched" and "I could not look" must not
    render as the same answer.
    """
    a, b, mounts = _fixture(tmp_path, outside=False)
    assert _run(a, mounts, since).returncode == 0, "precondition: clean before mutation"

    original = stat.S_IMODE(os.stat(b).st_mode)
    os.chmod(b, 0o555)
    try:
        r = _run(a, mounts, since)
    finally:
        os.chmod(b, original)

    assert r.returncode == 1, r.stdout + r.stderr
    assert "UNVERIFIED" in r.stdout
    assert "cannot-plant-sentinel" in r.stdout


@pytest.mark.red
def test_a_mount_find_cannot_list_refuses_instead_of_passing(tmp_path, since):
    """The branch the first version of this file left undefended.

    `test_an_unreadable_mount_refuses_instead_of_passing` passes through the earlier
    `cannot-plant-sentinel` check, so deleting the sentinel *verification* left every
    test green — a decorative guard, found by mutating the script (M1). This exercises
    the other branch: a mount where the sentinel CAN be written and then cannot be
    found, which is the shape of the 2026-10-25 failure (the file was there; the sweep
    could not see it).

    Mode 0o333 is write+traverse without read: `touch` succeeds, `find` cannot list.
    """
    a, b, mounts = _fixture(tmp_path, outside=False)
    assert _run(a, mounts, since).returncode == 0, "precondition: clean before mutation"

    original = stat.S_IMODE(os.stat(b).st_mode)
    os.chmod(b, 0o333)
    try:
        r = _run(a, mounts, since)
    finally:
        os.chmod(b, original)

    assert r.returncode == 1, r.stdout + r.stderr
    assert "UNVERIFIED" in r.stdout
    assert "sentinel-not-found" in r.stdout, (
        "the sweep wrote a sentinel it could not then see, and did not say so"
    )


@pytest.mark.red
def test_a_mount_that_is_not_there_refuses_instead_of_passing(tmp_path, since):
    a, b, mounts = _fixture(tmp_path, outside=False)
    mounts.write_text(mounts.read_text().replace(str(b), str(tmp_path / "mntGONE")))
    r = _run(a, mounts, since)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "not-a-directory" in r.stdout


@pytest.mark.red
@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["testmodel"], id="since-missing"),
        pytest.param(["--since", "not a date", "testmodel"], id="since-unparseable"),
        pytest.param(["--since", "@0"], id="model-missing"),
    ],
)
def test_usage_errors_are_distinguishable_from_a_verdict(tmp_path, args):
    """Exit 3 is usage. It must never collide with 0 (safe) or 1 (could not look)."""
    _a, _b, mounts = _fixture(tmp_path, outside=False)
    r = subprocess.run(
        [SCRIPT, *args, "--mounts", str(mounts)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PODRUN_ROOT": str(tmp_path / "mntA")},
    )
    assert r.returncode == 3, r.stdout + r.stderr


@pytest.mark.beige
def test_a_future_since_is_rejected(tmp_path):
    """C-171: the watcher whose hardcoded epoch was a year off read a stale file as
    fresh. A --since that is not in the past cannot bracket a finished run."""
    a, _b, mounts = _fixture(tmp_path, outside=False)
    r = _run(a, mounts, int(time.time()) + 86400)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "not before now" in (r.stdout + r.stderr)
