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
    """A mount table naming a path that is not there is "I could not look".

    ⚠️ The label changed on 2026-10-27, and the rename IS the fix. This case used to be
    reported as `not-a-directory`, which lumped two opposite situations together: a
    mountpoint that is **absent** (refuse — something is wrong with the host) and a
    **bind-mounted file** like /etc/resolv.conf (benign — every container has several).
    Because both refused, the script could never return 0 on a real pod, which is the
    cry-wolf inverse of C-169. See `test_bind_mounted_files_do_not_make_every_pod_unverified`
    for the other half; this test is what stops that fix going too far.
    """
    a, b, mounts = _fixture(tmp_path, outside=False)
    mounts.write_text(mounts.read_text().replace(str(b), str(tmp_path / "mntGONE")))
    r = _run(a, mounts, since)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "mountpoint-missing" in r.stdout, r.stdout


@pytest.mark.green
def test_an_overlay_mount_is_swept_and_not_filtered_away(tmp_path, since):
    """⛔ `/` on a container is overlay. Excluding it reproduced the original incident.

    Measured 2026-10-27: with `/` declared overlay and `/workspace` a network mount, a
    file written to /root was invisible to the sweep and the script printed "safe to
    destroy", exit 0. That is `find / -xdev` again with a mount-type filter in place of
    the device check — the same shape, one layer over. /root is also where this
    platform's convention puts hand-written files, because /workspace ignores chmod.

    The assertion is the whole point: an overlay mount holding an uncollected file must
    produce exit 2, never 0. A regression here is silent and costs irrecoverable weights.
    """
    a, b, _m = _fixture(tmp_path, outside=False)
    (b / "root").mkdir()
    (b / "root" / "by_sb_best.npz").touch()          # the gate posterior, outside $OUT

    mounts = tmp_path / "mounts_overlay"
    mounts.write_text(
        f"/dev/sda1 {a} ext4 rw 0 0\n"
        f"overlay {b} overlay rw 0 0\n"               # <- the container root
        "proc /proc proc rw 0 0\n"
    )
    r = _run(a, mounts, since)
    assert r.returncode == 2, (
        "an overlay mount was filtered out of the sweep, so a file outside $OUT was "
        "invisible and the host was declared safe to destroy\n" + r.stdout + r.stderr
    )
    assert "root" in r.stdout, r.stdout


@pytest.mark.green
def test_a_mount_whose_subdirectory_cannot_be_read_refuses(tmp_path, since):
    """⛔ The control must exercise RECURSION, because the real sweep recurses.

    The earlier version planted its sentinel at depth 1 and looked for it with
    `-maxdepth 1`, so the positive control proved only that the top of the mount was
    visible. Measured 2026-10-27: a chmod-000 SUBdirectory holding an uncollected file
    was skipped in silence and the script printed "safe to destroy", exit 0 — because
    find's stderr went to /dev/null and its exit code was swallowed by `$(...)`.

    C-113's lesson in a new subsystem: read the exit status before interpreting output.
    The mount root is readable here; only a child is not.
    """
    a, b, mounts = _fixture(tmp_path, outside=False)
    hidden = b / "hidden"
    hidden.mkdir()
    (hidden / "by_sb_best.npz").touch()
    os.chmod(hidden, 0o000)
    try:
        r = _run(a, mounts, since)
        assert r.returncode == 1, (
            "a subdirectory the sweep could not read was skipped in silence\n"
            + r.stdout + r.stderr
        )
        assert "find-could-not-read-all-of-it" in r.stdout + r.stderr, r.stdout + r.stderr
    finally:
        os.chmod(hidden, stat.S_IRWXU)


@pytest.mark.green
def test_the_verdict_names_what_it_did_not_sweep(tmp_path, since):
    """A verdict that says "every mount" while a filter removed a class is the defect.

    tmpfs, ramfs and squashfs remain excluded — a stated judgement, not a measurement —
    so the exclusion has to be visible in the output rather than implied by silence
    (ADR-025 §2.4), and the clean verdict must stop claiming more than it swept.
    """
    a, _b, mounts = _fixture(tmp_path, outside=False)
    r = _run(a, mounts, since)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "excluded as kernel pseudo filesystems" in r.stdout, r.stdout
    assert "/dev/shm" in r.stdout, "the excluded tmpfs mount is not named\n" + r.stdout
    assert "every mount listed above" in r.stdout, (
        'the clean verdict still claims "every mount"\n' + r.stdout
    )


@pytest.mark.green
@pytest.mark.parametrize("opt", ["--since", "--mounts"])
def test_a_trailing_option_exits_instead_of_hanging(opt):
    """⛔ `shift 2` with one argument left fails WITHOUT shifting — the loop never ended.

    Measured 2026-10-27: `pod_sweep_outputs.sh --since` never returned. This runs on a pod,
    under time pressure, before a host is destroyed; a typo must cost one line of output,
    not the terminal. The timeout is the assertion — if this test ever hangs, that is the
    regression, so it is bounded rather than trusting the exit code alone.
    """
    r = subprocess.run([SCRIPT, opt], capture_output=True, text=True, timeout=15)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "requires a value" in r.stderr, r.stderr


@pytest.mark.green
def test_bind_mounted_files_do_not_make_every_pod_unverified(tmp_path, since):
    """⛔ Every container has /etc/hosts, /etc/hostname and /etc/resolv.conf as bind-mounted
    FILES, fstype ext4 or overlay — not pseudo, so they reach the loop.

    Classifying each as UNVERIFIED meant the script could never return 0 on a real pod: it
    printed "DO NOT DESTROY" every single time. That is the cry-wolf inverse of C-169 and
    just as dangerous, because a refusal nobody reads is not a refusal. Measured
    2026-10-27 against a realistic mount table: rc=1 with the only real mount clean.

    A single file needs no sentinel and no recursion — it is stat'd, reported on its own
    line, and deliberately kept out of the exit-2 verdict, because /etc/resolv.conf is
    rewritten by the container runtime and is not a deliverable.
    """
    a, b, _m = _fixture(tmp_path, outside=False)
    hosts, resolv = tmp_path / "etc_hosts", tmp_path / "etc_resolv"
    hosts.touch()
    resolv.touch()

    mounts = tmp_path / "mounts_container"
    mounts.write_text(
        f"overlay {a} overlay rw 0 0\n"
        f"/dev/sdb1 {b} ext4 rw 0 0\n"
        f"/dev/sda1 {hosts} ext4 rw 0 0\n"          # <- bind-mounted FILE
        f"/dev/sda1 {resolv} ext4 rw 0 0\n"         # <- bind-mounted FILE
        "proc /proc proc rw 0 0\n"
    )
    r = _run(a, mounts, since)
    assert r.returncode == 0, (
        "bind-mounted files were treated as unsweepable mounts, so the script refuses on "
        "every real pod\n" + r.stdout + r.stderr
    )
    assert "file mounts touched" in r.stdout, (
        "the touched file mounts are not named — they must be reported, not hidden\n"
        + r.stdout
    )


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
