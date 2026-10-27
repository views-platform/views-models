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

⛔ **The scenario table this file is held to.** An independent audit on 2026-10-27 found FIVE
ways to make the script return 0 while a file the run touched sat outside `$OUT`. Each row below
is one scenario and its required exit code; the guards in this file are the durable form of it.

| | scenario | required |
|---|---|---|
| S1 | everything inside `$OUT` | **0** safe to destroy |
| S2 | a file outside `$OUT`, same mount | **2** collect first |
| S3 | a declared mountpoint that is not there | **1** could not look |
| S4 | a child directory the sweep cannot read | **1** |
| S5 | a mount the sentinel cannot be planted on | **1** |
| S6 | a mount table with nothing sweepable | **1** |
| S7 | the mountpoint is a symlink | **1** |
| S8 | a trailing option with no value | **3** usage, and no hang |
| **S9** | **`--since` set after the run** | **1** — the sweep saw nothing of the deliverable |
| **S10** | **a model name containing `.`** | **2** — matched literally, not as a regex |
| **S11** | **`$OUT`'s own mount excluded** | **1** — the deliverable's filesystem was never read |
| **S12** | **mtime equal to `--since`** | **2** — the boundary is not dropped |
| **S13** | **a signal mid-sweep** | **143**, and nothing left behind |
| **S14** | **nested mountpoints on one device** | counted **once** |

⛔ **And how these guards are proved.** Each is killed by a mutation that changes ONLY the
property its docstring names. A whole-file revert is NOT a proof: it shows that *something* in
the fix matters, not that each guard pins its own property, and it is exactly what let
`test_a_mount_whose_subdirectory_cannot_be_read_refuses` sit here decorative — passing on one
half of its stated reasoning while the other half was pinned by nothing. That is C-159 and
C-164. 22 of 22 per-property mutations are killed as of 2026-10-27; the log is on PR #589.
"""

import os
import re
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
    """⛔ `find`'s exit code and stderr are read before its output is interpreted (C-113).

    Measured 2026-10-27: a chmod-000 SUBdirectory holding an uncollected file was skipped in
    silence and the script printed "safe to destroy", exit 0 — because find's stderr went to
    /dev/null and its exit code was swallowed by `$(...)`. The mount root is readable here;
    only a child is not.

    ⛔ CORRECTED 2026-10-27, and the correction is the point. This docstring used to lead on
    the positive control exercising RECURSION. An independent audit showed that was false: it
    reverted the control to the exact depth-1 proxy and all fifteen tests stayed green, so
    this test passes on the `FIND_RC`/`ERRFILE` half of its reasoning and on nothing else. It
    was DECORATIVE with respect to the property it named.

    Why that slipped through: the mutation proof reverted the WHOLE script and counted seven
    failures, which shows that *something* in the fix matters — not that *each guard pins its
    stated property*. That is C-159 and C-164, committed in the session that documented them.
    The recursion property is now pinned structurally by
    `test_the_positive_control_uses_the_same_find_shape_as_the_real_sweep`, which also records
    why no behavioural test for it exists.
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


@pytest.mark.green
def test_a_sweep_that_saw_nothing_under_out_refuses(tmp_path, since):
    """⛔ A1 — the sentinel is not the control §2.4 asks for, and this is.

    The sentinel is created NOW and only requires SINCE < NOW, so it passes for ANY accepted
    `--since`: it proves the mount and the find predicates work, never that the window
    brackets the run. Measured 2026-10-27: `--since` one hour ago over a two-hour-old tree
    swept 0 files and printed "0 touched file(s), all inside $OUT. This is the only result on
    which destroying the host is justified" — with a gate posterior sitting outside the bundle.

    The floor is independent because another party guarantees it: `pod_run_model.sh:546`
    refuses to report OK unless `$OUT` holds a non-empty `predictions_forecasting_*`. So a
    sweep that sees nothing under `$OUT` has the wrong window, not a clean host.
    """
    a, b, mounts = _fixture(tmp_path, outside=True)
    old = time.time() - 7200
    for f in list(a.rglob("*")) + list(b.rglob("*")):
        if f.is_file():
            os.utime(f, (old, old))
    r = _run(a, mounts, int(time.time()) - 3600)
    assert r.returncode == 1, (
        "a sweep that found nothing at all reported the host safe to destroy\n"
        + r.stdout + r.stderr
    )
    assert "swept-but-empty-in-this-window" in r.stdout, r.stdout


@pytest.mark.green
def test_a_model_name_with_a_regex_metacharacter_is_matched_literally(tmp_path, since):
    """⛔ A2 — `$OUT` was interpolated into a `grep` REGEX, not a literal prefix.

    With `MODEL='a.c'` the `.` matched any character, so `deliver/aXc/` — a DIFFERENT model's
    directory — satisfied `^$OUT/` and its file was counted as INSIDE the bundle. Measured
    2026-10-27 against a control: literal name `aXc` correctly gave exit 2, dotted name `a.c`
    gave exit 0 over the same tree. It did not merely miss the file; it counted it as safe.
    """
    a = tmp_path / "mntA"
    (a / "deliver" / "a.c").mkdir(parents=True)
    (a / "deliver" / "a.c" / "y_pred.npy").touch()
    (a / "deliver" / "aXc").mkdir(parents=True)
    (a / "deliver" / "aXc" / "gate_LEFT_BEHIND.nc").touch()
    mounts = tmp_path / "mounts"
    mounts.write_text(f"/dev/sda1 {a} ext4 rw 0 0\n")

    r = subprocess.run(
        [SCRIPT, "--since", f"@{since}", "--mounts", str(mounts), "a.c"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PODRUN_ROOT": str(a)},
    )
    assert r.returncode == 2, (
        "a file in another model's directory was counted as inside $OUT because the model "
        "name was treated as a regex\n" + r.stdout + r.stderr
    )
    assert "aXc" in r.stdout, r.stdout


@pytest.mark.green
def test_out_on_an_excluded_mount_refuses(tmp_path, since):
    """⛔ A3 — the deliverable's own filesystem was never examined, and it still said yes.

    Measured 2026-10-27: with `$OUT`'s mount declared `tmpfs` — which the pseudo filter
    excludes — the sweep covered one unrelated empty mount and printed "safe to destroy",
    exit 0. Nothing checked that `$OUT` was inside a mount this sweep actually verified.
    """
    work, empty = tmp_path / "work", tmp_path / "empty"
    (work / "deliver" / "testmodel").mkdir(parents=True)
    (work / "deliver" / "testmodel" / "y_pred.npy").touch()
    empty.mkdir()
    mounts = tmp_path / "mounts"
    mounts.write_text(f"tmpfs {work} tmpfs rw 0 0\n/dev/sda1 {empty} ext4 rw 0 0\n")

    r = _run(work, mounts, since)
    assert r.returncode == 1, (
        "$OUT's own mount was excluded from the sweep and the host was still declared safe\n"
        + r.stdout + r.stderr
    )
    assert "not-on-any-verified-mount" in r.stdout, r.stdout


@pytest.mark.green
def test_a_file_whose_mtime_equals_since_is_not_dropped(tmp_path, since):
    """⛔ A4 — `find -newermt` is strictly greater, and the boundary is the common case.

    `--since` is usually a whole-second `@epoch` read from a log, so a file written in that
    same second sits exactly on the boundary and `-newermt @S` excludes it. The sweep now runs
    from `SINCE-1` and accepts one second of slack rather than dropping a boundary file.

    ⚠️ This does NOT fix the preserved-mtime blind spot (`cp -p`, `rsync -a`, `tar -xp`), which
    is a real bound on what the script can answer and is now stated in its header instead of
    being claimed away by "which files did this run touch".
    """
    a, b, mounts = _fixture(tmp_path, outside=False)
    (b / "wandb").mkdir()
    boundary = b / "wandb" / "by_sb_best.npz"
    boundary.touch()
    os.utime(boundary, (since, since))

    r = _run(a, mounts, since)
    assert r.returncode == 2, (
        "a file whose mtime equals --since exactly was invisible to the sweep\n"
        + r.stdout + r.stderr
    )


@pytest.mark.green
def test_a_signal_leaves_no_litter_and_reaches_no_verdict(tmp_path):
    """⛔ A5 — a Ctrl-C mid-sweep must stop, not carry on with its working files deleted.

    Two defects in one. There was no `trap`, so an interrupt left a sentinel directory on the
    mount — and because the sweep's own exclusion was not PID-scoped, that leftover became a
    permanent, unreported blind spot, against a header claiming "removes each one it writes".
    Then `trap cleanup EXIT INT TERM` runs the handler and lets the script **continue** —
    verified 2026-10-27 — so it would compute a verdict from emptied lists.

    ⚠️ SIGTERM, not SIGINT: bash sets SIGINT to ignore for `&`-started commands in a
    non-interactive shell. And the padding goes INSIDE `$OUT` so a completed run would exit 0,
    making an interrupt distinguishable from a legitimate exit 2 (which deliberately keeps its
    outside list). An earlier version of this test failed for both of those reasons instead of
    the one it names.
    """
    a, _b, mounts = _fixture(tmp_path, outside=False)
    out = a / "deliver" / "testmodel"
    for i in range(3000):
        d = out / "pad" / str(i)
        d.mkdir(parents=True)
        (d / "f").touch()
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()

    proc = subprocess.Popen(
        [SCRIPT, "--since", f"@{int(time.time()) - 600}", "--mounts", str(mounts), "testmodel"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PODRUN_ROOT": str(a), "TMPDIR": str(tmpdir)},
    )
    time.sleep(0.08)
    proc.terminate()
    proc.communicate(timeout=60)

    assert proc.returncode == 143, (
        f"a signalled run returned {proc.returncode}; it must stop at 143 and reach no verdict"
    )
    litter = list(a.rglob(".podsweep_sentinel.*")) + list(tmpdir.glob(".podsweep_tmp.*"))
    assert litter == [], f"the trap left files behind: {litter}"


@pytest.mark.green
def test_nested_mountpoints_do_not_double_count(tmp_path, since):
    """A bind-mounted directory is swept by its parent AND itself, so one file was counted twice.

    `-xdev` prunes at DEVICE boundaries while the loop iterates DECLARED mountpoints. The
    running totals therefore drifted and the "Full list" held duplicates. Counts are now
    derived from deduplicated lists, which removes the arithmetic rather than correcting it.
    """
    a, b, _m = _fixture(tmp_path, outside=False)
    (b / "sub" / "wandb").mkdir(parents=True)
    (b / "sub" / "wandb" / "table.json").touch()
    mounts = tmp_path / "mounts_nested"
    mounts.write_text(
        f"/dev/sda1 {a} ext4 rw 0 0\n"
        f"/dev/sdb1 {b} ext4 rw 0 0\n"
        f"/dev/sdb1 {b / 'sub'} ext4 rw 0 0\n"
    )
    r = _run(a, mounts, since)
    assert r.returncode == 2, r.stdout + r.stderr
    counts = re.findall(r"^⛔ (\d+) file\(s\) touched", r.stdout, re.M)
    assert counts == ["1"], f"expected one outside file, got {counts}\n{r.stdout}"


@pytest.mark.green
def test_two_model_names_are_refused(tmp_path, since):
    """Two positional arguments silently took the LAST as the model, so `$OUT` pointed at a
    directory the operator did not name and the verdict was about the wrong bundle."""
    _a, _b, mounts = _fixture(tmp_path, outside=False)
    r = subprocess.run(
        [SCRIPT, "--since", f"@{since}", "--mounts", str(mounts), "testmodel", "othermodel"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PODRUN_ROOT": str(tmp_path / "mntA")},
    )
    assert r.returncode == 3, r.stdout + r.stderr
    assert "two model names" in r.stderr, r.stderr


@pytest.mark.green
def test_the_positive_control_uses_the_same_find_shape_as_the_real_sweep():
    """⛔ The control must exercise the instrument, not a shallower proxy — pinned structurally.

    ADR-025 §4 holds this script up as its first exemplar specifically because its control
    "plants a sentinel with the same `find` predicates as the real sweep, so the control
    exercises the instrument rather than a proxy". An earlier version planted at depth 1 and
    looked with `-maxdepth 1` while the real sweep recursed.

    ⚠️ STATED LIMIT, per C-164. This is a SOURCE assertion, not a behavioural one, and that is
    deliberate: the recursive control has no externally observable consequence, because the
    `find`-exit-code-and-stderr check already refuses on any subtree the sweep cannot read. I
    could not construct a scenario in which the depth-1 control alone changes the verdict, so
    the recursive control is defence in depth. Recording that honestly is better than a
    behavioural test that would pass for a different reason —
    `test_a_mount_whose_subdirectory_cannot_be_read_refuses` is exactly that test, and its
    docstring now says which half of it does the work.
    """
    src = open(SCRIPT, encoding="utf-8").read()
    finds = re.findall(r"find \"\$MP\"[^\n]*", src)
    assert len(finds) == 2, f"expected one control find and one sweep find, got {finds}"
    control, sweep = finds
    for predicate in ("-xdev", "-type f", '-newermt "@$SWEEP_SINCE"'):
        assert predicate in control, f"the control lost {predicate}: {control}"
        assert predicate in sweep, f"the real sweep lost {predicate}: {sweep}"
    assert "-maxdepth" not in control, (
        "the control is depth-limited while the real sweep recurses — a proxy, not a control: "
        + control
    )


@pytest.mark.green
def test_an_empty_mount_table_refuses(tmp_path, since):
    """ADR-025 §5.2 holds this script's "the mount table yielded no sweepable mounts" up as the
    model refusal message, and nothing pinned it. A table with only pseudo filesystems leaves
    nothing to sweep, which is "I could not look"."""
    a, _b, _m = _fixture(tmp_path, outside=False)
    mounts = tmp_path / "mounts_pseudo"
    mounts.write_text("proc /proc proc rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n")
    r = _run(a, mounts, since)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no sweepable mounts" in r.stdout + r.stderr, r.stdout + r.stderr


@pytest.mark.green
def test_unverified_is_decided_before_the_outside_count(tmp_path, since):
    """⛔ ADR-025 §2.3: "the ordering is the requirement, not an implementation detail."

    A run that is BOTH unverified and dirty must report UNVERIFIED (exit 1), not
    collect-them-first (exit 2) — unverified dominates. Nothing pinned the order, so making
    the UNVERIFIED branch conditional on a clean outside count left the whole suite green.
    """
    # ⚠️ The unreadable directory must be on a THIRD mount, not on the one holding the outside
    #    files. An earlier version put both on mount B: the find error makes the script skip B
    #    entirely, so B's outside files were never counted, OUTSIDE_COUNT stayed 0, and the
    #    test passed whether or not the ordering held. It could not distinguish the property it
    #    names — caught by the per-property mutation proof, which is what that proof is for.
    a, b, _m = _fixture(tmp_path, outside=True)           # dirty: files outside $OUT, on B
    c = tmp_path / "mntC"                                  # unverified: a third mount
    hidden = c / "hidden"
    hidden.mkdir(parents=True)
    (hidden / "x").touch()
    mounts = tmp_path / "mounts_both"
    mounts.write_text(
        f"/dev/sda1 {a} ext4 rw 0 0\n"
        f"/dev/sdb1 {b} ext4 rw 0 0\n"
        f"/dev/sdc1 {c} ext4 rw 0 0\n"
    )
    os.chmod(hidden, 0o000)
    try:
        r = _run(a, mounts, since)
        assert r.returncode == 1, (
            "a run that was both unverified and dirty reported the dirt and swallowed the "
            "unverified — unverified must dominate (ADR-025 §2.3)\n" + r.stdout + r.stderr
        )
        assert "OUTSIDE" in r.stdout, (
            "precondition: mount B's outside files must actually have been counted, or this "
            "test cannot distinguish the ordering\n" + r.stdout
        )
    finally:
        os.chmod(hidden, stat.S_IRWXU)


@pytest.mark.green
@pytest.mark.parametrize("form", ["path", "iso"])
def test_every_documented_since_form_resolves(tmp_path, since, form):
    """The header documents three `--since` forms; only `@epoch` was exercised. A silent
    failure in either of the others resolves `--since` to 0 or to now, which is A1's defect
    arriving by a different road."""
    a, _b, mounts = _fixture(tmp_path, outside=False)
    if form == "path":
        ref = a / "deliver" / "testmodel" / ".lessons"
        ref.touch()
        os.utime(ref, (since, since))
        arg = str(ref)
    else:
        arg = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since))
    r = subprocess.run(
        [SCRIPT, "--since", arg, "--mounts", str(mounts), "testmodel"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PODRUN_ROOT": str(a)},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"since {since}" in r.stdout, (
        f"--since as {form} did not resolve to {since}\n" + r.stdout
    )


@pytest.mark.green
def test_a_clean_run_leaves_no_sentinel_behind(tmp_path, since):
    """The sentinel cleanup had no test, so removing it left the suite green while every run
    littered the mount it had just certified — and the sweep's own exclusion then hid it."""
    a, b, mounts = _fixture(tmp_path, outside=False)
    r = _run(a, mounts, since)
    assert r.returncode == 0, r.stdout + r.stderr
    litter = list(a.rglob(".podsweep_sentinel*")) + list(b.rglob(".podsweep_sentinel*"))
    assert litter == [], f"a clean run left its sentinel behind: {litter}"


@pytest.mark.green
def test_out_on_the_root_mount_is_recognised_as_covered(tmp_path, since):
    """⚠️ The companion to `test_out_on_an_excluded_mount_refuses`: the A3 check must not refuse
    when `$OUT` IS covered.

    The first version compared with `"$MP"/*`, which for the root mount `/` expands to `//*` and
    matches nothing — so `$OUT` on the root filesystem was reported
    `not-on-any-verified-mount` and refused. A false refusal is the cry-wolf direction this
    script's own comments call "just as dangerous", and it was the defect the A3 check exists to
    fix, reintroduced one line below it.

    The mountpoint here ends in a slash, which is the shape that breaks the naive pattern while
    still being a legitimate mount table entry.
    """
    a, _b, _m = _fixture(tmp_path, outside=False)
    mounts = tmp_path / "mounts_trailing_slash"
    mounts.write_text(f"/dev/sda1 {a}/ ext4 rw 0 0\n")
    r = _run(a, mounts, since)
    assert r.returncode == 0, (
        "$OUT was on the swept mount but the coverage check did not recognise it, so a clean "
        "host was refused\n" + r.stdout + r.stderr
    )
    assert "not-on-any-verified-mount" not in r.stdout, r.stdout


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
