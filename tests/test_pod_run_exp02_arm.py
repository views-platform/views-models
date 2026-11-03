"""`tools/podrun/pod_run_exp02_arm.sh` must make two prose rules structural.

EXP-02 runs four arms on four pods, and two things have to be true that `pod_run_model.sh`
cannot make true on its own, because it is read-only by standing instruction and correct for
what it claims:

1. **The gate must come home.** The runner archives `-path '*/lr_*'` and leaves `by_*` to die
   with the pod while writing `STATUS: OK` (views-models#558, C-161). The rule "slice the gate
   before teardown" lived only in `docs/runpod_run_guide.md` and in an xfail stub, and on
   2026-10-26 it was not followed. ⛔ `test_a_run_whose_gate_was_never_written_does_not_report_ok`
   is the one that matters here: prose cannot gate a teardown, an exit code can.

2. **The four arms must be on one data pull.** The train leg carries no `-sa` and refetches on
   every pod, so shipping the parquet changes nothing. The pull is therefore CHECKED after the
   fetch, against a digest read from the prepared arms' own provenance sidecars.

Every test drives the real script with a STUB runner, because the behaviour under test is the
orchestration — what it refuses, what it kills, and what it declines to certify — not HydraNet.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.beige

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "podrun" / "pod_run_exp02_arm.sh"
SLICER = REPO_ROOT / "tools" / "podrun" / "slice_gate_posterior.py"

ARM = "ctlfullzero_fortytwo"
DIGEST = "b32a221b70e5255d3d85e683b483d9398ae08ed3b5e9d1be71c6e8f7aa13422c"
ROWS, STEPS, DRAWS, FIRST_MONTH = 4, 3, 2, 457
GATE_TARGETS = ("by_sb_best", "by_ns_best", "by_os_best")


def _write_posterior(origin: Path, *, gate: bool = True):
    names = list(GATE_TARGETS) if gate else []
    for target in [*names, "lr_sb_best"]:
        d = origin / target
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        np.save(d / "y_pred.npy", rng.random((ROWS * STEPS, DRAWS), dtype=np.float32))
        np.savez(
            d / "identifiers.npz",
            time=np.repeat(np.arange(FIRST_MONTH, FIRST_MONTH + STEPS, dtype=np.int32), ROWS),
            unit=np.tile(np.arange(1, ROWS + 1, dtype=np.int32), STEPS),
        )


def _stub_runner(
    path: Path, repo: Path, out: Path, *,
    digest: str = DIGEST, status: str = "OK", rc: int = 0,
    sidecar: str | None = "json", gate: bool = True,
    sleep_seconds: float = 0.0, month_last: int = 504,
) -> Path:
    """A `pod_run_model.sh` stand-in: writes the sidecar, the posterior, STATUS, then exits.

    `sleep_seconds` sits BETWEEN the sidecar and the posterior, which is where the real run
    spends ~12.5 h of GPU — so a test can assert the digest gate stopped the run by observing
    that the posterior was never written.
    """
    prov = repo / "models" / ARM / "data" / "raw"
    prov.mkdir(parents=True, exist_ok=True)
    sidecar_path = prov / "calibration_datafactory_df.parquet.provenance.json"
    body = json.dumps({
        "queryset_digest": digest, "level": "pgm", "month_first": 121,
        "month_last": month_last, "partition": "calibration",
        "source": "datafactory", "provenance_version": 2, "drift_detection_ran": False,
    })

    source = repo / "models" / ARM / "data" / "generated" / "predictions_calibration_20261102_000000"
    marker = out / ".stub_wrote_posterior"

    lines = ["#!/usr/bin/env bash", "set -u", f'mkdir -p "{out}"']
    if sidecar == "json":
        lines.append(f"cat > {sidecar_path!s} <<'JSON'\n{body}\nJSON")
    elif sidecar == "garbage":
        lines.append(f"printf 'not json at all' > {sidecar_path!s}")
    lines.append(f"sleep {sleep_seconds}")
    lines.append(f'python3 -c "import pathlib; pathlib.Path(r\'{marker}\').write_text(\'yes\')"')
    lines.append(f'printf "%s\\n" "{status}" > "{out}/STATUS"')
    lines.append(f"exit {rc}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return source


def _pod(tmp_path, **runner_kwargs):
    repo = tmp_path / "views-models"
    root = tmp_path / "workspace"
    out = root / "deliver" / ARM
    out.mkdir(parents=True)
    (repo / "models" / ARM).mkdir(parents=True)
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 204.168.219.108 login x password y\n")

    runner = tmp_path / "stub_runner.sh"
    source = _stub_runner(runner, repo, out, **runner_kwargs)
    # The posterior is written up front: the stub cannot easily build numpy arrays in bash,
    # and what the tests exercise is the orchestration around it.
    if runner_kwargs.get("gate", True):
        for origin in ("origin_0", "origin_1"):
            _write_posterior(source / origin)
    else:
        for origin in ("origin_0", "origin_1"):
            _write_posterior(source / origin, gate=False)
    return repo, root, out, runner, netrc


def _transcript(result) -> str:
    """stdout and stderr together, because the script keeps ONE transcript.

    `pod_run_exp02_arm.sh` opens with `exec > >(tee -a "$EXP_LOG") 2>&1`, the same line
    `pod_run_model.sh` uses: on a pod there is one log and an operator reads it with `cat`,
    so a refusal written to fd 2 still arrives on fd 1. Asserting against `result.stderr`
    alone found nothing and read as "the message is missing" rather than "I looked in the
    wrong stream" — which is the shape of mistake ADR-025 is about, committed by the test
    rather than by the code.
    """
    return result.stdout + result.stderr


def _run(repo, root, runner, netrc, *extra, arm=ARM):
    env = {**os.environ, "PODRUN_ROOT": str(root)}
    return subprocess.run(
        [
            "bash", str(SCRIPT),
            "--repo", str(repo), "--root", str(root),
            "--runner", str(runner), "--slicer", str(SLICER),
            "--netrc", str(netrc),
            "--poll-seconds", "1", "--fetch-timeout-seconds", "5",
            "--max-cores", "100000",
            *extra, arm,
        ],
        capture_output=True, text=True, env=env, timeout=180,
    )


# ── the structural rule this file exists for ──────────────────────────────


def test_a_run_whose_gate_was_never_written_does_not_report_ok(tmp_path):
    """⛔ THE POINT. The runner says OK; the gate is absent; this must NOT say OK.

    This is C-161 made into an exit code. `pod_run_model.sh` writes `STATUS: OK` for a run
    whose `by_*` directories were never produced, because its own N_DRAWS/N_ARCHIVED guard
    counts only the half it archives. Before this script the only thing standing between that
    and a destroyed pod was a sentence in the guide, and on 2026-10-26 the sentence lost.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, gate=False)
    result = _run(repo, root, runner, netrc)

    assert (out / "STATUS").read_text().strip() == "OK", "the runner certified itself"
    assert result.returncode == 1
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:gate_slice"
    assert "DO NOT DESTROY THIS POD" in _transcript(result)
    assert not (out / "EXP02_MANIFEST").exists()


def test_it_never_writes_the_runners_own_status_file(tmp_path):
    """Two writers on one file is how a stale verdict outlives the run that wrote it.

    `$OUT/STATUS` belongs to `pod_run_model.sh`, and `pod_run_fao_delivery.sh` reads it to
    decide whether a model may be pooled. This script's verdict goes to `EXP02_STATUS`.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, status="FAILED:train", rc=1)
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert (out / "STATUS").read_text().strip() == "FAILED:train", "unchanged by this script"
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:train"


# ── the digest gate ───────────────────────────────────────────────────────


def test_it_stops_a_mismatched_pull_before_the_gpu_is_spent(tmp_path):
    """⛔ The money AND the experiment. A wrong pull is caught at the sidecar, not at the end.

    The stub sleeps between writing the sidecar and writing its posterior marker — which is
    where the real run spends ~12.5 h. If the gate worked, the marker is absent: the run was
    killed during that window rather than after it.
    """
    repo, root, out, runner, netrc = _pod(
        tmp_path, digest="0" * 64, sleep_seconds=60,
    )
    result = _run(repo, root, runner, netrc)

    assert result.returncode == 1
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:digest_gate"
    assert not (out / ".stub_wrote_posterior").exists(), (
        "the run was allowed to continue past the gate — the gate saved no GPU time"
    )

    quarantined = (out / "QUARANTINED").read_text()
    assert "MISMATCH" in quarantined
    assert "0" * 64 in quarantined and DIGEST in quarantined
    assert "P-23" in quarantined and "cross-population" in quarantined


def test_it_checks_the_window_and_not_only_the_digest(tmp_path):
    """A digest says "some pull"; it does not say which window.

    A digest pinned by hand can be pinned to the wrong thing, and `month_first`/`month_last`/
    `level` are independently checkable facts about the same file. So they are checked, and a
    sidecar whose digest matches but whose window does not still refuses.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, month_last=552, sleep_seconds=60)
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:digest_gate"
    detail = (out / "QUARANTINED").read_text()
    assert "month_last" in detail and "552" in detail


def test_an_unreadable_sidecar_refuses_rather_than_passing(tmp_path):
    """ADR-025: a check that cannot read its subject must FAIL, not pass.

    "I could not parse the provenance" and "the pull is correct" are the same thing to a
    `try/except: pass`, and the register holds eight instances of that substitution.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, sidecar="garbage", sleep_seconds=60)
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:digest_gate"
    assert "UNREADABLE" in (out / "QUARANTINED").read_text()


def test_a_sidecar_that_never_appears_refuses_rather_than_training_blind(tmp_path):
    repo, root, out, runner, netrc = _pod(tmp_path, sidecar=None, sleep_seconds=60)
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:digest_gate"
    assert "I could not look" in _transcript(result)
    assert not (out / ".stub_wrote_posterior").exists()


def test_the_expected_digest_is_overridable_without_a_code_edit(tmp_path):
    """So a corrected pin does not need a commit at 2am — and the refusal prints it."""
    repo, root, out, runner, netrc = _pod(tmp_path, digest="a" * 64, sleep_seconds=60)
    ok = _run(repo, root, runner, netrc, "--expect-digest", "a" * 64)
    # It gets past the gate; whether it then completes depends on the slicer, which is a
    # different test's subject. What matters here is that the gate did not refuse.
    assert not (out / "QUARANTINED").exists(), (
        (out / "QUARANTINED").read_text() if (out / "QUARANTINED").exists() else ""
    )
    assert "digest gate OK" in ok.stdout


# ── preflight ─────────────────────────────────────────────────────────────


def test_it_refuses_an_arm_that_is_not_one_of_the_declared_four(tmp_path):
    """⛔ The `*tinyzero*` rehearsal arms share the pull and the config shape.

    A glob over `models/*fullzero*` would have excluded them by accident; a glob over
    `models/*zero*` would have included them. The set is the experiment's definition, so it is
    written down and checked by name.
    """
    repo, root, out, runner, netrc = _pod(tmp_path)
    (repo / "models" / "ctltinyzero_fortytwo").mkdir(parents=True, exist_ok=True)
    result = _run(repo, root, runner, netrc, arm="ctltinyzero_fortytwo")
    assert result.returncode == 1
    assert "is not one of the 12 declared arms" in _transcript(result)
    assert "tinyzero" in _transcript(result)


def test_it_refuses_before_training_when_the_slicer_is_absent(tmp_path):
    """Fatal BEFORE the GPU, not after. Discovering it at teardown is discovering it too late."""
    repo, root, out, runner, netrc = _pod(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--repo", str(repo), "--root", str(root),
         "--runner", str(runner), "--slicer", str(tmp_path / "no-such-slicer.py"),
         "--netrc", str(netrc), "--max-cores", "100000", ARM],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "the gate slicer is missing" in _transcript(result)
    assert not (out / ".stub_wrote_posterior").exists()


def test_it_refuses_a_box_whose_core_count_breaks_the_runners_archive(tmp_path):
    """`pod_run_model.sh:638` uses `zstd -3 -T0` — one thread per core.

    On a 128-core pod that died in 2.3 s and left a truncated 1 MB archive from 182 MB of
    input, and the runner's own guard reported success (#558 addendum). The runner is
    read-only, so the bound is enforced on the BOX.
    """
    repo, root, out, runner, netrc = _pod(tmp_path)
    result = _run(repo, root, runner, netrc, "--max-cores", "0")
    assert result.returncode == 1
    assert "zstd -3 -T0" in _transcript(result)
    assert "the limit is 0" in _transcript(result)
    assert not (out / ".stub_wrote_posterior").exists()


def test_it_refuses_when_the_arm_directory_is_not_in_the_clone(tmp_path):
    """views-models#685: four pods clone `development`; without the arms they train nothing."""
    repo, root, out, runner, netrc = _pod(tmp_path)
    shutil.rmtree(repo / "models" / ARM)
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert "does not exist" in _transcript(result)
    assert "685" in _transcript(result)


def test_it_refuses_without_the_datafactory_credential(tmp_path):
    repo, root, out, runner, netrc = _pod(tmp_path)
    netrc.unlink()
    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1
    assert "missing or empty" in _transcript(result)
    assert "C-154" in _transcript(result)


# ── the declared constants, pinned ────────────────────────────────────────


def test_the_declared_pull_and_arm_set_are_what_the_experiment_registered():
    """Read from the script's own text, so a silent edit to either shows up here.

    The digest was read 2026-11-02 from all six prepared arm directories' provenance sidecars,
    which agreed on all 64 characters. The views-hydranet seat independently supplied the
    first 16 and they match — so this value has two sources, not one relayed string.
    """
    text = SCRIPT.read_text()
    assert f'EXPECTED_DIGEST="{DIGEST}"' in text
    assert 'EXPECTED_LEVEL="pgm"' in text
    assert "EXPECTED_MONTH_FIRST=121" in text
    assert "EXPECTED_MONTH_LAST=504" in text
    assert "CORES_MAX=32" in text, (
        "the bound exists because the read-only runner archives with zstd -T0"
    )
    for arm in (
        "ctlfullzero_fortytwo", "ctlfullzero_fortythree",
        "upctwofullzero_fortytwo", "upctwofullzero_fortythree",
        "ctlfullzero_fortyfour", "ctlfullzero_fortyfive",
        "ctlfullzero_fortysix", "ctlfullzero_fortyseven",
        "upctwofullzero_fortyfour", "upctwofullzero_fortyfive",
        "upctwofullzero_fortysix", "upctwofullzero_fortyseven",
    ):
        assert arm in text, f"{arm} is one of the four and must be named"
    assert "ctltinyzero" not in text.split("ARMS=(")[1].split(")")[0]


def test_the_digest_matches_every_prepared_arms_own_sidecar():
    """⛔ The pin is checked against the data, not against the message that supplied it.

    A peer relayed the first 16 characters. That is a claim, and a pinned constant sourced
    from a claim is a constant nobody can check later. These files are the source; if the
    experiment is re-pulled they change and this test says so.

    Skips rather than fails when the sidecars are absent — they are untracked local fetch
    output, so on a clean checkout there is nothing to compare and a failure here would mean
    "I could not look", which is the thing ADR-025 forbids reporting as a verdict either way.
    """
    sidecars = sorted(
        REPO_ROOT.glob("models/*zero*/data/raw/calibration_datafactory_df.parquet.provenance.json")
    )
    if not sidecars:
        pytest.skip("no prepared arm sidecars in this checkout (untracked fetch output)")
    seen = {}
    for path in sidecars:
        prov = json.loads(path.read_text())
        seen[path.parts[-4]] = (
            prov["queryset_digest"], prov["level"], prov["month_first"], prov["month_last"]
        )
    assert len(set(seen.values())) == 1, f"the prepared arms disagree on the pull: {seen}"
    assert next(iter(seen.values())) == (DIGEST, "pgm", 121, 504), seen


# ── the happy path, at the real dimensions ────────────────────────────────


def test_a_clean_run_reports_ok_and_leaves_a_verifiable_gate(tmp_path):
    """⛔ WITHOUT THIS TEST, A SCRIPT THAT ALWAYS REFUSED WOULD PASS EVERY OTHER TEST HERE.

    Fourteen tests above assert that it refuses. None of them asserts that it can succeed, and
    "every refusal fires" is not evidence of a working gate — it is the same shape as a check
    that cannot see its subject and reports a verdict anyway (ADR-025, and the eight instances
    the register holds of it).

    So this one runs at the PRODUCTION dimensions — 64,818 land cells, 36 months — because the
    script passes `slice_gate_posterior.py` no overrides and the defaults are what the pod
    uses. A fixture-sized posterior would be refused by the slicer, and the only way to make
    this pass with a small fixture would be to let the script weaken the slicer's bounds,
    which is the guarantee itself.

    One origin, one gate target, 1 draw: 9.3 MB of float32, which is the smallest array that
    is still the real shape.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, gate=False)
    source = (
        repo / "models" / ARM / "data" / "generated" / "predictions_calibration_20261102_000000"
    )
    shutil.rmtree(source)
    target = source / "origin_0" / "by_sb_best"
    target.mkdir(parents=True)
    rows, steps = 64_818, 36
    rng = np.random.default_rng(0)
    np.save(target / "y_pred.npy", rng.random((rows * steps, 1), dtype=np.float32))
    np.savez(
        target / "identifiers.npz",
        time=np.repeat(np.arange(457, 457 + steps, dtype=np.int32), rows),
        unit=np.tile(np.arange(1, rows + 1, dtype=np.int32), steps),
    )

    result = _run(repo, root, runner, netrc)
    transcript = _transcript(result)
    assert result.returncode == 0, transcript
    assert (out / "EXP02_STATUS").read_text().strip() == "OK"

    # Both endpoints, and the far one is h36 rather than whatever the last month happened
    # to be — the #686 correctness point, reached through the real command.
    manifest = (out / "EXP02_MANIFEST").read_text()
    assert "h1 and h36" in manifest
    assert DIGEST in manifest
    # ⛔ The manifest's gate line is written by the gate, not inferred from which branch
    # ran. The stub exits inside the first poll, so the sidecar is read AFTER the runner is
    # gone — and the manifest must say the pull was verified, because it was. An earlier
    # version printed "runner-exited-first" here, which described the path taken rather than
    # the comparison made, and read as "the gate never got to the pull".
    assert "digest gate:     verified after the runner exited" in manifest

    # The digest the operator verifies the transfer against must exist and be right, because
    # the check at the other end is `sha256sum -c` and nothing else re-derives it.
    import hashlib
    archive = out / "gate_endpoints.tar.zst"
    assert archive.is_file()
    recorded = (out / "gate_endpoints.sha256").read_text().split()[0]
    assert recorded == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert recorded in manifest


def test_the_script_hands_the_slicer_no_bound_of_its_own(tmp_path):
    """The production bounds must come from the slicer, not be re-declared here.

    If this script passed `--expected-rows` or `--steps`, there would be two places to change
    the land-cell count and the window, and the one that is wrong would be the one in effect.
    `slice_gate_posterior.py` owns them and `test_the_production_defaults_are_what_the_pod_needs`
    pins them there.
    """
    text = SCRIPT.read_text()
    slice_call = [line for line in text.splitlines() if "$SLICER" in line and "python3" in line]
    assert slice_call, "the slicer must actually be invoked"
    assert not any("--expected-rows" in line or "--steps" in line for line in slice_call), (
        f"the script re-declares a slicer bound: {slice_call}"
    )


def test_it_refuses_to_certify_an_arm_whose_pull_was_never_compared(tmp_path):
    """⛔ The hole the happy-path test exposed: every refusal skipped, so nothing refused.

    If the runner exits before writing a sidecar and none ever appears, the watcher's timeout
    never fires (the runner is already gone) and the post-wait check finds no file. Before
    2026-11-02 the script then sliced the gate and wrote `EXP02_STATUS: OK` with NOTHING
    having compared the pull against the pin.

    "I could not look" must not render as "the pull is fine" (ADR-025). The guard is on
    `GATE_RESULT` — the variable the comparison itself sets — rather than on which branch was
    taken, because the branches are what was wrong.
    """
    repo, root, out, runner, netrc = _pod(tmp_path, sidecar=None, gate=False)
    source = (
        repo / "models" / ARM / "data" / "generated" / "predictions_calibration_20261102_000000"
    )
    shutil.rmtree(source)
    target = source / "origin_0" / "by_sb_best"
    target.mkdir(parents=True)
    rows, steps = 64_818, 36
    rng = np.random.default_rng(0)
    np.save(target / "y_pred.npy", rng.random((rows * steps, 1), dtype=np.float32))
    np.savez(
        target / "identifiers.npz",
        time=np.repeat(np.arange(457, 457 + steps, dtype=np.int32), rows),
        unit=np.tile(np.arange(1, rows + 1, dtype=np.int32), steps),
    )

    result = _run(repo, root, runner, netrc)
    assert result.returncode == 1, _transcript(result)
    assert (out / "EXP02_STATUS").read_text().strip() == "FAILED:digest_gate"
    assert "nothing ever compared" in _transcript(result)
    assert not (out / "EXP02_MANIFEST").exists(), (
        "a manifest for an arm whose pull is unknown is a manifest that certifies nothing"
    )
