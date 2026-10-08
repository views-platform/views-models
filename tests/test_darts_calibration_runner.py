"""The darts pod runner (#534) must refuse before it spends money, not after.

Three techniques, in descending order of how much they prove, and the ordering is the point.
`tests/test_falsification_40_lesson_run_readiness.py` exists because the first version of the
guards for its subject were satisfied by the script's own comments — ten of fourteen of them.
Its conclusion, which this file inherits: **the better the comment, the weaker the guard.**

1. **Execute the script.** `--preflight` with a minimal environment and `PODRUN_ROOT` pointed
   at a temporary directory. The real checks genuinely fail on a dev machine, and that failure
   *is* the assertion. Nothing is mocked.
2. **Execute the embedded config check as a program.** It is lifted out of its heredoc and run
   under the same interface the shell gives it, against fixture model directories — including
   an adversarial one whose config *text* is correct and whose returned value is not.
3. **Comment-stripped text**, via `_code_only()`, only for invariants that text can carry: that
   no credential is installed, that a pin is present, that one command precedes another.

What none of this proves: that a real darts model trains to completion at global pgm. Nothing
can, without #537 and a rented machine.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "podrun" / "pod_run_darts_calibration.sh"

#: Stage names that mean money is being spent. None may appear in a --preflight run.
RUN_STAGES = ("install_system", "install_python", "train_and_evaluate", "collapse", "verify_parquet")

#: The nine publish variables the FAO chain needs. This runner must read none of them.
PUBLISH_VARS = (
    "DATASTORE_API_KEY",
    "DATASTORE_PROJECT_ID",
    "PREDICTION_STORE",
    "APPWRITE",
)


def _code_only(src: str) -> str:
    """Shell source with comments removed.

    Any assertion about what the script DOES must read only what the script RUNS. Full-line
    and trailing comments both go. Deliberately cruder than a shell parser, because cruder is
    the safe direction: it removes text an assertion might otherwise lean on.
    """
    out = []
    for line in src.splitlines():
        stripped = re.sub(r"(?<![$\\])#.*$", "", line)
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


def _extract_cfgcheck() -> str:
    """The config-check program, lifted out of its heredoc so it can be RUN."""
    src = RUNNER.read_text()
    m = re.search(r"<<'CFGCHECK'[^\n]*\n(.*?)\nCFGCHECK\s*?\n", src, re.S)
    assert m, "the CFGCHECK heredoc is gone from pod_run_darts_calibration.sh — re-read it"
    return m.group(1)


HP = """\
def get_hp_config():
    return {{
        "steps": list(range(1, 37)),
        "n_epochs": {epochs},
        "num_samples": {samples},
        "mc_dropout": {dropout},
    }}
"""

META = """\
def get_meta_config():
    return {{
        "name": "fixture",
        "algorithm": "NBEATSModel",
        "level": {level},
        "entity_id": {entity},
        "prediction_format": {fmt},
        "regression_point_metrics": {point_metrics},
    }}
"""


def _make_model(
    tmp_path: Path,
    *,
    epochs: int = 300,
    samples: int = 1,
    dropout: str = "False",
    level: str = '"pgm"',
    entity: str = '"priogrid_id"',
    fmt: str = '"dataframe"',
    point_metrics: str = '["MCR_point", "MSE", "MSLE", "y_hat_bar"]',
    hp_body: str | None = None,
) -> Path:
    model = tmp_path / "model"
    (model / "configs").mkdir(parents=True, exist_ok=True)
    (model / "configs" / "config_hyperparameters.py").write_text(
        hp_body
        if hp_body is not None
        else HP.format(epochs=epochs, samples=samples, dropout=dropout)
    )
    (model / "configs" / "config_meta.py").write_text(
        META.format(level=level, entity=entity, fmt=fmt, point_metrics=point_metrics)
    )
    return model


def _run_cfgcheck(model: Path, out: Path) -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    script = out / "_cfgcheck.py"
    script.write_text(_extract_cfgcheck())
    return subprocess.run(
        [sys.executable, str(script), str(model)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "MODEL": "fixture", "HOME": str(out)},
    )


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Execute the script for real, with a deliberately minimal environment."""
    return subprocess.run(
        ["bash", str(RUNNER), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp", "PODRUN_ROOT": str(tmp_path)},
    )


# ── the script, executed ──────────────────────────────────────────────────────────


def test_it_refuses_rather_than_proceeding(tmp_path):
    r = _run(tmp_path, "--preflight", "dark_river")
    assert r.returncode != 0, f"preflight passed on a machine with nothing set up:\n{r.stdout}"
    assert "preflight failed" in r.stdout


def test_it_reports_every_problem_not_only_the_first(tmp_path):
    """One round trip instead of six. This is the whole reason --preflight accumulates."""
    r = _run(tmp_path, "--preflight", "dark_river")
    assert r.stdout.count("MISSING:") >= 3, f"only reported one problem:\n{r.stdout}"


def test_preflight_never_reaches_a_run_stage(tmp_path):
    """Matched on the STAGE banner `=== [HH:MM:SS] <name> ===`, not on the bare word.

    A substring check fails here for the wrong reason: 'collapse' appears in the path
    `tools/collapse/collapse_darts_predictions.py` that preflight reports as missing. A test
    that passes only because the script happens not to mention a filename is not testing the
    stages.
    """
    r = _run(tmp_path, "--preflight", "dark_river")
    reached = re.findall(r"=== \[\d\d:\d\d:\d\d\] (\S+) ===", r.stdout)
    for stage in RUN_STAGES:
        assert stage not in reached, f"--preflight reached {stage}: stages were {reached}"
    assert "preflight" in reached, f"preflight itself never ran: {reached}"


def test_only_a_finished_run_writes_OK_to_status():
    """STATUS is what an orchestrator reads to decide whether output may be used.

    `pod_run_fao_delivery.sh` already does precisely that with `pod_run_model.sh`'s STATUS, to
    refuse pooling a partial roster. #538 runs ten of these in sequence. A `--preflight` that
    wrote plain `OK` would be indistinguishable on disk from a run that produced 13 verified
    parquets.
    """
    code = _code_only(RUNNER.read_text())
    writes = re.findall(r'echo (\S+) > "\$OUT/STATUS"', code)
    assert writes, "nothing writes STATUS"
    assert writes.count("OK") == 1, f"more than one path writes a bare OK: {writes}"
    assert "PREFLIGHT_OK" in writes, "the preflight path does not distinguish itself"

    # And the single bare OK must come after the deliverable has been verified.
    ok = re.search(r'echo OK > "\$OUT/STATUS"', code)
    verify = re.search(r"stage verify_parquet", code)
    assert verify and ok and verify.start() < ok.start(), "OK is written before verification"


def test_preflight_only_exits_before_anything_is_installed():
    """Structural, and the reason it has to be is worth stating.

    The executing test above cannot reach this branch on a dev machine: preflight dies at the
    missing `/root/.netrc` first, and a test cannot create that file without root. So a
    mutation that deleted the `--preflight` early exit entirely survived
    `test_preflight_never_reaches_a_run_stage` — on a pod where preflight PASSES, that
    mutation would run the full training. This assertion is what catches it.

    What it leaves unproven: that the exit actually fires. Only a machine where preflight
    passes can show that, which is #537.
    """
    code = _code_only(RUNNER.read_text())
    branch = re.search(r'if \[ "\$PREFLIGHT_ONLY" = "1" \]; then(.*?)\nfi', code, re.S)
    assert branch, "the --preflight early exit is gone — a preflight would run the whole job"
    assert "exit 0" in branch.group(1), f"the branch does not exit:\n{branch.group(1)}"
    install = re.search(r"uv venv", code)
    assert install, "the venv is never built"
    assert branch.end() < install.start(), "the --preflight exit comes after the install begins"


def test_it_says_how_to_run_off_a_pod_rather_than_blaming_a_lock(tmp_path):
    """An unchecked mkdir would report 'another run is in progress' for an unwritable root."""
    unwritable = tmp_path / "nope" / "deeper"
    unwritable.parent.mkdir()
    unwritable.parent.chmod(0o500)
    try:
        r = subprocess.run(
            ["bash", str(RUNNER), "--preflight", "dark_river"],
            capture_output=True, text=True, timeout=120,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PODRUN_ROOT": str(unwritable)},
        )
        assert r.returncode != 0
        assert "PODRUN_ROOT" in r.stdout + r.stderr
        assert "in progress" not in r.stdout + r.stderr
    finally:
        unwritable.parent.chmod(0o700)


@pytest.mark.parametrize(
    "args",
    [
        ([]),
        (["--preflight"]),
        (["--nonsense", "dark_river"]),
        (["dark_river", "blue_ocean"]),
    ],
)
def test_bad_invocations_are_refused_before_anything_is_touched(tmp_path, args):
    r = _run(tmp_path, *args)
    assert r.returncode != 0, f"accepted {args}"
    assert not (tmp_path / "deliver").exists(), f"{args} created state before refusing"


# ── the config gate, executed as a program ────────────────────────────────────────


def test_a_well_formed_model_passes_the_gate(tmp_path):
    r = _run_cfgcheck(_make_model(tmp_path), tmp_path / "out")
    assert r.returncode == 0, f"a correct config was refused:\n{r.stdout}\n{r.stderr}"


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"samples": 100}, "num_samples"),
        ({"dropout": "True"}, "mc_dropout"),
        ({"point_metrics": "[]"}, "regression_point_metrics"),
        ({"epochs": 40}, "n_epochs"),
        ({"fmt": '"prediction_frame"'}, "prediction_format"),
        ({"entity": '"country_id"'}, "entity_id"),
        ({"level": '"cm"'}, "level"),
    ],
)
def test_the_gate_refuses_each_wrong_value_on_its_own(tmp_path, kwargs, needle):
    """Independently, because a guard that passes on two of three is how a half-fix ships."""
    r = _run_cfgcheck(_make_model(tmp_path, **kwargs), tmp_path / "out")
    assert r.returncode != 0, f"{kwargs} was accepted:\n{r.stdout}"
    assert needle in r.stdout, f"the refusal did not name {needle}:\n{r.stdout}"


def test_the_two_sample_models_get_all_three_problems_at_once(tmp_path):
    """little_talks and mister_bluesky are wrong in three ways; naming one wastes a round trip."""
    model = _make_model(tmp_path, samples=100, dropout="True", point_metrics="[]")
    r = _run_cfgcheck(model, tmp_path / "out")
    assert r.returncode != 0
    assert r.stdout.count("MISSING:") == 3, f"expected all three:\n{r.stdout}"


def test_the_refusal_points_at_the_issue_that_decides_it(tmp_path):
    r = _run_cfgcheck(_make_model(tmp_path, samples=100), tmp_path / "out")
    assert "536" in r.stdout, "the operator is not told where the decision lives"


def test_a_config_whose_TEXT_is_right_and_VALUE_is_wrong_is_refused(tmp_path):
    """The adversarial case the load-and-call technique exists for.

    `num_samples` reads 1 in the file. `get_hp_config()` returns 100, because a second
    assignment wins. A guard that grepped the file — which is what #501 shipped once, and what
    this repo keeps rediscovering — would green-light a run needing ~303 GB of RAM.
    """
    sneaky = (
        "def get_hp_config():\n"
        "    d = {'n_epochs': 300, 'num_samples': 1, 'mc_dropout': False}\n"
        "    d['num_samples'] = 100\n"
        "    return d\n"
    )
    r = _run_cfgcheck(_make_model(tmp_path, hp_body=sneaky), tmp_path / "out")
    assert r.returncode != 0, (
        "the file says num_samples=1 and the config RETURNS 100; the gate believed the text.\n"
        f"{r.stdout}"
    )
    assert "num_samples is 100" in r.stdout


def test_the_gate_reads_config_meta_too_not_only_hyperparameters(tmp_path):
    """Both files matter, and a gate that loaded one would pass the other's traps silently."""
    model = _make_model(tmp_path)
    (model / "configs" / "config_meta.py").unlink()
    r = _run_cfgcheck(model, tmp_path / "out")
    assert r.returncode != 0, "a missing config_meta.py was not noticed"


# ── invariants that text can carry ────────────────────────────────────────────────


def test_no_appwrite_extra_is_installed(tmp_path):
    """A calibration run uploads nothing, so it must not carry a credential that could."""
    code = _code_only(RUNNER.read_text())
    assert "appwrite" not in code.lower(), "the runner installs or imports an Appwrite path"


def test_no_publish_variable_is_read(tmp_path):
    code = _code_only(RUNNER.read_text())
    for var in PUBLISH_VARS:
        assert var not in code, f"the runner reads {var}; it has nothing to publish"


def test_the_engine_is_pinned_to_the_git_tag_not_pypi():
    """PyPI's 0.2.3 never frees the prediction scratch dir (views-r2darts2#54)."""
    code = _code_only(RUNNER.read_text())
    assert "git+https://github.com/views-platform/views-r2darts2@0.2.4" in code
    assert "[manager]" in code, "without the extra there is no views-pipeline-core"


def test_the_installed_version_is_asserted_not_assumed():
    """A silent fall back to 0.2.3 does not fail — it fills the disk hours later."""
    code = _code_only(RUNNER.read_text())
    assert re.search(r'version\s*==\s*"0\.2\.4"', code), "the version is never checked"


def test_the_toolz_override_is_an_install_and_is_last():
    """Register C-151. Any install after it can silently pull toolz back under 0.12."""
    code = _code_only(RUNNER.read_text())
    installs = [m.start() for m in re.finditer(r"pip install", code)]
    overrides = [m.start() for m in re.finditer(r"pip install[^\n]*toolz>=0\.12", code)]
    assert overrides, "the toolz override is gone"
    assert max(overrides) == max(installs), "something is installed after the toolz override"


def test_a_real_cuda_kernel_is_launched_not_only_queried():
    """darts pins torch>=2.0.0 with no ceiling; is_available() is true on a bad build (#494)."""
    code = _code_only(RUNNER.read_text())
    assert "torch.cuda.is_available()" in code
    assert 'device="cuda"' in code, "no kernel is ever launched, so a driver mismatch survives"


def test_scratch_is_on_local_disk_and_never_the_network_volume():
    """The inversion of an earlier test, and the reason is a model we lost.

    The first version asserted `export TMPDIR="$ROOT/tmp"` — scratch on the volume — so that
    the existing disk-floor check, which measured `$ROOT`, would be meaningful. The principle
    was right (a floor must measure what the workload writes) and the application was
    backwards: it moved the workload to the filesystem the check already watched.

    `$ROOT` is `/workspace`, a NETWORK filesystem — which is why `chmod` silently does nothing
    there (C-154). The engine writes its Zarr store and prediction memmaps into `TMPDIR`. At 3
    covariates that is ~1 GB and five models completed without anyone noticing. At 71
    covariates it is ~15 GB: `blue_ocean` spent 100 minutes at 0% GPU and 11.6% CPU, blocked
    on I/O, never reached the GPU, and was killed having produced nothing.

    So scratch goes on local disk and the floor follows it there. The deliverable still lands
    under `$ROOT` — that is the volume that survives a pod stop, and only the throwaway
    intermediates move.
    """
    code = _code_only(RUNNER.read_text())
    assert not re.search(r'export TMPDIR="\$ROOT', code), (
        "TMPDIR points back at $ROOT — that is the network volume, and it is what stalled "
        "blue_ocean for 100 minutes"
    )
    m_tmp = re.search(r'export TMPDIR="\$SCRATCH"', code)
    assert m_tmp, "TMPDIR is not set to $SCRATCH"
    m_scr = re.search(r'SCRATCH=\$\{PODRUN_SCRATCH:-(/[^}]+)\}', code)
    assert m_scr, "SCRATCH has no default"
    assert not m_scr.group(1).startswith("/workspace"), (
        f"the scratch default is {m_scr.group(1)}, which is on the network volume"
    )
    m_run = re.search(r"main\.py -r calibration", code)
    assert m_run and m_tmp.start() < m_run.start(), "TMPDIR is set after the run starts"


def test_the_disk_floor_measures_the_scratch_filesystem_not_the_volume():
    """A floor is only worth having if it watches what the work actually fills."""
    code = _code_only(RUNNER.read_text())
    m = re.search(r'AVAIL_GB=\$\(df[^\n]*"\$(\w+)"', code)
    assert m, "the disk floor does not read a df of any named path"
    assert m.group(1) == "SCRATCH", (
        f"the floor measures ${m.group(1)} but the engine writes into $SCRATCH; on this "
        f"platform those are different filesystems and one of them is a network mount"
    )


def test_a_heartbeat_reports_gpu_cpu_and_scratch_during_the_run():
    """Silence was the failure this runner could not explain.

    `blue_ocean` logged nothing for 100 minutes. Distinguishing training from CPU-bound
    conversion from blocked I/O needed an SSH session and /proc, after the fact. These three
    numbers separate them, and a stalled run now says so itself.
    """
    code = _code_only(RUNNER.read_text())
    hb = re.search(r"HEARTBEAT", code)
    assert hb, "no heartbeat — a stalled run is undiagnosable from the log alone"
    window = code[hb.start(): hb.start() + 600]
    for probe, why in (("utilization.gpu", "GPU busy = training"),
                       ("pcpu", "CPU pegged = conversion"),
                       ("TMPDIR", "scratch growing = I/O")):
        assert probe in window, f"the heartbeat omits {probe} ({why})"
    assert "kill $HEARTBEAT_PID" in code, "the heartbeat is never stopped"


def test_the_converter_is_invoked_from_the_repo_root():
    """`python -m tools.collapse...` resolves `tools` from the cwd and nothing is installed.
    The same slip in the FAO script made every tool call raise ModuleNotFoundError while an
    `|| echo` fallback reported that its subject was broken."""
    code = _code_only(RUNNER.read_text())
    m_cd = [m.start() for m in re.finditer(r'cd "\$REPO" \|\|', code)]
    m_mod = re.search(r"-m tools\.collapse\.collapse_darts_predictions", code)
    assert m_mod, "the converter is never invoked"
    assert any(c < m_mod.start() for c in m_cd), "no `cd $REPO` precedes the converter call"


def test_the_venv_build_is_serialised_across_models_on_one_pod():
    """The per-model lock does not cover $VENV, which every model on the pod shares.

    Two models started together on a fresh pod would both enter the install block. Found by
    Simon asking whether the eleven could run in parallel.
    """
    code = _code_only(RUNNER.read_text())
    lock = re.search(r'mkdir "\$VENV_LOCK"', code)
    install = re.search(r"uv venv", code)
    assert lock, "the venv build is not locked"
    assert install and lock.start() < install.start(), "the lock is taken after the build starts"


def test_the_exit_trap_only_frees_a_venv_lock_this_process_holds():
    """An unconditional rmdir in the EXIT trap would free a lock another run depends on."""
    code = _code_only(RUNNER.read_text())
    trap = re.search(r"trap '([^']*)' EXIT", code)
    assert trap, "no EXIT trap"
    body = trap.group(1)
    assert "VENV_LOCK" in body, "the trap never releases the venv lock"
    assert "HELD_VENV_LOCK" in body, (
        "the trap frees $VENV_LOCK unconditionally, so a run that never held it would release "
        f"another run's lock: {body}"
    )


# ── cross-file: the runner and the converter must agree ───────────────────────────


def test_the_runner_and_the_converter_agree_on_the_row_count():
    from tools.collapse.collapse_darts_predictions import EXPECTED_ROWS

    code = RUNNER.read_text()
    assert f"{EXPECTED_ROWS:_}" in code or str(EXPECTED_ROWS) in code, (
        f"the runner's verification does not check {EXPECTED_ROWS} rows, which is what the "
        f"converter promises"
    )


def test_the_runner_and_the_converter_agree_on_the_origin_count():
    from tools.collapse.collapse_darts_predictions import EXPECTED_ORIGINS

    code = _code_only(RUNNER.read_text())
    assert re.search(rf'-eq {EXPECTED_ORIGINS}\b', code), "the runner does not check 13 parquets"


def test_the_converter_the_runner_calls_actually_exists_and_imports():
    """The runner dies at the last stage otherwise, after the whole training run."""
    assert (REPO / "tools" / "collapse" / "collapse_darts_predictions.py").is_file()
    r = subprocess.run(
        [sys.executable, "-m", "tools.collapse.collapse_darts_predictions", "--help"],
        cwd=str(REPO), capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"the converter is not importable from the repo root:\n{r.stderr}"
