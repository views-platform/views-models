"""Guards on pod_run_model.sh's artefact collection (views-models#549).

THESE TESTS EXECUTE THE SHELL. They do not grep the script for a word.

A source-level assertion that `pod_run_model.sh` mentions ".pt" would have passed
against every version of this script that loses the artefact, including the one
that lost eight models on 2026-09-28. So the collection block is extracted from
the real script, run against fabricated artefact directories with `stage` and
`die` stubbed, and judged on what it leaves on disk.

WHAT THESE CANNOT DO. The block is located by its `stage collect_artifact`
marker and run in isolation, so they prove the block behaves; they do not prove
it is still reached on both legs of the script, and they cannot run the pod. The
reachability of the stage is asserted separately below, on position rather than
on presence.
"""

import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "podrun" / "pod_run_model.sh"

STAGE_MARKER = "stage collect_artifact"
BLOCK_END = "these weights are NOT fit to measure"

def _block() -> str:
    """The collection block, lifted out of the real script.

    Extraction failing IS a test failure: it means the block was renamed,
    removed, or its terminating branch was restructured.
    """
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == STAGE_MARKER]
    assert len(starts) == 1, f"expected exactly one {STAGE_MARKER!r}, found {len(starts)}"
    ends = [i for i, line in enumerate(lines) if BLOCK_END in line]
    assert len(ends) == 1, "the rehearsal branch's closing echo moved or multiplied"
    assert ends[0] > starts[0], "the INCOMPLETE branch precedes the stage it belongs to"
    return "\n".join(lines[starts[0] : ends[0] + 2])

def _planted_contents(name: str) -> str:
    """Fixture bytes that differ per file, so reading the wrong one is detectable."""
    if name.endswith(".sha256"):
        return "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0\n"
    if name.endswith(".config.json"):
        return '{"model": "HydraBNUNet06_LSTM4"}\n'
    return "not really a tensor\n"


def _commit_something(repo: Path) -> str:
    """Give the fixture repo a HEAD, and return its short sha."""
    (repo / "README").write_text("fixture\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ.get("PATH", ""), "HOME": str(repo),
    }
    subprocess.run(["git", "-C", str(repo), "add", "README"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"],
                   check=True, env=env)
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True, env=env,
    ).stdout.strip()


def _run(
    tmp_path: Path,
    artefacts: dict[str, int | None],
    *,
    rehearsal: str = "",
    forecast: str = "0",
) -> subprocess.CompletedProcess:
    """Run the block with `artefacts` planted. Values are mtime offsets from START."""
    root = tmp_path / "workspace"
    model = "violet_visitor"
    adir = root / "views-models" / "models" / model / "artifacts"
    out = root / "deliver" / model
    adir.mkdir(parents=True)
    out.mkdir(parents=True)
    (out / ".lessons").write_text("300" if not rehearsal else rehearsal, encoding="utf-8")
    # The block runs `git -C $REPO rev-parse --short HEAD`, and that needs an actual
    # COMMIT. An empty repo returns nothing, the script falls back to "unknown", and a
    # PROVENANCE assertion against the fallback cannot tell a real sha from a hardcoded
    # one — which is how `git: unknown` survived mutation once already.
    repo = root / "views-models"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit_something(repo)

    start = int(time.time())
    for name, offset in artefacts.items():
        f = adir / name
        # DISTINCT per file. Identical bytes everywhere meant PROVENANCE could read its
        # sha256 from `.config.json` and still print the right string.
        f.write_text(_planted_contents(name), encoding="utf-8")
        if offset is not None:
            os.utime(f, (start + offset, start + offset))

    # A previous attempt's output, planted in the directory the operator rsyncs. Every
    # fixture used to start from an empty artifact/, so `rm -rf "$OUT/artifact"` could be
    # deleted with the suite green — leaving a stale .pt, its sidecars and a stale
    # INCOMPLETE beside a PROVENANCE that describes only one of them.
    stale_dir = out / "artifact"
    stale_dir.mkdir()
    (stale_dir / "calibration_model_19990101_000000.pt").write_bytes(b"a previous attempt")
    (stale_dir / "INCOMPLETE").write_text("from a previous attempt", encoding="utf-8")

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        f'ROOT="{root}"\nREPO=$ROOT/views-models\nMODEL="{model}"\nOUT=$ROOT/deliver/$MODEL\n'
        f"START={start}\n"
        f'FORECAST="{forecast}"\nREHEARSAL_LESSONS="{rehearsal}"\n'
        'stage() { echo "STAGE:$1"; }\n'
        'die()   { echo "DIE:$1"; exit 1; }\n' + _block() + "\n",
        encoding="utf-8",
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )
    adir_out = out / "artifact"
    proc.collected = sorted(p.name for p in adir_out.iterdir()) if adir_out.is_dir() else []
    prov = adir_out / "PROVENANCE"
    proc.provenance = prov.read_text(encoding="utf-8") if prov.is_file() else ""
    return proc

PT = "calibration_model_20261012_220000.pt"

def test_the_state_dict_and_both_sidecars_are_collected():
    """The .pt alone is not loadable by our own code, so all three come home."""
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Exact, not a subset: anything extra arriving in the directory the operator rsyncs is a
    # thing the consumer has to interpret, so a new file here is a deliberate decision.
    assert proc.collected == [
        "PROVENANCE",
        PT,
        f"{PT}.config.json",
        f"{PT}.sha256",
    ], proc.collected

def test_a_training_run_that_produced_no_artefact_fails_loudly():
    """The 2026-09-28 defect: eight models trained, STATUS said OK, weights gone.

    An empty artefacts directory after a `-t` run means training produced
    nothing, and the run must not be allowed to report success.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {})
    assert proc.returncode == 1, "a run with no artefact exited 0"
    assert "DIE:no .pt in models/violet_visitor/artifacts" in proc.stdout

def test_a_previous_runs_artefact_is_not_passed_off_as_this_runs():
    """The source is the repo tree, which this script never clears.

    A second run for the same model on the same pod sees the first run's .pt.
    Collecting it would hand back weights whose MANIFEST describes a different
    training run — the stale-artefact failure the parquet and draws stages
    already guard with `rm -rf`.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: -3600})
    assert proc.returncode == 1, "a stale artefact was collected as this run's"
    assert "DIE:newest artefact" in proc.stdout and "predates this run" in proc.stdout

def test_a_missing_config_sidecar_is_recorded_on_disk_not_only_warned():
    """Degraded, not fatal — but not silent either.

    The weights are worth keeping without the sidecar; refusing would throw a
    trained model away over metadata. A warning inside a 4 MB run.log is not a
    signal, so the condition is written as a file, the way REHEARSAL is.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.sha256": 0})
        assert proc.returncode == 0, "a missing sidecar discarded the weights"
        assert PT in proc.collected, "the weights were not collected"
        assert "INCOMPLETE" in proc.collected, "nothing on disk records the gap"

def test_a_missing_sha256_alone_is_not_flagged_as_incomplete():
    """Only the config sidecar is load-bearing.

    model_artifact_fetcher.py warns and loads on a missing .sha256 ("legacy
    artifact"); it cannot reconstruct the model at all without the config. So
    the marker tracks the one that stops the artefact being usable.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0})
    assert proc.returncode == 0
    assert "INCOMPLETE" not in proc.collected, "a missing .sha256 was treated as fatal metadata"

SECTION_START = "stage train"
SECTION_END = 'echo "train + inference took'


def _run_section() -> str:
    """The script from `stage train` through the end of inference, lifted out verbatim.

    Wider than `_block()` on purpose. The question these tests answer is not what the
    collection block does — that is covered above — but whether it is REACHED, between
    the right two things, on BOTH legs. That is a question about control flow, so the
    control flow has to run: an `if [ "$FORECAST" != "1" ]` wrapper around the collection
    block moves no string offset, and a guard built on `src.index(a) < src.index(b)`
    cannot see it.
    """
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == SECTION_START]
    assert len(starts) == 1, f"expected exactly one {SECTION_START!r}, found {len(starts)}"
    ends = [i for i, line in enumerate(lines) if SECTION_END in line]
    assert len(ends) == 1, "the inference section's closing timing line moved or multiplied"
    assert ends[0] > starts[0], "inference precedes training"
    return "\n".join(lines[starts[0] : ends[0] + 1])


def _trace_section(tmp_path: Path, *, forecast: str = "0", rehearsal: str = "") -> list[str]:
    """Run the section with main.py stubbed, and return what happened, in order.

    The stub stands in for `$VENV/bin/python`. On a `-t` call it creates the artefact
    triple in the pod's repo tree, exactly where train_model.py puts it. On every call it
    records its own flags AND whether the collected artefact is already sitting in
    `$OUT/artifact` at that moment — which is the property under test, stated as the
    script's own invariant rather than as a position in a file.
    """
    root = tmp_path / "workspace"
    model = "violet_visitor"
    repo = root / "views-models"
    adir = repo / "models" / model / "artifacts"
    out = root / "deliver" / model
    adir.mkdir(parents=True)
    out.mkdir(parents=True)
    (out / ".lessons").write_text(rehearsal or "300", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    trace = tmp_path / "trace"
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    stub = venv_bin / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'TRACE="{trace}"\nOUT="{out}"\nARTEFACTS="{adir}"\n'
        # What the collection stage is for: at the moment inference starts, the weights
        # must already be out of the repo tree and in the directory that comes home.
        'if ls "$OUT"/artifact/*.pt >/dev/null 2>&1; then P=collected; else P=absent; fi\n'
        'echo "python $* | artefact_in_OUT=$P" >> "$TRACE"\n'
        # Mimic train_model.py: the state dict and its two sidecars, in the repo tree.
        'for a in "$@"; do if [ "$a" = "-t" ]; then\n'
        '  PT=$ARTEFACTS/calibration_model_20261012_220000.pt\n'
        '  echo weights > "$PT"; echo "{}" > "$PT.config.json"; echo deadbeef > "$PT.sha256"\n'
        'fi; done\n',
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    harness = tmp_path / "section.sh"
    harness.write_text(
        "set -uo pipefail\n"
        f'ROOT="{root}"\nREPO=$ROOT/views-models\nMODEL="{model}"\nOUT=$ROOT/deliver/$MODEL\n'
        f'VENV="{tmp_path / "venv"}"\nFORECAST="{forecast}"\nREHEARSAL_LESSONS="{rehearsal}"\n'
        'stage() { echo "stage $1" >> ' + f'"{trace}"' + '; }\n'
        'die()   { echo "die $1" >> ' + f'"{trace}"' + '; exit 1; }\n' + _run_section() + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=120
    )
    recorded = trace.read_text(encoding="utf-8").splitlines() if trace.is_file() else []
    assert proc.returncode == 0, "\n".join(recorded) + "\n" + proc.stdout + proc.stderr
    return recorded


def test_the_calibration_leg_collects_between_training_and_evaluation():
    """Executed control flow, not a string offset — and this is the regression.

    The previous version of this guard asserted only that `stage collect_artifact`
    appeared before `stage collapse` in the file. That was TRUE of the script that ran
    `main.py -t -e` as one invocation and collected afterwards, which left the weights
    unprotected for the 54 minutes between `.pt` written at 03:31:12 and collection at
    04:25:10 on the 2026-10-12 run. The guard passed on the defect it was written for.

    What is asserted now is the property itself: when the invocation that runs inference
    starts, the artefact is already in the directory that comes home.
    """
    with tempfile.TemporaryDirectory() as td:
        trace = _trace_section(Path(td))

    assert "stage train" in trace, trace
    assert "stage collect_artifact" in trace, trace
    assert "stage evaluate" in trace, trace
    assert trace.index("stage train") < trace.index("stage collect_artifact") \
        < trace.index("stage evaluate"), trace

    calls = [line for line in trace if line.startswith("python ")]
    assert len(calls) == 2, f"expected a training call and an inference call, got {calls}"
    training, inference = calls
    assert " -t" in training and "artefact_in_OUT=absent" in training, training
    assert "artefact_in_OUT=collected" in inference, (
        "inference began with no artefact in $OUT/artifact — a crash in it destroys the "
        f"trained model, which is views-models#549\n  {inference}"
    )


def test_the_forecast_leg_collects_too():
    """The leg the position guard claimed to cover and could not.

    Wrapping the collection block in `if [ "$FORECAST" != "1" ]` moves no string offset,
    so the old assertion `collect < src.index('exit 0')` stayed green while the forecast
    leg trained a model and never collected it — the #549 loss, on one of two legs,
    silently. Both legs train, so both must collect.
    """
    with tempfile.TemporaryDirectory() as td:
        trace = _trace_section(Path(td), forecast="1")

    assert "stage collect_artifact" in trace, (
        f"the forecast leg never reached the collection stage\n  {trace}"
    )
    assert trace.index("stage train") < trace.index("stage collect_artifact") \
        < trace.index("stage forecast"), trace

    calls = [line for line in trace if line.startswith("python ")]
    assert len(calls) == 2, f"expected a training call and a forecast call, got {calls}"
    assert "artefact_in_OUT=collected" in calls[1], (
        f"the forecast leg began forecasting with no artefact collected\n  {calls[1]}"
    )


def test_training_and_inference_are_never_one_invocation():
    """The flag combination the split exists to prevent.

    A combined invocation puts inference INSIDE the training call, so no amount of
    reordering stages can protect the artefact. Checked on the invocation lines; the
    prose above them has to be free to name `-t -e` as the thing that was wrong.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    invocations = [
        ln.strip()
        for ln in src.splitlines()
        if "main.py" in ln and not ln.lstrip().startswith("#")
    ]
    assert invocations, "no main.py invocation found — the run is launched some other way"
    for line in invocations:
        flags = set(line.split())
        assert not ({"-t"} & flags) or not ({"-e", "-f"} & flags), (
            f"training and inference are one invocation again, so the collection stage "
            f"cannot sit between them\n  {line}"
        )


def test_inference_reads_the_dataframe_training_fetched_rather_than_a_fresh_one():
    """`--saved` is a correctness flag here, not a speed one.

    Splitting the run means the second invocation fetches its own data unless told not
    to. Evaluating a different viewser snapshot than was trained on would be a silent
    mismatch: both halves succeed, the parquets are well-formed, and nothing records that
    the model never saw this data. pipeline-core also refuses a non-training run without
    it (`cli/args.py:414-417`), so dropping the flag fails loudly — but AFTER training,
    which is an hour of GPU time to learn it.

    Both spellings are accepted. The first version of this guard required the literal
    `-sa`, which would have failed on the long form and on `-r "$RUN_TYPE"` — correct
    edits both. A guard that fires on a correct change gets edited out of the way.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    inference = [
        ln.strip()
        for ln in src.splitlines()
        if "main.py" in ln
        and not ln.lstrip().startswith("#")
        and ({"-e", "-f"} & set(ln.split()))
    ]
    assert len(inference) == 2, f"expected one evaluation and one forecast call, got {inference}"
    for line in inference:
        flags = set(line.split())
        assert flags & {"-sa", "--saved"}, (
            f"an inference invocation runs without --saved: it will fetch a fresh snapshot "
            f"and evaluate a model on data it was not trained on\n  {line}"
        )


def _startup_clear_block() -> str:
    """The lines that clear the previous attempt's markers, from the real script."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == 'rm -f "$OUT/STATUS"')
    end = next(i for i, ln in enumerate(lines) if ln.strip() == 'rm -f "$OUT/FAILURE"')
    assert end > start, "FAILURE is cleared before STATUS — the block was restructured"
    return "\n".join(lines[start : end + 1])


def test_every_marker_from_a_previous_attempt_is_cleared_before_the_run_starts():
    """A stale FAILURE beside a fresh `STATUS: OK` is the state we shipped in.

    STATUS is overwritten by the next `die`, so a stale one is at least replaced
    on the next failure. FAILURE is written only BY a `die`, so it survives a
    successful rerun — and the 2026-10-12 vehicle came home with an appwrite
    import error in FAILURE sitting next to a clean STATUS, with nothing on disk
    to say which run the artefact beside them belonged to.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "deliver" / "violet_visitor"
        out.mkdir(parents=True)
        stale = ("STATUS", "FAILURE", "REHEARSAL", ".lessons")
        for name in stale:
            (out / name).write_text("from the previous attempt", encoding="utf-8")

        harness = Path(td) / "clear.sh"
        harness.write_text(
            f'set -uo pipefail\nOUT="{out}"\n' + _startup_clear_block() + "\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        survivors = [name for name in stale if (out / name).exists()]
    assert survivors == [], f"a previous attempt's markers survived into this run: {survivors}"


def _venv_guard_line() -> str:
    """The refusal that keeps a hand-built venv from being reused."""
    src = SCRIPT.read_text(encoding="utf-8")
    marker = '[ -f "$VENV/.built-by-pod-run-model" ]'
    assert src.count(marker) == 1, "the venv-provenance guard moved or multiplied"
    i = src.index(marker)
    return src[i : src.index("\n", src.index("die", i))]


def test_a_venv_this_script_did_not_build_is_refused_rather_than_reused():
    """The failure that cost the 2026-10-12 launch, guarded by execution.

    The install block is ordered — the appwrite extra before the toolz override,
    because reversing them silently downgrades toolz below 0.12.1 and breaks
    every datafactory fetch (C-151). The order is not recoverable by inspecting
    the result, so "a venv exists" is not evidence that it is the right venv. A
    venv built from the runbook's own Step 2.2 was reused, the install block was
    skipped, and verify_env then failed on an import that block provides.
    """
    guard = _venv_guard_line()
    with tempfile.TemporaryDirectory() as td:
        venv = Path(td) / "venv"
        venv.mkdir()

        def run() -> subprocess.CompletedProcess:
            harness = Path(td) / "guard.sh"
            harness.write_text(
                f'set -uo pipefail\nVENV="{venv}"\n'
                'die() { echo "DIE:$1"; exit 1; }\n' + guard + "\n",
                encoding="utf-8",
            )
            return subprocess.run(
                ["bash", str(harness)], capture_output=True, text=True, timeout=60
            )

        unmarked = run()
        (venv / ".built-by-pod-run-model").touch()
        marked = run()

    assert unmarked.returncode == 1, "a venv built by hand was accepted"
    assert "was not built by this script" in unmarked.stdout, unmarked.stdout
    # The inverse, which is what makes the guard a guard rather than a refusal to run.
    assert marked.returncode == 0, marked.stdout + marked.stderr


def test_the_venv_refusal_sits_where_it_can_actually_fire():
    """Behaviour is not enough for this one: placement IS the guarantee.

    The line is extracted and run in isolation above, which proves what it does and
    nothing about whether it is reached. Two placements leave it correct and useless:
    moved into the branch that BUILDS the venv, it fires on every fresh pod (the marker
    does not exist yet) and never on the reuse path it exists for; moved to the end of
    the script, a hand-built venv is reused through training and inference and the
    refusal arrives after the money is spent. Both left the extracted-line test green.

    So the refusal must sit inside the `else` of the venv-exists branch, and before
    anything is verified or trained.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    marker = '[ -f "$VENV/.built-by-pod-run-model" ]'
    assert src.count(marker) == 1, "the venv-provenance guard moved or multiplied"
    refusal = src.index(marker)

    reuse_branch = src.index('echo "venv already present')
    install_block = src.index('touch "$VENV/.built-by-pod-run-model"')
    assert install_block < refusal < reuse_branch, (
        "the venv refusal is not in the reuse branch: it must follow the install block's "
        "marker write and precede the 'reusing' message, or it fires on fresh pods and "
        "not on the path it exists for"
    )
    assert refusal < src.index("stage verify_env"), (
        "the venv refusal comes after verify_env; a hand-built venv is already in use "
        "by then, and after the run it is after the money is spent"
    )
    # Reached unconditionally within that branch, not nested behind a further condition.
    line = next(ln for ln in src.splitlines() if marker in ln)
    assert line.startswith("  ["), (
        f"the venv refusal is nested deeper than the reuse branch itself\n  {line!r}"
    )


def _appwrite_check() -> str:
    """The Appwrite import check and its fallback, lifted out of the real script."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [
        i for i, ln in enumerate(lines)
        if "Appwrite SDK is not importable" in ln and "<<'PY'" in ln
    ]
    assert len(starts) == 1, "the Appwrite check's opening line moved or multiplied"
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "PY" and i > starts[0]]
    assert ends, "the Appwrite heredoc is unterminated"
    return "\n".join(lines[starts[0] : ends[0] + 1])


def test_the_publish_path_check_cannot_stop_a_run_that_publishes_nothing():
    """Advisory here, fatal where a publish follows — and EXECUTED, not spell-checked.

    The header promises at :8 that this script uploads nothing and no stage in it reaches
    a datastore. The import check is still worth running — missing it is how the first FAO
    delivery failed after hours of training (#517) — but making it fatal means a
    calibration run that needs no credentials refuses to start over a capability it never
    uses, which cost a launch on 2026-10-12.

    The first version of this guard grepped the line for the substring "die". That made it
    a spell-checker for one failure idiom: `|| { echo "..."; exit 1; }` restored the
    fatality with the guard green. So the check is run against a python that fails the
    import, and judged on its exit status.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        stub = venv_bin / "python"
        # Stands in for a venv without the appwrite extra: every import fails.
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'echo "ModuleNotFoundError: No module named \'appwrite\'" >&2\n'
            "cat > /dev/null\nexit 1\n",
            encoding="utf-8",
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

        harness = tmp_path / "appwrite.sh"
        harness.write_text(
            "set -uo pipefail\n"
            f'VENV="{tmp_path / "venv"}"\n'
            'die() { echo "DIE:$1"; exit 1; }\n'
            + _appwrite_check()
            + '\necho "REACHED THE NEXT STAGE"\n',
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, timeout=60
        )

    assert proc.returncode == 0, (
        "a missing Appwrite SDK stopped a run that publishes nothing\n"
        + proc.stdout + proc.stderr
    )
    assert "REACHED THE NEXT STAGE" in proc.stdout, (
        "the Appwrite check exited the script rather than continuing\n"
        + proc.stdout + proc.stderr
    )
    assert "NOTE" in proc.stdout, "the missing SDK was not reported at all"


def test_the_environment_verification_itself_stays_fatal():
    """The inverse of the test above, or it would pass on a script with no checks left.

    Demoting the Appwrite import is a statement about one capability this script does not
    use. It is not a statement about CUDA, torch, tlz or the pipeline packages, whose
    absence means the run cannot work at all.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    verify = src[src.index("stage verify_env") :]
    opening = verify.splitlines()[1]
    assert "<<'PY'" in opening and "|| die" in opening, (
        f"verify_env's own failure is no longer fatal\n  {opening}"
    )

def test_the_header_contract_lists_the_artefact():
    """The header block is what an operator reads to know what comes home."""
    src = SCRIPT.read_text(encoding="utf-8")
    header = src[: src.index("set -uo pipefail")]
    assert "/workspace/deliver/<model>/artifact/" in header, (
        "the script collects the artefact without promising it"
    )


def test_the_artefact_carries_its_own_lesson_count_and_commit():
    """artifact/ is what the operator rsyncs, so it must describe itself.

    The config sidecar records architecture and nothing else — no lessons, no
    region, no commit. Separated from $OUT, a 5-lesson artefact and a 300-lesson
    one are byte-indistinguishable, which is the hazard the script names at :38
    and then guards in $OUT/REHEARSAL, a file that does not travel.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PROVENANCE" in proc.collected
        prov = proc.provenance
    assert "lessons:     300" in prov, prov
    assert "mode:        production" in prov, prov
    assert "region:      land" in prov, prov


def test_a_rehearsal_artefact_says_so_in_the_directory_that_travels():
    """The one that matters: a rehearsal must not arrive looking real.

    $OUT/REHEARSAL is written at the END of the run — after this stage — and
    lives outside artifact/, so copying it here would have copied nothing. The
    marker is generated instead, from $REHEARSAL_LESSONS, which is known here.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(
            Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0}, rehearsal="5"
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "REHEARSAL" in proc.collected, "a rehearsal artefact travels unmarked"
        prov = proc.provenance
        marker = (
            Path(td) / "workspace" / "deliver" / "violet_visitor" / "artifact" / "REHEARSAL"
        ).read_text(encoding="utf-8")
    assert "mode:        REHEARSAL -- NOT FIT FOR MEASUREMENT OR DELIVERY" in prov, prov
    assert "lessons:     5" in prov, prov
    assert "DO NOT MEASURE OR DELIVER IT" in marker


def test_a_stale_artefact_directory_is_cleared_rather_than_added_to():
    """`rm -rf "$OUT/artifact"` is load-bearing, and nothing used to assert it.

    A second run for the same model on the same pod inherits the previous attempt's
    output in the directory the operator rsyncs. The exact-contents assertion in
    test_the_state_dict_and_both_sidecars_are_collected says it exists so that "a new
    file here is a deliberate decision" — but with every fixture starting from an empty
    directory, no stale file could ever appear for it to catch.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "calibration_model_19990101_000000.pt" not in proc.collected, (
        "a previous attempt's weights came home beside this run's, described by a "
        f"PROVENANCE that names only one of them\n  {proc.collected}"
    )
    assert "INCOMPLETE" not in proc.collected, (
        "a previous attempt's INCOMPLETE marker survived into a complete collection"
    )


def test_the_newest_artefact_is_the_one_collected():
    """Which of two artefacts, not whether there is one.

    The script picks with `ls -1t | head -1`. No fixture planted two `.pt` files, so
    `head -1` could become `tail -1` with the suite green: the mtime check then refuses
    the run with "newest artefact ... predates this run", which is both a failure where
    there should be none AND a message that sends the operator looking for a missing
    artefact that is in fact sitting right there.
    """
    older = "calibration_model_20261012_100000.pt"
    with tempfile.TemporaryDirectory() as td:
        proc = _run(
            Path(td),
            {
                older: 10,
                f"{older}.config.json": 10,
                f"{older}.sha256": 10,
                PT: 60,
                f"{PT}.config.json": 60,
                f"{PT}.sha256": 60,
            },
        )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert PT in proc.collected, f"the newest artefact was not collected: {proc.collected}"
    assert older not in proc.collected, (
        f"the older of two artefacts was collected as this run's: {proc.collected}"
    )


def test_provenance_records_the_artefacts_own_checksum_and_commit():
    """The two fields the guard above is named for and never checked.

    `sha256:` can be read from the wrong sidecar and `git:` can be hardcoded to
    "unknown" with the suite green — and `git:` is the one field
    `artifact/INCOMPLETE` depends on: it tells the reader to reconstruct the
    architecture "from the model's own config at the git sha in PROVENANCE".
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
        assert proc.returncode == 0, proc.stdout + proc.stderr
        repo = Path(td) / "workspace" / "views-models"
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        planted = (
            repo / "models" / "violet_visitor" / "artifacts" / f"{PT}.sha256"
        ).read_text(encoding="utf-8").strip()
        prov = proc.provenance

    assert head, "the fixture repo has no HEAD, so the git: assertion below proves nothing"
    assert f"artefact:    {PT}" in prov, prov
    assert f"sha256:      {planted}" in prov, (
        f"PROVENANCE does not carry the artefact's own recorded checksum — it is reading "
        f"the wrong file, or none\n{prov}"
    )
    assert f"git:         {head}" in prov, (
        f"PROVENANCE does not carry the commit it was built at (expected {head}). This is "
        f"the one field artifact/INCOMPLETE depends on: it tells the reader to rebuild the "
        f"architecture from the config at that sha.\n{prov}"
    )


def test_a_production_run_writes_no_rehearsal_marker():
    """The inverse, which is the half that makes the marker mean anything.

    A marker present on every run would be ignored on every run.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
    assert "REHEARSAL" not in proc.collected, "a production artefact was marked a rehearsal"
