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
BLOCK_END = "config sidecar is missing"

def _block() -> str:
    """The collection block, lifted out of the real script.

    Extraction failing IS a test failure: it means the block was renamed,
    removed, or its terminating branch was restructured.
    """
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == STAGE_MARKER]
    assert len(starts) == 1, f"expected exactly one {STAGE_MARKER!r}, found {len(starts)}"
    ends = [i for i, line in enumerate(lines) if BLOCK_END in line]
    assert len(ends) == 1, "the INCOMPLETE branch's closing echo moved or multiplied"
    assert ends[0] > starts[0], "the INCOMPLETE branch precedes the stage it belongs to"
    return "\n".join(lines[starts[0] : ends[0] + 2])

def _run(tmp_path: Path, artefacts: dict[str, int | None]) -> subprocess.CompletedProcess:
    """Run the block with `artefacts` planted. Values are mtime offsets from START."""
    root = tmp_path / "workspace"
    model = "violet_visitor"
    adir = root / "views-models" / "models" / model / "artifacts"
    out = root / "deliver" / model
    adir.mkdir(parents=True)
    out.mkdir(parents=True)

    start = int(time.time())
    for name, offset in artefacts.items():
        f = adir / name
        f.write_bytes(b"not really a tensor")
        if offset is not None:
            os.utime(f, (start + offset, start + offset))

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -uo pipefail\n"
        f'ROOT="{root}"\nREPO=$ROOT/views-models\nMODEL="{model}"\nOUT=$ROOT/deliver/$MODEL\n'
        f"START={start}\n"
        'stage() { echo "STAGE:$1"; }\n'
        'die()   { echo "DIE:$1"; exit 1; }\n' + _block() + "\n",
        encoding="utf-8",
    )
    harness.chmod(harness.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        ["bash", str(harness)], capture_output=True, text=True, timeout=60
    )
    proc.collected = sorted(p.name for p in out.joinpath("artifact").iterdir()) if (
        out / "artifact"
    ).is_dir() else []
    return proc

PT = "calibration_model_20261012_220000.pt"

def test_the_state_dict_and_both_sidecars_are_collected():
    """The .pt alone is not loadable by our own code, so all three come home."""
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.collected == [PT, f"{PT}.config.json", f"{PT}.sha256"], proc.collected

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

def test_the_collection_runs_before_the_stages_that_can_die_expensively():
    """Position, not presence — and this is the half the executed tests cannot see.

    Training is the cheap-to-survive part. The 25x posterior-sampling collapse
    (postmortem 2.3) and the OOM the memory guard cannot see (2.5) both strike
    during EVALUATION, after the weights exist. Collecting after the collapse
    stage would mean a run that dies in evaluation still loses the artefact it
    had already produced.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    collect = src.index(STAGE_MARKER)
    assert collect < src.index("stage collapse"), "collection moved after the collapse stage"
    assert collect < src.index('rm -rf "$OUT/parquet"'), "collection moved after the parquet stage"
    # Both legs train, so both must collect: the forecast leg exits early, and that exit
    # must come after the collection rather than before it.
    assert collect < src.index('exit 0'), "the forecast leg exits before the artefact is collected"

def test_the_header_contract_lists_the_artefact():
    """The header block is what an operator reads to know what comes home."""
    src = SCRIPT.read_text(encoding="utf-8")
    header = src[: src.index("set -uo pipefail")]
    assert "/workspace/deliver/<model>/artifact/" in header, (
        "the script collects the artefact without promising it"
    )
