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
    # the block runs `git -C $REPO rev-parse`; an empty repo is enough for it not to hang
    subprocess.run(['git', 'init', '-q', str(root / 'views-models')], check=True)

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


def test_a_production_run_writes_no_rehearsal_marker():
    """The inverse, which is the half that makes the marker mean anything.

    A marker present on every run would be ignored on every run.
    """
    with tempfile.TemporaryDirectory() as td:
        proc = _run(Path(td), {PT: 0, f"{PT}.config.json": 0, f"{PT}.sha256": 0})
    assert "REHEARSAL" not in proc.collected, "a production artefact was marked a rehearsal"
