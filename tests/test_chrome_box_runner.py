"""`tools/podrun/pod_run_chrome_box.sh` must keep the disciplines it was ported with.

Phase 2 of the CRAF'd chain — the eight `*_sparrow` members and `chrome_box`, which reconciles
against `chrome_crate`. CI cannot run it: that needs a rented 87 GiB pod and a datafactory
credential. What CI can check is that its declared disciplines are still declared, and that
`--preflight` really refuses — the latter is exercised for real, against a staged root, because
a preflight nobody has seen fail is a preflight nobody has seen.

**Every property below is a failure that has actually happened**, which is why they are pinned
rather than trusted:

- a pgm figure used against a cm floor, and the reverse (views-models#648, and a config that
  told an operator to rent 87 GiB for a seven-minute job)
- a venv reused across phases carrying a pipeline-core that cannot reconcile, discovered after
  ~3h05m of paid member training (the 2026-10-28 pod; `crafd_chain.out:210` records 3.3.4)
- a missing partner forecast discovered at reconcile rather than at preflight (`phase2.out`
  printed "Cannot reconcile: no local forecast found" three times, after the members were paid
  for)
- `-t` on the ensemble, which shells out to `models/*/run.sh` and re-enters the conda lane
- a publish credential on rented hardware (RunPod guide rule 5)

Sibling of `tests/test_crafd_chain_runner.py`, deliberately duplicating its fixture and some of
its assertions. The two runners are ~350 and ~450 lines of shell for different phases on
different hardware; a shared test helper would put both on the path of every change to either,
and WET is the cheaper trade at this size.
"""

from pathlib import Path
import re
import subprocess

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "podrun" / "pod_run_chrome_box.sh"
ENSEMBLE = "chrome_box"
PARTNER = "chrome_crate"


@pytest.fixture(scope="module")
def script() -> str:
    assert RUNNER.is_file(), (
        f"{RUNNER} does not exist. Phase 2 was ported out of "
        f"tools/podrun/evidence/2026-10-28_chrome_box/phase2_pgm.sh; if it has been removed, "
        f"this whole file is vacuous and refuses rather than passing (ADR-025)."
    )
    return RUNNER.read_text()


def test_the_runner_exists_and_is_executable() -> None:
    assert RUNNER.is_file(), f"{RUNNER} does not exist"
    mode = RUNNER.stat().st_mode & 0o777
    assert mode & 0o111, f"{RUNNER} is not executable (mode {mode:o}); a pod invokes it directly"


# ── the floors ────────────────────────────────────────────────────────────────────────


def test_the_memory_floor_is_sized_for_the_pgm_peak_and_not_the_cm_one(script: str) -> None:
    """80 GiB, from the measured 68.2 plus headroom. ⛔ Never phase 1's 16.

    `pod_run_crafd_chain.sh` floors at 16 and is right to: `chrome_crate` is country-month,
    96 MB of pool, ~2.0 MB peak RSS per member, 7 minutes. `chrome_box` is pgm global-land and
    peaked at **>= 68.2 GiB** on 2026-10-28. Using either phase's figure for the other is the
    mistake that produced views-models#648 in one direction and, in the other, a config block
    that told an operator to rent 87 GiB for a seven-minute job.

    The floor is above the peak rather than equal to it because headroom is the property: the
    runpod postmortem's 31 GB machine never OOM'd, it collapsed 25x from kernel reclaim and
    ran for eight hours.
    """
    match = re.search(r"^MEM_FLOOR_GB=(\d+)", script, re.MULTILINE)
    assert match, "MEM_FLOOR_GB is not set"
    floor = int(match.group(1))
    assert floor >= 80, (
        f"MEM_FLOOR_GB is {floor}; chrome_box's pool peaked at >= 68.2 GiB MEASURED and that "
        f"figure is a lower bound. A floor at or below the peak admits a box that collapses "
        f"into reclaim after the member training is paid for. 80 admits the 87 GiB class and "
        f"refuses the 57, which sits at 120% of the measured peak."
    )
    assert floor != 16, (
        "MEM_FLOOR_GB is 16, which is phase 1's floor for a country-month pool. This phase is "
        "pgm global-land."
    )
    assert "68.2" in script, (
        "the measured peak no longer appears in the script, so the floor has no stated basis "
        "and the next person to touch it cannot tell a measurement from a guess."
    )


def test_the_disk_floor_is_labelled_derived_because_it_is(script: str) -> None:
    """The one floor here that was never measured must say so.

    `phase2_pgm.sh` died before its manifest stage, so no run has ever reported this pool's
    size on disk. The floor is derived from raw float32 and the script shows the arithmetic.
    Mislabelling it as measured invites over-trust; the first successful run's `pool:` line is
    what replaces it.
    """
    match = re.search(r"^DISK_FLOOR_GB=(\d+)", script, re.MULTILINE)
    assert match, "DISK_FLOOR_GB is not set"
    disk_line = next(
        (ln for ln in script.splitlines() if "DISK_FLOOR_GB}" in ln and "echo" in ln), None
    )
    assert disk_line, (
        "could not find the line that reports the disk floor to the operator, so this guard "
        "cannot see its subject and refuses rather than passing (ADR-025)"
    )
    assert "DERIVED" in disk_line, (
        f"the disk floor is not labelled derived: {disk_line.strip()!r}. No run has measured "
        f"this pool on disk — `phase2_pgm.sh` died before its manifest stage."
    )


# ── the inherited venv, which is the expensive failure ────────────────────────────────


def test_the_resolved_pipeline_core_version_is_asserted_not_assumed(script: str) -> None:
    """This script installs nothing, so the only honest check is what is INSTALLED.

    ⛔ The failure in full, because it is the one that costs three hours. The install set lives
    in `pod_run_crafd_chain.sh`, that script only re-resolves when `$VENV` is absent, and this
    script installs nothing at all. So a surviving workspace carries whatever pipeline-core it
    was built with — and this repository's own evidence records `views-pipeline-core==3.3.4` in
    exactly that venv (`evidence/2026-10-28_chrome_box/crafd_chain.out:210`). Below 3.3.5,
    `load_cm_frame` cannot read a PredictionFrame cm forecast, so the reconcile dies after the
    members are paid for.

    Verifying that a range PERMITS 3.3.5 is not verifying that 3.3.5 is INSTALLED.
    """
    assert "PC_FLOOR=" in script, (
        "the pipeline-core floor is no longer declared as a single variable, so the assert and "
        "the message an operator reads can disagree."
    )
    floor = next(
        (ln.split("=", 1)[1].strip() for ln in script.splitlines() if ln.startswith("PC_FLOOR=")),
        None,
    )
    assert floor and tuple(int(n) for n in floor.split(".")[:3]) >= (3, 3, 5), (
        f"PC_FLOOR is {floor!r}; it must be at least 3.3.5, the release whose `load_cm_frame` "
        f"can read a PredictionFrame cm forecast (views-pipeline-core#562). This phase "
        f"reconciles against one."
    )

    assert 'm.version("views-pipeline-core")' in script and "sys.exit" in script, (
        "the script does not compare the INSTALLED views-pipeline-core version to the floor. "
        "Printing it is not asserting it."
    )
    #: This script must not BUILD or INSTALL anything — phase 1 owns the venv, and two
    #: scripts racing to create one prefix is a failure mode worth not having. (The first
    #: version of this assertion forbade the string `[ ! -d "$VENV" ]` instead, and fired on
    #: the preflight's perfectly correct *check* that the venv exists. Forbidding a syntax is
    #: not the same as forbidding an action.)
    for forbidden in ("uv venv", "uv pip install", "pip install"):
        assert forbidden not in script, (
            f"this script runs {forbidden!r}. Phase 2 must not build or install: phase 1 owns "
            f"the venv. An install here also re-resolves it, which can silently pull toolz "
            f"back under 0.12 (C-151, observed 2026-09-29)."
        )

    #: The version assert must sit in the unconditional body, after the tee, so no branch can
    #: skip it. That is the whole lesson of the 3.3.4 workspace: the install block it would
    #: otherwise live beside only runs when the venv is absent.
    tee_at = script.index("exec > >(tee")
    assert 'm.version("views-pipeline-core")' in script[tee_at:], (
        "the resolved-version assert is not in the unconditional body after the tee. Anywhere "
        "a branch can skip it, it will be skipped on exactly the runs that need it — a "
        "surviving workspace is the case this check exists for."
    )
    assert "import tlz.curried" in script, (
        "the `import tlz.curried` check is gone. C-151's reversion is silent, so it is verified "
        "by import rather than by version string."
    )


# ── the partner forecast: the check this phase exists to have ─────────────────────────


def test_the_partner_forecast_is_required_before_anything_is_trained(script: str) -> None:
    """`chrome_box` reconciles against `chrome_crate`, read from local disk.

    Without it the run pools and then dies at reconcile — after ~3h05m of paid member training.
    `phase2.out` printed "Cannot reconcile: no local forecast found for country model
    'chrome_crate'" three times on 2026-10-28 for exactly this reason.
    """
    assert PARTNER in script, f"the runner no longer mentions {PARTNER} at all"
    assert "PARTNER_GENERATED" in script, (
        f"the preflight no longer checks for {PARTNER}'s forecast on local disk. That check is "
        f"the difference between failing in seconds and failing after three paid hours."
    )
    preflight_end = script.index("preflight OK")
    preflight = script[:preflight_end]
    assert "PARTNER_GENERATED" in preflight, (
        "the partner check has moved out of the preflight. It must run before any member "
        "trains, which is the entire point of it."
    )
    #: Again the glob and not the word: "forecasting" also appears in this check's own error
    #: message, so a word search passes on the prose that describes the check while the check
    #: itself looks for something else.
    assert re.search(r"find .*-name 'predictions_forecasting_\*'", preflight), (
        "the partner check no longer looks for `predictions_forecasting_*`. A calibration-only "
        "data/generated, or a `forecasting_log.txt` from an interrupted phase 1, would pass "
        "the preflight and the reconcile would still fail — after the members are paid for."
    )

    #: And it must read phase 1's own verdict. A PredictionFrame directory is created before
    #: it is filled, so a phase 1 killed mid-save leaves a matching directory and nothing
    #: else. An earlier version of this check TOLD the operator "its STATUS must read OK" and
    #: never read it — a message describing a check that was not performed, which is the
    #: defect class this whole branch keeps closing.
    assert "PARTNER_STATUS" in preflight and "deliver/$PARTNER/STATUS" in preflight.replace(
        '$ROOT/deliver/$PARTNER/STATUS', 'deliver/$PARTNER/STATUS'
    ), (
        "the preflight does not read phase 1's STATUS. Phase 1 writes its verdict to "
        "$ROOT/deliver/chrome_crate/STATUS on the same PODRUN_ROOT this script honours; "
        "without reading it, a partial partner forecast passes and produces a plausible "
        "wrong reconciled number."
    )
    assert re.search(r"grep -q '\^OK\$'", preflight), (
        "the partner STATUS is referenced but not compared to OK, so a FAILED verdict would "
        "satisfy the check."
    )

    #: The datafactory credential, which phase 2 needs more than phase 1 does.
    assert "netrc" in preflight, (
        "the preflight does not check /root/.netrc. All eight *_sparrow members fetch pgm "
        "global-land data before anything else happens, so without it a box passes "
        "--preflight OK and dies inside member 1 — the opposite of 'every problem at once'."
    )


# ── the ensemble invocation ───────────────────────────────────────────────────────────


def test_the_ensemble_is_pooled_with_saved_artifacts_and_never_trained(script: str) -> None:
    """`-f -sa`, never `-t`.

    With `-t` the manager calls `_train_model_artifact`, which shells out to each member's
    `run.sh` and re-enters the conda lane — where C-151 and the per-launcher prefixes live
    (#643, #644). `-sa` with the artifacts on disk makes it load them instead.
    """
    ensemble_calls = [
        ln for ln in script.splitlines()
        if "main.py" in ln and "-sa" in ln
    ]
    assert ensemble_calls, "no `-sa` ensemble invocation found"
    for line in ensemble_calls:
        assert re.search(r"-f\s+-sa|-sa\s+-f", line), (
            f"the ensemble invocation is not `-f -sa`: {line.strip()!r}"
        )
        assert not re.search(r"\s-t(\s|$)", line), (
            f"the ensemble is invoked with -t: {line.strip()!r}. That shells out to each "
            f"member's run.sh and re-enters the conda lane."
        )


def test_the_member_list_matches_the_ensembles_config(script: str) -> None:
    """A chain that silently follows a reordered config is worse than one that disagrees."""
    match = re.search(r'^MODELS="([^"]+)"', script, re.MULTILINE)
    assert match, "MODELS is not set"
    declared = match.group(1).split()

    config = REPO / "ensembles" / ENSEMBLE / "configs" / "config_modelset.py"
    assert config.is_file(), f"{config} is gone; this guard cannot see its subject (ADR-025)"
    in_config = re.findall(r'"([a-z]+_sparrow)"', config.read_text())

    assert declared == in_config, (
        f"the runner's member list and {ENSEMBLE}'s config_modelset.py disagree.\n"
        f"  runner: {declared}\n"
        f"  config: {in_config}\n"
        f"Order matters: PFE concat pools on the sample axis and the pairing against the "
        f"*_condor twins is positional."
    )


def test_a_member_already_forecast_is_skipped(script: str) -> None:
    """Re-running after a mid-chain failure must not retrain 184 minutes.

    Not a hypothetical optimisation: it is the state the 2026-10-28 pod was left in — eight
    members trained, then the reconcile died on #562. The manager does the same for the
    ensemble (`prediction_frame_ensemble.py:998` loads an existing PredictionFrame rather than
    regenerating), so the member loop agrees with it.
    """
    #: Scoped to the member loop, and asserting the MECHANISM.
    #:
    #: ⛔ The first version of this test asserted `"SKIPPED" in script` and `"continue" in
    #: script`. A mutation that deleted the counter left it green, because the word still
    #: appears in the echo and the manifest — a guard that searched for vocabulary rather than
    #: behaviour. That mutation turned out to be harmless to the skip itself but not to the
    #: MANIFEST, which would then report "0 already present" after skipping eight. Both halves
    #: are pinned below.
    loop_start = script.index("for M in $MODELS; do")
    loop_end = script.index("echo \"  members:", loop_start)
    loop = script[loop_start:loop_end]

    #: The GLOB, not the word. A mutation changing `-name '*forecasting*'` to a pattern that
    #: never matches survived an earlier `"forecasting" in loop` assertion, because the word
    #: also appears in `main.py -r forecasting -t -f` two lines below. Asserting vocabulary
    #: that the subject happens to contain for an unrelated reason is a guard that cannot fail.
    #: `predictions_forecasting_*`, and NOT the looser `*forecasting*` this guard asserted
    #: when it was written. That glob also matches `forecasting_log.txt`, which the run writes
    #: into the same directory BEFORE the predictions are saved — the sibling run type proves
    #: the shape on disk today (`models/bronze_kestrel/data/generated/` holds
    #: `calibration_log.txt` beside `predictions_calibration_<ts>`). So a member whose
    #: training finished and whose forecast stage then died would have been reported "already
    #: present" and dropped from the pool, which is the one case this skip exists for.
    assert re.search(r"find .*-name 'predictions_forecasting_\*'", loop), (
        "the member skip is not keyed on `predictions_forecasting_*`, the artefact a finished "
        "forecast actually leaves. Anything looser also matches `forecasting_log.txt`, which "
        "exists as soon as the run starts — so an interrupted member is skipped and 7 of 8 "
        "reach the pool, where concat ordering is positionally paired against the *_condor "
        "twins. Anything stricter never fires and a re-run retrains 184 minutes."
    )
    assert "continue" in loop, (
        "the member loop detects an existing forecast and then trains anyway: there is no "
        "`continue` in the loop body, so the skip does not skip."
    )
    assert re.search(r"SKIPPED=\$\(\(SKIPPED\+1\)\)", loop), (
        "nothing increments SKIPPED inside the loop, so the MANIFEST's `members_run:` line "
        "reports 0 already present however many were skipped. A status field that cannot be "
        "wrong is worth more than one that is usually right (`feedback_status_fields_lie`)."
    )
    assert re.search(r"TRAINED=\$\(\(TRAINED\+1\)\)", loop), (
        "nothing increments TRAINED inside the loop, so the MANIFEST cannot say how much of "
        "this run was paid for."
    )
    assert "members_run:" in script, (
        "the MANIFEST no longer records how many members were trained versus reused, which is "
        "the only record of what a re-run actually cost."
    )


# ── it publishes nothing ──────────────────────────────────────────────────────────────


def test_no_publish_flag_is_passed_to_anything(script: str) -> None:
    """Guide rule 5: publish credentials never go on rented hardware."""
    for line in script.splitlines():
        if "main.py" not in line:
            continue
        assert not re.search(r"\s-p(\s|$)", line), f"a -p publish flag is passed: {line.strip()!r}"
        assert "--prediction_store" not in line, (
            f"--prediction_store is passed: {line.strip()!r}. This chain publishes nothing."
        )


def test_publish_variables_are_refused_rather_than_read(script: str) -> None:
    """Their presence is a preflight FAILURE, not a warning and not an input."""
    for var in ("APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID", "APPWRITE_DATASTORE_API_KEY"):
        assert var in script, (
            f"{var} is no longer named in the preflight, so a publish credential could sit on "
            f"the pod unremarked (guide rule 5)."
        )
    assert "guide rule 5" in script, (
        "the rule the refusal enforces is no longer cited, so the next person reads it as "
        "defensive clutter and deletes it."
    )


def test_wandb_is_forced_offline(script: str) -> None:
    """The manager opens a W&B run before any member trains (#645)."""
    assert "WANDB_MODE=offline" in script, (
        "WANDB_MODE=offline is not forced. The ensemble manager opens a W&B run before any "
        "member trains, so a box with no key stops at a credential prompt having done no work."
    )


# ── and the preflight, exercised for real ─────────────────────────────────────────────


def test_preflight_refuses_an_empty_root_and_reports_every_problem(tmp_path: Path) -> None:
    """One round trip, not six. And it must touch nothing.

    This is the only test here that runs the script. `PODRUN_ROOT` is honoured precisely so
    the preflight can be exercised off a pod.
    """
    done = subprocess.run(
        [str(RUNNER), "--preflight"],
        env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 1, (
        f"--preflight on an empty root exited {done.returncode}, expected 1.\n{done.stdout}"
    )
    out = done.stdout + done.stderr

    #: Every distinct problem, not just the first — the property that makes it one round trip.
    for expected in ("is not a checkout", "no such model", "no such ensemble",
                     "has produced no forecast", "does not exist"):
        assert expected in out, (
            f"--preflight did not report {expected!r}. It must report EVERY problem at once; "
            f"one per attempt is six round trips on rented hardware.\n{out}"
        )
    assert out.count("MISSING") >= 10, (
        f"--preflight reported only {out.count('MISSING')} problems on a completely empty "
        f"root; it stopped early.\n{out}"
    )
    assert "preflight failed" in out
    #: It must not have run anything.
    assert "HEARTBEAT" not in out and "pool_and_reconcile" not in out, (
        f"--preflight ran part of the chain.\n{out}"
    )


def test_preflight_refuses_a_publish_credential(tmp_path: Path) -> None:
    """Set the key and the run must stop, whatever else is in order."""
    done = subprocess.run(
        [str(RUNNER), "--preflight"],
        env={
            "PATH": "/usr/bin:/bin",
            "PODRUN_ROOT": str(tmp_path),
            "APPWRITE_DATASTORE_API_KEY": "not-a-real-key",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = done.stdout + done.stderr
    assert done.returncode == 1
    assert "APPWRITE_DATASTORE_API_KEY is set" in out, (
        f"a publish credential in the environment was not refused (guide rule 5).\n{out}"
    )


def test_a_bad_invocation_is_refused_before_anything_is_touched(tmp_path: Path) -> None:
    """A positional argument means the operator expects a different script."""
    for argv in (["chrome_crate"], ["--publish"]):
        done = subprocess.run(
            [str(RUNNER), *argv],
            env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = done.stdout + done.stderr
        assert done.returncode == 1, f"{argv} was accepted"
        assert "usage:" in out, f"{argv} was refused without saying how to invoke it:\n{out}"
        assert "preflight" not in out.lower() or "usage" in out, (
            f"{argv} got as far as the preflight before being refused:\n{out}"
        )
