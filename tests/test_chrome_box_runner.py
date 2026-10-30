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


# ─────────────────────────────────────────────────────────────────────────────────────────
# The three defects the 2026-10-30 run exposed in this runner's own output.
# All three were found by reading the MANIFEST the run produced, not by reading the script.
# ─────────────────────────────────────────────────────────────────────────────────────────


def _manifest_echo_lines(script: str) -> list[str]:
    """The lines the MANIFEST is actually built from — comments excluded, deliberately.

    ⛔ Do not match these properties against the script's prose. This repository keeps a
    superseded claim in the file that corrected it, so the file holding the fix also holds
    the defect's vocabulary, and no text matcher can tell them apart (C-189, which cost
    three failed attempts at the sibling guard). The MANIFEST block is a structure: the
    `echo` lines between `stage manifest` and the redirect. Read that.
    """
    body = script.split("stage manifest", 1)[1].split('} > "$OUT/MANIFEST"', 1)[0]
    return [
        ln.strip()
        for ln in body.splitlines()
        if ln.strip().startswith("echo ")
    ]


def test_the_memory_sampler_records_anon_and_not_only_memory_current(script: str) -> None:
    """The sampler must read `anon` from memory.stat, not `memory.current` alone.

    ⛔ Measured live on the 2026-10-30 pod, same cgroup, same second:

        member stage      anon  4,313 MiB   memory.current 22,021 MiB   (5.1x)
        pool_and_reconcile anon 65,404 MiB  memory.current 89,609 MiB   (1.4x)

    `memory.current` counts reclaimable page cache, which climbs monotonically as the
    members write 9.5 GB arrays and is not given back until something needs it. At the peak
    it had pinned against the 87 GiB cgroup limit, so it reports the size of the box rather
    than the size of the job.

    This matters because `MEM_FLOOR_GB` is derived from this number. Both figures that set
    it were `memory.current` readings: the 68.2 GiB "lower bound" from 2026-10-28, and the
    19.8 GiB since withdrawn from `chrome_box/configs/config_hyperparameters.py` — whose own
    comment names `memory.current` as the instrument that produced it. A sibling file in the
    same evidence directory, `anon_sampler.sh`, opens with the explanation; this runner was
    ported from its neighbour and took the wrong one of the two.
    """
    sampler = script.split("# ── memory sampler", 1)[1].split("& SAMPLER=", 1)[0]
    assert "/sys/fs/cgroup/memory.stat" in sampler, (
        "the sampler does not read memory.stat, so it cannot see `anon` — and "
        "memory.current alone was 5x the requirement during the member stage"
    )
    assert "current=" in sampler, (
        "drop memory.current and the two series can no longer be compared; both are logged "
        "on purpose, and the MANIFEST says which one is the requirement"
    )

    # ⛔ EXECUTE the field extraction; do not look for the string "anon=" in this block.
    # A first version of this guard did, and a mutation that renamed the printf labels to
    # `a=`/`f=` SURVIVED it — because `anon=` also occurs in the sampler's own
    # `${MS:-anon=NA file=NA}` fallback three lines down. Same family as C-189: the text
    # that documents a format is not the format.
    cmd = re.search(r"MS=\$\((awk .*?)\)\n", sampler, re.S)
    assert cmd, "the sampler no longer builds its line with a single awk over memory.stat"

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        stat = Path(d) / "memory.stat"
        # real field order and names from the 2026-10-30 pod, anon/file at the measured peak
        # Exact MiB multiples, so the expected string below is arithmetic and not a
        # rounding coincidence: 65404 and 23833 MiB are the 2026-10-30 peak readings.
        # `active_anon` is deliberately LARGER than `anon` — an awk keyed on /anon/ rather
        # than /^anon / would pick it up, and the assertion would show it.
        stat.write_text(
            f"anon {65404 * 1048576}\nfile {23833 * 1048576}\nkernel_stack 9879552\n"
            f"slab 1288490188\nsock 262144\nactive_anon {99999 * 1048576}\n"
        )
        out = subprocess.run(
            ["bash", "-c", cmd.group(1).replace("/sys/fs/cgroup/memory.stat", str(stat))],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    assert out == "anon=65404MiB file=23833MiB", (
        f"the sampler emitted {out!r}. It must label the two fields exactly `anon=` and "
        f"`file=` in MiB, because peak_of() keys on those names and the MANIFEST reports "
        f"anon as the requirement. `active_anon` must not be picked up instead of `anon`."
    )


def test_the_manifest_never_claims_expected_models_enforces_the_draw_count(script: str) -> None:
    """`expected_models` is read by nothing at run time, and the MANIFEST must not say it is.

    ⛔ `grep -rn "expected_models"` over installed views-pipeline-core 3.3.5 returns **zero**
    matches. `expected_samples_per_model` IS enforced (`_assert_expected_sample_count`,
    `prediction_frame_ensemble.py:153`) and the #160 balance guard rejects constituents whose
    sample counts differ *from each other* — but neither can see a MISSING constituent. Seven
    members at 128 draws are mutually balanced and individually correct, so a short pool
    passes both guards and delivers 896 draws where the config, the epic and the delivery
    contract all say 1,024.

    The claim was asserted four times before it was measured: in #649's commit message, in
    register entry C-188, in a test docstring, and in this MANIFEST line — which on
    2026-10-30 printed it into a real delivery artifact that a UN partner's forecast is built
    from. `members_pooled:` is the only verification of the count, and it is this script's own.

    Pinned against the MANIFEST's `echo` lines, not its prose, for the reason in
    `_manifest_echo_lines`.
    """
    offenders = [ln for ln in _manifest_echo_lines(script) if "expected_models" in ln]
    assert not offenders, (
        "the MANIFEST names `expected_models` as a backstop for the draw count. Nothing "
        "reads it at run time, so the line promises a guarantee no code provides:\n  "
        + "\n  ".join(offenders)
    )
    # ⛔ Anchored to the start of the echoed string. A substring test SURVIVED deleting
    # this field, because the `draws:` line above refers to `members_pooled:` by name —
    # the guard matched the cross-reference that explains the field, not the field.
    pooled = [ln for ln in _manifest_echo_lines(script)
              if re.search(r'echo\s+"members_pooled:', ln)]
    assert pooled, (
        "the MANIFEST must report members_pooled: — with expected_models enforcing nothing, "
        "the runner's own pre-pool count is the only check that eight members were pooled"
    )


def test_a_peak_that_could_not_be_read_reports_unreadable_and_not_zero(script: str) -> None:
    """The peak parsers must refuse when nothing matched, not return `0MiB` (ADR-025).

    Executed against the shipped function rather than asserted about its text, because the
    failure this guards is an arithmetic one: `awk`'s `m` is unset when no field matches, so
    `printf "%dMiB", m` yields **`0MiB`** — which prints in the MANIFEST beside the real
    figures and reads as a measured peak of nothing. An operator sizing the next box from a
    MANIFEST saying `PEAK anon: 0MiB` is the exact shape of
    `feedback_absence_is_not_a_pass`, whose seventh instance was recorded the same night.

    Three ways the parser legitimately sees nothing, all tested below: the sampler could not
    read memory.stat and logged `anon=NA`; the log is from a runner whose format predates the
    two-series sampler; the log is empty because the run died before the first sample.
    """
    body = re.search(r"peak_of\(\)\s*\{(.*?)\n\}", script, re.S)
    assert body, "peak_of() is gone — the MANIFEST's peak figures are no longer parsed here"

    def run(log: str, key: str, tmp: Path) -> str:
        f = tmp / "memory.log"
        f.write_text(log)
        snippet = f'OUT={tmp}\npeak_of() {{{body.group(1)}\n}}\npeak_of {key}\n'
        return subprocess.run(
            ["bash", "-c", snippet], capture_output=True, text=True, check=True
        ).stdout.strip()

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        good = (
            "07:28:00 anon=65404MiB file=23833MiB current=89609MiB pool_and_reconcile\n"
            "06:06:09 anon=4311MiB file=17533MiB current=22021MiB member:eager_sparrow\n"
        )
        assert run(good, "anon", tmp) == "65404MiB", "the parser cannot read a good log"
        assert run(good, "current", tmp) == "89609MiB"

        for label, log in (
            ("sampler could not read memory.stat", "07:28:00 anon=NA file=NA current=89609MiB x\n"),
            ("log predates the two-series sampler", "04:17:17 3190MiB verify_install\n"),
            ("run died before the first sample", ""),
        ):
            assert run(log, "anon", tmp) == "UNREADABLE", (
                f"{label}: the parser returned a number instead of refusing. A peak of "
                f"0MiB in the MANIFEST is indistinguishable from a job that used no memory"
            )


def test_the_residual_check_runs_unconditionally_and_its_failure_is_fatal(script: str) -> None:
    """The residual measurement may not be skippable, and may not be advisory.

    It exists because the 2026-10-30 delivery lost 2.9%-7.3% of the country forecast's
    fatalities and reported `reconciled: against chrome_crate` with no qualification. A
    check that can be skipped would have been skipped on exactly that run.

    ⚠️ **What this guard does NOT prove.** An earlier version claimed the column-0 anchors
    detected *nesting* — a residual step moved inside `if [ ... ]; then`. They do not:
    bash ignores indentation, so a nested step can sit at column 0 and a mutation that
    wrapped it in `if true; then ... fi` SURVIVED this guard. Detecting that needs a shell
    parser, which is not worth building here. What protects the property instead is the
    fatality below — a skipped step cannot reach `die`, and `RESIDUAL_LINES` then renders
    the MANIFEST's refusal line, which `test_the_manifest_cannot_...` pins.

    So this asserts exactly four things, each mutation-proved: the stage exists, the exit
    code is captured, a non-zero code is fatal, and the module is the one invoked.
    """
    assert re.search(r"^stage reconciliation_residual$", script, re.M), (
        "the residual step is gone, or is no longer a top-level stage"
    )
    assert re.search(r"^RESIDUAL_RC=\$\?$", script, re.M), (
        "the residual's exit code is not captured at the top level — if this moved inside "
        "a conditional it can be skipped, which is how the finding happened"
    )
    assert re.search(r'^\[ "\$RESIDUAL_RC" = 0 \] \|\| die ', script, re.M), (
        "a non-zero residual exit is not fatal. Non-zero means either the scaling failed "
        "(country totals wrong) or the residual could not be measured — ADR-025 forbids "
        "reporting OK in either case"
    )
    # ⛔ Anchored to the INVOCATION, not to the module name. A bare
    # `"-m reconciliation.residual" in script` SURVIVED deleting the real call, because
    # the `die` message helpfully tells the operator how to re-run the check by hand and
    # contains the same words. Matching explanatory text instead of mechanism is the
    # C-189 family and this is its third instance in one evening.
    invocation = re.search(
        r'"\$VENV/bin/python" -m reconciliation\.residual', script
    )
    assert invocation, (
        "the runner no longer INVOKES the residual module (the module name may still "
        "appear in the die message, which is not an invocation)"
    )

    # Order matters for a reason that is not stylistic: the MANIFEST interpolates
    # `$RESIDUAL_LINES`, so a residual computed AFTER the manifest block would render an
    # empty field, and the delivery would again carry no deficit figure.
    assert (
        script.index('cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/."')
        < invocation.start()
        < script.index('} > "$OUT/MANIFEST"')
    ), (
        "the residual must be measured after the pool is copied and before the MANIFEST "
        "is written; outside that window the MANIFEST's residual field renders empty"
    )


def test_the_manifest_cannot_describe_the_pool_as_simply_reconciled(script: str) -> None:
    """⛔ The exact line that shipped on 2026-10-30, and what replaced it.

    `reconciled:    against chrome_crate, read from local disk` is a statement about what
    was *attempted*, in a field every reader takes as a statement about what was
    *achieved*. The MANIFEST goes to a UN partner. It must carry the measured deficit.
    """
    echoes = _manifest_echo_lines(script)
    unqualified = [
        ln for ln in echoes
        if re.search(r'echo\s+"reconciled:', ln) and "residual" not in ln.lower()
    ]
    assert not unqualified, (
        "the MANIFEST states `reconciled:` without pointing at the residual:\n  "
        + "\n  ".join(unqualified)
    )
    assert [ln for ln in echoes if re.search(r'echo\s+"residual:', ln)], (
        "the MANIFEST carries no `residual:` field, so a reader cannot see how much of "
        "the country forecast never reached the grid"
    )
    assert [ln for ln in echoes if "$RESIDUAL_LINES" in ln], (
        "the per-target measured figures are not interpolated into the MANIFEST — a "
        "prose warning without numbers is the thing being fixed, not the fix"
    )


def test_the_trained_member_artefacts_are_collected_and_a_missing_one_is_fatal(script: str) -> None:
    """⛔ The 2026-10-30 teardown audit found 16 trained models about to die with the box.

    The pool is the product; the member artefacts are the only thing here that cannot be
    regenerated. `docs/runpod_run_guide.md:429` states it in bold and cites 2026-09-28, when
    eight models trained, delivered their parquets, reported `OK`, and their weights died
    with the pods (views-models#549). `pod_run_model.sh:395` has had a `collect_artifact`
    stage since before this runner existed; this runner had none, because the deliberate
    (and correct) decision not to copy 26.7 GiB of member *forecasts* silently took
    `artifacts/` and `wandb/` with it — 542 MB, and the half that cannot be remade.

    Fatal rather than advisory, and the ordering makes that safe: the pool is copied first,
    so dying here costs attention rather than work, and a missing irreplaceable artefact is
    exactly what an operator must learn **before** destroying the host.
    """
    assert re.search(r"^stage collect_members$", script, re.M), (
        "the member-artefact collection stage is gone — the trained models will die with "
        "the pod, which is views-models#549 verbatim"
    )
    assert re.search(r'artifacts"/\*\.pkl', script), (
        "the stage no longer reads each member's artifacts/*.pkl (the sparrow members are "
        "pickled baselines, not .pt state dicts)"
    )
    assert re.search(r'-lt "\$START"', script), (
        "the artefact's mtime is no longer compared against this run's start. The repo tree "
        "is not cleared between runs on one pod, so a previous run's .pkl would be collected "
        "and reported as this run's (C-171)"
    )
    assert re.search(r'^\[ -z "\$MISSING" \] \|\| die ', script, re.M), (
        "a member with no artefact from this run is no longer fatal. A warning in a log is "
        "not a signal when the next action is destroying the host"
    )
    assert re.search(r'cp -r "\$REPO/models/\$M/wandb"', script), (
        "the offline W&B record is no longer collected. WANDB_MODE=offline is forced, so it "
        "is written outside $OUT by construction — C-170's second class"
    )
    pool_copy = script.index('cp -r "$REPO/ensembles/$ENSEMBLE/data/generated/."')
    collect = script.index("stage collect_members")
    manifest = script.index('} > "$OUT/MANIFEST"')
    assert pool_copy < collect < manifest, (
        "collect_members must run after the pool is copied (so a failure here loses nothing) "
        "and before the MANIFEST is written (so the MANIFEST can report what was collected)"
    )


def test_the_package_set_is_recorded_and_cannot_be_silently_empty(script: str) -> None:
    """The MANIFEST pins 2 versions; the venv has ~143, and `pip freeze` here returns none.

    ⛔ The venv is uv-built and has no pip, so `pip freeze` writes an **empty file** and
    exits 0 — the absence that reads as success, which this repository has recorded seven
    times. `importlib.metadata` is asked instead: the same source the pipeline-core version
    assert already uses, needing neither pip nor uv.

    Why it is load-bearing rather than tidy: the reconciliation mass deficit this runner
    measures and reports is a property of a **specific views-frames** (1.11.0 on the
    2026-10-30 run), so a reproduction that re-resolves it is not a reproduction of anything.

    The `pip freeze` check reads non-comment lines only. The explanatory comment above
    necessarily contains the phrase it forbids — matching it would be C-189's trap, which
    this same file already has three instances of.
    """
    # ⛔ Anchored to the heredoc that RUNS it, not to the words. `importlib.metadata` also
    # appears in the explanatory comment above it, so a bare substring test is satisfied by
    # the explanation while the code that does the work is gone — C-189, instance four.
    freeze = re.search(
        r'"\$VENV/bin/python" - > "\$OUT/FREEZE" <<.PYFREEZE.\n(.*?)\nPYFREEZE',
        script, re.S,
    )
    assert freeze, "the FREEZE heredoc is gone, or no longer writes $OUT/FREEZE"
    assert "importlib.metadata" in freeze.group(1), (
        "the FREEZE heredoc no longer asks importlib.metadata for the package set"
    )
    assert re.search(r'^\[ -s "\$OUT/FREEZE" \] \|\| die ', script, re.M), (
        "an empty FREEZE is no longer fatal — which is the only failure mode this has, "
        "since the mechanism exits 0 having written nothing"
    )
    # ⛔ A regex, not a substring. `"$VENV/bin/pip" freeze` — the form anyone would actually
    # write here — does not contain the string "pip freeze", and a substring test SURVIVED a
    # mutation that added exactly that line. C-189, instance five.
    executed = [
        ln for ln in script.splitlines()
        if re.search(r"""pip["']?\s+freeze""", ln) and not ln.lstrip().startswith("#")
    ]
    assert not executed, (
        "`pip freeze` is executed somewhere. This venv has no pip: it writes an empty file "
        "and exits 0.\n  " + "\n  ".join(executed)
    )
    echoes = _manifest_echo_lines(script)
    assert [ln for ln in echoes if "$OUT/FREEZE" in ln], (
        "the MANIFEST does not mention FREEZE, so a reader of the delivery has no pointer "
        "to the versions it was produced with"
    )
    assert [ln for ln in echoes if '"artefacts:' in ln], (
        "the MANIFEST does not report how many member artefacts were collected"
    )
