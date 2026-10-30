"""`tools/podrun/pod_run_crafd_chain.sh` — the disciplines that must not rot (#640, epic #595).

The sibling of `tests/test_darts_calibration_runner.py`, and written for the same reason: every
property asserted here is one a comment would state and the next edit would quietly break.

Four of them are load-bearing enough to name up front, because each one is a defect this
platform has already paid for:

    its own venv            The other three podrun scripts all use `$ROOT/venv`, so a pod has
                            one virtualenv whose contents are whatever ran last — the uv twin
                            of C-116. A baseline chain reusing it finds views-hydranet and no
                            views_baseline. If this script's VENV is ever "simplified" back to
                            the shared path, that is the bug returning.
    never calls run.sh      `models/*/run.sh` activates a conda prefix shared by 45 tenants and
                            does not check pip's exit status (#644, pipeline-core#560). The pod
                            lane's immunity comes entirely from running main.py directly.
    pools with -f -sa       With `-t`, `_train_model_artifact` shells out to the member's
                            run.sh and lands straight back in the conda lane. `-f` alone takes
                            `_forecast_model_artifact`, which does not shell out.
    publishes nothing       Guide rule 5. The specific way this breaks is copying
                            `pod_run_fao_delivery.sh`, which legitimately needs nine publish
                            variables because its job is to publish. This one's is not.

Tests here are text assertions on the script, not executions of it: the script's job needs a
rented pod with a datafactory credential, so what CI can check is that its declared
disciplines are still declared. `--preflight` behaviour is exercised for real where it can be.
"""

from pathlib import Path
import re
import subprocess

import pytest

pytestmark = pytest.mark.beige

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools" / "podrun" / "pod_run_crafd_chain.sh"
ENSEMBLE = "chrome_crate"


@pytest.fixture(scope="module")
def script() -> str:
    assert RUNNER.is_file(), f"{RUNNER} does not exist"
    return RUNNER.read_text()


def test_the_runner_exists_and_is_executable():
    assert RUNNER.is_file(), f"{RUNNER} is missing"
    assert RUNNER.stat().st_mode & 0o111, (
        f"{RUNNER} is not executable. The guide tells an operator to run it directly; a "
        "non-executable runner fails with 'Permission denied' on a rented pod."
    )


def test_it_uses_its_own_venv_and_not_the_shared_one(script):
    """The whole point of a fourth script. See the module docstring.

    `$ROOT/venv` is shared by pod_run_model.sh, pod_run_darts_calibration.sh and
    pod_run_fao_delivery.sh. This chain installs views-baseline; that one holds views-hydranet.
    """
    assigns = re.findall(r"^VENV=(\S+)", script, re.M)
    assert assigns, "the script sets no VENV at all"
    assert len(assigns) == 1, f"VENV is assigned {len(assigns)} times: {assigns}"
    assert assigns[0] != "$ROOT/venv", (
        "VENV is the SHARED $ROOT/venv. This chain installs views-baseline into it, and "
        "pod_run_model.sh installs views-hydranet into the same path — last writer wins, which "
        "is register C-116 in the uv lane. Give this chain its own venv."
    )
    assert "venv-crafd" in assigns[0], (
        f"VENV is {assigns[0]!r}; expected a chain-specific path such as $ROOT/venv-crafd so "
        "no other podrun script can inherit or clobber it."
    )


# A run.sh being EXECUTED, rather than merely named. The first version of this test matched any
# non-comment line containing "run.sh" and so fired on the preflight message that warns an
# operator their model's run.sh is broken — a diagnostic string, not an invocation. Matching the
# word was the lazy proxy; matching an execution is the property.
_EXECUTES_RUN_SH = re.compile(
    r"""(?:^|[;&|]|\$\()\s*(?:bash|sh|exec|source|\.)\s+\S*run\.sh|^\s*\S*/run\.sh\b|"\S*/run\.sh"\s""",
    re.M,
)


def test_it_never_invokes_a_models_run_sh(script):
    """run.sh is the conda lane. This script's immunity depends on not entering it."""
    offenders = [
        line.strip()
        for line in script.splitlines()
        if not line.lstrip().startswith("#") and _EXECUTES_RUN_SH.search(line)
    ]
    assert not offenders, (
        "the runner executes a run.sh. `models/*/run.sh` activates a conda prefix shared by 45 "
        "tenants and ignores pip's exit status (#644, pipeline-core#560); running main.py "
        "under this script's own venv is what avoids all of it:\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


def test_the_ensemble_is_pooled_without_train(script):
    """`-t` on the ensemble shells out to every member's run.sh. See the module docstring."""
    pool = [
        line.strip()
        for line in script.splitlines()
        if "main.py" in line and not line.lstrip().startswith("#") and "-sa" in line
    ]
    assert pool, (
        "no ensemble pool invocation found. The pool must be `main.py -r forecasting -f -sa`: "
        "-sa so the eight member forecasts are reused rather than refetched."
    )
    for line in pool:
        tokens = line.split()
        assert "-t" not in tokens and "--train" not in tokens, (
            f"the pool invocation passes --train:\n  {line}\n"
            "With -t the manager calls _train_model_artifact, which shells out to "
            "models/<member>/run.sh and re-enters the conda lane. Use -f alone."
        )


def test_no_publish_flag_is_passed_to_anything(script):
    """Guide rule 5: this chain produces a forecast and uploads nothing."""
    offenders = []
    for line in script.splitlines():
        if line.lstrip().startswith("#") or "main.py" not in line:
            continue
        tokens = line.split()
        if "-p" in tokens or "--prediction_store" in tokens:
            offenders.append(line.strip())
    assert not offenders, (
        "a run invocation passes -p/--prediction_store. This chain holds no publish "
        "credential and the CRAF'd delivery is gated separately (views-crafdapi#180, #599):\n"
        + "\n".join(f"  {line}" for line in offenders)
    )


def test_no_appwrite_extra_is_installed(script):
    """The sibling defect: starting from the script that already works.

    `pod_run_fao_delivery.sh` installs the appwrite extra because it publishes. Copying that
    line into a runner that does not publish puts a write path on rented hardware for nothing.
    """
    assert "[appwrite]" not in script, (
        "the runner installs the appwrite extra. It publishes nothing, so that is a write "
        "path on hardware we do not control, for no benefit (guide rule 5, ADR-024)."
    )


def test_publish_variables_are_refused_rather_than_read(script):
    """Their presence is a finding, not an input.

    `pod_run_fao_delivery.sh` reads these nine because publishing is its job. Here, a set
    publish variable means a credential was placed on a rented box that has no use for one, so
    preflight must say so rather than silently ignoring it.
    """
    for var in ("APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID", "APPWRITE_DATASTORE_API_KEY"):
        assert var in script, f"{var} is not mentioned; preflight cannot refuse what it never checks"
    # It must appear as a refusal, never as an export or an assignment.
    for line in script.splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert not re.match(r"^\s*(export\s+)?APPWRITE_\w+=", line), (
            f"the runner assigns an Appwrite variable:\n  {line.strip()}\n"
            "It must only CHECK for them and refuse."
        )


def test_the_toolz_override_is_an_install_and_is_last(script):
    """C-151. Any install appended after it re-resolves the venv and can revert it silently.

    This happened on 2026-09-29: toolz went 1.1.0 -> 0.11.2 with no error, and every
    datafactory fetch in that venv broke.
    """
    installs = [
        i
        for i, line in enumerate(script.splitlines())
        if "uv pip install" in line and not line.lstrip().startswith("#")
    ]
    assert installs, "the runner performs no uv pip install"
    toolz_lines = [
        i
        for i, line in enumerate(script.splitlines())
        if "uv pip install" in line and "toolz" in line and not line.lstrip().startswith("#")
    ]
    assert toolz_lines, (
        "no `uv pip install ... toolz>=0.12.1` found. Without it viewser's toolz<0.12 pin "
        "resolves to a toolz that cannot import tlz submodules on Python 3.11 (C-151) and "
        "every datafactory fetch in this venv fails."
    )
    assert max(installs) == max(toolz_lines), (
        "the toolz override is not the LAST install in the script. A later `uv pip install` "
        "re-resolves the venv and can pull toolz back under 0.12 with no error."
    )


def test_the_override_is_verified_by_import_not_by_version_string(script):
    """A version number can be right while the import is broken, and the reverse.

    ⛔ Asserted as an EXECUTED, `|| die`-guarded command, not as a substring. A mutation that
    replaced the real check with `true` left an earlier version of this test green, because
    "import tlz.curried" still appeared in the comment above it and in the die message below
    it. A guard that a deleted check satisfies is decorative.
    """
    verified = [
        line.strip()
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
        and re.search(r"\$VENV/bin/python\"?\s+-c\s+.{0,4}import tlz\.curried", line)
    ]
    assert verified, (
        "the runner does not EXECUTE `$VENV/bin/python -c 'import tlz.curried'`. That import "
        "is the exact thing C-151 breaks and the exact thing a datafactory fetch needs. A "
        "`toolz.__version__` check is not a substitute: it passes on a venv whose toolz "
        "package is missing or shadowed."
    )
    # The check must also be load-bearing. An unguarded probe prints and continues.
    guarded = [
        line
        for line in verified
        if "||" in line or "die" in line
    ]
    index = next(
        i
        for i, line in enumerate(script.splitlines())
        if re.search(r"\$VENV/bin/python\"?\s+-c\s+.{0,4}import tlz\.curried", line)
        and not line.lstrip().startswith("#")
    )
    following = "\n".join(script.splitlines()[index : index + 3])
    assert guarded or "die" in following, (
        "the `import tlz.curried` check runs but its failure is not fatal. Without `|| die` "
        f"the chain continues into eight paid model runs that cannot fetch:\n{following}"
    )


def _install_commands(script: str) -> list[str]:
    """Each `uv pip install` command, with `\\` continuations joined and comments dropped.

    Scoped deliberately. The first version of the pandas test searched every non-comment line
    and so fired on the preflight message that WARNS about the extra — a string that must stay.
    What the tests below assert is what gets installed, so that is what they must read.
    """
    commands, current = [], None
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if current is not None:
            current.append(stripped.rstrip("\\").strip())
            if not stripped.endswith("\\"):
                commands.append(" ".join(current))
                current = None
            continue
        if "uv pip install" in stripped:
            current = [stripped.rstrip("\\").strip()]
            if not stripped.endswith("\\"):
                commands.append(" ".join(current))
                current = None
    if current is not None:
        commands.append(" ".join(current))
    return commands


def test_it_does_not_declare_the_unsatisfiable_pandas_extra(script):
    """`views-datafactory[pandas]` cannot resolve beside views-pipeline-core 3.x.

    pipeline-core requires viewser (ADR-063 clause 1), viewser caps `pandas<2`, and the extra
    floors pandas at `>=2`. Measured in a clean venv 2026-10-28 (#644). pandas arrives through
    xarray regardless, which is what the proven rusty_bucket stand-ins relied on.
    """
    installs = _install_commands(script)
    assert installs, "the runner performs no uv pip install"
    offenders = [cmd for cmd in installs if "views-datafactory[pandas]" in cmd]
    assert not offenders, (
        "an install command declares views-datafactory[pandas], which is unsatisfiable "
        "alongside views-pipeline-core 3.x (#644, pipeline-core#561). Declare plain "
        "views-datafactory:\n" + "\n".join(f"  {cmd}" for cmd in offenders)
    )


def test_the_pipeline_core_floor_is_at_least_3_2_0(script):
    """Below 3.2.0 the loader cannot find configs/config_maturity.py (#594, #635)."""
    match = re.search(r'"views-pipeline-core([^"]*)"', script)
    assert match, "the install set does not declare views-pipeline-core"
    assert ">=3.2.0" in match.group(1), (
        f"views-pipeline-core is declared as {match.group(1)!r}. A floor below 3.2.0 admits a "
        "pipeline-core that cannot read configs/config_maturity.py, and the run dies on a "
        "missing config_deployment.py (#594)."
    )


def test_wandb_is_forced_offline(script):
    """The manager opens a W&B run before any member trains (#645).

    `prediction_frame_ensemble.py:413` wraps `_train_ensemble` in `initialize_run`, so a box
    with no credential stops at a prompt having done no work. Offline also keeps a W&B key off
    rented hardware (guide rule 5).
    """
    assert re.search(r"export\s+WANDB_MODE=offline", script), (
        "the runner does not export WANDB_MODE=offline. On a fresh pod the ensemble contacts "
        "W&B before the first member trains and the chain stops at a credential prompt (#645)."
    )


def test_the_member_list_matches_the_ensembles_config(script):
    """The script's literal list must agree with chrome_crate's config_modelset.py.

    The list is literal so the script reads on a pod with no python yet; this is the price of
    that choice. Order is compared too: `aggregation: "concat"` lays member k's draws at fixed
    sample-axis positions, and chrome_box reconciles position-wise against them.
    """
    match = re.search(r"^MODELS=\"([^\"]+)\"", script, re.M)
    assert match, "the script declares no MODELS list"
    declared = match.group(1).split()

    config = (REPO / "ensembles" / ENSEMBLE / "configs" / "config_modelset.py").read_text()
    in_config = re.findall(r'"(\w+_condor)"', config)
    assert in_config, f"found no members in {ENSEMBLE}/configs/config_modelset.py"
    assert declared == in_config, (
        "the runner's MODELS list disagrees with "
        f"ensembles/{ENSEMBLE}/configs/config_modelset.py.\n"
        f"  runner: {declared}\n  config: {in_config}\n"
        "Order is load-bearing, not only membership: concat pooling places member k at fixed "
        "sample-axis positions and chrome_box reconciles against them position-wise."
    )


def test_phase_two_is_declared_absent_rather_than_half_present(script):
    """chrome_box is not here, and the script must say so rather than leave it ambiguous.

    chrome_box at 8 x 128 = 1024 draws is the shape rusty_bucket's own config records as
    peaking ~28.6 GB. Its draw count is an open decision and it determines the machine class,
    so a memory floor for it cannot be written yet.
    """
    assert "chrome_box" in script, (
        "the runner never mentions chrome_box. Phase 2's absence should be explicit — an "
        "operator reading this script needs to know the pgm half is not covered."
    )
    offenders = [
        line.strip()
        for line in script.splitlines()
        if "chrome_box" in line and "main.py" in line and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "the runner actually invokes chrome_box. Phase 2 needs the draw count settled first, "
        "and a memory floor derived from a configuration nobody has chosen is a guess with a "
        "price tag:\n" + "\n".join(f"  {line}" for line in offenders)
    )


def test_each_floor_is_labelled_with_how_it_was_arrived_at(script):
    """One floor is measured and one is reasoned, and each must say which it is.

    `feedback_never_state_an_unmeasured_cost`: estimates here have been one-sidedly optimistic
    9 times out of 9, so labelling is what lets a real run replace them.

    ⛔ This replaces `test_the_derived_floors_are_labelled_as_derived`, which asserted
    `script.count("DERIVED") >= 2` on the premise that *both* floors were reasoned. The
    memory floor was measured on 2026-10-28 (96 MB pool, 176 MB members, ~2 MB peak RSS
    each, 7 minutes, 87 GiB box never strained), and the comment block was updated to say
    so — while the line the operator actually READS still printed "DERIVED not measured".
    A count-based assertion could not tell the two floors apart, so it held the stale
    label in place: the guard was the reason the defect survived its own correction.

    The check is now per-floor and two-directional, because mislabelling is harmful in
    both: a guess that claims measurement invites over-trust, and a measurement that
    claims to be a guess invites someone to "replace" a real number with another guess.
    """
    for floor in ("MEM_FLOOR_GB", "DISK_FLOOR_GB"):
        assert floor in script, f"{floor} is not set"

    mem_line = next(
        (ln for ln in script.splitlines() if "MEM_FLOOR_GB}" in ln and "echo" in ln), None
    )
    disk_line = next(
        (ln for ln in script.splitlines() if "DISK_FLOOR_GB}" in ln and "echo" in ln), None
    )
    assert mem_line and disk_line, (
        "could not find the lines that report the two floors to the operator, so this "
        "guard cannot see its subject and refuses rather than passing (ADR-025)"
    )

    assert "MEASURED" in mem_line and "DERIVED" not in mem_line, (
        f"the memory floor is reported as derived: {mem_line.strip()!r}. It was measured "
        f"on 2026-10-28 and the comment block above it says so; an operator reads this "
        f"line, not the comment."
    )
    assert "DERIVED" in disk_line, (
        f"the disk floor is no longer labelled derived: {disk_line.strip()!r}. It has "
        f"still never been observed — only the memory floor has. Label it, or measure it."
    )


def test_bad_invocations_are_refused_before_anything_is_touched(tmp_path):
    """A positional argument is a sign the operator expects a different script."""
    done = subprocess.run(
        ["bash", str(RUNNER), "some_model"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path)},
    )
    assert done.returncode != 0, "a positional argument was accepted"
    assert "no positional arguments" in (done.stdout + done.stderr), (
        "the refusal does not explain that the chain is fixed:\n" + done.stdout + done.stderr
    )
    assert not (tmp_path / "venv-crafd").exists(), "a bad invocation still created a venv"


def test_preflight_refuses_a_missing_checkout_without_installing(tmp_path):
    """Preflight must cost seconds and touch nothing. Run for real, not asserted from text."""
    done = subprocess.run(
        ["bash", str(RUNNER), "--preflight"],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path)},
    )
    combined = done.stdout + done.stderr
    assert done.returncode != 0, f"preflight passed with no checkout present:\n{combined}"
    assert "is not a checkout" in combined, f"preflight did not name the missing checkout:\n{combined}"
    assert not (tmp_path / "venv-crafd").exists(), "preflight created a venv"
    status = tmp_path / "deliver" / ENSEMBLE / "STATUS"
    assert status.is_file() and status.read_text().strip() == "FAILED:preflight", (
        "a failed preflight did not record FAILED:preflight in STATUS"
    )


def test_preflight_reports_every_problem_not_only_the_first(tmp_path):
    """One round trip, not six. The reason --preflight exists at all."""
    done = subprocess.run(
        ["bash", str(RUNNER), "--preflight"],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "PODRUN_ROOT": str(tmp_path)},
    )
    combined = done.stdout + done.stderr
    assert combined.count("MISSING:") >= 2, (
        "preflight reported fewer than two problems on a bare directory, so it is stopping at "
        f"the first one:\n{combined}"
    )


def test_a_run_clears_the_previous_runs_verdict(script):
    """STATUS must not outlive its run.

    Found on a live pod 2026-10-28: a `FAILED:preflight` written at 11:58:46 — before the
    credential was placed — was still being served at 12:00 while the install was healthily
    in progress. An operator reading STATUS, and the poller watching it, both concluded the
    run had failed. STAGE is the live field; STATUS is a verdict, and a stale verdict is worse
    than none (`feedback_status_fields_lie`).
    """
    cleared = [
        line.strip()
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
        and re.search(r"rm\s+-f\s+.*\$OUT/STATUS", line)
    ]
    assert cleared, (
        "the runner never clears $OUT/STATUS at startup, so a previous run's verdict is "
        "served during this one. Add `rm -f \"$OUT/STATUS\" \"$OUT/FAILURE\"` after $OUT is "
        "created and before the first stage."
    )
    body = script.splitlines()
    clear_index = next(
        i for i, line in enumerate(body)
        if not line.lstrip().startswith("#") and re.search(r"rm\s+-f\s+.*\$OUT/STATUS", line)
    )
    first_stage = next(
        i for i, line in enumerate(body)
        if not line.lstrip().startswith("#") and re.match(r"^\s*stage\s+\w", line)
    )
    assert clear_index < first_stage, (
        "STATUS is cleared after the first stage begins, so there is a window in which the "
        "previous run's verdict is served alongside this run's STAGE."
    )


def test_only_a_finished_run_writes_OK_to_status(script):
    """STATUS is what an operator greps after a long run. It must not lie."""
    ok_writes = [
        line.strip()
        for line in script.splitlines()
        if re.search(r"echo\s+OK\s*>\s*\"?\$OUT/STATUS", line) and not line.lstrip().startswith("#")
    ]
    assert len(ok_writes) == 1, (
        f"STATUS is set to OK in {len(ok_writes)} places; it must be written once, after the "
        f"manifest: {ok_writes}"
    )
    body = script.splitlines()
    ok_index = next(i for i, line in enumerate(body) if re.search(r"echo\s+OK\s*>\s*\"?\$OUT/STATUS", line))
    pool_index = max(i for i, line in enumerate(body) if "main.py" in line and not line.lstrip().startswith("#"))
    assert ok_index > pool_index, (
        "STATUS is set to OK before the last run invocation, so a failure after that point "
        "would leave a run reporting success."
    )
