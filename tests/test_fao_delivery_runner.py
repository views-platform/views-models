"""
Guards for `tools/podrun/pod_run_fao_delivery.sh` — the FAO delivery chain (views-models#499
Track B), which until 2026-09-29 existed only as commands typed by hand on a rented pod.

These are written after a /falsify guard-mode audit found that 10 of 14 guards for the
sibling script were DECORATIVE: they read the script as text, and the script's own comments
contained the strings they asserted, so deleting the code left the guard green. So here:

  * the argument parser is EXECUTED, not matched;
  * `--preflight` is EXECUTED, which is the whole mode's value — it must refuse on a machine
    that cannot deliver, and it must refuse for the right reason;
  * text assertions read COMMENT-STRIPPED source, and are limited to ordering facts that
    cannot be executed here (the steps themselves need a GPU, a store and a partner bucket).

What these guards cannot cover, stated rather than implied: no test here runs a forecast,
publishes, or reaches the FAO. The first real exercise of this script is a rehearsal on a
pod, and that is what `--rehearsal` and `--preflight` exist to make cheap.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FAO = REPO / "tools" / "podrun" / "pod_run_fao_delivery.sh"

HYDRANETS = {
    "purple_alien", "pink_pirate", "blue_stranger", "bright_starship",
    "heavy_freighter", "blazing_meteor", "bold_comet", "violet_visitor",
}


def _code_only(src: str) -> str:
    out = []
    for line in src.splitlines():
        stripped = re.sub(r"(?<![$\\])#.*$", "", line)
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


def _run(*args, env=None, root=None):
    """Run the script. Every case below is expected to exit in the parser or in preflight.

    PODRUN_ROOT relocates the workspace so this is exercisable off a pod. It does not relax
    any check: the credential, conda, region, GPU and disk checks all still run and all still
    fail here, which is exactly what these tests assert.
    """
    e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"}
    if root is not None:
        e["PODRUN_ROOT"] = str(root)
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", str(FAO), *args], capture_output=True, text=True, timeout=120, env=e,
    )


class TestTheArgumentParser:
    """Executed. Deleting a case arm left every text guard on the sibling script green."""

    def test_rehearsal_requires_a_count(self):
        r = _run("--rehearsal")
        assert r.returncode != 0
        assert "lesson count" in r.stdout + r.stderr

    def test_rehearsal_refuses_a_non_integer(self):
        r = _run("--rehearsal", "forty")
        assert r.returncode != 0
        assert "positive integer" in r.stdout + r.stderr

    def test_rehearsal_refuses_zero(self):
        r = _run("--rehearsal", "0")
        assert r.returncode != 0
        assert "nothing to rehearse" in r.stdout + r.stderr

    def test_an_unknown_option_is_refused(self):
        r = _run("--bogus")
        assert r.returncode != 0
        assert "unknown option" in r.stdout + r.stderr

    def test_a_stray_positional_is_refused(self):
        """This script takes no model name — it runs the whole roster. A stray argument is
        far more likely to be a mistake than an intention, and silently ignoring it on a
        script that PUBLISHES to a partner is not a good trade."""
        r = _run("purple_alien")
        assert r.returncode != 0
        assert "unexpected argument" in r.stdout + r.stderr


class TestPreflightRefusesAMachineThatCannotDeliver:
    """Executed on THIS machine, which is not a pod — so preflight must refuse, and the
    refusal must name what is missing.

    This is the mode that protects the money: at 300 lessons step 1 alone is ~16 GPU hours,
    and every condition it checks is knowable in seconds beforehand.
    """

    @pytest.fixture
    def result(self, tmp_path):
        return _run("--preflight", root=tmp_path)

    def test_it_refuses_rather_than_proceeding(self, result):
        assert result.returncode != 0, (
            "preflight passed on a machine with no pod checkout, no /workspace venv and no "
            "publish credentials. It would then have proceeded to spend GPU hours."
        )

    def test_it_builds_the_publish_config_rather_than_counting_variables(self):
        """The check that would have caught the real failure.

        PredictionStoreConfig.from_environment() requires NINE variables — the three secrets
        plus PROD_FORECASTS_BUCKET_ID/NAME, PROD_FORECASTS_COLLECTION_ID/NAME and
        METADATA_DATABASE_ID/NAME. The first version of this preflight checked the three
        secrets, so it would have reported ready with six of nine missing and the run would
        have failed at the publish, after the whole roster trained.

        Counting variables cannot be made correct by adding six more names either: the extra
        can be absent, the endpoint unreachable, the key expired (#359 — 2026-11-17). Only
        constructing the thing answers the question.
        """
        code = _code_only(FAO.read_text())
        assert "PredictionStoreConfig" in code and "from_environment()" in code, (
            "preflight must CONSTRUCT the publish config, not enumerate variable names"
        )
        assert "import appwrite" in code, (
            "preflight must also confirm the appwrite extra is importable — variables can all "
            "be set while the SDK that uses them is missing (#517)"
        )

    def test_it_tells_the_operator_the_variables_must_be_exported(self):
        """pipeline-core stopped auto-loading a .env from the working directory (#346, C-177),
        so a correct .env sitting beside the operator is not enough. Without this line the
        failure looks like wrong credentials rather than unexported ones."""
        code = _code_only(FAO.read_text())
        assert re.search(r"set -a", code) and re.search(r"C-177|#346", FAO.read_text()), (
            "preflight must say the nine variables need exporting, and name why"
        )

    def test_it_names_the_missing_publish_secrets(self, result):
        out = result.stdout + result.stderr
        for var in ("APPWRITE_ENDPOINT", "APPWRITE_DATASTORE_PROJECT_ID",
                    "APPWRITE_DATASTORE_API_KEY"):
            assert var in out, (
                f"{var} is not named. views-pipeline-core's PredictionStoreConfig claims to "
                "check these at startup and does not — it is called from _build_datastore, "
                "after training (views-pipeline-core#557). Until that moves, this preflight "
                "is the only check that happens before the money is spent."
            )

    def test_it_checks_conda_because_the_postprocessor_needs_it(self, result):
        """The two legs need different interpreters: this pod builds a uv venv, while
        tools/launcher/postprocessor.sh uses `conda shell.bash hook`. A pod can satisfy one
        and not the other, and the failure would land after every GPU hour was spent."""
        code = _code_only(FAO.read_text())
        assert "command -v conda" in code, (
            "nothing checks for conda. The un_fao postprocessor launcher requires it "
            "(tools/launcher/postprocessor.sh:72) and step 4 is the last step."
        )

    def test_the_disk_floor_is_not_below_the_floor_it_delegates_to(self):
        """Found by a falsification pass on the forecasting claim.

        This script delegates to pod_run_model.sh eight times, and that script refuses below
        40GB for ONE model. A floor below 40 here is a preflight that reports ready and then
        refuses at model 3 — the exact opposite of why this mode exists. The first version
        said 60GB for all eight, which was lower than the per-model transient for a run eight
        times the size.
        """
        fao = _code_only(FAO.read_text())
        model = _code_only((REPO / "tools" / "podrun" / "pod_run_model.sh").read_text())
        delegated = re.search(r'AVAIL_GB" -ge (\d+)', model)
        assert delegated, "pod_run_model.sh's disk floor moved; re-read it"
        mine = re.search(r"DISK_FLOOR_GB=(\d+)", fao)
        assert mine, "this script no longer declares DISK_FLOOR_GB"
        assert int(mine.group(1)) >= int(delegated.group(1)), (
            f"this orchestrator demands {mine.group(1)}GB for eight models while the script it "
            f"delegates to demands {delegated.group(1)}GB for one. Preflight would pass and a "
            "later model would refuse, after hours."
        )

    def test_it_checks_the_appwrite_coordinate_registry(self, result):
        """Found during the first rehearsal, with four of eight models already trained.

        The postprocessor leg is fatal without the Appwrite coordinate registry —
        `platform_env_require_registry()` says "the registry is the ONLY source of coordinates"
        (#308) — and its default path is a relative hop to a SIBLING checkout of views-appwrite,
        which a pod that cloned only views-models does not have. It would have failed at step 4,
        the last one, after every GPU hour was spent.

        The REGION check passing is not evidence for this: that resolves a COVERAGE declaration,
        a different registry entirely. Two registries, two checks.
        """
        out = result.stdout + result.stderr
        code = _code_only(FAO.read_text())
        assert "coordinate_registry.toml" in code, (
            "preflight does not check for the Appwrite coordinate registry"
        )
        assert "APPWRITE_REGISTRY" in code, (
            "preflight must name the override, or an operator on a machine with a different "
            "layout has no way to satisfy the check"
        )
        assert "registry" in out.lower(), (
            f"preflight ran on a machine with no registry and said nothing about it.\n{out}"
        )

    def test_conda_is_probed_for_CAPABILITY_not_presence(self):
        """The 2026-09-30 delivery failed at step 4 — the last — because miniconda 26.7.1 will
        not create an environment until its channel Terms of Service are accepted. Preflight
        had checked `command -v conda`, which said nothing about that. The binary existing is
        not the property the postprocessor needs."""
        code = _code_only(FAO.read_text())
        assert "conda create --dry-run" in code, (
            "preflight must probe that conda can CREATE an environment. `command -v conda` "
            "passed on the pod that then failed at the last step."
        )
        assert "tos accept" in code, (
            "the refusal must name the remedy — an operator who hits the ToS gate at 4am "
            "should not have to find the two commands themselves"
        )

    def test_it_reports_every_problem_not_only_the_first(self, result):
        """A preflight that dies on the first missing thing costs a round trip per problem,
        on hardware billed by the second."""
        out = result.stdout + result.stderr
        assert out.count("MISSING:") >= 2, (
            f"preflight reported {out.count('MISSING:')} problems and stopped. It should "
            "accumulate and report them together.\n" + out
        )

    def test_preflight_only_never_reaches_a_run_stage(self, result):
        out = result.stdout + result.stderr
        for forbidden in ("pool_and_publish", "un_fao_postprocessor", "forecast:"):
            assert forbidden not in out, f"--preflight reached stage {forbidden}"


class TestTheChainIsInTheRightOrderWithTheRightFlags:
    """Ordering facts, asserted on comment-stripped source because executing them needs a
    GPU, a prediction store and a partner bucket."""

    @pytest.fixture(scope="class")
    def code(self):
        return _code_only(FAO.read_text())

    def test_the_ensemble_pools_from_saved_member_forecasts(self, code):
        """Without -sa/--saved the ensemble refetches instead of pooling what the eight runs
        just produced, and those GPU hours are wasted."""
        m = re.search(r"main\.py -r forecasting[^\n|]*", code)
        assert m, "the ensemble invocation is not in the expected shape"
        assert " -sa" in m.group(0), f"--saved is missing from: {m.group(0)}"
        assert " -f" in m.group(0), f"--forecast is missing from: {m.group(0)}"
        assert " -p" in m.group(0), f"--prediction_store is missing from: {m.group(0)}"

    def test_the_models_run_before_the_pool_which_runs_before_the_postprocessor(self, code):
        forecast = code.index("--forecast")
        pool = code.index("main.py -r forecasting")
        postproc = code.index("postprocessors/un_fao/run.sh")
        assert forecast < pool < postproc, (
            "the chain is out of order: the eight forecasts must precede the pool, and the "
            "pool must precede the postprocessor that curates land -> land_gaul"
        )

    def test_the_whole_roster_is_named(self, code):
        declared = set(re.search(r'MODELS="([^"]+)"', code).group(1).split())
        assert declared == HYDRANETS, (
            f"the roster here is {sorted(declared)}; rusty_bucket's members are "
            f"{sorted(HYDRANETS)}. Pooling a partial roster changes the forecast silently."
        )

    def test_a_model_that_did_not_report_OK_stops_the_delivery(self, code):
        assert 'STATUS" 2>/dev/null)" = "OK"' in code, (
            "the orchestrator must check each model's STATUS before pooling. An exit code "
            "alone does not distinguish 'trained' from 'wrote a partial forecast'."
        )

    def test_wandb_is_forced_offline_on_the_ensemble_leg(self, code):
        """Without this, main.py calls wandb.login() and blocks on a prompt no one is
        watching. The first forecasting run on a pod died exactly there."""
        tail = code[code.index("pool_and_publish"):]
        assert "WANDB_MODE=offline" in tail


class TestThePosteriorIsInspectedBeforeTheDeliveryIsCalledDone:
    """The 2026-09-30 delivery passed every structural check and served a posterior whose point
    estimate was zero for all 2,333,448 cells. Nothing in the chain said so."""

    @pytest.fixture(scope="class")
    def code(self):
        return _code_only(FAO.read_text())

    def test_posterior_health_runs_on_every_delivery(self, code):
        assert "tools.prereg.posterior_health" in code, (
            "nothing reports whether the pooled posterior contains anything. A valid manifest "
            "over an empty posterior passes every other check in the chain."
        )

    def test_the_health_check_is_told_which_mode_the_run_is(self, code):
        """The numbers cannot distinguish an expected rehearsal from a broken production run."""
        seg = code[code.index("tools.prereg.posterior_health"):]
        assert "--mode" in seg[:300], (
            f"posterior_health is invoked without --mode:\n{seg[:300]}"
        )

    def test_anchors_are_captured_by_the_runner_not_by_hand(self, code):
        """P8's protection is that anchors are not chosen after seeing the API's answer. Running
        it from the runner makes that true by construction; running it by hand afterwards makes
        it true only if nobody looked first."""
        assert "tools.prereg.capture_anchors" in code

    def test_the_posterior_is_inspected_before_the_postprocessor_stage(self, code):
        """It must run as soon as the pooled frame exists, not after the last step — otherwise a
        failure in the postprocessor buries the one signal that the numbers were empty."""
        assert code.index("posterior_health") < code.index("un_fao_postprocessor")


class TestARehearsalIsMarkedEverywhereItCanBe:
    """A rehearsal reaches the FAO shelf, because nothing downstream refuses one (#523). So
    the marking is the only protection, and it has to be loud."""

    @pytest.fixture(scope="class")
    def code(self):
        return _code_only(FAO.read_text())

    def test_the_rehearsal_flag_reaches_the_per_model_runs(self, code):
        assert "REH_ARGS" in code and "--rehearsal $REHEARSAL_LESSONS" in code, (
            "the lesson count must be forwarded to each model run, or the models train in "
            "full while the delivery is labelled a rehearsal"
        )

    def test_the_marker_admits_the_forecasts_reached_the_shelf(self, code):
        """The dangerous half. A reader who sees 'REHEARSAL' may assume it was contained;
        it was not, and the file has to say so."""
        assert "ON THE FAO SHELF" in code, (
            "the rehearsal marker must state that the undertrained forecasts ARE published "
            "and servable, not merely that the run was a rehearsal"
        )

    def test_the_marker_is_cleared_at_the_start_of_every_run(self, code):
        clear = code.index('rm -f "$OUT/STATUS"')
        assert "REHEARSAL" in code[clear:clear + 120], (
            "a production delivery must not inherit a previous rehearsal's marker, and far "
            "worse, a rehearsal must not inherit a production run's absence of one"
        )

    def test_what_landed_is_read_back_by_name(self, code):
        """`tools.liveness` answers 'is it there?' rather than 'did we send it?'. On
        2026-09-29 the publish reported success having written nothing (C-155)."""
        assert "tools.liveness" in code
