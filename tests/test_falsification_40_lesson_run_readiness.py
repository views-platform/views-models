"""
Falsification audit of the claim, 2026-09-29:
"We are ready to start a new full 40-lesson FAO delivery run — nothing remains but
creating a pod."

Source: /falsify skill, claim mode. Verdict: FALSIFIED, three hard falsifications.

All three had ONE cause, and it is the finding worth keeping: every fix that made the
2026-09-29 run work was applied BY HAND to the pod, and the pod was destroyed. The
repository never learned any of them. A fresh pod reproduced the original broken state.
The environment that produced the first FAO delivery existed only as a running machine —
the pod was the artefact, and nothing in git described it.

  H1  the run could not be launched at all: pod_run_model.sh refuses total_lessons < 300
      and every config declares 300, so "40 lessons" was reachable only by editing the
      guard out — which produced output indistinguishable from a real run.
  H2  a fresh env built SUCCESSFULLY and WRONG: the declared requirements resolved to
      xarray 2025.12.0 / pandas 3.0.6 where the working pod had 2024.3.0 / 1.5.3, and
      neither was pinned anywhere. views-models#516 called this "unbuildable"; it was not,
      and the silent version is the more dangerous of the two.
  H3  `appwrite` appeared nowhere in pod_run_model.sh, so `_build_datastore` failed at
      publish — after the full training run. That is how 2026-09-29 failed (#517).

These tests now GUARD those three fixes rather than assert the defects. They are kept as a
single file because they share one cause: if a fourth instance appears, it belongs here.

Two assertions in this file were written as predicted falsifications and SURVIVED — the
toolz ordering and the guide's chmod warning. They are retained deliberately. A
falsification file containing only the author's successful predictions is a record of an
argument, not of an audit.

ONE ITEM IS DEFERRED, not fixed: a rehearsal's output is MARKED unfit to deliver but
nothing REFUSES to publish it. That guard belongs in views-pipeline-core, where the publish
happens, so this file cannot test it. See TestARehearsalCannotBeMistakenForADelivery.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PODRUN = REPO / "tools" / "podrun" / "pod_run_model.sh"

HYDRANETS = (
    "purple_alien", "pink_pirate", "blue_stranger", "bright_starship",
    "heavy_freighter", "blazing_meteor", "bold_comet", "violet_visitor",
)


class TestACheapRunIsReachableWithoutEditingATrackedFile:
    """H1. The floor was right; its absence of an escape hatch was not.

    The >=300 floor protects against spending a full budget by ACCIDENT — a config left at
    a sweep value, a stale checkout. It was never meant to forbid a deliberate cheap
    end-to-end test, which is routine and is exactly what you want after a run that failed
    late. Before `--rehearsal` the only way to get one was to delete the guard, and the
    result looked identical on disk to a real run: the cheap test and the accident were
    indistinguishable. That, not the wasted GPU time, was the hazard.
    """

    def test_the_rehearsal_flag_exists_and_takes_a_lesson_count(self):
        src = PODRUN.read_text()
        assert "--rehearsal" in src, "no escape hatch: a cheap end-to-end test needs the guard deleted"
        assert "REHEARSAL_LESSONS" in src, (
            "--rehearsal must carry the lesson count on the command line. main.py has no "
            "hyperparameter override, so a flag without a count still forces an edit to a "
            "tracked config — the hand-edit this audit exists to remove."
        )

    def test_the_production_floor_is_still_enforced(self):
        src = PODRUN.read_text()
        assert re.search(r"if not requested and lessons < 300:", src), (
            "the >=300 floor must still apply when --rehearsal is NOT given. Adding the "
            "escape hatch must not widen the hole it was protecting."
        )

    @pytest.mark.parametrize("model", HYDRANETS)
    def test_tracked_configs_still_declare_the_production_count(self, model):
        """A rehearsal patches the pod's clone. The committed config stays production truth."""
        cfg = REPO / "models" / model / "configs" / "config_hyperparameters.py"
        hit = re.search(r"'total_lessons':\s*(\d+)", cfg.read_text())
        assert hit is not None, f"{model}: no total_lessons in {cfg}"
        assert int(hit.group(1)) >= 300, (
            f"{model} declares total_lessons={hit.group(1)}. A rehearsal must never be "
            "obtained by committing a low count — that is the state this audit found and "
            "the reason --rehearsal patches only the pod's ephemeral checkout."
        )

    def test_a_leftover_rehearsal_patch_is_named_not_blamed_on_the_config(self):
        """Found reviewing this change, not in the original audit.

        Section 2 does not re-clone when `.git` already exists, so a production run started
        on a pod that has already rehearsed reads the LEFTOVER patch. The floor still refuses
        it — but it would blame the committed config and advise `--rehearsal`, which is the
        exact opposite of what the operator wants. A correct refusal for the wrong reason
        sends someone to edit the wrong file.
        """
        src = PODRUN.read_text()
        assert "is MODIFIED in this checkout" in src, (
            "a production run must detect a leftover rehearsal patch and name it, rather "
            "than reporting the patched value as though it were the committed config"
        )

    def test_the_patch_is_verified_by_re_reading_the_config(self):
        """A substitution that silently missed would produce a 300-lesson run wearing a
        rehearsal label, or a rehearsal wearing none. Both are worse than a refusal."""
        src = PODRUN.read_text()
        assert "importlib.invalidate_caches()" in src and "still reports total_lessons" in src, (
            "the config patch must be confirmed by re-importing the config, not trusted "
            "from re.subn's return value alone"
        )


class TestFreshEnvironmentReproducesTheOneThatWorked:
    """H2 (views-models#516, reframed). Measured, not reasoned about.

        declared:  views-datafactory>=1.9.0,<2.0.0 + numpy>=1.26.4,<2.0.0
        resolved:  numpy 1.26.4, xarray 2025.12.0, pandas 3.0.6
        the pod that delivered:  xarray 2024.3.0, pandas 1.5.3

    views-datafactory requires xarray outright and pandas only in an optional extra that is
    not installed, so pandas arrives THROUGH xarray — xarray is the sole carrier, as
    datafactory's own pyproject comment says. Pinning the carrier fixes it.

    The bound is measured: xarray 2024.3.0 is the LAST release accepting pandas 1.x; 2024.5.0
    moved to pandas>=2.0 and 2024.9.0 to pandas>=2.1. A tidy `<2025` cap would have been
    WRONG — the cliff is inside the 2024 line, not at the year boundary.
    """

    @pytest.mark.parametrize("postprocessor", ["un_fao", "un_crafd"])
    def test_the_carrier_and_the_passenger_are_both_pinned(self, postprocessor):
        req = REPO / "postprocessors" / postprocessor / "requirements.txt"
        text = req.read_text()
        missing = [p for p in ("xarray", "pandas") if not re.search(rf"^{p}[><=~]", text, re.M)]
        assert not missing, (
            f"{req.relative_to(REPO)} does not pin {missing}. Unpinned, this file resolves "
            "to pandas 3.0.6 — a different MAJOR from every run that has ever succeeded, "
            "with no error. See views-models#516."
        )

    def test_both_postprocessors_declare_identical_pins(self):
        """C-116: one shared prefix. Pinning one lets whichever runs last decide for both."""
        pins = {}
        for name in ("un_fao", "un_crafd"):
            text = (REPO / "postprocessors" / name / "requirements.txt").read_text()
            pins[name] = sorted(
                re.findall(r"^((?:numpy|pandas|xarray)[><=~][^\s#]*)", text, re.M)
            )
        assert pins["un_fao"] == pins["un_crafd"], (
            f"un_fao pins {pins['un_fao']} but un_crafd pins {pins['un_crafd']}. Both "
            "install into envs/views-postprocessing (C-116); divergent pins mean the last "
            "postprocessor to run silently decides the versions for the other."
        )


class TestThePodCanActuallyPublish:
    """H3 (views-models#517). The publish path has to be installed and PROVEN at preflight.

    Verified by a real resolve on 2026-09-29: requesting the extra yields appwrite 13.6.1
    alongside views-hydranet 0.1.2 and views-pipeline-core 3.3.4 — both of which carry the
    fixes shipped that day — with pandas 1.5.3 and numpy 1.26.4 intact.
    """

    def test_the_appwrite_extra_is_installed(self):
        src = PODRUN.read_text()
        assert "views-pipeline-core[appwrite]" in src, (
            "the appwrite extra is never installed, so the run trains for hours and then "
            "cannot publish. This is the failure of 2026-09-29. See views-models#517."
        )

    def test_the_publish_path_is_checked_before_the_money_is_spent(self):
        """An import in preflight costs seconds. Discovering it at publish costs the run."""
        src = PODRUN.read_text()
        verify = src.split("stage verify_env", 1)[-1].split("stage", 1)[0]
        assert "appwrite" in verify, (
            "pod_run_model.sh verifies torch, tlz, hydranet and datafactory at preflight but "
            "not the Appwrite client. The whole script is written to fail early; the one "
            "dependency that failed late was the one not checked here."
        )

    def test_the_toolz_override_is_reasserted_after_the_last_install(self):
        """GREEN, and it survived as a prediction. Confirmed necessary, not folklore: the
        resolver really does land on toolz 0.11.2 for this dependency set."""
        src = PODRUN.read_text()
        installs = [m.start() for m in re.finditer(r"pip install", src)]
        overrides = [m.start() for m in re.finditer(r"toolz>=0\.12", src)]
        assert overrides, "the C-151 toolz override is gone entirely"
        assert max(overrides) > max(installs), (
            "the C-151 toolz override must be the LAST install: anything after it "
            "re-resolves the prefix and can pull toolz back under 0.12, which happened by "
            "hand on 2026-09-29 (1.1.0 -> 0.11.2) with no error."
        )


class TestARehearsalCannotBeMistakenForADelivery:
    """The half that actually prevents harm.

    A rehearsal's parquets are structurally identical to a production run's — same columns,
    row counts, names, all finite and non-negative. Every structural check passes. Nothing
    in the DATA says the model behind it is undertrained.
    """

    def test_the_run_marks_its_output(self):
        src = PODRUN.read_text()
        assert '"$OUT/REHEARSAL"' in src and "NOT FIT TO DELIVER" in src, (
            "a rehearsal must leave an unmistakable marker beside its parquets"
        )

    def test_a_stale_marker_cannot_be_inherited(self):
        """Worse than a stale marker on a real run: a rehearsal inheriting a production
        run's ABSENCE of one."""
        src = PODRUN.read_text()
        assert 'rm -f "$OUT/REHEARSAL"' in src, (
            "the marker must be cleared with the other stale per-run state, or a rehearsal "
            "in a directory left by a production run carries no mark at all"
        )

    def test_the_manifest_admits_the_config_was_patched(self):
        src = PODRUN.read_text()
        assert "PATCHED after checkout" in src, (
            "on a rehearsal the MANIFEST's `git:` sha no longer describes the run — the "
            "config was patched after checkout. A manifest that must be cross-read with a "
            "command-line flag to be understood will be misread."
        )

    # NOT TESTED HERE, DELIBERATELY. pod_run_model.sh MARKS a rehearsal; it cannot REFUSE
    # to publish one, because the publish step is not in this script — the header is accurate
    # that this runner uploads nothing. The refusal belongs wherever a forecast is handed to
    # a store, which is views-pipeline-core, so no assertion in THIS repo can observe it.
    #
    # An xfail(strict) stub was written here and deleted: its regex spanned the whole file
    # under re.S and XPASSed by accident, i.e. it was the exact class of guard-that-cannot-
    # fire this audit exists to find. A test that cannot observe its subject is worse than no
    # test, because it reports coverage. Tracked as an issue instead; see the module
    # docstring. Until that guard exists, $OUT/REHEARSAL is the only thing between an
    # undertrained model and a partner, and that is weaker than it should be.


class TestPublishCredentialsHaveADurableHome:
    """GREEN (survived), and the prediction was wrong in the useful direction: the guide
    DOES record the /workspace chmod trap, so the operator-facing half of C-154 / #518 is
    already durable. What remains is that placement is still a manual step with a correct
    runbook — an operator task, not a defect, and not counted as a falsification."""

    def test_the_guide_warns_that_workspace_ignores_chmod(self):
        guide = REPO / "docs" / "runpod_run_guide.md"
        assert re.search(r"/workspace.*chmod|chmod.*/workspace", guide.read_text(), re.I | re.S), (
            "/workspace is a network filesystem that reports chmod success and leaves the "
            "file world-readable. C-154 / views-models#518."
        )
