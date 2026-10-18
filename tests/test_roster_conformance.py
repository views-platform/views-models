"""Roster conformance — the 8 HydraNet models and the rusty_bucket ensemble conform
to the Epic #242 roster and the shared v2 ``gated_NB`` foundation.

This supersedes ``test_datafactory_parity.py`` (the viewser↔datafactory trio mirror,
C-47), which is deleted in the same change. The two cannot coexist: the parity suite
asserts ``loss_reg == "tobit"``, a non-empty ``loss_reg_sigma`` and ``loss_class ==
"focal"``, while the roster foundation below sets ``mse``, no sigma, and
``weighted_bce``. The parity programme's premise — two 3-member trios differing only
in data source — no longer holds now that every member reads views-datafactory.

Roster (pre-registration 05, LOCKED 2026-08-08):

  gated_NB     (nb,         soft_gate)           violet_visitor 42 / bright_starship 43 / bold_comet 44
  th_gated_NB  (nb,         threshold_gate 0.5)  blazing_meteor 45 / heavy_freighter 46
  mixture_NB   (mixture_nb, soft_gate)           pink_pirate 42 / blue_stranger 43 / purple_alien 44

That table is the pre-registration as locked, kept as the record. It is SUPERSEDED: the
2026-09 reconfiguration (#463) moved families, compositions and seeds, and #466 replaced
the 0.5 gate threshold with per-member priors. ``ROSTER`` below is the live declaration.

Cross-repo references are qualified because a bare ``#`` number resolves against THIS
repository and would point at something unrelated: the roster is
**views-hydranet#246**, the family head is **views-hydranet ADR-067** (still *Proposed*
there), the ensemble gate-pooling fix is **views-pipeline-core#422**.

See ``reports/2026-08-08_hydranet_ensemble_dossier/05_analysis_plan.md`` and register
C-71 (violet reconstruction) / C-132 (ensemble gate pooling).

**What this file deliberately does NOT contain.** Two things, each for its own reason.
The fleet-wide C-132 guard and ``rusty_bucket``'s ``classification_targets`` assertion
live with the C-132 work, gated on views-pipeline-core#422 shipping and being pinned
here — asserting the pool carries the gate before the framework can honour it would be
a green test for a thing that does not happen. And the ``rusty_bucket`` membership
rewiring waits on violet_visitor settling; see the note at the foot of this file.
"""

import ast
import difflib
import importlib.util
from pathlib import Path

import re

import pytest

from tests.conftest import get_regression_targets

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"

pytestmark = [pytest.mark.green]

# --- The roster: model -> (output_distribution, forecast_composition,
#     gate_threshold, seed). gate_threshold is None for soft_gate. ---
# REVISED 2026-09-07 for the pre-deployment validation run (views-hydranet #324,
# reports/2026-09-06_ensemble_roster_dossier/04_run_plan.md). Chosen from a 20-arm emit over the
# previous roster: 8 models x 2 compositions with the cell clamp, scored on 3 targets x 6 horizons.
#
# What the measurement supported:
#   * mixture_nb + soft_gate + scheduled sampling OFF was best on BOTH axes (ranking AND mass
#     landing on real event cells) -- the old `purple_alien` configuration, now replicated.
#   * threshold_gate ranks slightly better and predicts measurably LESS on every model. A real
#     trade, so both compositions are carried rather than one being picked.
#   * `heavy_freighter` keeps scheduled sampling ON deliberately: it was the only configuration
#     predicting fatalities at a realistic scale, and that is the configuration that produced it.
#     With `bright_starship` (same family, same composition, ss OFF) it is the closest thing to
#     a controlled contrast in the design -- but NOT a one-variable pair: the seeds differ too
#     (47 vs 43), so any difference between the two runs is confounded with seed.
#
# What is NOT evidenced and is assumed: the 5/3 family split, the specific seeds, and that eight
# distinct trainings beat fewer models carrying more compositions.
ROSTER = {
    "purple_alien":    ("mixture_nb", "soft_gate", None, 44),
    "pink_pirate":     ("mixture_nb", "soft_gate", None, 42),
    "blue_stranger":   ("mixture_nb", "soft_gate", None, 43),
    "bold_comet":      ("mixture_nb", "threshold_gate", 0.14, 45),
    "blazing_meteor":  ("mixture_nb", "threshold_gate", 0.16, 46),
    "heavy_freighter": ("nb", "soft_gate", None, 47),
    "bright_starship": ("nb", "soft_gate", None, 43),
    "violet_visitor":  ("nb", "threshold_gate", 0.20, 42),
}
ROSTER_MODELS = list(ROSTER)

# --- Models exempt from the exact-value pins because their config declares
#     EXPERIMENT_IN_PROGRESS. Pinned as a SET so that both adding and removing an
#     exemption is a deliberate, reviewed edit rather than a silent config change.
#
#     This mechanism is inherited verbatim from test_datafactory_parity.py, which was
#     its only reader in the entire repository. Deleting that file without re-homing
#     this here would have removed the escape hatch silently — and the marker's own
#     text in models/violet_visitor/configs/config_hyperparameters.py instructed the
#     reader to remove it "when the roster lands", which was a decision for whoever
#     owns the experiment, not a side effect of a test rewrite.
#
#     **Empty since 2026-08-12.** violet_visitor was the only member, and the maintainer
#     un-fenced it: it is now a full roster member on the same foundation as the other
#     seven, pinned like the rest. The mechanism stays rather than being deleted — an
#     empty set still fails loudly if a marker reappears, which is the point of pinning
#     it as a set in both directions. ---
EXPERIMENTS_IN_PROGRESS: set[str] = set()

#: Roster members whose values ARE pinned — everything not mid-experiment.
PINNED_MODELS = [m for m in ROSTER_MODELS if m not in EXPERIMENTS_IN_PROGRESS]

# --- Experiment arms: pinned like the roster, but NOT roster members.
#
#     An arm is a tracked model directory that exists to be trained against one of the
#     eight as a control. It is kept out of ROSTER on purpose: ROSTER is read as ensemble
#     membership (`test_members_are_the_roster`), by the sample-count contract and by the
#     family-head checks, and an arm is not an ensemble constituent, not a delivery and not
#     part of the pre-registered eight. Putting one in ROSTER would change what the roster
#     means in four places in order to get a value pin in one.
#
#     It must still be pinned. An arm's whole identity is the small set of values that
#     differ from its control, and a config edit that quietly moved one of them would make
#     the comparison measure something else while every test stayed green.
#
#     arm -> (control, (output_distribution, forecast_composition, gate_threshold, seed))
#     gate_threshold is None where the composition is soft_gate.
EXPERIMENT_ARMS = {
    # views-hydranet#396. Occurrence moves from the body to the gate: a zero-truncated body
    # (P(body>0) = 1 by construction) composed continuously, so emitted P(y>0) equals the
    # gate. The control is NOT a second training run — it is the 2026-10-12 violet_visitor
    # vehicle (calibration_model_20261012_033112.pt, sha256 4acbf4d3c2b457b0...), which is
    # why every non-composition value has to stay identical.
    "amber_kestrel": ("violet_visitor", ("truncated_nb", "soft_gate", None, 42)),
    # views-hydranet reports/2026-10-16_pos_weight_prereg.md. The gate's objective, not its
    # composition: `weighted_bce` at pos_weight 2.0 has its optimum at s* = wp/(1 + (w-1)p),
    # so the loss asks the gate to overstate occurrence by ~2x — improper by Gneiting &
    # Ranjan (2011) Thm 1. This arm sets w = 1.0 and changes nothing else.
    #
    # ⛔ Its control is `amber_kestrel`, NOT `violet_visitor`. Editing amber_kestrel in place
    # was the alternative and was rejected (Simon, 2026-10-16): its control artefact cannot
    # be retrained (C-317), so a second free key there would have spent the composition
    # contrast permanently. The declared composition values below are amber_kestrel's,
    # inherited unchanged — they are what this arm must NOT vary.
    "copper_kestrel": ("amber_kestrel", ("truncated_nb", "soft_gate", None, 42)),
    # views-models#565 / #568, pre-registered in views-hydranet
    # reports/2026-10-18_weight_decay_arms_prereg.md §3. The two zero-weight-decay
    # interventions. Each varies `weight_decay` 0.1 -> 0.0 and NOTHING else.
    #
    # ⛔ THEIR CONTROLS ARE THE SEED-MATCHED REPLICATES, NOT amber_kestrel, and that is the
    # whole design. indigo and silver are both seed 42; russet and bronze are both seed 43;
    # so each pair differs on `weight_decay` ALONE. Declared against amber_kestrel instead,
    # each would differ on THREE keys — weight_decay, torch_seed, np_seed — which the
    # comment at :66-70 names as the confounded case.
    #
    # ⚠️ So an arm's control here is a SEED REPLICATE, which is deliberate and is why
    # `test_every_declared_control_is_itself_a_declared_model` forbids a replicate-of-a-
    # replicate but explicitly permits an arm-of-a-replicate.
    "indigo_kestrel": ("silver_kestrel", ("truncated_nb", "soft_gate", None, 42)),
    "russet_kestrel": ("bronze_kestrel", ("truncated_nb", "soft_gate", None, 43)),
}

# --- Seed replicates: the SAME config as their control, re-run to measure how much this
#     training pipeline varies run to run.
#
#     A replicate is not an arm and cannot be declared as one. `EXPERIMENT_ARMS` ends its
#     one-variable guard with `assert differing - INSTRUMENTATION_KEYS` — "there is no arm to
#     run" — and that assertion is correct: a replicate differs from its control on nothing
#     but a seed, or on nothing at all, so there is no contrast to measure. It measures the
#     NOISE against which a contrast is read.
#
#     Why it has to be declared at all, rather than left as untracked directories: every
#     pair guard in TestExperimentArmConformance is parametrized over EXPERIMENT_ARMS, so a
#     model outside it has no pair guard of ANY kind — not byte identity, not config_meta,
#     not maturity. And the cells with no guard would be exactly the cells the variance
#     estimate is built from: a config divergence among the replicates feeds straight into
#     that estimate as variance, INFLATING it, which raises every threshold and makes every
#     effect fail significance. A corrupted variance estimate is indistinguishable from an
#     honest inconclusive result. It does not look like a failure; it looks like a
#     measurement. See views-models#565.
#
#     replicate -> (control, seed)
SEED_REPLICATES: dict[str, tuple[str, int]] = {
    "silver_kestrel": ("amber_kestrel", 42),
    "bronze_kestrel": ("amber_kestrel", 43),
    "cobalt_kestrel": ("amber_kestrel", 44),
}

#: The hyperparameter keys a REPLICATE may differ from its control on.
#:
#: Module-level rather than per-pair, and that is not an inconsistency with ARM_FREE_KEYS.
#: ARM_FREE_KEYS is keyed per arm because an arm's free key is an experimental choice that
#: needs a justification written against that contrast, and because a shared set would
#: ratchet open. A replicate has no such choice: varying anything but the seed stops it
#: being a replicate, so there is no per-pair justification to write and nothing to widen.
#: Adding a key here would not loosen a contrast — it would delete the concept.
REPLICATE_FREE_KEYS = frozenset({"torch_seed", "np_seed"})

#: The hyperparameter keys an arm may differ from its control on. Anything else differing
#: means it is no longer a one-variable contrast.
#:
#: Pinned as an exact set below, in both directions, for the same reason KNOWN_GATELESS is:
#: this is an ESCAPE HATCH, and the assertion that uses it tells the reader to widen it. One
#: word here plus one config edit turns a 300-lesson arm into a 160-lesson one against a
#: 300-lesson control, with every test in the repository green. Widening it must be a visible
#: act, not a quiet one.
#: ONE SET PER PAIR, not one set for the repository, and that is the whole point.
#:
#: This was a single module-level set until 2026-10-16. The defect that shape has is not
#: hypothetical: `copper_kestrel` needs `loss_class_pos_weight` free, and adding it to a
#: shared set would have silently licensed the `amber_kestrel` ↔ `violet_visitor` pair to
#: differ on `loss_class_pos_weight` too — the exact key that pair must hold fixed, because
#: holding it at 2.0 is what made the composition contrast one-variable. Every future
#: widening would do that again, to every pair at once, and no single commit would look
#: wrong. The escape hatch would ratchet open.
#:
#: Keyed by ARM, so a widening reaches exactly one contrast and the justification is written
#: against that contrast rather than in the abstract.
ARM_FREE_KEYS = {
    # views-hydranet#396: the composition triple, and nothing else.
    "amber_kestrel": frozenset(
        {"output_distribution", "forecast_composition", "gate_threshold"}
    ),
    # The pos_weight arm varies exactly one value. `output_distribution` and
    # `forecast_composition` are NOT free here — they are inherited from the control and
    # must match it.
    "copper_kestrel": frozenset({"loss_class_pos_weight"}),
    # The two zero-decay arms vary exactly one value each, against a seed-matched control.
    "indigo_kestrel": frozenset({"weight_decay"}),
    "russet_kestrel": frozenset({"weight_decay"}),
}

#: Keys that may differ for ANY pair because they change what is RECORDED, never what is
#: computed. Separate from ARM_FREE_KEYS deliberately: a log path is not an experimental
#: variable, and filing it as one would make "this arm varies one key" false in the literal
#: reading that gives the guard its force.
#:
#: `trajectory_log_path` is here because it must differ — it resolves to an absolute path
#: inside the pod's rsynced deliver directory, and the deliver directory is named after the
#: model. A relative path dies with the pod (views-models#558 / C-161).
INSTRUMENTATION_KEYS = frozenset({"trajectory_log_path"})

#: Every declared pair in the repository: name -> control. An arm and its control, or a
#: replicate and its control.
#:
#: The guards below that assert a property of the PAIR — byte identity, config_meta, roster
#: separation, README survival, maturity — are parametrized over this rather than over
#: EXPERIMENT_ARMS, because those properties hold of any declared pair and a declaration
#: outside the parametrize list is guarded by nothing at all.
#:
#: The two guards that assert a property of the CONTRAST — the declared values and the
#: one-variable diff — stay on EXPERIMENT_ARMS, and have replicate counterparts, because a
#: replicate has no contrast to assert.
#:
#: Derived, not hand-written, and covered by a test that it spans both declarations: a third
#: declaration kind added later and forgotten here would silently inherit the same hole this
#: mapping exists to close.
PAIRED_MODELS = {
    **{arm: control for arm, (control, _declared) in EXPERIMENT_ARMS.items()},
    **{rep: control for rep, (control, _seed) in SEED_REPLICATES.items()},
}

#: Config files that must be BYTE-IDENTICAL between an arm and its control.
#:
#: Byte identity rather than a value diff, and that is deliberate. These four files are
#: model-agnostic — they resolve their own model name from the path — so the arm's copies
#: were already identical, and the hazard is not a wrong value but ANY divergence: the
#: control is an artefact that was already trained from the control's copy of these files.
#: `config_queryset.py` is the dangerous one. `REGION = "land"` -> `"africa_me_legacy"` is a
#: one-word edit that trains the arm on 13,110 cells against a control trained on 64,818,
#: and a value-level diff of hyperparameters cannot see it.
#:
#: The arm's own model name is normalised to the control's before comparing, because these
#: files name their model in a docstring. Nothing else is exempt.
#:
#: The cost of byte identity is that a comment fix to the CONTROL's copy turns this red. That
#: is the intended behaviour, not a wart: the arm must then be brought back into step, which
#: is the only way the pair stays comparable. The arm's copies therefore also carry the
#: control's stale docstrings verbatim (`config_queryset.py` describes a viewser migration
#: that is the control's history, not the arm's). Fixing them in the arm alone would break
#: the identity this guard exists to assert, and the control's own files are not ours to edit.
#:
#: Paths are relative to the MODEL directory, so this covers the model root as well as
#: configs/. `requirements.txt` is in here because it carries the views-datafactory floor,
#: and `run.sh` because it is the launcher; neither is a hyperparameter, and a value-level
#: diff of hyperparameters sees neither. All six are identical today after the arm's own
#: name is normalised out, which is what makes byte identity the right assertion rather
#: than an aspiration.
ARM_IDENTICAL_FILES = (
    "configs/config_queryset.py",
    "configs/config_partitions.py",
    "configs/config_sweep.py",
    "main.py",
    "run.sh",
    "requirements.txt",
)

#: Keys an arm's `config_meta` may differ from its control's on. `name` is the model's
#: identity; everything else — level, algorithm, targets, evaluation profile — is the
#: experimental setting and must match. `evaluation_profile` is the one that matters:
#: 'hydranet_ucdp' -> 'default' scores the two runs with different metrics.
ARM_FREE_META_KEYS = {"name"}

# --- The shared v2 gated_NB foundation every pinned member holds fixed. Values that
#     differ per member (family, composition, seed) live in ROSTER, not here.
#     total_lessons is a RUN-TIME budget (160 during the window-constrained smoke runs, now
#     300 -- 160 was not converged) and is deliberately NOT pinned. ---
FOUNDATION = {
    "loss_reg": "mse",
    "reg_activation": "softplus",
    "body_supervision": "all",
    "loss_class": "weighted_bce",
    "loss_class_pos_weight": 2.0,
    "rollout_feedback": "sample",
    "bn_recalibrate": True,
    "n_head_samples": 4,
    "n_posterior_samples": 4,
    "model": "HydraBNUNet06_LSTM4",
    # #484: a CPU device is a hard stop (views-hydranet 0.1.1). On 0.1.0 the key is accepted and
    # ignored (extra="allow"), which is why the floor moved with it.
    "require_cuda": True,
}

REGRESSION_TARGETS = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
CLASSIFICATION_TARGETS = ["by_sb_best", "by_ns_best", "by_os_best"]

def _load(path, fn):
    """Load ``fn`` from ``path``.

    Fails rather than skips when the file is absent. A roster member that has been
    renamed or half-applied must turn the suite red; ``pytest.skip`` here would report
    green-by-silence at exactly the moment the signal matters (C-113 class).
    """
    if not path.exists():
        pytest.fail(f"{path.relative_to(REPO_ROOT)} does not exist — a roster member is missing")
    spec = importlib.util.spec_from_file_location(
        f"_cfg_{path.parent.parent.name}_{path.stem}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)()


def _load_hp(model_name):
    return _load(MODELS_DIR / model_name / "configs" / "config_hyperparameters.py", "get_hp_config")


def _load_meta(name, base_dir):
    return _load(base_dir / name / "configs" / "config_meta.py", "get_meta_config")


def _queryset_text(model_name):
    path = MODELS_DIR / model_name / "configs" / "config_queryset.py"
    if not path.exists():
        pytest.fail(f"{model_name} has no config_queryset.py")
    return path.read_text()


def _experiment_in_progress(model_name):
    """True iff the model's config declares ``EXPERIMENT_IN_PROGRESS = True``."""
    path = MODELS_DIR / model_name / "configs" / "config_hyperparameters.py"
    if not path.exists():
        return False
    spec = importlib.util.spec_from_file_location(f"_eip_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return bool(getattr(mod, "EXPERIMENT_IN_PROGRESS", False))


# ── the exemption is itself pinned ────────────────────────────────────


def test_the_experiment_in_progress_roster_is_exactly_as_declared():
    """Both adding and removing an exemption must be a reviewed edit.

    Without this, marking a model EXPERIMENT_IN_PROGRESS silently removes it from every
    value pin below, and unmarking one silently subjects a churning config to them.
    """
    marked = {m for m in ROSTER_MODELS if _experiment_in_progress(m)}
    assert marked == EXPERIMENTS_IN_PROGRESS, (
        f"the EXPERIMENT_IN_PROGRESS roster changed: expected {EXPERIMENTS_IN_PROGRESS}, "
        f"found {marked}. Update EXPERIMENTS_IN_PROGRESS here in the same change, and say "
        f"why in the commit — an exemption that appears or disappears on its own is how a "
        f"pinned config stops being pinned without anyone deciding it."
    )


def test_the_value_pins_are_not_vacuous():
    """At least one member must actually be pinned.

    If every roster model were EXPERIMENT_IN_PROGRESS, all the pins below would pass
    over an empty parametrisation. A test that asserts nothing is worse than a missing
    test, because it reports success.
    """
    assert PINNED_MODELS, (
        "every roster model is EXPERIMENT_IN_PROGRESS, so the value pins assert nothing"
    )


# ── family, composition, gate, seed ───────────────────────────────────


class TestRosterFamilyConformance:
    """Each pinned model IS its roster entry: family, composition, gate, seed."""

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_family_and_composition(self, model_name):
        distribution, composition, _threshold, _seed = ROSTER[model_name]
        hp = _load_hp(model_name)
        assert hp["output_distribution"] == distribution, (
            f"{model_name}: output_distribution {hp['output_distribution']!r} != roster {distribution!r}"
        )
        assert hp["forecast_composition"] == composition, (
            f"{model_name}: forecast_composition {hp['forecast_composition']!r} != roster {composition!r}"
        )

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_gate_threshold(self, model_name):
        _distribution, composition, threshold, _seed = ROSTER[model_name]
        hp = _load_hp(model_name)
        if composition == "threshold_gate":
            assert hp.get("gate_threshold") == threshold, (
                f"{model_name}: threshold_gate needs gate_threshold={threshold}, "
                f"got {hp.get('gate_threshold')!r}"
            )
        else:
            # soft_gate composes gate * body continuously — no hard threshold.
            assert "gate_threshold" not in hp or hp["gate_threshold"] is None, (
                f"{model_name}: soft_gate must not carry a gate_threshold "
                f"(got {hp.get('gate_threshold')!r})"
            )

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_seed(self, model_name):
        _distribution, _composition, _threshold, seed = ROSTER[model_name]
        hp = _load_hp(model_name)
        assert hp["torch_seed"] == seed and hp["np_seed"] == seed, (
            f"{model_name}: seeds torch={hp.get('torch_seed')} np={hp.get('np_seed')} "
            f"!= roster seed {seed} (torch_seed and np_seed must match)"
        )

    def test_every_roster_member_has_a_family_head_config(self):
        """The roster's eight all exist on disk with a loadable hyperparameter config.

        Membership is asserted for all eight, exemption or not — being mid-experiment
        excuses a model's *values*, not its presence.
        """
        missing = [
            m for m in ROSTER_MODELS
            if not (MODELS_DIR / m / "configs" / "config_hyperparameters.py").exists()
        ]
        assert not missing, f"roster members missing a hyperparameter config: {missing}"


class TestSharedV2Foundation:
    """Every pinned member holds the v2 gated_NB foundation fixed."""

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    @pytest.mark.parametrize("key,expected", list(FOUNDATION.items()))
    def test_foundation_value(self, model_name, key, expected):
        hp = _load_hp(model_name)
        assert hp.get(key) == expected, (
            f"{model_name}: foundation {key}={hp.get(key)!r} != expected {expected!r}"
        )

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_no_tobit_sigma(self, model_name):
        """The roster is mse, not tobit — no leftover tobit-only loss_reg_sigma."""
        hp = _load_hp(model_name)
        assert hp["loss_reg"] == "mse"
        assert "loss_reg_sigma" not in hp, (
            f"{model_name}: carries loss_reg_sigma but loss_reg is mse (tobit-only knob)"
        )


# ── structural checks: applied to ALL eight, exemption or not ─────────
# These are not what an in-flight loss experiment churns. Grid, targets and metadata
# must hold for every member or the pooled ensemble is not well-formed, so the
# exemption does not reach them.


class TestGridAndTargets:
    """Region grid and target channels are identical across the whole roster."""

    @pytest.fixture()
    def hps(self):
        return {name: _load_hp(name) for name in ROSTER_MODELS}

    def test_identical_grid_topology(self, hps):
        # Compare members to each other rather than to hardcoded offsets — all eight
        # must share one grid for the concat pool to be valid.
        ref_name = ROSTER_MODELS[0]
        ref = {k: hps[ref_name][k] for k in ("row_offset", "col_offset", "height", "width")}
        for name, hp in hps.items():
            got = {k: hp[k] for k in ("row_offset", "col_offset", "height", "width")}
            assert got == ref, f"{name} grid {got} != {ref_name} {ref}"

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_regression_targets(self, model_name):
        hp = _load_hp(model_name)
        assert hp["regression_targets"] == REGRESSION_TARGETS, (
            f"{model_name}: regression_targets {hp['regression_targets']} != {REGRESSION_TARGETS}"
        )

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_classification_targets_gate_channel(self, model_name):
        # The by_* gate channel is 1:1 with the lr_* magnitudes — this is the occurrence
        # gate the ensemble must pool (C-132, pooled once views-pipeline-core#422 ships).
        hp = _load_hp(model_name)
        assert hp["classification_targets"] == CLASSIFICATION_TARGETS, (
            f"{model_name}: classification_targets {hp['classification_targets']} != {CLASSIFICATION_TARGETS}"
        )
        assert len(hp["classification_targets"]) == len(hp["regression_targets"]), (
            f"{model_name}: gate channels not 1:1 with magnitudes"
        )




class TestCollapseDeclaration:
    """How the roster's posterior draws collapse to a point — declared here, honoured by tools.collapse.

    Its own class rather than a line in TestGridAndTargets: that class is documented as
    "Region grid and target channels are identical across the whole roster", and how a draw
    axis is folded to a scalar is neither a grid nor a target channel. Same section, because
    it applies to all eight regardless of the loss-experiment exemption.
    """

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_collapse_declaration_matches_the_converter(self, model_name):
        """The roster declares how its draws collapse; `tools.collapse` must obey that, not guess.

        The pipeline itself would apply this at `inference_orchestrator.py:179`
        (views-hydranet `vhy_021` / `vhy_039` stage 5), but stage 5 is gated on
        `evaluation_mode == "point"` and all eight run `stochastic` — so the draws reach
        disk uncollapsed and views-models collapses them instead (ADR-023, #505).

        Two places therefore state one fact. This test is what stops them disagreeing: if a
        model moves to `median`, the converter must be invoked with `--aggregate-method
        median`, and this failure is where you find that out.
        """
        # Imported inside the test, and it must stay that way: `roster-configs-load` imports
        # this module (via tests/test_roster_configs_load.py) in a minimal env that installs
        # views-hydranet and nothing else. `tools.collapse` imports pandas at module level, so
        # a top-level import here breaks collection of a job that never runs this test.
        from tools.collapse.collapse_predictions import DEFAULT_AGGREGATE_METHOD

        hp = _load_hp(model_name)
        assert hp["evaluation_mode"] == "stochastic", (
            f"{model_name}: evaluation_mode={hp['evaluation_mode']!r} — the pipeline now "
            "collapses in-run, so tools.collapse would be averaging an already-point volume"
        )
        assert hp["aggregate_method"] == DEFAULT_AGGREGATE_METHOD, (
            f"{model_name}: declares aggregate_method={hp['aggregate_method']!r} but "
            f"tools.collapse defaults to {DEFAULT_AGGREGATE_METHOD!r} — pass "
            f"--aggregate-method {hp['aggregate_method']} or the delivered parquet will "
            "not be the estimator the model declares"
        )
class TestDatafactorySource:
    """Every pinned member reads views-datafactory at global land (``REGION = "land"``).

    The source migration is exempt for in-flight models, for the same reason the value
    pins are: violet_visitor was not migrated by S2 (#365), which covered the other
    three viewser models but not the one that was mid-experiment. Its queryset is
    therefore still a pure viewser queryset in git. Migrating it is an edit to a fenced
    config and belongs to whoever un-fences the model.

    ``test_declares_ged_features`` stays on all eight: the GED feature names are
    source-independent and already hold for every member.
    """

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_uses_datafactory(self, model_name):
        text = _queryset_text(model_name)
        assert (
            '"source": "views-datafactory"' in text
            or "'source': 'views-datafactory'" in text
        ), f"{model_name} does not declare a views-datafactory source"

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_global_land_region(self, model_name):
        """The ``REGION`` assignment itself, not a substring anywhere in the file.

        #499 Step 1 (2026-09-19): the roster moved from ``africa_me_legacy`` (13,110 cells)
        to ``land`` (64,818 — what the FAO delivery cuts ``land_gaul`` from). The old
        assertion was ``"africa_me_legacy" in text``, which a comment recording the history
        satisfies; this one reads the line that decides.
        """
        m = re.search(r'^REGION\s*=\s*"([a-z_]+)"', _queryset_text(model_name), re.MULTILINE)
        assert m, f"{model_name} has no REGION assignment"
        assert m.group(1) == "land", f"{model_name} REGION is {m.group(1)!r}, not 'land'"

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_no_viewser_import(self, model_name):
        """No *executable* viewser import — parsed, not grepped.

        This substring-matched `"from viewser"` until 2026-08-12, and violet_visitor's
        migration docstring opens *"Migrated from viewser to views-datafactory"*. The test
        failed on the sentence describing the fix. That is C-57 — a regex cannot tell a
        commented or quoted mention from a live statement — committed inside the very
        check meant to catch it. The AST can, so it does.
        """
        tree = ast.parse(_queryset_text(model_name))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names if a.name.split(".")[0] == "viewser"]
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "viewser":
                    offenders.append(node.module)
        assert not offenders, (
            f"{model_name} imports viewser ({offenders}) — should use datafactory"
        )

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_declares_ged_features(self, model_name):
        text = _queryset_text(model_name)
        for feat in ("ged_sb_best", "ged_ns_best", "ged_os_best"):
            assert feat in text, f"{model_name} missing {feat}"


class TestModelMeta:
    """Per-model meta consistency."""

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_meta(self, model_name):
        meta = _load_meta(model_name, MODELS_DIR)
        assert meta["algorithm"] == "HydraNet", f"{model_name} algorithm"
        assert meta["level"] == "pgm", f"{model_name} level"
        assert meta["prediction_format"] == "prediction_frame", f"{model_name} prediction_format"
        assert meta["name"] == model_name, f"{model_name} name != directory"


# ── the rusty_bucket rewiring, landed ──────────────────────────────────
# Deferred from #369 and done here (#146, #372 item 3). It waited on two things, both
# now true: violet_visitor emits D×K = 4×4 = 16 like the rest, so the ADR-015 contract
# holds across all eight; and views-pipeline-core 3.0.1 carries #422, so a declared gate
# is actually pooled rather than silently dropped.
#
# Order matters and gets it wrong LOUDLY, which is the one mercy here: declaring the gate
# while the members are the `temporary_*` stand-ins — which declare no
# classification_targets — raises `Model 'X' did not produce a forecast for target
# 'by_sb_best'` on the PredictionFrame path. A sequencing mistake costs a failed run, not
# a wrong number.

ENSEMBLE = "rusty_bucket"


def _load_modelset(name):
    return _load(ENSEMBLES_DIR / name / "configs" / "config_modelset.py", "get_modelset_config")


def _load_partitions(name, base_dir):
    return _load(base_dir / name / "configs" / "config_partitions.py", "generate")


#: HydraNet/PF concat ensembles that declare no `by_*` gate today, and why they are not
#: fixed. **This set may only ever SHRINK** — an entry here is an ensemble whose AP is
#: understated, so adding one means shipping the defect C-132 describes.
#:
#: Both are the retired viewser-vs-datafactory parity ensembles. The programme they served
#: ended when S2 (#365) migrated the trio, so declaring a gate on them would be work on
#: something scheduled for removal. They are slated to be reduced to README-only
#: placeholders (the maintainer's decision; #367 proposes deleting them outright). Neither
#: is scheduled by `monthly_run.sh`, and neither has produced an artifact since 2026-06-03,
#: so nothing is currently scoring occurrence off them.
#:
#: This is recorded rather than filtered silently: the guard found a real pre-existing
#: defect, and hiding that to make the suite green would be the exact failure the guard
#: exists to prevent.
KNOWN_GATELESS = frozenset({"golden_hour", "stellar_horizon"})


def test_every_hydranet_pf_ensemble_declares_the_occurrence_gate():
    """C-132 class guard, fleet-wide rather than just rusty_bucket.

    A `prediction_frame` / `hydranet_ucdp` concat ensemble pools its constituents'
    per-sample channels, and the occurrence gate rides `classification_targets` (`by_*`).
    An ensemble that declares `regression_targets` and omits the gate pools the magnitudes
    only: its AP and Brier are understated with **no error anywhere**, which is the whole
    of C-132.

    views-pipeline-core#422 makes the pool *respect* a declared gate; it does not
    synthesise one. So this is the fail-loud that stops a NEW gate-less HydraNet ensemble
    from reintroducing the defect the framework fix cannot see.
    """
    offenders = {}
    for meta_path in ENSEMBLES_DIR.rglob("configs/config_meta.py"):
        if meta_path.parent.parent.name in KNOWN_GATELESS:
            continue
        name = meta_path.parent.parent.name
        meta = _load_meta(name, ENSEMBLES_DIR)
        is_hydranet_pf = (
            meta.get("prediction_format") == "prediction_frame"
            or meta.get("evaluation_profile") == "hydranet_ucdp"
        )
        reg = meta.get("regression_targets") or []
        if not (is_hydranet_pf and reg):
            continue
        gate = meta.get("classification_targets") or []
        if len(gate) != len(reg):
            offenders[name] = {"regression_targets": reg, "classification_targets": gate}
    assert not offenders, (
        "these HydraNet/PF concat ensembles do not declare a 1:1 by_* gate channel, so "
        f"the concat pool silently drops occurrence (C-132): {offenders}. Declare "
        f"classification_targets in the ensemble config_meta."
    )


class TestRustyBucketEnsemble:
    """The 8-member concat ensemble — the epic's delivery unit."""

    @pytest.fixture()
    def ens_meta(self):
        return _load_meta(ENSEMBLE, ENSEMBLES_DIR)

    def test_members_are_the_roster(self):
        modelset = _load_modelset(ENSEMBLE)
        assert modelset["models"] == ROSTER_MODELS, (
            f"{ENSEMBLE} members {modelset['models']} != roster {ROSTER_MODELS}"
        )

    def test_no_temporary_stand_ins_remain(self):
        """The `temporary_*` clones existed to exercise the machinery; they are retired."""
        leftovers = [m for m in _load_modelset(ENSEMBLE)["models"] if m.startswith("temporary_")]
        assert not leftovers, f"{ENSEMBLE} still lists stand-ins: {leftovers}"

    def test_aggregation_and_level(self, ens_meta):
        assert ens_meta["aggregation"] == "concat"
        assert ens_meta["level"] == "pgm"

    def test_pools_regression_targets(self, ens_meta):
        assert ens_meta["regression_targets"] == REGRESSION_TARGETS
        constituent = {
            tuple(get_regression_targets(MODELS_DIR / m)) for m in _load_modelset(ENSEMBLE)["models"]
        }
        assert constituent == {tuple(REGRESSION_TARGETS)}, (
            f"constituents disagree on regression_targets: {constituent}"
        )

    def test_pools_the_occurrence_gate_channel(self, ens_meta):
        assert ens_meta.get("classification_targets") == CLASSIFICATION_TARGETS, (
            f"{ENSEMBLE} must declare classification_targets={CLASSIFICATION_TARGETS} so the "
            f"concat pool carries the gate (C-132); got {ens_meta.get('classification_targets')!r}"
        )
        assert "targets" not in ens_meta, (
            f"{ENSEMBLE} carries a retired synthesised `targets` key (#380 upstream) — "
            f"declare regression_targets / classification_targets instead"
        )

    def test_the_gate_carries_a_classification_metric_in_the_right_cell(self, ens_meta):
        """Both keys, and AP under **point** — the pair verified against both gates.

        `classification_targets` with no classification metric key is refused at load by
        `CoreConfigSniffer._check_targets_and_metrics` (the defect #367 shipped). And AP
        under `classification_sample_metrics` passes the sniffer and then fails
        `NativeEvaluator._validate_config`, because METRIC_MEMBERSHIP puts AP in
        ("classification", "point") — moving the failure later and quieter.
        """
        assert ens_meta.get("classification_point_metrics") == ["AP"]
        assert ens_meta.get("classification_sample_metrics") == ["Brier_cls_sample"]

    def test_every_member_produces_the_declared_count(self):
        """8 x 16 = 128, equally weighted — the reason the rewiring waited on violet."""
        from tests.conftest import get_produced_sample_count

        expected = _load(
            ENSEMBLES_DIR / ENSEMBLE / "configs" / "config_hyperparameters.py", "get_hp_config"
        )["expected_samples_per_model"]
        produced = {
            m: get_produced_sample_count(MODELS_DIR / m)
            for m in _load_modelset(ENSEMBLE)["models"]
        }
        wrong = {m: n for m, n in produced.items() if n != expected}
        assert not wrong, (
            f"these constituents do not emit the declared {expected} draws: {wrong}. "
            f"Unequal counts weight the pooled mixture unequally (ADR-015 §2/§3)."
        )

    def test_metrics_profile(self, ens_meta):
        assert ens_meta["regression_sample_metrics"] == ["CRPS", "QS_sample", "MCR_sample"]
        assert ens_meta["evaluation_profile"] == "hydranet_ucdp"

    def test_uses_prediction_frame_manager(self):
        main_path = ENSEMBLES_DIR / ENSEMBLE / "main.py"
        assert main_path.exists(), f"{ENSEMBLE}/main.py is missing"
        assert "PredictionFrameEnsembleManager" in main_path.read_text()

    def test_partition_boundaries_match_a_member(self):
        ens_parts = _load_partitions(ENSEMBLE, ENSEMBLES_DIR)
        member_parts = _load_partitions(ROSTER_MODELS[0], MODELS_DIR)
        assert ens_parts["calibration"] == member_parts["calibration"]
        assert ens_parts["validation"] == member_parts["validation"]


def test_the_known_gateless_set_only_shrinks():
    """Both directions are a reviewed edit.

    A new entry means shipping an ensemble whose occurrence is silently understated. A
    removed entry means the ensemble was fixed or retired — good, and the pin should say
    so rather than quietly agreeing.
    """
    present = {p.parent.parent.name for p in ENSEMBLES_DIR.rglob("configs/config_meta.py")}
    stale = KNOWN_GATELESS - present
    assert not stale, (
        f"{sorted(stale)} no longer exist — remove them from KNOWN_GATELESS so the pin "
        f"keeps meaning something."
    )
    for name in KNOWN_GATELESS:
        meta = _load_meta(name, ENSEMBLES_DIR)
        reg = meta.get("regression_targets") or []
        gate = meta.get("classification_targets") or []
        assert len(gate) != len(reg), (
            f"{name} now declares a 1:1 gate channel — good. Remove it from "
            f"KNOWN_GATELESS so the fleet guard covers it."
        )


# ── experiment arms ───────────────────────────────────────────────────

_KEY_ABSENT = object()


def _differing_keys(left: dict, right: dict) -> set[str]:
    """Keys whose values differ between two configs, treating ABSENT as a value of its own.

    Extracted at the THIRD occurrence, not the second: the arm one-variable guard, the
    `config_meta` guard and the replicate seed-only guard all computed this identically.
    CLAUDE.md's rule is to *examine* whether an abstraction is real on the third incident,
    and by then the shape was observed rather than guessed.

    ⛔ **The sentinel is the whole reason this is a function.** The obvious spelling,
    `left.get(key) != right.get(key)`, reads an ABSENT key as `None` — so a key deleted from
    one config and set to `None` in the other compares EQUAL, and a guard built on it passes
    on a real divergence. Three hand-written copies were three chances to forget that, and a
    fourth would have been a fourth.
    """
    return {
        key
        for key in set(left) | set(right)
        if left.get(key, _KEY_ABSENT) != right.get(key, _KEY_ABSENT)
    }



#: The parameter name every PAIR guard is parametrized on. Read by the scope guard below,
#: which discovers guards by this name rather than listing them.
_PAIR_ARGNAME = "paired"


def _parametrized_values(func, argname: str) -> set[str]:
    """The values pytest will collect for `argname` on `func`, read from the DECORATOR.

    Deliberately a second implementation rather than an import of
    `test_roster_configs_load._parametrized_scope`: that module imports the declarations
    FROM this one, so importing its helper back would be a cycle. Two four-line copies is
    the cheaper side of that trade, and a third module to hold one comprehension is not.
    """
    marks = [
        m
        for m in getattr(func, "pytestmark", [])
        if m.name == "parametrize" and m.args[0] == argname
    ]
    assert len(marks) == 1, (
        f"{func.__name__} is not parametrized exactly once on {argname!r}, so its real "
        f"scope cannot be read here and this guard cannot do its job"
    )
    return set(marks[0].args[1])

def _manual_blocks(text):
    """The `<!-- manual -->` … `<!-- /manual -->` bodies, read the way the tool reads them.

    Re-implemented here rather than imported: `tools/catalogs/readme_preserve.py` is
    importable, but keeping the markers' literal spelling in this file means a change to
    either side has to be deliberate on both.
    """
    return re.findall(r"<!-- manual -->(.*?)<!-- /manual -->", text, flags=re.DOTALL)


class TestExperimentArmConformance:
    """Declared pairs: an arm asserts a contrast, a seed replicate asserts the absence of one.

    ⚠️ The class name says "arm" and its contents are wider than that — it holds the ARM
    guards (declared values, one-variable contrast), the PAIR guards that hold of any
    declared pair (byte identity, meta, roster separation, README survival, maturity), and
    the REPLICATE guards. The boundary is drawn by section comments rather than structure,
    which is a real seam in the wrong place and is deliberately NOT fixed here: splitting it
    is its own change, and doing it inside a change that widens five guards would make both
    unreviewable. Noted so the next reader knows it was seen.

    An arm is its declared values, and differs from its control ONLY on its declared variable.

    The second half is the one that matters, and the one a roster-shaped pin would miss.
    `amber_kestrel`'s control is a trained artefact that already exists, so the arm is a
    valid contrast only while every value except the composition triple is identical to
    `violet_visitor`'s. A well-meant edit — bumping lessons, changing a seed to "get an
    independent draw", adding a static channel — would turn a one-variable experiment into
    a two-variable one, and none of the roster pins above would notice, because an arm is
    deliberately not in the roster.
    """

    @pytest.mark.parametrize("arm", sorted(EXPERIMENT_ARMS))
    def test_the_arm_is_its_declared_values(self, arm):
        _control, (distribution, composition, threshold, seed) = EXPERIMENT_ARMS[arm]
        hp = _load_hp(arm)
        assert hp["output_distribution"] == distribution, (
            f"{arm}: output_distribution {hp['output_distribution']!r} != declared {distribution!r}"
        )
        assert hp["forecast_composition"] == composition, (
            f"{arm}: forecast_composition {hp['forecast_composition']!r} != declared {composition!r}"
        )
        if threshold is None:
            # Stricter than the engine, deliberately. ADR-069's rule is
            # `elif self.gate_threshold is not None:` — so a soft_gate config carrying
            # `gate_threshold = None` loads fine and only a non-None value raises. This
            # assertion requires the key to be ABSENT anyway, because an arm's config is
            # read by people deciding what the experiment was: a `None` sitting next to
            # `soft_gate` invites someone to fill it in.
            #
            # An earlier version of this comment claimed the engine refuses the key at all.
            # It does not. Checked by constructing both against the pinned engine version.
            assert "gate_threshold" not in hp, (
                f"{arm}: soft_gate must not carry a gate_threshold key at all "
                f"(got {hp.get('gate_threshold')!r}); a non-None value would also be "
                f"refused by the engine (ADR-069)"
            )
        else:
            assert hp.get("gate_threshold") == threshold, (
                f"{arm}: gate_threshold {hp.get('gate_threshold')!r} != declared {threshold!r}"
            )
        assert hp["torch_seed"] == seed and hp["np_seed"] == seed, (
            f"{arm}: seeds torch={hp.get('torch_seed')} np={hp.get('np_seed')} != declared {seed}"
        )

    @pytest.mark.parametrize("arm", sorted(EXPERIMENT_ARMS))
    def test_the_arm_differs_from_its_control_only_on_its_declared_variable(self, arm):
        control, _declared = EXPERIMENT_ARMS[arm]
        free = ARM_FREE_KEYS[arm]
        admissible = free | INSTRUMENTATION_KEYS
        differing = _differing_keys(_load_hp(arm), _load_hp(control))
        assert differing <= admissible, (
            f"{arm} differs from its control {control} on "
            f"{sorted(differing - admissible)}, which is outside the keys declared free for "
            f"THIS pair ({sorted(free)}). The control for this arm is an artefact that has "
            f"already been trained and cannot be retrained (C-317), so any other difference "
            f"confounds the comparison — revert it, or add it to ARM_FREE_KEYS[{arm!r}] and "
            f"say in the commit why the contrast survives. Adding it to another pair's set "
            f"will not help, and that is deliberate."
        )
        # And the inverse, so this cannot pass on an arm that is merely a copy of its control.
        # Instrumentation does not count as an arm: a pair that differs ONLY on a log path has
        # nothing to run, and before the per-pair split this would have read as a live arm.
        assert differing - INSTRUMENTATION_KEYS, (
            f"{arm} is identical to its control {control} apart from instrumentation "
            f"({sorted(differing & INSTRUMENTATION_KEYS)}): there is no arm to run"
        )

    def test_the_escape_hatch_is_pinned_in_both_directions(self):
        """`ARM_FREE_KEYS` is the one thing that can widen the guard above.

        The assertion above names widening it as the documented route ("add it to
        ARM_FREE_KEYS and say in the commit why the contrast survives"), which is correct
        advice and a silent one-word hole. Adding `total_lessons` here plus the config edit
        it licenses makes the arm a 160-lesson run against a 300-lesson control, with the
        full suite green — the config edit alone fails loudly, so the hatch is the whole
        attack surface.

        Pinned exactly, like KNOWN_GATELESS: both adding and removing a key is then a
        reviewed edit to this literal rather than a side effect.
        """
        assert ARM_FREE_KEYS == {
            "amber_kestrel": frozenset(
                {"output_distribution", "forecast_composition", "gate_threshold"}
            ),
            "copper_kestrel": frozenset({"loss_class_pos_weight"}),
            "indigo_kestrel": frozenset({"weight_decay"}),
            "russet_kestrel": frozenset({"weight_decay"}),
        }, (
            f"ARM_FREE_KEYS is { {k: sorted(v) for k, v in ARM_FREE_KEYS.items()} }. "
            f"Widening any pair's set licenses a second variable in a one-variable "
            f"experiment whose control is already trained and cannot be retrained (C-317). "
            f"If an arm genuinely needs another free key, change this literal too and say in "
            f"the commit why THAT contrast survives."
        )

        # Every arm must declare a set, and no set may be empty: an arm with no declared
        # variable would pass the guard above by admitting nothing and failing the inverse
        # assertion for a reason that does not name the cause.
        assert set(ARM_FREE_KEYS) == set(EXPERIMENT_ARMS), (
            f"ARM_FREE_KEYS covers {sorted(ARM_FREE_KEYS)} but the arms are "
            f"{sorted(EXPERIMENT_ARMS)}. Every arm declares exactly which keys its own "
            f"contrast may vary; a missing entry is a KeyError at test time, not a pass."
        )
        for arm, keys in ARM_FREE_KEYS.items():
            assert keys, f"{arm} declares an empty free-key set, so it has no variable"

        # Instrumentation is pinned on its own, for the same reason: moving a key into it
        # would exempt that key for EVERY pair at once, which is the ratchet the per-pair
        # split exists to close.
        assert INSTRUMENTATION_KEYS == frozenset({"trajectory_log_path"}), (
            f"INSTRUMENTATION_KEYS is {sorted(INSTRUMENTATION_KEYS)}. A key here is exempt "
            f"for every arm, so it must change what is RECORDED and never what is computed. "
            f"If it can alter a forecast it belongs in that pair's ARM_FREE_KEYS instead."
        )

        # And the two sets must not overlap, or a key would be both "the variable" and
        # "exempt for everyone" and the guard's meaning would depend on which check ran.
        for arm, keys in ARM_FREE_KEYS.items():
            assert not (keys & INSTRUMENTATION_KEYS), (
                f"{arm} declares {sorted(keys & INSTRUMENTATION_KEYS)} as its experimental "
                f"variable while INSTRUMENTATION_KEYS exempts it for every arm"
            )

        # REPLICATE_FREE_KEYS is the THIRD hatch and is pinned here with the other two,
        # because this test is where the file keeps them. It was introduced unpinned, with a
        # comment arguing a pin was unnecessary since "adding a key here would delete the
        # concept" — which is the shape of argument the two pins above replaced. #554 found
        # ARM_FREE_KEYS unpinned and recorded why: "the hatch is the whole attack surface".
        # One word added here, plus the config edit it licenses, makes a "replicate" a
        # contrast whose difference is then counted as run-to-run variance.
        assert REPLICATE_FREE_KEYS == frozenset({"torch_seed", "np_seed"}), (
            f"REPLICATE_FREE_KEYS is {sorted(REPLICATE_FREE_KEYS)}. A replicate re-runs the "
            f"SAME config, so the seeds are the only keys that may differ. Widening this "
            f"makes the replicate a contrast and silently inflates the variance estimate "
            f"every threshold in views-models#565 is divided by."
        )

        # The replicate hatch must not overlap either of the others. A key in both
        # REPLICATE_FREE_KEYS and an arm's set would be "the variable" for one pair kind and
        # "expected to differ" for the other, and whether a divergence was a finding would
        # depend on which declaration the model happened to be in.
        assert not (REPLICATE_FREE_KEYS & INSTRUMENTATION_KEYS), (
            f"{sorted(REPLICATE_FREE_KEYS & INSTRUMENTATION_KEYS)} is both a replicate's "
            f"free key and exempt for every pair; the seed is neither instrumentation nor "
            f"exempt — it is the replicate's entire identity"
        )
        for arm, keys in ARM_FREE_KEYS.items():
            assert not (keys & REPLICATE_FREE_KEYS), (
                f"{arm} declares {sorted(keys & REPLICATE_FREE_KEYS)} as its experimental "
                f"variable, which is also a replicate's free key. A seed contrast is a "
                f"replicate, not an arm — declare it in SEED_REPLICATES instead."
            )

    @pytest.mark.parametrize("paired", sorted(PAIRED_MODELS))
    @pytest.mark.parametrize("filename", ARM_IDENTICAL_FILES)
    def test_the_paired_model_shares_its_controls_data_configuration_byte_for_byte(self, paired, filename):
        """The confound the hyperparameter diff cannot see.

        `test_the_arm_differs_from_its_control_only_on_its_declared_variable` reads
        `config_hyperparameters.py` and nothing else, while the README claims
        *everything* else is copied unchanged. So `REGION = "land"` -> `"africa_me_legacy"`
        in the paired model's queryset — one word — trains it on 13,110 Africa/ME cells against a
        control trained on 64,818 global land cells, makes the comparison meaningless, and
        leaves every test in the repository green.

        These files are model-agnostic (they resolve their own name from the path), so byte
        identity is achievable and anything less is a divergence. `run.sh` is asserted here,
        never edited — it is production infrastructure.
        """
        control = PAIRED_MODELS[paired]
        paired_path = MODELS_DIR / paired / filename
        control_path = MODELS_DIR / control / filename
        for path in (paired_path, control_path):
            assert path.exists(), f"{path.relative_to(REPO_ROOT)} does not exist"
        # The paired model's own name is the ONE permitted substitution: these files name the model
        # they belong to in a docstring. Normalised rather than exempted by line number, so
        # a second difference on the same line is still caught.
        paired_text = paired_path.read_text(encoding="utf-8").replace(paired, control)
        control_text = control_path.read_text(encoding="utf-8")
        if paired_text == control_text:
            return
        diff = "\n".join(
            difflib.unified_diff(
                control_text.splitlines(),
                paired_text.splitlines(),
                fromfile=str(control_path.relative_to(REPO_ROOT)),
                tofile=str(paired_path.relative_to(REPO_ROOT)),
                lineterm="",
            )
        )
        pytest.fail(
            f"{paired}'s {filename} differs from its control {control}'s. These files set the "
            f"DATA the two runs see — region, grid, features, partition windows — and the "
            f"control is an artefact that has already been trained and cannot be retrained "
            f"(C-317). Any difference here confounds the experiment rather than varying it."
            f"\n\nThe paired model's own name has already been normalised to the control's, so what "
            f"follows is a real divergence:\n\n{diff}"
        )

    @pytest.mark.parametrize("paired", sorted(PAIRED_MODELS))
    def test_the_paired_model_is_evaluated_the_same_way_as_its_control(self, paired):
        """`config_meta` is documentation except for the parts that are not.

        `evaluation_profile` selects the metric set. 'hydranet_ucdp' -> 'default' scores the
        paired and its control with different metrics, which is a confound that looks like a
        result. Nothing covered this file before.
        """
        control = PAIRED_MODELS[paired]
        differing = _differing_keys(
            _load_meta(paired, MODELS_DIR), _load_meta(control, MODELS_DIR)
        )
        assert differing <= ARM_FREE_META_KEYS, (
            f"{paired}'s config_meta differs from its control {control}'s on "
            f"{sorted(differing - ARM_FREE_META_KEYS)}. Only {sorted(ARM_FREE_META_KEYS)} "
            f"may differ — everything else is the experimental setting, and the control has "
            f"already been trained under it."
        )

    @pytest.mark.parametrize("paired", sorted(PAIRED_MODELS))
    def test_a_paired_model_is_neither_a_roster_member_nor_an_ensemble_constituent(self, paired):
        """The separation this structure exists for, asserted rather than assumed.

        Named for the PAIR, not the arm: this runs over `PAIRED_MODELS`, so a seed replicate
        reaches it too, and a failure reporting "experiment arm" would send the reader to the
        concept `test_a_replicate_is_not_also_an_arm` exists to keep separate.
        """
        assert paired not in ROSTER, (
            f"{paired} is a declared pair member (an arm or a seed replicate) AND a roster "
            f"member at once; the roster is the "
            f"pre-registered eight and is read as ensemble membership"
        )
        assert paired not in _load_modelset(ENSEMBLE)["models"], (
            f"{paired} is pooled into {ENSEMBLE}; a declared pair member is trained to be "
            f"measured against a "
            f"control, not shipped"
        )

    @pytest.mark.parametrize("paired", sorted(PAIRED_MODELS))
    def test_the_paired_models_rationale_survives_readme_regeneration(self, paired):
        """A paired model's README is the only place its reasoning lives, and CI deletes READMEs.

        `tools/catalogs/update_readme.py` rebuilds every model README from a scaffold and
        discards anything outside a `<!-- manual -->` block, and
        `.github/workflows/update_catalogs.yml` runs it on every push to development that
        touches `models/*/configs/config_*.py` — which is every push that adds one. So an
        unmarked README is deleted by the merge that introduces it, silently, and the answer
        to "why is this a directory rather than an edit, and why is the kill one-sided"
        survives only in git history.

        Not hypothetical: it is C-77/C-78, and it took synthetic_chant's evaluation-semantics
        documentation on 2026-06-04. That is why the preservation mechanism exists; this test
        is what makes a paired model use it.
        """
        readme = MODELS_DIR / paired / "README.md"
        assert readme.exists(), f"{paired} has no README.md, so it documents nothing"
        text = readme.read_text(encoding="utf-8")
        blocks = _manual_blocks(text)
        assert blocks, (
            f"{paired}/README.md has no <!-- manual --> block, so update_readme.py will discard "
            f"all of it on the next push to development that touches its configs (C-77/C-78)."
        )
        # Not merely "a block exists": the hand-written content has to be INSIDE one, because
        # a marker pair wrapped round nothing satisfies the assertion above.
        #
        # The first version of this compared every `## ` heading against the block and
        # required none outside it. That was true of the README as committed and FALSE the
        # moment CI regenerated it: the scaffold's own headings are supposed to sit outside
        # the block. So it passed pre-merge, turned `development` red on the regeneration
        # commit, and reported a correct file as broken. A guard that fires on the correct
        # state is worse than no guard — it gets edited out of the way.
        #
        # The baseline is DERIVED rather than typed: it is the README of some model that is
        # still PURELY generated, so its headings are exactly what the scaffold emits.
        # Anything outside the paired model's block that the baseline does not also have is
        # hand-written text standing where CI will delete it.
        #
        # It used to be the paired model's own CONTROL, which was true while every control was a
        # roster model. It stopped being true on 2026-10-16, when copper_kestrel's control
        # became amber_kestrel — itself an arm, with a manual block of its own. The test then
        # failed on a correct paired and named the control as the problem, which is the shape of
        # failure its own comment above warns about. So the baseline is now ANY purely
        # generated README, searched deterministically rather than assumed of a role.
        baselines = sorted(
            d.name
            for d in MODELS_DIR.iterdir()
            if (d / "README.md").is_file()
            and not _manual_blocks((d / "README.md").read_text(encoding="utf-8"))
        )
        assert baselines, (
            "every model README now carries a manual block, so there is no purely generated "
            "baseline left to derive the scaffold's heading set from. Pin the heading set "
            "explicitly here instead — but note that pinning it means a scaffold change "
            "turns this red, which is the trade being made."
        )
        baseline = baselines[0]
        baseline_text = (MODELS_DIR / baseline / "README.md").read_text(encoding="utf-8")
        generated_headings = {
            line.strip() for line in baseline_text.splitlines() if line.startswith("## ")
        }
        assert generated_headings, (
            f"no `## ` headings found in {baseline}/README.md, so the baseline is empty and "
            f"this check would pass on anything"
        )

        preserved = "\n".join(blocks)
        orphaned = [
            line.strip()
            for line in text.splitlines()
            if line.startswith("## ")
            and line.strip() not in generated_headings
            and line not in preserved
        ]
        assert not orphaned, (
            f"{paired}/README.md has hand-written sections outside its manual block(s), which "
            f"update_readme.py will delete on the next regeneration: {orphaned}. Move them "
            f"inside <!-- manual --> … <!-- /manual -->. (Scaffold headings were taken from "
            f"{baseline}/README.md, the baseline this run derived.)"
        )
        # And the block must hold something, not just exist.
        assert preserved.strip(), (
            f"{paired}/README.md has an EMPTY manual block, which preserves nothing"
        )

    @pytest.mark.parametrize("paired", sorted(PAIRED_MODELS))
    def test_a_paired_model_is_a_candidate(self, paired):
        """Maturity is a person's act (ADR-017). An paired has not earned anything yet."""
        maturity = _load(
            MODELS_DIR / paired / "configs" / "config_maturity.py", "get_maturity_config"
        )
        assert maturity["maturity"] == "candidate", (
            f"{paired} declares maturity {maturity['maturity']!r}; a model declared against a "
            f"control — arm or seed replicate — is a candidate, never graduated"
        )

    # ── seed replicates: the pair guards above cover them; these cover what is theirs ──

    def test_the_replicate_roster_is_exactly_as_declared(self):
        """Both adding and removing a replicate must be a reviewed edit.

        Pinned as an exact literal in both directions, for the reason ARM_FREE_KEYS is: a
        replicate silently removed from this mapping keeps its directory and loses every
        pair guard, and one silently added subjects a directory to guards nobody chose for
        it. Full dict equality, so a changed SEED or a changed CONTROL fails here too — not
        only an added or removed name.
        """
        assert SEED_REPLICATES == {
            "silver_kestrel": ("amber_kestrel", 42),
            "bronze_kestrel": ("amber_kestrel", 43),
            "cobalt_kestrel": ("amber_kestrel", 44),
        }, (
            f"SEED_REPLICATES changed: {SEED_REPLICATES}. Update this literal in the same "
            f"commit and say in the message which variance estimate the replicate serves. "
            f"⛔ The three seeds must stay DISTINCT: identical seeds deflate σ̂, drive every "
            f"threshold toward zero and manufacture a positive (views-models#567)."
        )

    def test_paired_models_spans_every_declaration(self):
        """PAIRED_MODELS is derived; this asserts it is derived from everything.

        The hole this mapping closes is that a model outside the parametrize list has no
        pair guard at all. A third declaration kind added later and left out of PAIRED_MODELS
        would inherit exactly that hole, which is why the mapping is checked against its
        sources rather than trusted for being a comprehension.
        """
        assert set(PAIRED_MODELS) == set(EXPERIMENT_ARMS) | set(SEED_REPLICATES), (
            "PAIRED_MODELS does not span EXPERIMENT_ARMS | SEED_REPLICATES; a declared pair "
            "outside it is guarded by nothing — not byte identity, not meta, not maturity"
        )

    def test_every_pair_guard_runs_over_every_declared_pair(self):
        """The guard on the five pair guards' own SCOPE — the weak point of this change.

        A guard's real scope is its `parametrize` decorator, not the mapping the decorator
        happens to be handed today. Narrowing one back to `sorted(EXPERIMENT_ARMS)` is a
        one-line edit that drops every replicate from it and leaves the entire suite green,
        because with no replicate declared the collected cases are identical either way.

        ⛔ Not hypothetical. `e874be0f` records that exact mutation SURVIVING in
        `test_roster_configs_load.py` — "TRACKED_MODELS narrowed to the roster -> SURVIVED" —
        which is why `_parametrized_scope` exists there to read `pytestmark`. This file had
        no equivalent, and `test_paired_models_spans_every_declaration` asserts on the
        MAPPING, which is the one thing the mutation does not touch.

        Guards are DISCOVERED by their argname rather than listed by name, and the count is
        pinned, so the two ways out are both closed: narrowing a decorator fails the scope
        check, and renaming its argname out of discovery fails the count.
        """
        cls = type(self)
        guards = {
            name: getattr(cls, name)
            for name in dir(cls)
            if name.startswith("test_")
            and any(
                m.name == "parametrize" and m.args[0] == _PAIR_ARGNAME
                for m in getattr(getattr(cls, name), "pytestmark", [])
            )
        }
        assert len(guards) == 5, (
            f"{len(guards)} guards are parametrized on {_PAIR_ARGNAME!r}, expected 5: "
            f"{sorted(guards)}. A pair guard renamed off this parameter is no longer "
            f"discovered here and loses its scope check — the hole this test exists to "
            f"close. Add the new guard and update this count in the same commit."
        )
        for name, func in sorted(guards.items()):
            assert _parametrized_values(func, _PAIR_ARGNAME) == set(PAIRED_MODELS), (
                f"{name} runs over {sorted(_parametrized_values(func, _PAIR_ARGNAME))}, "
                f"but the declared pairs are {sorted(PAIRED_MODELS)}. A pair outside a "
                f"guard's parametrize list is not checked by it at all, and a smaller "
                f"parametrization is not a failure — the only trace is one fewer case."
            )

    def test_a_replicate_is_not_also_an_arm(self):
        """A model is a contrast or a noise measurement, never both.

        Declared as both, it would be read as an arm by the one-variable guard and as a
        replicate by the seed guard, and the two make opposite demands: an arm must differ
        from its control, a replicate must not.

        ⛔ Asserted on the set INTERSECTION rather than parametrized over the replicates,
        because parametrizing a collision check over one of the two colliding collections
        makes it collect zero cases while that collection is empty — and the collision it
        guards against is also what `PAIRED_MODELS`' `{{**arms, **replicates}}` merge
        resolves SILENTLY, in the replicate's favour, with
        `test_paired_models_spans_every_declaration` unable to see it because the key sets
        are identical either way.
        """
        both = set(EXPERIMENT_ARMS) & set(SEED_REPLICATES)
        assert not both, (
            f"{sorted(both)} declared as both a seed replicate and an experiment arm. An "
            f"arm must differ from its control and a replicate must not, so the two guards "
            f"make opposite demands — and PAIRED_MODELS would silently resolve the name to "
            f"the REPLICATE's control, so the arm would be checked against the wrong model."
        )

    def test_every_declared_control_is_itself_a_declared_model(self):
        """A control named in a declaration must resolve, and a replicate's must be a real run.

        Nothing checked this. A typo in a control name failed only as a FileNotFoundError
        raised from inside a byte-identity assertion, which names a path and not the
        declaration that is wrong.

        ⛔ And the second half is load-bearing for `σ̂`: a replicate whose control is itself
        a replicate makes the "three runs of one config" a chain rather than a cell, and the
        spread computed over it is not the quantity views-models#565 pays for. The arms are
        deliberately NOT held to that rule — `indigo_kestrel`'s control is `silver_kestrel`,
        a replicate, and that pairing is the whole point of the seed-matched design.
        """
        declared = set(ROSTER_MODELS) | set(EXPERIMENT_ARMS) | set(SEED_REPLICATES)
        for model, control in sorted(PAIRED_MODELS.items()):
            assert control in declared, (
                f"{model} is declared against control {control!r}, which is not a declared "
                f"model. Declared: {sorted(declared)}. A control that does not resolve "
                f"fails later as a missing file and names the path, not this declaration."
            )
        for replicate, (control, _seed) in sorted(SEED_REPLICATES.items()):
            assert control not in SEED_REPLICATES, (
                f"{replicate} is a replicate of {control}, which is itself a replicate. "
                f"Replicates must all point at the SAME control, or the spread over them is "
                f"a chain of comparisons rather than the run-to-run variance of one config."
            )

        # ⛔ And the sentence above is now ASSERTED rather than merely written down. The
        # docstring claimed "replicates must all point at the SAME control" while the loop
        # only forbade a replicate-of-a-replicate, so two cells against two different
        # controls would have passed — and `σ̂` pooled across them would be a spread of two
        # populations, not the run-to-run variance of one config. A docstring that states a
        # rule the code does not check is the shape C-163 is registered for.
        # ⛔ And an ARM must share its control's SEED. Carried from views-models#567, where
        # review found nothing asserted it. It is what makes indigo/silver and
        # russet/bronze one-variable contrasts: same seed, one differing key. An arm whose
        # seed drifted from its control's would silently become a TWO-variable contrast,
        # and the one-variable guard cannot see it — the seed keys are exempt there only
        # because REPLICATE_FREE_KEYS and INSTRUMENTATION_KEYS are, so a seed difference
        # would read as admissible rather than as a confound.
        #
        # Universal here, not a special case: an arm can never declare the seed as its
        # variable, because ARM_FREE_KEYS is asserted disjoint from REPLICATE_FREE_KEYS.
        # A seed contrast is a replicate by construction.
        for arm, (control, _declared) in sorted(EXPERIMENT_ARMS.items()):
            arm_hp, control_hp = _load_hp(arm), _load_hp(control)
            for key in sorted(REPLICATE_FREE_KEYS):
                assert arm_hp.get(key) == control_hp.get(key), (
                    f"{arm} has {key}={arm_hp.get(key)!r} but its control {control} has "
                    f"{control_hp.get(key)!r}. An arm must share its control's seed, or the "
                    f"contrast varies TWO things and the comparison is confounded "
                    f"(tests/test_roster_conformance.py:66-70). If a seed contrast is what "
                    f"is wanted, declare a SEED REPLICATE instead."
                )

        controls = {control for control, _seed in SEED_REPLICATES.values()}
        assert len(controls) <= 1, (
            f"the replicates point at {sorted(controls)} — more than one control. σ̂ is the "
            f"spread of ONE config re-run, so a cell split across two controls pools two "
            f"populations and the estimate is not the quantity views-models#565 pays for. "
            f"If two cells are genuinely wanted, they need separate declarations and a "
            f"separate σ̂ each."
        )

    @pytest.mark.parametrize("replicate", sorted(SEED_REPLICATES))
    def test_the_replicate_is_its_declared_seed(self, replicate):
        """The seed is the replicate's whole identity, so it is pinned like an arm's values.

        Both keys must be PRESENT and both must equal the declared value. Presence is checked
        first and separately: the previous version subscripted `hp` inside its own failure
        message, so a missing key raised `KeyError` from the assertion and then `KeyError`
        again from the message meant to diagnose it, leaving the reader a bare key name and
        no model.
        """
        control, seed = SEED_REPLICATES[replicate]
        hp = _load_hp(replicate)
        missing = sorted(k for k in ("torch_seed", "np_seed") if k not in hp)
        assert not missing, (
            f"{replicate} declares no {missing}, so its seed is whatever the library "
            f"defaults to and the run it is meant to replicate is not pinned at all"
        )
        assert hp["torch_seed"] == seed and hp["np_seed"] == seed, (
            f"{replicate}: declared seed {seed}, config has torch_seed "
            f"{hp.get('torch_seed')!r} / np_seed {hp.get('np_seed')!r}"
        )

        # ⚠️ WITHDRAWN, deliberately: an earlier version of this guard also required the
        # seed to DIFFER from the control's. That is wrong here, and the reason is worth
        # keeping because the assertion looked obviously right.
        #
        # It assumed the control is a MEMBER of the variance cell, so that a replicate
        # sharing its seed would be a duplicate measuring nothing. The control is not a
        # member. views-hydranet's pre-registration §7 rules it out explicitly:
        # `amber_kestrel` "was trained on a different pod session at a different time under
        # a different code sha", so pooling it would import every difference between then
        # and now INTO the variance estimate — the one quantity the design rests on. It is a
        # CONFIG REFERENCE ("base config amber_kestrel, so controls are directly comparable
        # to the published b = 1.1875"), not a run in the cell.
        #
        # So `silver_kestrel` carrying seed 42, which is also amber's, is the approved
        # design (§3's table: 42 / 43 / 44) and not a defect. The guard would have refused
        # the first real replicate it ever saw.
        #
        # ⛔ The invariant that IS real lives in `test_the_replicates_are_distinct_runs`:
        # distinctness WITHIN the cell. That is the direction that can hurt us, and it is
        # not symmetric with config divergence — see that test.

    def test_the_replicates_are_distinct_runs(self):
        """No two replicates may share a seed. The cell is the subject, not the control.

        Not parametrized: a pairwise property cannot be checked one subject at a time, and
        parametrizing it over `SEED_REPLICATES` would also make it collect zero cases while
        the mapping is empty.

        ⛔ **This is the one corruption in this design that fails UNSAFE, and the direction
        is why it matters more than the config-divergence guard beside it.** Every threshold
        in the pre-registration is `C = k·σ̂`, so `σ̂` sits in the DENOMINATOR:

            config divergence  ->  INFLATES σ̂  ->  thresholds rise  ->  effects fail
                                   = mimics the null we already expect.   CONSERVATIVE.
            seed collision     ->  DEFLATES σ̂  ->  thresholds -> 0    ->  anything clears
                                   = mimics a RESOLVED EFFECT.          ANTI-CONSERVATIVE.

        Identical seeds can only make runs MORE alike than distinct seeds would, so `σ̂` can
        only be under-stated and a threshold divided by an under-stated spread can only be
        too permissive. ⚠️ **That argument needs no determinism assumption**, which matters
        because Amendment 6 withdrew determinism as a precondition of this design.

        Independently reached by the views-hydranet seat as Amendment 11, which added the
        same requirement to F13 as a third clause: its `(b)` EXEMPTS the seed keys from the
        identity check, and an exemption cannot express a requirement that they differ.
        """
        seeds: dict[int, list[str]] = {}
        for replicate, (_control, seed) in sorted(SEED_REPLICATES.items()):
            seeds.setdefault(seed, []).append(replicate)
        collisions = {seed: names for seed, names in seeds.items() if len(names) > 1}
        assert not collisions, (
            f"replicates share a seed: {collisions}. Identical seeds mean identical runs, "
            f"so they add no spread — the variance estimate shrinks toward zero and so does "
            f"every threshold divided by it, making a null read as a resolved effect."
        )

    @pytest.mark.parametrize("replicate", sorted(SEED_REPLICATES))
    def test_the_replicate_differs_from_its_control_only_on_the_seed(self, replicate):
        """The inverse of the arm guard, and the reason this declaration exists.

        An arm must differ from its control; a replicate must not, beyond the seed. So there
        is no "there is no arm to run" inverse assertion here — a replicate that is config-
        identical to its control is the valid case, not the degenerate one.

        ⚠️ This guard permits differing on NOTHING at all, and that is correct: the control
        is a config reference, not a member of the variance cell, so a replicate whose
        config matches it exactly — seed included — is the valid case. ⛔ What must not
        happen is two replicates matching EACH OTHER, and that is asserted in
        `test_the_replicates_are_distinct_runs`, over the cell rather than against the
        control. An earlier version of this file asserted the control-relative form; it was
        withdrawn in #567 because it is both wrong and weaker — see the comment in
        `test_the_replicate_is_its_declared_seed`.

        What makes this load-bearing rather than tidy: the variance estimate these runs exist
        to produce is the spread of their results, so a config divergence among them enters
        that estimate as variance. It inflates it, every threshold derived from it rises, and
        every effect fails significance — which is indistinguishable from an honest null. The
        corruption would look like a measurement, not like a bug.
        """
        control = SEED_REPLICATES[replicate][0]
        admissible = REPLICATE_FREE_KEYS | INSTRUMENTATION_KEYS
        differing = _differing_keys(_load_hp(replicate), _load_hp(control))
        assert differing <= admissible, (
            f"{replicate} differs from its control {control} on "
            f"{sorted(differing - admissible)}. A replicate re-runs the SAME config to "
            f"measure run-to-run variance, so any other difference is counted as variance "
            f"and inflates the estimate — raising every threshold derived from it and making "
            f"a real effect fail significance, indistinguishably from an honest null. Revert "
            f"it, or declare an experiment arm instead and say what it varies."
        )
