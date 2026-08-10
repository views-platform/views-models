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

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

pytestmark = [pytest.mark.green]

# --- The roster: model -> (output_distribution, forecast_composition,
#     gate_threshold, seed). gate_threshold is None for soft_gate. ---
ROSTER = {
    "violet_visitor":  ("nb", "soft_gate", None, 42),
    "bright_starship": ("nb", "soft_gate", None, 43),
    "bold_comet":      ("nb", "soft_gate", None, 44),
    "blazing_meteor":  ("nb", "threshold_gate", 0.5, 45),
    "heavy_freighter": ("nb", "threshold_gate", 0.5, 46),
    "pink_pirate":     ("mixture_nb", "soft_gate", None, 42),
    "blue_stranger":   ("mixture_nb", "soft_gate", None, 43),
    "purple_alien":    ("mixture_nb", "soft_gate", None, 44),
}
ROSTER_MODELS = list(ROSTER)

# --- Models exempt from the exact-value pins because their config declares
#     EXPERIMENT_IN_PROGRESS. Pinned as a SET so that both adding and removing an
#     exemption is a deliberate, reviewed edit rather than a silent config change.
#
#     This mechanism is inherited verbatim from test_datafactory_parity.py, which was
#     its only reader in the entire repository. Deleting that file without re-homing
#     this here would have removed the escape hatch silently — and the marker's own
#     text in models/violet_visitor/configs/config_hyperparameters.py instructs the
#     reader to remove it "when the roster lands", which is a decision for whoever
#     owns the experiment, not a side effect of a test rewrite. ---
EXPERIMENTS_IN_PROGRESS = {"violet_visitor"}

#: Roster members whose values ARE pinned — everything not mid-experiment.
PINNED_MODELS = [m for m in ROSTER_MODELS if m not in EXPERIMENTS_IN_PROGRESS]

# --- The shared v2 gated_NB foundation every pinned member holds fixed. Values that
#     differ per member (family, composition, seed) live in ROSTER, not here.
#     total_lessons is a RUN-TIME budget (amended 300->160, window-constrained) and is
#     deliberately NOT pinned. ---
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


class TestDatafactorySource:
    """Every member reads views-datafactory / africa_me_legacy."""

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_uses_datafactory(self, model_name):
        text = _queryset_text(model_name)
        assert (
            '"source": "views-datafactory"' in text
            or "'source': 'views-datafactory'" in text
        ), f"{model_name} does not declare a views-datafactory source"

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_africa_region(self, model_name):
        assert "africa_me_legacy" in _queryset_text(model_name), (
            f"{model_name} missing africa_me_legacy region"
        )

    @pytest.mark.parametrize("model_name", PINNED_MODELS)
    def test_no_viewser_import(self, model_name):
        """Exempt for in-flight models: retiring the import is part of settling them.

        violet_visitor declares a views-datafactory source AND still imports viewser —
        a leftover from before the S2 migration, which covered the other three viewser
        models but not the one that was mid-experiment. Removing it is an edit to a
        fenced config, so it belongs to whoever un-fences the model.
        """
        text = _queryset_text(model_name)
        assert "from viewser" not in text and "import viewser" not in text, (
            f"{model_name} imports viewser — should use datafactory"
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


# ── the rusty_bucket rewiring is NOT here ─────────────────────────────
# Swapping the ensemble's 8 `temporary_*` stand-ins for the 8 roster models is S3's
# delivery unit, and it cannot land while violet_visitor is exempt: rusty_bucket
# declares `expected_samples_per_model: 16` (ADR-015 §2/§3), the seven pinned members
# emit D×K = 4×4 = 16, and violet emits 8 (D=8, no head sampler). Rewiring now would
# fail `test_ensemble_configs.py::test_declared_modelset_and_sample_counts_match_reality`
# — correctly. The ensemble is rewired in the change that settles violet.
