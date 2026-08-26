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

import ast
import importlib.util
from pathlib import Path

import pytest

from tests.conftest import get_regression_targets

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"

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
    """Every pinned member reads views-datafactory / africa_me_legacy.

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
    def test_africa_region(self, model_name):
        assert "africa_me_legacy" in _queryset_text(model_name), (
            f"{model_name} missing africa_me_legacy region"
        )

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
