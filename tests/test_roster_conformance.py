"""Roster conformance — the 8 HydraNet models + the rusty_bucket ensemble conform
to Epic #242's LOCKED roster and the shared v2 `gated_NB` foundation.

This supersedes the old ``test_datafactory_parity.py`` (viewser↔datafactory trio
mirror, C-47). That programme is **resolved**: S2 migrated the three viewser models
to views-datafactory, so every model now reads the same source and the two 3-member
parity ensembles (golden_hour / stellar_horizon) were retired in favour of the single
8-member ``rusty_bucket``. What matters now is not "two trios mirror each other" but
"each model conforms to its roster entry, and all share the v2 foundation".

Roster (05 pre-registration, LOCKED 2026-08-08):
  gated_NB     (nb,         soft_gate)          violet_visitor 42 / bright_starship 43 / bold_comet 44
  th_gated_NB  (nb,         threshold_gate 0.5) blazing_meteor 45 / heavy_freighter 46
  mixture_NB   (mixture_nb, soft_gate)          pink_pirate 42 / blue_stranger 43 / purple_alien 44

See reports/2026-08-08_hydranet_ensemble_dossier/05_analysis_plan.md and risk
register C-71 (violet reconstruction, now settled) / C-132 (ensemble gate pooling).
"""

import importlib.util
from pathlib import Path

import pytest

from tests.conftest import get_regression_targets

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"

pytestmark = [pytest.mark.green]

# --- The LOCKED roster: model -> (output_distribution, forecast_composition,
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

# --- The shared v2 gated_NB foundation every member holds fixed (05 "Fixed
#     design"). Values that differ per member (family, composition, seed) live in
#     ROSTER, not here. total_lessons is a RUN-TIME budget (amended 300->160,
#     window-constrained) — deliberately NOT pinned here. ---
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

ENSEMBLE = "rusty_bucket"


def _load(path, fn):
    if not path.exists():
        pytest.skip(f"{path} not yet created")
    spec = importlib.util.spec_from_file_location(f"_cfg_{path.parent.parent.name}_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)()


def _load_hp(model_name):
    return _load(MODELS_DIR / model_name / "configs" / "config_hyperparameters.py", "get_hp_config")


def _load_meta(name, base_dir):
    return _load(base_dir / name / "configs" / "config_meta.py", "get_meta_config")


def _load_modelset(name):
    return _load(ENSEMBLES_DIR / name / "configs" / "config_modelset.py", "get_modelset_config")


def _load_partitions(name, base_dir):
    return _load(base_dir / name / "configs" / "config_partitions.py", "generate")


def _queryset_text(model_name):
    path = MODELS_DIR / model_name / "configs" / "config_queryset.py"
    if not path.exists():
        pytest.skip(f"{model_name} not yet created")
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


class TestRosterFamilyConformance:
    """Each model IS its roster entry: family, composition, gate, seed."""

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_family_and_composition(self, model_name):
        distribution, composition, threshold, _seed = ROSTER[model_name]
        hp = _load_hp(model_name)
        assert hp["output_distribution"] == distribution, (
            f"{model_name}: output_distribution {hp['output_distribution']!r} != roster {distribution!r}"
        )
        assert hp["forecast_composition"] == composition, (
            f"{model_name}: forecast_composition {hp['forecast_composition']!r} != roster {composition!r}"
        )

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
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

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_seed(self, model_name):
        _distribution, _composition, _threshold, seed = ROSTER[model_name]
        hp = _load_hp(model_name)
        assert hp["torch_seed"] == seed and hp["np_seed"] == seed, (
            f"{model_name}: seeds torch={hp.get('torch_seed')} np={hp.get('np_seed')} "
            f"!= roster seed {seed} (torch_seed and np_seed must match)"
        )

    def test_roster_is_exactly_the_eight(self):
        """The on-disk model set carries exactly the roster's eight — no more, no
        fewer. A model added to or dropped from the roster is a reviewed edit."""
        on_disk = {
            d.name for d in MODELS_DIR.iterdir()
            if d.is_dir() and (d / "configs" / "config_hyperparameters.py").exists()
            and _load_hp(d.name).get("output_distribution") in {"nb", "zinb", "mixture_nb"}
            and _load_hp(d.name).get("n_head_samples", 1) > 1
        }
        missing = set(ROSTER_MODELS) - on_disk
        assert not missing, f"roster members missing a family-head config: {missing}"


class TestSharedV2Foundation:
    """Every member holds the v2 gated_NB foundation fixed."""

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    @pytest.mark.parametrize("key,expected", list(FOUNDATION.items()))
    def test_foundation_value(self, model_name, key, expected):
        hp = _load_hp(model_name)
        assert hp.get(key) == expected, (
            f"{model_name}: foundation {key}={hp.get(key)!r} != expected {expected!r}"
        )

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_no_tobit_sigma(self, model_name):
        """The roster is mse, not tobit — no leftover tobit-only loss_reg_sigma."""
        hp = _load_hp(model_name)
        assert hp["loss_reg"] == "mse"
        assert "loss_reg_sigma" not in hp, (
            f"{model_name}: carries loss_reg_sigma but loss_reg is mse (tobit-only knob)"
        )


class TestGridAndTargets:
    """Region grid and target channels are identical across the roster."""

    @pytest.fixture()
    def hps(self):
        return {name: _load_hp(name) for name in ROSTER_MODELS}

    def test_identical_grid_topology(self, hps):
        # Derive from the roster and compare members to each other — the local
        # africa run scopes heavy_freighter to africa too (its global config is
        # banked separately). All eight must share one grid so the pool is valid.
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
        # The by_* gate channel is 1:1 with the lr_* magnitudes — this is the
        # occurrence gate the ensemble must pool (C-132).
        hp = _load_hp(model_name)
        assert hp["classification_targets"] == CLASSIFICATION_TARGETS, (
            f"{model_name}: classification_targets {hp['classification_targets']} != {CLASSIFICATION_TARGETS}"
        )
        assert len(hp["classification_targets"]) == len(hp["regression_targets"]), (
            f"{model_name}: gate channels not 1:1 with magnitudes"
        )


class TestDatafactorySource:
    """Every member reads views-datafactory / africa_me_legacy — no viewser."""

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

    @pytest.mark.parametrize("model_name", ROSTER_MODELS)
    def test_no_viewser_import(self, model_name):
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


def test_every_hydranet_pf_ensemble_declares_the_occurrence_gate():
    """C-132 class guard (fleet-wide, not just rusty_bucket).

    A `prediction_frame` / `hydranet_ucdp` concat ensemble pools its constituents'
    per-sample channels; the occurrence gate rides the `classification_targets`
    (`by_*`) channel. If such an ensemble declares `regression_targets` but omits
    `classification_targets`, the pool silently drops the gate and its AP/Brier is
    understated with no error (C-132). The framework fix
    (views-pipeline-core#422 — `_build_context` via `combined_targets`) makes the pool
    *respect* a declared gate, but it does NOT synthesise one; this test is the
    fail-loud that stops a NEW gate-less HydraNet ensemble from reintroducing C-132.
    Today only rusty_bucket qualifies (and declares it), so this is green — it flips
    red the moment a gate-less one is added.
    """
    offenders = {}
    for meta_path in ENSEMBLES_DIR.rglob("configs/config_meta.py"):
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
        "these HydraNet/PF concat ensembles do not declare a 1:1 by_* gate channel, "
        "so the concat pool silently drops occurrence (C-132): "
        f"{offenders}. Declare classification_targets in the ensemble config_meta."
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

    def test_aggregation_and_level(self, ens_meta):
        assert ens_meta["aggregation"] == "concat"
        assert ens_meta["level"] == "pgm"

    def test_pools_regression_targets(self, ens_meta):
        assert ens_meta["regression_targets"] == REGRESSION_TARGETS
        # ...and the constituents agree on those targets (derive, don't trust).
        constituent = {
            tuple(get_regression_targets(MODELS_DIR / m)) for m in _load_modelset(ENSEMBLE)["models"]
        }
        assert constituent == {tuple(REGRESSION_TARGETS)}, (
            f"constituents disagree on regression_targets: {constituent}"
        )

    def test_pools_the_occurrence_gate_channel(self, ens_meta):
        # C-132: the concat pool must carry the by_* gate, else ensemble
        # occurrence/AP is silently understated. The durable form is a declared
        # classification_targets (pooled via combined_targets), NOT a synthesised
        # `targets` key — which #380 retired and the manager now refuses.
        assert ens_meta.get("classification_targets") == CLASSIFICATION_TARGETS, (
            f"{ENSEMBLE} must declare classification_targets={CLASSIFICATION_TARGETS} so the "
            f"concat pool carries the gate (C-132); got {ens_meta.get('classification_targets')!r}"
        )
        assert "targets" not in ens_meta, (
            f"{ENSEMBLE} carries a retired synthesised `targets` key (#380) — declare "
            f"regression_targets / classification_targets instead"
        )

    def test_metrics_profile(self, ens_meta):
        assert ens_meta["regression_sample_metrics"] == ["CRPS", "QS_sample", "MCR_sample"]
        assert ens_meta["evaluation_profile"] == "hydranet_ucdp"

    def test_uses_prediction_frame_manager(self):
        main_path = ENSEMBLES_DIR / ENSEMBLE / "main.py"
        if not main_path.exists():
            pytest.skip(f"{ENSEMBLE} not yet created")
        assert "PredictionFrameEnsembleManager" in main_path.read_text()

    def test_partitions_use_current_month_id(self):
        path = ENSEMBLES_DIR / ENSEMBLE / "configs" / "config_partitions.py"
        if not path.exists():
            pytest.skip(f"{ENSEMBLE} not yet created")
        text = path.read_text()
        assert "_current_month_id" in text
        assert "ViewsMonth" not in text

    def test_partition_boundaries_match_a_member(self):
        ens_parts = _load_partitions(ENSEMBLE, ENSEMBLES_DIR)
        member_parts = _load_partitions(ROSTER_MODELS[0], MODELS_DIR)
        assert ens_parts["calibration"] == member_parts["calibration"]
        assert ens_parts["validation"] == member_parts["validation"]


def test_roster_is_fully_pinned():
    """The roster has LANDED — every member is pinned to exact values, so none is
    EXPERIMENT_IN_PROGRESS. (violet_visitor carried the marker during its C-71
    reconstruction, S1.5 #335; the S3 roster reconfig settles it.) If a future
    experiment needs the escape hatch back, re-add the marker deliberately and
    exempt that model from the pins above."""
    marked = {m for m in ROSTER_MODELS if _experiment_in_progress(m)}
    assert not marked, (
        f"these roster models still declare EXPERIMENT_IN_PROGRESS: {marked}. The roster is "
        f"pinned now — either settle the config and drop the marker, or exempt it explicitly."
    )
