"""Datafactory parity tests — verify the datafactory HydraNet trio and ensemble
mirror the viewser trio (golden_hour) in every dimension except data source.

The datafactory trio (all tobit, mirroring pink_pirate):
  - bright_starship
  - bold_comet
  - blazing_meteor

The datafactory ensemble:
  - stellar_horizon (concat, PredictionFrame) — new

See risk register C-47 and golden_hour pre-analysis plan for context.
"""

import importlib.util
from pathlib import Path

import pytest

from tests.conftest import get_regression_targets

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"

DATAFACTORY_TRIO = ["bright_starship", "bold_comet", "blazing_meteor"]
VIEWSER_TRIO = ["pink_pirate", "blue_stranger", "violet_visitor"]
DF_ENSEMBLE = "stellar_horizon"
VS_ENSEMBLE = "golden_hour"

pytestmark = [pytest.mark.green]

ALL_LOSS_KEYS = {"loss_reg", "loss_reg_sigma"}


def _load_hp(model_name):
    path = MODELS_DIR / model_name / "configs" / "config_hyperparameters.py"
    if not path.exists():
        pytest.skip(f"{model_name} not yet created")
    spec = importlib.util.spec_from_file_location(f"_hp_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


def _load_meta(model_name, base_dir=None):
    if base_dir is None:
        base_dir = MODELS_DIR
    path = base_dir / model_name / "configs" / "config_meta.py"
    if not path.exists():
        pytest.skip(f"{model_name} not yet created")
    spec = importlib.util.spec_from_file_location(f"_meta_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_meta_config()


def _load_modelset(ensemble_name, base_dir=None):
    if base_dir is None:
        base_dir = ENSEMBLES_DIR
    path = base_dir / ensemble_name / "configs" / "config_modelset.py"
    if not path.exists():
        pytest.skip(f"{ensemble_name} config_modelset.py not yet created")
    spec = importlib.util.spec_from_file_location(f"_modelset_{ensemble_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_modelset_config()


def _load_partitions(model_name, base_dir=None):
    if base_dir is None:
        base_dir = MODELS_DIR
    path = base_dir / model_name / "configs" / "config_partitions.py"
    if not path.exists():
        pytest.skip(f"{model_name} not yet created")
    spec = importlib.util.spec_from_file_location(f"_part_{model_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate()


class TestDatafactoryTrioConfigParity:
    """All three datafactory models must share identical hyperparams."""

    @pytest.fixture()
    def trio_hps(self):
        return {name: _load_hp(name) for name in DATAFACTORY_TRIO}

    def _shared_keys(self, hp):
        return {k: v for k, v in hp.items() if k not in ALL_LOSS_KEYS}

    def test_identical_shared_hyperparameters(self, trio_hps):
        shared = [self._shared_keys(hp) for hp in trio_hps.values()]
        for i in range(1, len(shared)):
            assert shared[0] == shared[i], (
                f"{DATAFACTORY_TRIO[0]} vs {DATAFACTORY_TRIO[i]} shared hyperparams differ"
            )

    def test_identical_model_architecture(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["model"] == "HydraBNUNet06_LSTM4", f"{name} wrong architecture"

    def test_identical_features(self, trio_hps):
        # Parity intent: the trio are identical in features and targets. Derive from
        # config and compare members to EACH OTHER — no hardcoded target-name literal.
        items = list(trio_hps.items())
        ref_name, ref = items[0]
        for name, hp in items[1:]:
            for key in ("features", "regression_targets", "classification_targets"):
                assert hp[key] == ref[key], (
                    f"{name} {key} {hp[key]} != {ref_name} {ref[key]}"
                )
        # Structural: targets are declared, and the hurdle gate's classification
        # targets are 1:1 with the regression targets (not name-based).
        for name, hp in trio_hps.items():
            assert hp["regression_targets"], f"{name} declares no regression_targets"
            assert len(hp["classification_targets"]) == len(hp["regression_targets"]), (
                f"{name}: classification_targets {hp['classification_targets']} not 1:1 with "
                f"regression_targets {hp['regression_targets']}"
            )

    def test_identical_grid_topology(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["row_offset"] == 87, f"{name} row_offset"
            assert hp["col_offset"] == 310, f"{name} col_offset"
            assert hp["height"] == 180, f"{name} height"
            assert hp["width"] == 180, f"{name} width"

    def test_identical_posterior_samples(self, trio_hps):
        values = {name: hp["n_posterior_samples"] for name, hp in trio_hps.items()}
        assert len(set(values.values())) == 1, f"n_posterior_samples mismatch: {values}"

    def test_identical_classification_loss(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["loss_class"] == "focal", f"{name} loss_class"
            assert hp["loss_class_alpha"] == 0.75, f"{name} loss_class_alpha"
            assert hp["loss_class_gamma"] == 1.5, f"{name} loss_class_gamma"

    def test_loss_reg_matches(self, trio_hps):
        loss_values = {hp["loss_reg"] for hp in trio_hps.values()}
        assert loss_values == {"tobit"}

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_loss_params_correct(self, model_name):
        hp = _load_hp(model_name)
        assert hp["loss_reg"] == "tobit", f"{model_name}: loss_reg expected tobit, got {hp.get('loss_reg')}"
        sigma = hp.get("loss_reg_sigma")
        assert isinstance(sigma, dict) and sigma, (
            f"{model_name}: loss_reg_sigma must be a non-empty dict, got {sigma!r}"
        )
        # loss_reg_sigma is keyed by the model's OWN regression targets — derive,
        # don't hardcode the names.
        assert set(sigma) == set(hp["regression_targets"]), (
            f"{model_name}: loss_reg_sigma keys {set(sigma)} != regression_targets "
            f"{set(hp['regression_targets'])}"
        )

    def test_identical_loss_reg_sigma(self, trio_hps):
        # Parity: the trio share the same loss_reg_sigma (values too) — compare
        # members to each other rather than to a hardcoded mapping.
        sigmas = [hp.get("loss_reg_sigma") for hp in trio_hps.values()]
        for i in range(1, len(sigmas)):
            assert sigmas[0] == sigmas[i], (
                f"trio loss_reg_sigma differ: {DATAFACTORY_TRIO[0]}={sigmas[0]} "
                f"vs {DATAFACTORY_TRIO[i]}={sigmas[i]}"
            )


class TestDatafactoryTrioDataSource:
    """All three must use datafactory, same zarr store, same region."""

    @pytest.fixture()
    def trio_queryset_texts(self):
        texts = {}
        for name in DATAFACTORY_TRIO:
            path = MODELS_DIR / name / "configs" / "config_queryset.py"
            if not path.exists():
                pytest.skip(f"{name} not yet created")
            texts[name] = path.read_text()
        return texts

    def test_all_use_datafactory_source(self, trio_queryset_texts):
        for name, text in trio_queryset_texts.items():
            assert '"source": "views-datafactory"' in text or "'source': 'views-datafactory'" in text, (
                f"{name} does not declare views-datafactory source"
            )

    def test_identical_region(self, trio_queryset_texts):
        for name, text in trio_queryset_texts.items():
            assert "africa_me_legacy" in text, f"{name} missing africa_me_legacy region"

    def test_identical_feature_rename(self, trio_queryset_texts):
        for name, text in trio_queryset_texts.items():
            assert "ged_sb_best" in text, f"{name} missing ged_sb_best"
            assert "ged_ns_best" in text, f"{name} missing ged_ns_best"
            assert "ged_os_best" in text, f"{name} missing ged_os_best"

    def test_no_viewser_import(self, trio_queryset_texts):
        for name, text in trio_queryset_texts.items():
            assert "from viewser" not in text and "import viewser" not in text, (
                f"{name} imports viewser — should use datafactory"
            )


class TestDatafactoryTrioMetaConfig:
    """Meta config consistency across the trio."""

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_algorithm_is_hydranet(self, model_name):
        meta = _load_meta(model_name)
        assert meta["algorithm"] == "HydraNet"

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_level_is_pgm(self, model_name):
        meta = _load_meta(model_name)
        assert meta["level"] == "pgm"

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_prediction_format_is_prediction_frame(self, model_name):
        meta = _load_meta(model_name)
        assert meta["prediction_format"] == "prediction_frame"

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_name_matches_directory(self, model_name):
        meta = _load_meta(model_name)
        assert meta["name"] == model_name


class TestStellarHorizonEnsembleConfig:
    """Verify stellar_horizon ensemble is correctly configured."""

    @pytest.fixture()
    def ens_meta(self):
        return _load_meta(DF_ENSEMBLE, ENSEMBLES_DIR)

    def test_models_list(self):
        modelset = _load_modelset(DF_ENSEMBLE)
        assert modelset["models"] == ["bright_starship", "bold_comet", "blazing_meteor"]

    def test_regression_targets_match_constituents(self, ens_meta):
        # The ensemble's targets must match its constituents' (the datafactory trio).
        # Derive both — no hardcoded literal.
        modelset = _load_modelset(DF_ENSEMBLE)
        constituent_targets = {
            tuple(get_regression_targets(MODELS_DIR / m)) for m in modelset["models"]
        }
        assert len(constituent_targets) == 1, (
            f"datafactory constituents disagree on regression_targets: {constituent_targets}"
        )
        ens_targets = ens_meta.get("regression_targets")
        assert ens_targets, "stellar_horizon declares no regression_targets"
        assert tuple(ens_targets) == next(iter(constituent_targets)), (
            f"stellar_horizon regression_targets {ens_targets} != constituents' "
            f"{next(iter(constituent_targets))}"
        )

    def test_aggregation_is_concat(self, ens_meta):
        assert ens_meta["aggregation"] == "concat"

    def test_level_is_pgm(self, ens_meta):
        assert ens_meta["level"] == "pgm"

    def test_metrics(self, ens_meta):
        assert ens_meta["regression_sample_metrics"] == ["CRPS", "QS_sample", "MCR_sample"]

    def test_uses_prediction_frame_manager(self):
        main_path = ENSEMBLES_DIR / DF_ENSEMBLE / "main.py"
        if not main_path.exists():
            pytest.skip("stellar_horizon not yet created")
        text = main_path.read_text()
        assert "PredictionFrameEnsembleManager" in text

    def test_partitions_use_current_month_id(self):
        path = ENSEMBLES_DIR / DF_ENSEMBLE / "configs" / "config_partitions.py"
        if not path.exists():
            pytest.skip("stellar_horizon not yet created")
        text = path.read_text()
        assert "_current_month_id" in text
        assert "ViewsMonth" not in text

    def test_partition_boundaries_match_bright_starship(self):
        ens_parts = _load_partitions(DF_ENSEMBLE, ENSEMBLES_DIR)
        bs_parts = _load_partitions("bright_starship", MODELS_DIR)
        assert ens_parts["calibration"] == bs_parts["calibration"]
        assert ens_parts["validation"] == bs_parts["validation"]


class TestCrossEnsembleParityReadiness:
    """golden_hour and stellar_horizon must have matching config for valid CRPS comparison."""

    @pytest.fixture()
    def both_metas(self):
        vs = _load_meta(VS_ENSEMBLE, ENSEMBLES_DIR)
        df = _load_meta(DF_ENSEMBLE, ENSEMBLES_DIR)
        return vs, df

    def test_same_regression_targets(self, both_metas):
        vs, df = both_metas
        assert vs["regression_targets"] == df["regression_targets"]

    def test_same_metrics(self, both_metas):
        vs, df = both_metas
        assert vs["regression_sample_metrics"] == df["regression_sample_metrics"]

    def test_same_aggregation(self, both_metas):
        vs, df = both_metas
        assert vs["aggregation"] == df["aggregation"]

    def test_same_level(self, both_metas):
        vs, df = both_metas
        assert vs["level"] == df["level"]

    def test_same_model_count(self):
        vs_modelset = _load_modelset(VS_ENSEMBLE)
        df_modelset = _load_modelset(DF_ENSEMBLE)
        assert len(vs_modelset["models"]) == len(df_modelset["models"]) == 3

    def test_names_differ(self, both_metas):
        vs, df = both_metas
        assert vs["name"] != df["name"]

    def test_both_trios_use_same_loss(self):
        # 2026-06-09 (PR #116): violet_visitor was intentionally diverged from the
        # tobit baseline for the magnitude_calibration experiments (issue #85),
        # deliberately breaking viewser<->datafactory loss parity for the
        # golden_hour<->stellar_horizon comparison — tracked as C-71 in
        # reports/technical_risk_register.md.
        # 2026-06-11: the divergence moved from the Arm-1 hurdle loss
        # (lognormal_nll) to the hurdle-NB stack (ZINB epic #102, decision A):
        # TruncatedNBLoss body + weighted-BCE gate. We pin the expected per-model
        # state (five tobit + violet_visitor hurdle_nb) instead of asserting
        # strict uniformity, so the known divergence is documented in-place and any
        # *other* drift is still caught. Revert to {"tobit"} when the experiment
        # concludes.
        EXPERIMENT_DIVERGED = {"violet_visitor": "hurdle_nb"}
        expected = {
            name: EXPERIMENT_DIVERGED.get(name, "tobit")
            for name in VIEWSER_TRIO + DATAFACTORY_TRIO
        }
        actual = {}
        for name in VIEWSER_TRIO + DATAFACTORY_TRIO:
            hp = _load_hp(name)
            actual[name] = hp["loss_reg"]
        assert actual == expected, f"loss functions: {actual}; expected: {expected}"

    def test_constituent_posterior_samples_match(self):
        # 2026-06-16: violet_visitor's n_posterior_samples was reduced 16 -> 8 as an
        # interim eval-stage OOM workaround (tracked as C-116/#124 in views-hydranet;
        # to be restored to 16 once fixed) — tracked as C-87 in
        # reports/technical_risk_register.md. This intentionally breaks the trio
        # sample-count parity for the golden_hour<->stellar_horizon comparison. We pin
        # the expected per-model state (five at 16 + violet_visitor at 8) instead of
        # asserting strict uniformity, so the known divergence is documented in-place
        # and any *other* drift is still caught. Restore to a strict single-value
        # assertion when violet_visitor returns to 16. (Same pattern as
        # test_both_trios_use_same_loss / C-71.)
        EXPERIMENT_DIVERGED = {"violet_visitor": 8}
        expected = {
            name: EXPERIMENT_DIVERGED.get(name, 16)
            for name in VIEWSER_TRIO + DATAFACTORY_TRIO
        }
        actual = {}
        for name in VIEWSER_TRIO + DATAFACTORY_TRIO:
            hp = _load_hp(name)
            actual[name] = hp["n_posterior_samples"]
        assert actual == expected, f"n_posterior_samples: {actual}; expected: {expected}"


class TestDatafactoryTrioPartitions:
    """All three datafactory models must use identical partition structure."""

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_uses_current_month_id(self, model_name):
        path = MODELS_DIR / model_name / "configs" / "config_partitions.py"
        if not path.exists():
            pytest.skip(f"{model_name} not yet created")
        text = path.read_text()
        assert "_current_month_id" in text

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_no_ingester3_dependency(self, model_name):
        path = MODELS_DIR / model_name / "configs" / "config_partitions.py"
        if not path.exists():
            pytest.skip(f"{model_name} not yet created")
        text = path.read_text()
        assert "ingester3" not in text

    def test_calibration_validation_boundaries_identical(self):
        partitions = {}
        for name in DATAFACTORY_TRIO:
            partitions[name] = _load_partitions(name)
        for name in DATAFACTORY_TRIO[1:]:
            assert partitions[DATAFACTORY_TRIO[0]]["calibration"] == partitions[name]["calibration"], (
                f"calibration mismatch: {DATAFACTORY_TRIO[0]} vs {name}"
            )
            assert partitions[DATAFACTORY_TRIO[0]]["validation"] == partitions[name]["validation"], (
                f"validation mismatch: {DATAFACTORY_TRIO[0]} vs {name}"
            )
