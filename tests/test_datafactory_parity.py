"""Datafactory parity tests — verify the datafactory HydraNet trio and ensemble
mirror the viewser trio (golden_hour) in every dimension except data source.

The datafactory trio:
  - bright_starship (shrinkage) — exists
  - bold_comet (basu_dpd) — new
  - blazing_meteor (lognormal_nll) — new

The datafactory ensemble:
  - stellar_horizon (concat, PredictionFrame) — new

See risk register C-47 and golden_hour pre-analysis plan for context.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ENSEMBLES_DIR = REPO_ROOT / "ensembles"

DATAFACTORY_TRIO = ["bright_starship", "bold_comet", "blazing_meteor"]
VIEWSER_TRIO = ["purple_alien", "blue_stranger", "violet_visitor"]
DF_ENSEMBLE = "stellar_horizon"
VS_ENSEMBLE = "golden_hour"

pytestmark = [pytest.mark.green]

LOSS_PARAMS = {
    "bright_starship": {
        "loss_reg": "shrinkage",
        "loss_reg_a": 258,
        "loss_reg_c": 0.001,
    },
    "bold_comet": {
        "loss_reg": "basu_dpd",
        "loss_reg_alpha": 0.3,
        "loss_reg_sigma": 3.0,
    },
    "blazing_meteor": {
        "loss_reg": "lognormal_nll",
        "loss_reg_sigma": 0.9,
    },
}

ALL_LOSS_KEYS = {"loss_reg", "loss_reg_a", "loss_reg_c", "loss_reg_alpha", "loss_reg_sigma"}


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
    """All three datafactory models must share identical hyperparams except loss."""

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
        expected_features = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
        expected_reg = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
        expected_cls = ["by_sb_best", "by_ns_best", "by_os_best"]
        for name, hp in trio_hps.items():
            assert hp["features"] == expected_features, f"{name} features mismatch"
            assert hp["regression_targets"] == expected_reg, f"{name} regression targets mismatch"
            assert hp["classification_targets"] == expected_cls, f"{name} classification targets mismatch"

    def test_identical_grid_topology(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["row_offset"] == 87, f"{name} row_offset"
            assert hp["col_offset"] == 310, f"{name} col_offset"
            assert hp["height"] == 180, f"{name} height"
            assert hp["width"] == 180, f"{name} width"

    def test_identical_posterior_samples(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["n_posterior_samples"] == 64, f"{name} posterior samples"

    def test_identical_classification_loss(self, trio_hps):
        for name, hp in trio_hps.items():
            assert hp["loss_class"] == "focal", f"{name} loss_class"
            assert hp["loss_class_alpha"] == 0.75, f"{name} loss_class_alpha"
            assert hp["loss_class_gamma"] == 1.5, f"{name} loss_class_gamma"

    def test_loss_reg_differs(self, trio_hps):
        loss_values = {hp["loss_reg"] for hp in trio_hps.values()}
        assert loss_values == {"shrinkage", "basu_dpd", "lognormal_nll"}

    @pytest.mark.parametrize("model_name", DATAFACTORY_TRIO)
    def test_loss_params_correct(self, model_name):
        hp = _load_hp(model_name)
        expected = LOSS_PARAMS[model_name]
        for key, value in expected.items():
            assert hp[key] == value, f"{model_name}: {key} expected {value}, got {hp.get(key)}"
        unexpected = ALL_LOSS_KEYS - set(expected.keys())
        for key in unexpected:
            assert key not in hp, f"{model_name}: unexpected loss param {key}={hp[key]}"


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

    def test_models_list(self, ens_meta):
        assert ens_meta["models"] == ["bright_starship", "bold_comet", "blazing_meteor"]

    def test_regression_targets(self, ens_meta):
        assert ens_meta["regression_targets"] == ["lr_sb_best", "lr_ns_best", "lr_os_best"]

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

    def test_same_model_count(self, both_metas):
        vs, df = both_metas
        assert len(vs["models"]) == len(df["models"]) == 3

    def test_names_differ(self, both_metas):
        vs, df = both_metas
        assert vs["name"] != df["name"]

    def test_constituent_loss_functions_parallel(self):
        vs_losses = set()
        df_losses = set()
        for name in VIEWSER_TRIO:
            hp = _load_hp(name)
            vs_losses.add(hp["loss_reg"])
        for name in DATAFACTORY_TRIO:
            hp = _load_hp(name)
            df_losses.add(hp["loss_reg"])
        assert vs_losses == df_losses == {"shrinkage", "basu_dpd", "lognormal_nll"}

    def test_constituent_posterior_samples_match(self):
        for name in VIEWSER_TRIO + DATAFACTORY_TRIO:
            hp = _load_hp(name)
            assert hp["n_posterior_samples"] == 64, f"{name} has {hp['n_posterior_samples']} samples"


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
    def test_partition_override_declared(self, model_name):
        path = MODELS_DIR / model_name / "configs" / "config_partitions.py"
        if not path.exists():
            pytest.skip(f"{model_name} not yet created")
        text = path.read_text()
        assert "PARTITION_OVERRIDE" in text

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
