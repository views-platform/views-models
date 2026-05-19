"""Tests for ensemble configuration completeness and dependency validation.

Ensembles have different required config keys than individual models:
- config_meta.py: name, models, regression_targets, level, aggregation
- config_deployment.py: deployment_status
- config_hyperparameters.py: steps
- config_partitions.py: generate() function

Ensembles do NOT have config_sweep.py or config_queryset.py.
"""
import pytest

from tests.conftest import (
    load_config_module,
    MODELS_DIR,
    ENSEMBLES_DIR,
)


REQUIRED_ENSEMBLE_META_KEYS = {"name", "models", "regression_targets", "level", "aggregation"}

REQUIRED_ENSEMBLE_CONFIG_FILES = [
    "config_meta.py",
    "config_deployment.py",
    "config_hyperparameters.py",
    "config_partitions.py",
]

VALID_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", "deprecated"}


# ── Structure ──────────────────────────────────────────────────────────

class TestEnsembleStructure:
    def test_main_py_exists(self, ensemble_dir):
        assert (ensemble_dir / "main.py").exists()

    def test_run_sh_exists(self, ensemble_dir):
        assert (ensemble_dir / "run.sh").exists()

    def test_configs_directory_exists(self, ensemble_dir):
        assert (ensemble_dir / "configs").is_dir()

    @pytest.mark.parametrize("config_file", REQUIRED_ENSEMBLE_CONFIG_FILES)
    def test_required_config_file_exists(self, ensemble_dir, config_file):
        cfg_path = ensemble_dir / "configs" / config_file
        assert cfg_path.exists(), (
            f"{ensemble_dir.name} missing config file: {config_file}"
        )


# ── Config Completeness ───────────────────────────────────────────────

class TestEnsembleConfigMeta:
    def test_meta_has_required_keys(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        missing = REQUIRED_ENSEMBLE_META_KEYS - set(meta.keys())
        assert not missing, (
            f"{ensemble_dir.name} config_meta missing keys: {missing}"
        )

    def test_meta_name_matches_directory(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert meta["name"] == ensemble_dir.name

    def test_meta_level_is_valid(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert meta["level"] in ("cm", "pgm")

    def test_meta_models_is_nonempty_list(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert isinstance(meta["models"], list) and len(meta["models"]) > 0, (
            f"{ensemble_dir.name} config_meta.models must be a non-empty list"
        )

    def test_no_old_targets_key(self, ensemble_dir):
        """Ensembles must use 'regression_targets', not the old 'targets' key."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert "targets" not in meta, (
            f"{ensemble_dir.name} still has old 'targets' key"
        )

    def test_no_old_metrics_key(self, ensemble_dir):
        """Ensembles must use 'regression_point_metrics', not old 'metrics' key."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        assert "metrics" not in meta, (
            f"{ensemble_dir.name} still has old 'metrics' key"
        )


class TestEnsembleConfigDeployment:
    def test_deployment_has_valid_status(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_deployment.py"
        module = load_config_module(cfg_path)
        dep = module.get_deployment_config()
        assert dep.get("deployment_status") in VALID_DEPLOYMENT_STATUSES, (
            f"{ensemble_dir.name} has invalid deployment_status"
        )


class TestEnsembleConfigHyperparameters:
    def test_hp_has_steps(self, ensemble_dir):
        cfg_path = ensemble_dir / "configs" / "config_hyperparameters.py"
        module = load_config_module(cfg_path)
        hp = module.get_hp_config()
        assert "steps" in hp, f"{ensemble_dir.name} config_hp missing 'steps'"


# ── Dependency Validation ─────────────────────────────────────────────

class TestEnsembleDependencies:
    def test_all_constituent_models_exist(self, ensemble_dir):
        """Every model listed in config_meta.models must exist as a model directory."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        missing = [
            m for m in meta["models"]
            if not (MODELS_DIR / m).is_dir()
        ]
        assert not missing, (
            f"{ensemble_dir.name} references non-existent models: {missing}"
        )

    def test_constituent_models_match_ensemble_level(self, ensemble_dir):
        """All models in an ensemble must have the same level (cm/pgm) as the ensemble."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        ensemble_level = meta["level"]

        mismatched = []
        for model_name in meta["models"]:
            model_meta_path = MODELS_DIR / model_name / "configs" / "config_meta.py"
            if model_meta_path.exists():
                model_module = load_config_module(model_meta_path)
                model_level = model_module.get_meta_config().get("level")
                if model_level != ensemble_level:
                    mismatched.append(f"{model_name} (level={model_level})")
        assert not mismatched, (
            f"{ensemble_dir.name} is level='{ensemble_level}' but contains "
            f"models with different levels: {mismatched}"
        )

    def test_reconcile_with_target_exists(self, ensemble_dir):
        """If reconcile_with is declared, the target must exist as an ensemble."""
        cfg_path = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(cfg_path)
        meta = module.get_meta_config()
        target = meta.get("reconcile_with")
        if target is not None:
            assert (ENSEMBLES_DIR / target).is_dir(), (
                f"{ensemble_dir.name} declares reconcile_with='{target}' "
                f"but no ensemble directory '{target}' exists"
            )
