"""Tests that every model has complete and consistent config files."""
import pytest

from tests.conftest import load_config_module


# ── Required keys ──────────────────────────────────────────────────────

REQUIRED_META_KEYS = {
    "name", "algorithm", "level", "creator",
    "prediction_format", "rolling_origin_stride",
}

REQUIRED_HP_KEYS = {"steps", "time_steps"}

VALID_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", "deprecated"}


# ── Fixtures for pre-loaded configs ────────────────────────────────────

@pytest.fixture
def meta_config(model_dir):
    module = load_config_module(model_dir / "configs" / "config_meta.py")
    return module.get_meta_config()


@pytest.fixture
def deployment_config(model_dir):
    module = load_config_module(model_dir / "configs" / "config_deployment.py")
    return module.get_deployment_config()


@pytest.fixture
def hp_config(model_dir):
    module = load_config_module(model_dir / "configs" / "config_hyperparameters.py")
    return module.get_hp_config()


# ── config_meta.py ─────────────────────────────────────────────────────

class TestConfigMeta:
    def test_meta_config_exists(self, model_dir):
        assert (model_dir / "configs" / "config_meta.py").exists()

    def test_meta_config_has_required_keys(self, model_dir, meta_config):
        missing = REQUIRED_META_KEYS - set(meta_config.keys())
        assert not missing, f"{model_dir.name} config_meta missing keys: {missing}"

    def test_meta_name_matches_directory(self, model_dir, meta_config):
        assert meta_config["name"] == model_dir.name, (
            f"config_meta name '{meta_config['name']}' does not match "
            f"directory '{model_dir.name}'"
        )

    def test_meta_level_is_valid(self, model_dir, meta_config):
        assert meta_config["level"] in ("cm", "pgm"), (
            f"{model_dir.name} has invalid level: {meta_config['level']}"
        )


# ── config_deployment.py ───────────────────────────────────────────────

class TestConfigDeployment:
    def test_deployment_config_exists(self, model_dir):
        assert (model_dir / "configs" / "config_deployment.py").exists()

    def test_deployment_has_status(self, model_dir, deployment_config):
        assert "deployment_status" in deployment_config

    def test_deployment_status_is_valid(self, model_dir, deployment_config):
        status = deployment_config["deployment_status"]
        assert status in VALID_DEPLOYMENT_STATUSES, (
            f"{model_dir.name} has invalid deployment_status: '{status}'"
        )


# ── config_hyperparameters.py ──────────────────────────────────────────

class TestConfigHyperparameters:
    def test_hp_config_exists(self, model_dir):
        assert (model_dir / "configs" / "config_hyperparameters.py").exists()

    def test_hp_config_has_required_keys(self, model_dir, hp_config):
        missing = REQUIRED_HP_KEYS - set(hp_config.keys())
        assert not missing, f"{model_dir.name} config_hp missing keys: {missing}"

    def test_time_steps_matches_steps_length(self, model_dir, hp_config):
        steps = hp_config.get("steps")
        time_steps = hp_config.get("time_steps")
        if isinstance(steps, list) and time_steps is not None:
            assert time_steps == len(steps), (
                f"{model_dir.name}: time_steps={time_steps} but len(steps)={len(steps)}"
            )
