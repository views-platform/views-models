"""Tests that every model has complete and consistent config files."""
import pytest

from tests.conftest import load_config_module

pytestmark = pytest.mark.beige


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

    def test_no_old_targets_key(self, model_dir, meta_config):
        """Models must use 'regression_targets', not the old 'targets' key."""
        assert "targets" not in meta_config, (
            f"{model_dir.name} still has old 'targets' key — "
            f"rename to 'regression_targets'"
        )

    def test_no_old_metrics_key(self, model_dir, meta_config):
        """Models must use 'regression_point_metrics', not the old 'metrics' key."""
        assert "metrics" not in meta_config, (
            f"{model_dir.name} still has old 'metrics' key — "
            f"rename to 'regression_point_metrics'"
        )

    def test_regression_targets_canonical(self, model_dir, meta_config):
        """Regression targets must use the canonical lr_ged_{sb,ns,os} name (#151).

        Guards against the historical drift (lr_sb_best, lr_ged_sb_dep, lr_*_best)
        re-entering as a *target*. Checks regression_targets only — feature columns
        may legitimately use other names. Stays until the pipeline-core scaffold
        template (views-pipeline-core#201) stops seeding non-canonical names.
        """
        canonical = {"lr_ged_sb", "lr_ged_ns", "lr_ged_os"}
        synthetic = {"synth_target"}  # synthetic pipeline-test models (e.g. *_dream) are exempt
        # Non-canonical targets that can't be flipped config-only because stored
        # prediction artifacts embed the old name — pending regeneration (#152).
        pending_regeneration = {"black_ranger", "green_ranger", "pink_ranger", "yellow_ranger"}
        targets = meta_config.get("regression_targets")
        if not targets or any(t in synthetic for t in targets):
            pytest.skip(f"{model_dir.name} has no real regression_targets")
        if model_dir.name in pending_regeneration:
            pytest.skip(f"{model_dir.name}: ns/os target rename pending prediction regeneration (#152)")
        noncanonical = [t for t in targets if t not in canonical]
        assert not noncanonical, (
            f"{model_dir.name} uses non-canonical regression target(s) {noncanonical} — "
            f"standardize to lr_ged_sb/ns/os (#151)"
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

    def test_hydranet_has_sampling_strategy(self, model_dir, hp_config, meta_config):
        if meta_config.get("algorithm") != "HydraNet":
            pytest.skip("not a HydraNet model")
        assert "sampling_strategy" in hp_config, (
            f"{model_dir.name} is HydraNet but missing 'sampling_strategy' (ADR-049)"
        )

    def test_time_steps_matches_steps_length(self, model_dir, hp_config):
        steps = hp_config.get("steps")
        time_steps = hp_config.get("time_steps")
        if isinstance(steps, list) and time_steps is not None:
            assert time_steps == len(steps), (
                f"{model_dir.name}: time_steps={time_steps} but len(steps)={len(steps)}"
            )
