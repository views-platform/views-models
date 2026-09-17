"""Tests that every model has complete and consistent config files."""
import pytest

from tests.conftest import get_regression_targets, load_config_module

pytestmark = pytest.mark.beige


# ── Required keys ──────────────────────────────────────────────────────

REQUIRED_META_KEYS = {
    "name", "algorithm", "level", "creator",
    "prediction_format", "rolling_origin_stride",
}

REQUIRED_HP_KEYS = {"steps", "time_steps"}

#: The two vocabularies of ADR-017 Phase 2. A source carries exactly ONE of the two files.
#: `config_maturity.py` is the destination; `config_deployment.py` is the legacy file a
#: source keeps until its engine runs on pipeline-core >= 3.2.0 (the 38 stepshifter and 31
#: r2darts2 models are on 2.3.0, which requires the legacy file and knows nothing of
#: maturity — views-models#473, ADR-017 §11). Readers translate; files do not coexist.
VALID_MATURITIES = {"candidate", "graduate", "retired"}
VALID_DEPLOYMENT_STATUSES = {"shadow", "deployed", "baseline", "deprecated"}


# ── Fixtures for pre-loaded configs ────────────────────────────────────

@pytest.fixture
def meta_config(model_dir):
    module = load_config_module(model_dir / "configs" / "config_meta.py")
    return module.get_meta_config()


@pytest.fixture
def maturity_file(model_dir):
    """(path, getter, key, valid_values) for whichever maturity file this source carries."""
    configs = model_dir / "configs"
    new = configs / "config_maturity.py"
    legacy = configs / "config_deployment.py"
    if new.exists():
        return new, "get_maturity_config", "maturity", VALID_MATURITIES
    return legacy, "get_deployment_config", "deployment_status", VALID_DEPLOYMENT_STATUSES


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

    def test_regression_targets_present_if_metrics(self, model_dir, meta_config):
        """A model that declares regression evaluation must resolve at least one
        regression target (so the metric has something to score).

        Name-agnostic (EPIC #154): derives targets via the single accessor
        (``conftest.get_regression_targets``) and asserts nothing about what they
        are *called*. Replaces the former hardcoded-canonical guard. Cross-location
        agreement is enforced in ``tests/test_regression_targets.py``.
        """
        if not meta_config.get("regression_point_metrics"):
            pytest.skip(f"{model_dir.name} declares no regression_point_metrics")
        targets = get_regression_targets(model_dir)
        assert targets, (
            f"{model_dir.name} declares regression_point_metrics but no resolvable "
            f"regression_targets in config_meta or config_hyperparameters — the metric "
            f"has nothing to score"
        )


# ── config_maturity.py / config_deployment.py ─────────────────────────

class TestMaturityConfig:
    """Exactly one maturity file per source, in one of the two vocabularies."""

    def test_exactly_one_maturity_file(self, any_model_dir):
        """The #455 guard. ADR-017 Phase 2 is a RENAME: a source carries config_maturity.py
        OR config_deployment.py, never both. Two files is the state PR #444 left 14 models in
        — pipeline-core >= 3.0.1 reads the new one and silently ignores the legacy one, so
        the two can disagree with nothing noticing. Runs over fixture models too, on purpose:
        the scaffold must not produce a two-file model either."""
        configs = any_model_dir / "configs"
        new, legacy = configs / "config_maturity.py", configs / "config_deployment.py"
        assert not (new.exists() and legacy.exists()), (
            f"{any_model_dir.name} carries BOTH config_maturity.py and config_deployment.py. "
            f"ADR-017 Phase 2 is a rename; delete the legacy file. (#455)"
        )
        assert new.exists() or legacy.exists(), (
            f"{any_model_dir.name} has neither config_maturity.py nor config_deployment.py"
        )

    def test_maturity_file_has_its_key(self, model_dir, maturity_file):
        path, getter, key, _ = maturity_file
        config = getattr(load_config_module(path), getter)()
        assert key in config, f"{model_dir.name}: {path.name} does not declare '{key}'"

    def test_maturity_value_is_valid(self, model_dir, maturity_file):
        path, getter, key, valid = maturity_file
        value = getattr(load_config_module(path), getter)()[key]
        assert value in valid, (
            f"{model_dir.name}: {path.name} has invalid {key}: '{value}' (valid: {sorted(valid)})"
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
