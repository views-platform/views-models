"""Tests that all stepshifter models declare a valid ``target_transform`` and pass
the views_stepshifter ``ReproducibilityGate``.

Mirrors ``test_darts_reproducibility.py`` (which covers the views_r2darts2 package)
for the views_stepshifter package. Enforces ADR-003 / views-stepshifter#52:
``target_transform`` is a required config key validated against a closed registry;
``HurdleModel``/``ShurfModel`` are restricted to ``identity`` (deferred, risk
register D-26). Skipped when views_stepshifter is not installed.
"""
import pytest

from tests.conftest import ALL_MODEL_DIRS, load_config_module

try:
    from views_stepshifter.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )
    from views_stepshifter.infrastructure.transforms import TRANSFORMS

    _HAS_STEPSHIFTER = True
except ImportError:
    _HAS_STEPSHIFTER = False

STEPSHIFTER_ALGORITHMS = {
    "XGBRegressor",
    "XGBRFRegressor",
    "LGBMRegressor",
    "HurdleModel",
    "ShurfModel",
}
DEFERRED_ALGORITHMS = {"HurdleModel", "ShurfModel"}

pytestmark = [
    pytest.mark.green,
    pytest.mark.skipif(
        not _HAS_STEPSHIFTER,
        reason="views_stepshifter not installed — ReproducibilityGate unavailable",
    ),
]


def _algorithm(model_dir):
    module = load_config_module(model_dir / "configs" / "config_meta.py")
    return module.get_meta_config().get("algorithm")


def _hyperparameters(model_dir):
    module = load_config_module(model_dir / "configs" / "config_hyperparameters.py")
    return module.get_hp_config()


STEPSHIFTER_MODELS = [
    d for d in ALL_MODEL_DIRS if _algorithm(d) in STEPSHIFTER_ALGORITHMS
]
STEPSHIFTER_MODEL_NAMES = [d.name for d in STEPSHIFTER_MODELS]


class TestStepshifterTargetTransform:
    @pytest.mark.parametrize(
        "model_dir", STEPSHIFTER_MODELS, ids=STEPSHIFTER_MODEL_NAMES
    )
    def test_declares_valid_target_transform(self, model_dir):
        """Every stepshifter model must declare target_transform as a registry member."""
        hp = _hyperparameters(model_dir)
        assert "target_transform" in hp, (
            f"{model_dir.name} config_hyperparameters is missing the required "
            "'target_transform' key"
        )
        assert hp["target_transform"] in TRANSFORMS, (
            f"{model_dir.name} target_transform={hp['target_transform']!r} is not a "
            f"registered transform ({sorted(TRANSFORMS)})"
        )

    @pytest.mark.parametrize(
        "model_dir", STEPSHIFTER_MODELS, ids=STEPSHIFTER_MODEL_NAMES
    )
    def test_deferred_models_declare_identity(self, model_dir):
        """HurdleModel/ShurfModel must declare identity (non-identity deferred, D-26)."""
        if _algorithm(model_dir) in DEFERRED_ALGORITHMS:
            hp = _hyperparameters(model_dir)
            assert hp.get("target_transform") == "identity", (
                f"{model_dir.name} ({_algorithm(model_dir)}) must declare "
                "target_transform='identity' (non-identity is deferred, D-26)"
            )

    @pytest.mark.parametrize(
        "model_dir", STEPSHIFTER_MODELS, ids=STEPSHIFTER_MODEL_NAMES
    )
    def test_passes_reproducibility_gate(self, model_dir):
        """The full config must pass the stepshifter ReproducibilityGate (prod parity)."""
        config = {**_hyperparameters(model_dir), "algorithm": _algorithm(model_dir)}
        ReproducibilityGate.Config.audit_manifest(config)
