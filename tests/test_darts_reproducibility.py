"""Tests that all darts models have complete ReproducibilityGate parameters.

Imports the canonical param definitions from views_r2darts2 to avoid DRY
violation. Skipped when views_r2darts2 is not installed.

This test prevents regression of the missing-params bug fixed in 6 models
on 2026-03-16.
"""
import pytest

from tests.conftest import ALL_MODEL_DIRS, load_config_module

try:
    from views_r2darts2.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    DARTS_CORE_PARAMS = set(ReproducibilityGate.Config.CORE_GENOME)
    ARCH_PARAMS = {
        algo: set(params)
        for algo, params in ReproducibilityGate.Config.ALGORITHM_GENOMES.items()
    }
    _HAS_R2DARTS2 = True
except ImportError:
    _HAS_R2DARTS2 = False
    DARTS_CORE_PARAMS = set()
    ARCH_PARAMS = {}

# These are set at runtime by the framework, not in config files
RUNTIME_PARAMS = {"run_type", "name", "algorithm"}

pytestmark = pytest.mark.skipif(
    not _HAS_R2DARTS2,
    reason="views_r2darts2 not installed — ReproducibilityGate params unavailable",
)


def _is_darts_model(model_dir):
    """Check if a model imports from views_r2darts2."""
    source = (model_dir / "main.py").read_text()
    return "views_r2darts2" in source


def _get_algorithm(model_dir):
    """Get algorithm name from config_meta.py."""
    module = load_config_module(model_dir / "configs" / "config_meta.py")
    return module.get_meta_config().get("algorithm")


DARTS_MODELS = [d for d in ALL_MODEL_DIRS if _is_darts_model(d)]
DARTS_MODEL_NAMES = [d.name for d in DARTS_MODELS]


class TestDartsReproducibilityGate:
    @pytest.mark.parametrize("model_dir", DARTS_MODELS, ids=DARTS_MODEL_NAMES)
    def test_has_core_params(self, model_dir):
        """Every darts model must have all core ReproducibilityGate params."""
        hp = load_config_module(
            model_dir / "configs" / "config_hyperparameters.py"
        ).get_hp_config()
        missing = (DARTS_CORE_PARAMS - RUNTIME_PARAMS) - set(hp.keys())
        assert not missing, (
            f"{model_dir.name} missing core ReproducibilityGate params: "
            f"{sorted(missing)}"
        )

    @pytest.mark.parametrize("model_dir", DARTS_MODELS, ids=DARTS_MODEL_NAMES)
    def test_has_architecture_params(self, model_dir):
        """Every darts model must have all params required by its algorithm."""
        algorithm = _get_algorithm(model_dir)
        if algorithm not in ARCH_PARAMS:
            pytest.skip(f"Unknown algorithm '{algorithm}' — no genome defined")

        hp = load_config_module(
            model_dir / "configs" / "config_hyperparameters.py"
        ).get_hp_config()
        required = ARCH_PARAMS[algorithm]
        missing = required - set(hp.keys())
        assert not missing, (
            f"{model_dir.name} ({algorithm}) missing architecture params: "
            f"{sorted(missing)}"
        )
