"""Tests for cross-artifact coherence: algorithm, manager import, and requirements.

Validates that config_meta['algorithm'], the manager import in main.py,
and the package declared in requirements.txt all agree with each other.

Addresses risks C-04 and C-08 from the technical risk register.
"""
import ast
import re
from pathlib import Path

import pytest

from tests.conftest import load_config_module

# Verified empirically from all 66 active models on 2026-04-04.
# Update this mapping when a new algorithm is added to a package.
ALGORITHM_TO_PACKAGE = {
    # views_stepshifter algorithms
    "HurdleModel": "views_stepshifter",
    "XGBRegressor": "views_stepshifter",
    "XGBRFRegressor": "views_stepshifter",
    "LGBMRegressor": "views_stepshifter",
    "RandomForestRegressor": "views_stepshifter",
    "ShurfModel": "views_stepshifter",
    # views_r2darts2 algorithms
    "TCNModel": "views_r2darts2",
    "TFTModel": "views_r2darts2",
    "BlockRNNModel": "views_r2darts2",
    "NBEATSModel": "views_r2darts2",
    "TransformerModel": "views_r2darts2",
    "TiDEModel": "views_r2darts2",
    "TSMixerModel": "views_r2darts2",
    # views_baseline algorithms
    "MixtureBaseline": "views_baseline",
    "AverageModel": "views_baseline",
    "ZeroModel": "views_baseline",
    "LocfModel": "views_baseline",
    "ConflictologyModel": "views_baseline",
    # views_hydranet algorithms
    "HydraNet": "views_hydranet",
}


def _extract_manager_package(main_path: Path) -> str | None:
    """Extract the top-level package name from the non-pipeline-core import in main.py.

    Handles both patterns:
      - from views_r2darts2 import DartsForecastingModelManager
      - from views_stepshifter.manager.stepshifter_manager import StepshifterManager

    Returns the top-level package (e.g., 'views_r2darts2') or None.
    """
    source = main_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top_package = node.module.split(".")[0]
            if top_package.startswith("views_") and top_package != "views_pipeline_core":
                return top_package
    return None


def _extract_requirements_package(req_path: Path) -> str | None:
    """Extract package name from requirements.txt, normalizing hyphens to underscores."""
    if not req_path.exists():
        return None
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pkg_name = re.split(r"[><=!~@]", line)[0].strip()
            return pkg_name.replace("-", "_")
    return None


class TestAlgorithmManagerCoherence:
    """Risk C-04: config_meta['algorithm'] must be consistent with main.py manager import."""

    def test_algorithm_is_in_known_mapping(self, model_dir):
        """The declared algorithm must appear in the known algorithm-to-package mapping."""
        module = load_config_module(model_dir / "configs" / "config_meta.py")
        algorithm = module.get_meta_config()["algorithm"]
        assert algorithm in ALGORITHM_TO_PACKAGE, (
            f"{model_dir.name}: algorithm '{algorithm}' is not in the known "
            f"algorithm-to-package mapping. If this is a new algorithm, add it "
            f"to ALGORITHM_TO_PACKAGE in test_algorithm_coherence.py."
        )

    def test_algorithm_matches_manager_package(self, model_dir):
        """The algorithm's expected package must match the package actually imported in main.py."""
        module = load_config_module(model_dir / "configs" / "config_meta.py")
        algorithm = module.get_meta_config()["algorithm"]
        if algorithm not in ALGORITHM_TO_PACKAGE:
            pytest.skip(f"Algorithm '{algorithm}' not in mapping")

        expected_package = ALGORITHM_TO_PACKAGE[algorithm]
        actual_package = _extract_manager_package(model_dir / "main.py")

        assert actual_package is not None, (
            f"{model_dir.name}: could not extract manager package from main.py"
        )
        assert actual_package == expected_package, (
            f"{model_dir.name}: algorithm '{algorithm}' belongs to '{expected_package}' "
            f"but main.py imports from '{actual_package}'"
        )


class TestRequirementsCoherence:
    """Risk C-08: requirements.txt package must match main.py import package."""

    def test_requirements_package_matches_import(self, model_dir):
        """The package in requirements.txt must match the package imported in main.py."""
        req_path = model_dir / "requirements.txt"
        if not req_path.exists():
            pytest.skip(f"{model_dir.name} has no requirements.txt")

        req_package = _extract_requirements_package(req_path)
        import_package = _extract_manager_package(model_dir / "main.py")

        assert req_package is not None, (
            f"{model_dir.name}: could not extract package from requirements.txt"
        )
        assert import_package is not None, (
            f"{model_dir.name}: could not extract manager package from main.py"
        )
        assert req_package == import_package, (
            f"{model_dir.name}: requirements.txt declares '{req_package}' "
            f"but main.py imports from '{import_package}'"
        )
