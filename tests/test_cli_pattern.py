"""Tests that all model main.py files use the current CLI API pattern.

Uses AST-based detection rather than string matching for robustness
against false positives from comments, docstrings, or multi-line strings.
"""
import ast

import pytest

from tests.conftest import (
    ALL_MODEL_DIRS, ALL_ENSEMBLE_DIRS,
    MODEL_NAMES, ENSEMBLE_NAMES,
)

ALL_DIRS = ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS
ALL_NAMES = MODEL_NAMES + ENSEMBLE_NAMES


def _find_imports_from(tree: ast.AST, module: str) -> list[ast.ImportFrom]:
    """Find all 'from <module> import ...' nodes in an AST."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == module
    ]


def _find_attribute_calls(tree: ast.AST, obj: str, method: str) -> list[ast.Call]:
    """Find all '<obj>.<method>()' call nodes in an AST."""
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == obj
        and node.func.attr == method
    ]


class TestCLIPattern:
    @pytest.mark.parametrize("any_dir", ALL_DIRS, ids=ALL_NAMES)
    def test_uses_new_cli_import(self, any_dir):
        """All models and ensembles should import from views_pipeline_core.cli, not cli.utils."""
        source = (any_dir / "main.py").read_text()
        tree = ast.parse(source)
        old_imports = _find_imports_from(tree, "views_pipeline_core.cli.utils")
        assert len(old_imports) == 0, (
            f"{any_dir.name}/main.py uses old CLI pattern "
            "(views_pipeline_core.cli.utils). Migrate to ForecastingModelArgs."
        )

    @pytest.mark.parametrize("any_dir", ALL_DIRS, ids=ALL_NAMES)
    def test_no_explicit_wandb_login(self, any_dir):
        """Models and ensembles should not call wandb.login() directly — the manager handles it."""
        source = (any_dir / "main.py").read_text()
        tree = ast.parse(source)
        wandb_login_calls = _find_attribute_calls(tree, "wandb", "login")
        assert len(wandb_login_calls) == 0, (
            f"{any_dir.name}/main.py calls wandb.login() explicitly. "
            "Remove it — the manager handles authentication."
        )

    @pytest.mark.parametrize("any_dir", ALL_DIRS, ids=ALL_NAMES)
    def test_no_global_warning_suppression(self, any_dir):
        """Models and ensembles must not globally suppress warnings (ADR-008)."""
        source = (any_dir / "main.py").read_text()
        tree = ast.parse(source)
        suppression_calls = _find_attribute_calls(tree, "warnings", "filterwarnings")
        assert len(suppression_calls) == 0, (
            f"{any_dir.name}/main.py calls warnings.filterwarnings() — "
            f"this contradicts ADR-008 (Observability)"
        )
