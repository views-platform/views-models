"""Tests for scaffold builder injection seams and I/O decoupling.

The scaffold builders (build_model_scaffold.py, build_ensemble_scaffold.py)
require views_pipeline_core at import time. Tests that instantiate the builders
are skipped when the package is unavailable.

Tests that verify the injection seam contract (callback signatures, default
behavior) work regardless of package availability by testing the pattern
at the source level.
"""
import ast
import re

import pytest

from tests.conftest import REPO_ROOT


# ---------------------------------------------------------------------------
# Tests that work WITHOUT views_pipeline_core (AST-based verification)
# ---------------------------------------------------------------------------


class TestModelScaffoldInjectionSeams:
    """Verify build_model_scaffold.py has the expected injection seams."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (REPO_ROOT / "build_model_scaffold.py").read_text()
        self.tree = ast.parse(self.source)

    def test_build_model_scripts_accepts_input_fn(self):
        """build_model_scripts must accept an input_fn keyword argument."""
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "build_model_scripts"):
                kwonly_names = [arg.arg for arg in node.args.kwonlyargs]
                assert "input_fn" in kwonly_names, (
                    "build_model_scripts() missing input_fn keyword argument"
                )
                return
        pytest.fail("build_model_scripts() not found in build_model_scaffold.py")

    def test_build_model_scripts_accepts_get_version_fn(self):
        """build_model_scripts must accept a get_version_fn keyword argument."""
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "build_model_scripts"):
                kwonly_names = [arg.arg for arg in node.args.kwonlyargs]
                assert "get_version_fn" in kwonly_names, (
                    "build_model_scripts() missing get_version_fn keyword argument"
                )
                return
        pytest.fail("build_model_scripts() not found in build_model_scaffold.py")

    def test_no_bare_input_calls(self):
        """build_model_scripts should use input_fn, not bare input()."""
        # Extract just the method body
        method_match = re.search(
            r'def build_model_scripts\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.source, re.DOTALL
        )
        assert method_match is not None
        body = method_match.group(1)
        # Find input() calls that are NOT input_fn(
        lines_with_bare_input = []
        for line in body.split('\n'):
            if 'input(' in line and 'input_fn' not in line and 'or input' not in line:
                lines_with_bare_input.append(line.strip())
        assert lines_with_bare_input == [], (
            f"Found bare input() calls (should use input_fn): {lines_with_bare_input}"
        )

    def test_no_bare_github_version_calls(self):
        """build_model_scripts should use get_version_fn, not bare GitHub call."""
        method_match = re.search(
            r'def build_model_scripts\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.source, re.DOTALL
        )
        assert method_match is not None
        body = method_match.group(1)
        for line in body.split('\n'):
            if 'get_latest_release_version_from_github(' in line:
                assert 'get_version_fn' in line or 'or ' in line, (
                    f"Bare GitHub API call found (should use get_version_fn): {line.strip()}"
                )

    def test_package_validation_uses_not_instead_of_eq_false(self):
        """Package name validation should use 'not' instead of '== False'."""
        method_match = re.search(
            r'def build_model_scripts\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.source, re.DOTALL
        )
        assert method_match is not None
        body = method_match.group(1)
        assert '== False' not in body, (
            "Package validation still uses '== False' instead of 'not'"
        )


class TestEnsembleScaffoldInjectionSeams:
    """Verify build_ensemble_scaffold.py has the expected injection seam."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (REPO_ROOT / "build_ensemble_scaffold.py").read_text()
        self.tree = ast.parse(self.source)

    def test_build_model_scripts_accepts_pipeline_config(self):
        """build_model_scripts must accept a pipeline_config keyword argument."""
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "build_model_scripts"):
                kwonly_names = [arg.arg for arg in node.args.kwonlyargs]
                assert "pipeline_config" in kwonly_names, (
                    "build_model_scripts() missing pipeline_config keyword argument"
                )
                return
        pytest.fail("build_model_scripts() not found in build_ensemble_scaffold.py")

    def test_no_bare_pipeline_config_instantiation(self):
        """build_model_scripts should use injected pipeline_config, not PipelineConfig()."""
        method_match = re.search(
            r'def build_model_scripts\(self.*?\n(.*?)(?=\n    def |\nclass |\nif |\Z)',
            self.source, re.DOTALL
        )
        assert method_match is not None
        body = method_match.group(1)
        for line in body.split('\n'):
            stripped = line.strip()
            # Skip docstring lines and comments
            if stripped.startswith('#') or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if 'Defaults to' in stripped:
                continue
            if 'PipelineConfig()' in stripped:
                assert 'pipeline_config' in stripped or 'or ' in stripped, (
                    f"Bare PipelineConfig() instantiation (should use injected param): {stripped}"
                )


# ---------------------------------------------------------------------------
# Tests that REQUIRE views_pipeline_core (functional tests with mocked I/O)
# ---------------------------------------------------------------------------

class TestModelScaffoldBuilderFunctional:
    """Functional tests using the injection seams. Skipped without views_pipeline_core."""

    @pytest.fixture(autouse=True)
    def _skip_without_vpc(self):
        pytest.importorskip("views_pipeline_core")

    def test_build_model_scripts_uses_injected_input(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder

        builder = ModelScaffoldBuilder("fake_model")
        # Override the model directory to tmp_path
        builder._model._model_dir = tmp_path / "models" / "fake_model"
        builder._model.model_dir.mkdir(parents=True, exist_ok=True)
        (builder._model.model_dir / "configs").mkdir(exist_ok=True)

        responses = iter(["XGBModel", "views-stepshifter"])
        def mock_input(prompt):
            return next(responses)

        def mock_version(package_name):
            return ">=1.0.0,<2.0.0"

        builder.build_model_scripts(
            input_fn=mock_input,
            get_version_fn=mock_version,
        )
        assert builder._model_algorithm == "XGBModel"
        assert builder.package_name == "views-stepshifter"

    def test_build_model_scripts_github_failure_graceful(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder

        builder = ModelScaffoldBuilder("fake_model")
        builder._model._model_dir = tmp_path / "models" / "fake_model"
        builder._model.model_dir.mkdir(parents=True, exist_ok=True)
        (builder._model.model_dir / "configs").mkdir(exist_ok=True)

        responses = iter(["XGBModel", "views-stepshifter"])

        def mock_version(package_name):
            raise ConnectionError("No network")

        builder.build_model_scripts(
            input_fn=lambda prompt: next(responses),
            get_version_fn=mock_version,
        )
        # Should not raise — the existing try/except handles this gracefully
