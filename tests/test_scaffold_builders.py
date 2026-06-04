"""Tests for scaffold builder injection seams and I/O decoupling.

The scaffold builders (build_model_scaffold.py, build_ensemble_scaffold.py,
build_package_scaffold.py) require views_pipeline_core at import time.
Tests that instantiate the builders are skipped when the package is
unavailable.

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


@pytest.mark.beige
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


@pytest.mark.beige
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

@pytest.mark.green
class TestModelScaffoldBuilderFunctional:
    """Functional tests using the injection seams. Skipped without views_pipeline_core.

    NOTE: These tests access internal attributes (_model.model_dir, _model_algorithm)
    because ModelScaffoldBuilder has no public API for overriding the model directory.
    Changes to ModelPathManager internals in views_pipeline_core could break these tests.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_vpc(self):
        pytest.importorskip("views_pipeline_core")

    def test_build_model_scripts_uses_injected_input(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder

        builder = ModelScaffoldBuilder("fake_model")
        # Override the model directory to tmp_path
        builder._model.model_dir = tmp_path / "models" / "fake_model"
        builder._model.model_dir.mkdir(parents=True, exist_ok=True)
        (builder._model.model_dir / "configs").mkdir(exist_ok=True)
        # requirements_path is normally set by build_model_directory()
        builder.requirements_path = builder._model.model_dir / "requirements.txt"

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
        builder._model.model_dir = tmp_path / "models" / "fake_model"
        builder._model.model_dir.mkdir(parents=True, exist_ok=True)
        (builder._model.model_dir / "configs").mkdir(exist_ok=True)
        # requirements_path is normally set by build_model_directory()
        builder.requirements_path = builder._model.model_dir / "requirements.txt"

        responses = iter(["XGBModel", "views-stepshifter"])

        def mock_version(package_name):
            raise ConnectionError("No network")

        builder.build_model_scripts(
            input_fn=lambda prompt: next(responses),
            get_version_fn=mock_version,
        )
        # Should not raise — the existing try/except handles this gracefully


@pytest.mark.green
class TestModelScaffoldBuilderDirectoryCreation:
    """CIC: ModelScaffoldBuilder must create directories and README."""

    @pytest.fixture(autouse=True)
    def _skip_without_vpc(self):
        pytest.importorskip("views_pipeline_core")

    def test_build_model_directory_creates_dir(self, tmp_path, monkeypatch):
        from build_model_scaffold import ModelScaffoldBuilder
        builder = ModelScaffoldBuilder("test_model")
        target = tmp_path / "models" / "test_model"
        builder._model.model_dir = target
        builder._subdirs = [
            target / "configs",
            target / "data" / "raw",
            target / "data" / "generated",
            target / "data" / "processed",
            target / "reports",
            target / "logs",
        ]
        result = builder.build_model_directory()
        assert result.exists()
        assert result == target

    def test_build_model_directory_creates_readme(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder
        builder = ModelScaffoldBuilder("test_model")
        target = tmp_path / "models" / "test_model"
        builder._model.model_dir = target
        builder._subdirs = [target / "configs"]
        builder.build_model_directory()
        readme = target / "README.md"
        assert readme.exists()
        content = readme.read_text()
        assert "test_model" in content

    def test_build_model_directory_creates_subdirs(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder
        builder = ModelScaffoldBuilder("test_model")
        target = tmp_path / "models" / "test_model"
        builder._model.model_dir = target
        sub1 = target / "configs"
        sub2 = target / "data" / "raw"
        builder._subdirs = [sub1, sub2]
        builder.build_model_directory()
        assert sub1.is_dir()
        assert sub2.is_dir()

    def test_build_model_directory_creates_gitkeep(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder
        builder = ModelScaffoldBuilder("test_model")
        target = tmp_path / "models" / "test_model"
        builder._model.model_dir = target
        sub = target / "data" / "raw"
        builder._subdirs = [sub]
        builder.build_model_directory()
        assert (sub / ".gitkeep").exists()

    def test_build_model_scripts_without_directory_raises(self, tmp_path):
        from build_model_scaffold import ModelScaffoldBuilder
        builder = ModelScaffoldBuilder("nonexistent_model")
        builder._model.model_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            builder.build_model_scripts()


@pytest.mark.green
class TestEnsembleScaffoldBuilderDirectoryCreation:
    """CIC: EnsembleScaffoldBuilder must create directories and configs."""

    @pytest.fixture(autouse=True)
    def _skip_without_vpc(self):
        pytest.importorskip("views_pipeline_core")

    def test_ensemble_inherits_from_model_scaffold(self):
        from build_model_scaffold import ModelScaffoldBuilder
        from build_ensemble_scaffold import EnsembleScaffoldBuilder
        assert issubclass(EnsembleScaffoldBuilder, ModelScaffoldBuilder)

    def test_build_model_scripts_without_directory_raises(self, tmp_path):
        from build_ensemble_scaffold import EnsembleScaffoldBuilder
        builder = EnsembleScaffoldBuilder("nonexistent_ensemble")
        builder._model.model_dir = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            builder.build_model_scripts()

    def test_build_model_directory_creates_dir(self, tmp_path):
        from build_ensemble_scaffold import EnsembleScaffoldBuilder
        builder = EnsembleScaffoldBuilder("test_ensemble")
        target = tmp_path / "ensembles" / "test_ensemble"
        builder._model.model_dir = target
        builder._subdirs = [target / "configs"]
        result = builder.build_model_directory()
        assert result.exists()


# ---------------------------------------------------------------------------
# PackageScaffoldBuilder tests
# ---------------------------------------------------------------------------


@pytest.mark.beige
class TestPackageScaffoldBuilderStructure:
    """AST-based tests for build_package_scaffold.py — no VPC needed."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source_path = REPO_ROOT / "build_package_scaffold.py"
        self.source = self.source_path.read_text()
        self.tree = ast.parse(self.source)

    def test_has_package_scaffold_builder_class(self):
        classes = [
            n.name for n in ast.walk(self.tree)
            if isinstance(n, ast.ClassDef)
        ]
        assert "PackageScaffoldBuilder" in classes

    def test_has_build_package_scaffold_method(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "PackageScaffoldBuilder":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert "build_package_scaffold" in methods
                return
        pytest.fail("PackageScaffoldBuilder class not found")

    def test_has_add_gitignore_method(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "PackageScaffoldBuilder":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert "add_gitignore" in methods
                return
        pytest.fail("PackageScaffoldBuilder class not found")

    def test_has_build_package_directories_method(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "PackageScaffoldBuilder":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert "build_package_directories" in methods
                return
        pytest.fail("PackageScaffoldBuilder class not found")

    def test_has_build_package_scripts_method(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "PackageScaffoldBuilder":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                assert "build_package_scripts" in methods
                return
        pytest.fail("PackageScaffoldBuilder class not found")

    def test_build_package_scaffold_calls_create_and_validate(self):
        """build_package_scaffold must call create_views_package and validate_views_package."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_package_scaffold":
                calls = [
                    n.attr for n in ast.walk(node)
                    if isinstance(n, ast.Attribute)
                ]
                assert "create_views_package" in calls, (
                    "build_package_scaffold must call create_views_package"
                )
                assert "validate_views_package" in calls, (
                    "build_package_scaffold must call validate_views_package"
                )
                return
        pytest.fail("build_package_scaffold method not found")

    def test_build_package_scaffold_propagates_exceptions(self):
        """build_package_scaffold must re-raise exceptions after logging."""
        source = self.source
        method_match = re.search(
            r'def build_package_scaffold\(self\).*?\n(.*?)(?=\n    def |\nclass |\nif |\Z)',
            source, re.DOTALL
        )
        assert method_match is not None
        body = method_match.group(1)
        assert "raise" in body, (
            "build_package_scaffold must re-raise exceptions, not swallow them"
        )

    def test_main_block_validates_package_name(self):
        """The __main__ block must validate package names before proceeding."""
        assert "validate_package_name" in self.source
