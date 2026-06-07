"""Tests for create_catalogs.py — catalog generation utilities."""
import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.catalogs.create_catalogs import (
        replace_table_in_section,
        generate_model_table,
        generate_ensemble_table,
        create_link,
    )
    _HAS_PIPELINE_CORE = True
except (ImportError, ModuleNotFoundError):
    _HAS_PIPELINE_CORE = False

_skip_no_pipeline = pytest.mark.skipif(
    not _HAS_PIPELINE_CORE,
    reason="views_pipeline_core not installed"
)


@pytest.mark.beige
class TestNoExecUsage:
    def test_create_catalogs_does_not_use_exec(self):
        """create_catalogs.py should use importlib, not raw exec()."""
        source = (REPO_ROOT / "tools" / "catalogs" / "create_catalogs.py").read_text()
        tree = ast.parse(source)
        exec_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "exec"
        ]
        assert len(exec_calls) == 0, (
            f"create_catalogs.py uses exec() {len(exec_calls)} time(s). "
            "Replace with importlib.util for safer config loading."
        )


@_skip_no_pipeline
@pytest.mark.green
class TestReplaceTableInSection:
    def test_replaces_content_between_markers(self):
        content = (
            "before\n"
            "<!-- FOO_START -->\n"
            "old table\n"
            "<!-- FOO_END -->\n"
            "after"
        )
        result = replace_table_in_section(content, "FOO", "new table")
        assert "new table" in result
        assert "old table" not in result
        assert "before" in result
        assert "after" in result

    def test_preserves_markers(self):
        content = "<!-- X_START -->old<!-- X_END -->"
        result = replace_table_in_section(content, "X", "new")
        assert "<!-- X_START -->" in result
        assert "<!-- X_END -->" in result

    def test_missing_markers_leaves_content_unchanged(self):
        """If markers don't exist, the content should pass through unchanged."""
        content = "no markers here"
        result = replace_table_in_section(content, "MISSING", "new table")
        assert "new table" in result

    def test_empty_new_table(self):
        content = "<!-- T_START -->old<!-- T_END -->"
        result = replace_table_in_section(content, "T", "")
        assert "old" not in result
        assert "<!-- T_START -->" in result
        assert "<!-- T_END -->" in result

    def test_multiple_sections_independent(self):
        """Replacing one section must not affect another."""
        content = (
            "<!-- A_START -->a<!-- A_END -->\n"
            "<!-- B_START -->b<!-- B_END -->"
        )
        result = replace_table_in_section(content, "A", "new_a")
        assert "new_a" in result
        assert "b" in result


@_skip_no_pipeline
@pytest.mark.green
class TestGenerateModelTable:
    def test_produces_valid_markdown_table(self):
        models_list = [
            {
                "name": "test_model",
                "algorithm": "XGBRegressor",
                "targets": "lr_ged_sb",
                "queryset": "test_qs",
                "hyperparameters": "test_hp",
                "deployment_status": "shadow",
                "creator": "Test",
            }
        ]
        table = generate_model_table(models_list)
        assert "test_model" in table
        assert "XGBRegressor" in table
        assert "|" in table
        lines = [line for line in table.strip().split("\n") if line.strip()]
        assert len(lines) >= 3

    def test_header_row_has_expected_columns(self):
        table = generate_model_table([])
        first_line = table.strip().split("\n")[0]
        assert "Model Name" in first_line
        assert "Algorithm" in first_line
        assert "Input Features" in first_line
        assert "Hyperparameters" in first_line
        assert "Forecasting Type" not in first_line

    def test_separator_row_is_valid_markdown(self):
        table = generate_model_table([])
        lines = table.strip().split("\n")
        separator = lines[1]
        cells = [c.strip() for c in separator.split("|") if c.strip()]
        for cell in cells:
            assert re.match(r'^-+$', cell), (
                f"Separator cell '{cell}' is not valid markdown"
            )

    def test_targets_list_rendered_as_comma_separated(self):
        models = [{"name": "m", "targets": ["a", "b", "c"]}]
        table = generate_model_table(models)
        assert "a, b, c" in table

    def test_missing_keys_produce_empty_cells(self):
        models = [{"name": "minimal"}]
        table = generate_model_table(models)
        assert "minimal" in table

    def test_empty_model_list_produces_header_only(self):
        table = generate_model_table([])
        lines = [line for line in table.strip().split("\n") if line.strip()]
        assert len(lines) == 2


@_skip_no_pipeline
@pytest.mark.green
class TestGenerateEnsembleTable:
    def test_header_has_constituent_models(self):
        table = generate_ensemble_table([])
        first_line = table.strip().split("\n")[0]
        assert "Ensemble Name" in first_line
        assert "Constituent Models" in first_line
        assert "Input Features" not in first_line

    def test_shows_aggregation_as_algorithm(self):
        ensembles = [{"name": "test_ens", "aggregation": "mean"}]
        table = generate_ensemble_table(ensembles)
        assert "mean" in table

    def test_shows_modelset_link(self):
        ensembles = [{"name": "test_ens", "modelset_link": "- [link](url)"}]
        table = generate_ensemble_table(ensembles)
        assert "link" in table


@_skip_no_pipeline
@pytest.mark.green
class TestCreateLink:
    def test_produces_markdown_link_format(self):
        from views_pipeline_core.managers.model import ModelPathManager
        root = ModelPathManager.get_root()
        test_path = root / "models" / "test_model" / "configs" / "config_hp.py"
        result = create_link("hp_link", test_path)
        assert result.startswith("- [hp_link](")
        assert "config_hp.py" in result

    def test_marker_appears_in_link_text(self):
        from views_pipeline_core.managers.model import ModelPathManager
        root = ModelPathManager.get_root()
        test_path = root / "models" / "x" / "file.py"
        result = create_link("my_marker", test_path)
        assert "[my_marker]" in result

    def test_link_contains_github_url(self):
        from views_pipeline_core.managers.model import ModelPathManager
        from tools.catalogs.create_catalogs import GITHUB_URL
        root = ModelPathManager.get_root()
        test_path = root / "README.md"
        result = create_link("readme", test_path)
        assert GITHUB_URL in result


@_skip_no_pipeline
@pytest.mark.red
class TestCatalogAdversarialWithPipeline:
    """Red tests for create_catalogs.py functions that need views_pipeline_core."""

    def test_create_link_empty_marker(self):
        from views_pipeline_core.managers.model import ModelPathManager
        root = ModelPathManager.get_root()
        result = create_link("", root / "file.py")
        assert "[](" in result

    def test_generate_model_table_missing_all_keys(self):
        table = generate_model_table([{}])
        lines = [ln for ln in table.strip().split("\n") if ln.strip()]
        assert len(lines) == 3

    def test_generate_model_table_targets_is_none(self):
        table = generate_model_table([{"name": "x", "targets": None}])
        assert "x" in table

    def test_generate_ensemble_table_missing_aggregation(self):
        table = generate_ensemble_table([{"name": "e"}])
        assert "e" in table
