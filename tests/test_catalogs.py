"""Tests for create_catalogs.py — catalog generation utilities."""
import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from create_catalogs import replace_table_in_section, generate_markdown_table
    _HAS_PIPELINE_CORE = True
except (ImportError, ModuleNotFoundError):
    _HAS_PIPELINE_CORE = False

_skip_no_pipeline = pytest.mark.skipif(
    not _HAS_PIPELINE_CORE,
    reason="views_pipeline_core not installed"
)


class TestNoExecUsage:
    def test_create_catalogs_does_not_use_exec(self):
        """create_catalogs.py should use importlib, not raw exec()."""
        source = (REPO_ROOT / "create_catalogs.py").read_text()
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


@_skip_no_pipeline
class TestGenerateMarkdownTable:
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
        table = generate_markdown_table(models_list)
        assert "test_model" in table
        assert "XGBRegressor" in table
        assert "|" in table
        lines = [line for line in table.strip().split("\n") if line.strip()]
        assert len(lines) >= 3
