"""Characterization tests for pure functions in tooling scripts.

These scripts (create_catalogs.py, update_readme.py, generate_features_catalog.py)
cannot be imported directly because they have top-level imports from
views_pipeline_core. Instead, we duplicate the pure function logic here as
characterization tests — pinning the current behavior so regressions are caught.

If the algorithm in the source script changes, these tests must be updated
to match. The source of truth is the script; the test documents the contract.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Characterization: create_catalogs.py :: replace_table_in_section (lines 156-177)
# ---------------------------------------------------------------------------

def _replace_table_in_section(content, section_name, new_table):
    """Exact copy of create_catalogs.py::replace_table_in_section."""
    start_marker = f"<!-- {section_name}_START -->"
    end_marker = f"<!-- {section_name}_END -->"
    before, _, after = content.partition(start_marker)
    _, _, after = after.partition(end_marker)
    return before + start_marker + "\n" + new_table + "\n" + end_marker + after


class TestReplaceTableInSection:
    def test_basic_replacement(self):
        content = (
            "# README\n"
            "<!-- CM_TABLE_START -->\n"
            "old table here\n"
            "<!-- CM_TABLE_END -->\n"
            "footer\n"
        )
        result = _replace_table_in_section(content, "CM_TABLE", "| new | table |")
        assert "| new | table |" in result
        assert "old table here" not in result
        assert result.startswith("# README\n")
        assert result.endswith("footer\n")

    def test_missing_marker_returns_content_with_appended_markers(self):
        content = "# README\nNo markers here\n"
        result = _replace_table_in_section(content, "MISSING", "| table |")
        # partition() returns full string + two empty strings when marker not found
        # so the result is: content + start_marker + "\n" + table + "\n" + end_marker
        assert "<!-- MISSING_START -->" in result
        assert "<!-- MISSING_END -->" in result
        assert "| table |" in result

    def test_empty_table(self):
        content = (
            "before\n"
            "<!-- X_START -->\nold\n<!-- X_END -->\n"
            "after\n"
        )
        result = _replace_table_in_section(content, "X", "")
        assert "<!-- X_START -->\n\n<!-- X_END -->" in result
        assert "old" not in result

    def test_preserves_content_outside_markers(self):
        content = (
            "header\n"
            "<!-- T_START -->middle<!-- T_END -->\n"
            "footer"
        )
        result = _replace_table_in_section(content, "T", "replaced")
        assert result.startswith("header\n")
        assert result.endswith("\nfooter")


# ---------------------------------------------------------------------------
# Characterization: create_catalogs.py :: generate_markdown_table (lines 87-123)
# ---------------------------------------------------------------------------

def _generate_markdown_table(models_list):
    """Exact copy of create_catalogs.py::generate_markdown_table."""
    headers = [
        'Model Name', 'Algorithm', 'Targets', 'Input Features',
        'Non-default Hyperparameters', 'Forecasting Type',
        'Implementation Status', 'Implementation Date', 'Author',
    ]
    markdown_table = '| ' + ' '.join([f"{header} |" for header in headers]) + '\n'
    markdown_table += '| ' + ' '.join(['-' * len(header) + ' |' for header in headers]) + '\n'
    for model in models_list:
        targets = model.get('targets', '')
        if isinstance(targets, list):
            targets = ', '.join(targets)
        row = [
            model.get('name', ''),
            str(model.get('algorithm', '')).split('(')[0],
            targets,
            model.get('queryset', ''),
            model.get('hyperparameters', ''),
            'None',
            model.get('deployment_status', ''),
            'NA',
            model.get('creator', ''),
        ]
        markdown_table += '| ' + ' | '.join(row) + ' |\n'
    return markdown_table


class TestGenerateMarkdownTable:
    def test_basic_table(self):
        models = [
            {
                'name': 'test_model',
                'algorithm': 'RandomForest(n=100)',
                'targets': ['fatalities', 'ged_sb'],
                'queryset': 'link_to_qs',
                'hyperparameters': 'hp_link',
                'deployment_status': 'shadow',
                'creator': 'alice',
            }
        ]
        result = _generate_markdown_table(models)
        lines = result.strip().split('\n')
        assert len(lines) == 3  # header + separator + 1 row
        assert 'Model Name' in lines[0]
        assert 'test_model' in lines[2]
        assert 'RandomForest' in lines[2]
        # Algorithm should strip parenthesized part
        assert 'n=100' not in lines[2]
        # Targets list should be joined
        assert 'fatalities, ged_sb' in lines[2]

    def test_empty_list(self):
        result = _generate_markdown_table([])
        lines = result.strip().split('\n')
        assert len(lines) == 2  # header + separator only

    def test_missing_keys_use_empty_string(self):
        models = [{}]
        result = _generate_markdown_table(models)
        lines = result.strip().split('\n')
        assert len(lines) == 3
        # Row should have 9 pipe-separated cells (all empty except 'None' and 'NA')
        assert 'None' in lines[2]
        assert 'NA' in lines[2]


# ---------------------------------------------------------------------------
# Characterization: update_readme.py :: generate_repo_structure (lines 12-58)
# ---------------------------------------------------------------------------

def _generate_repo_structure(folders, scripts, model_name):
    """Exact copy of update_readme.py::generate_repo_structure."""
    root_path = Path(folders["model_dir"])
    tree = [model_name]
    sorted_folders = sorted(folders.values(), key=lambda x: x.count("/"))
    folder_structure = {folder: [] for folder in sorted_folders}
    for script, script_path in scripts.items():
        parent_folder = str(Path(script_path).parent)
        if parent_folder in folder_structure:
            folder_structure[parent_folder].append(script)

    def build_tree(current_path, depth=0):
        indent = "│   " * depth
        rel_path = Path(current_path).relative_to(root_path)
        tree.append(f"{indent}├── {rel_path.name}")
        for script in sorted(folder_structure[current_path]):
            tree.append(f"{indent}│   ├── {script}")
        subfolders = [f for f in sorted_folders if Path(f).parent == Path(current_path)]
        for subfolder in subfolders:
            build_tree(subfolder, depth + 1)

    root_scripts = []
    for key, value in scripts.items():
        if Path(value).parent == Path(root_path):
            root_scripts.append(key)
    for script in sorted(root_scripts):
        tree.append(f"├── {script}")
    for folder in sorted_folders:
        if Path(folder).parent == root_path:
            build_tree(folder)
    return "\n".join(tree)


class TestGenerateRepoStructure:
    def test_basic_structure(self, tmp_path):
        model_dir = str(tmp_path / "models" / "test_model")
        configs_dir = str(tmp_path / "models" / "test_model" / "configs")
        folders = {
            "model_dir": model_dir,
            "configs": configs_dir,
        }
        scripts = {
            "main.py": str(tmp_path / "models" / "test_model" / "main.py"),
            "config_meta.py": str(tmp_path / "models" / "test_model" / "configs" / "config_meta.py"),
        }
        result = _generate_repo_structure(folders, scripts, "test_model")
        assert result.startswith("test_model")
        assert "main.py" in result
        assert "configs" in result
        assert "config_meta.py" in result

    def test_empty_scripts(self, tmp_path):
        model_dir = str(tmp_path / "models" / "empty_model")
        folders = {"model_dir": model_dir}
        scripts = {}
        result = _generate_repo_structure(folders, scripts, "empty_model")
        assert result == "empty_model"


# ---------------------------------------------------------------------------
# Characterization: update_readme.py :: "Created on" regex (line 123)
# ---------------------------------------------------------------------------

CREATED_ON_PATTERN = re.compile(r"(## Created on.*)", re.DOTALL)


class TestCreatedOnRegex:
    def test_extracts_section(self):
        content = (
            "# Model README\n\n"
            "Some content here.\n\n"
            "## Created on 2025-01-15\n"
            "By: alice\n"
            "Notes: initial version\n"
        )
        match = CREATED_ON_PATTERN.search(content)
        assert match is not None
        assert match.group(1).startswith("## Created on 2025-01-15")
        # DOTALL means it captures everything after "## Created on"
        assert "By: alice" in match.group(1)

    def test_no_match_without_section(self):
        content = "# Model README\n\nNo created section here.\n"
        match = CREATED_ON_PATTERN.search(content)
        assert match is None

    def test_empty_content(self):
        match = CREATED_ON_PATTERN.search("")
        assert match is None


# ---------------------------------------------------------------------------
# Characterization: generate_features_catalog.py :: Column regex (lines 57-60)
# ---------------------------------------------------------------------------

COLUMN_PATTERN = re.compile(r'Column\((.*?)\)')
COLUMN_NAME_PATTERN = re.compile(r'"(.*?)"')
LOA_PATTERN = re.compile(r'from_loa="(.*?)"')


class TestColumnRegexExtraction:
    SAMPLE_QUERYSET = '''
qs_test = (Queryset("test_queryset", "priogrid_month")
    .with_column(Column("ged_sb_dep", from_loa="priogrid_month", from_column="ged_sb_best_sum_nokgi"))
    .with_column(Column("acled_count", from_loa="country_month", from_column="acled_count_pr"))
    .with_column(Column("simple_col"))
)
'''

    def test_finds_all_columns(self):
        matches = COLUMN_PATTERN.findall(self.SAMPLE_QUERYSET)
        assert len(matches) == 3

    def test_extracts_column_names(self):
        matches = COLUMN_PATTERN.findall(self.SAMPLE_QUERYSET)
        names = [COLUMN_NAME_PATTERN.search(m).group(1) for m in matches]
        assert names == ["ged_sb_dep", "acled_count", "simple_col"]

    def test_extracts_loa(self):
        matches = COLUMN_PATTERN.findall(self.SAMPLE_QUERYSET)
        loas = []
        for m in matches:
            loa_match = LOA_PATTERN.search(m)
            loas.append(loa_match.group(1) if loa_match else None)
        assert loas == ["priogrid_month", "country_month", None]

    def test_no_columns_returns_empty(self):
        source = "x = 42\nno columns here\n"
        matches = COLUMN_PATTERN.findall(source)
        assert matches == []
