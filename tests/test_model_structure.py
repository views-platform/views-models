"""Tests that model directories follow the required structure and naming conventions."""
import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import MODEL_NAMES, REPO_ROOT

MODEL_NAME_PATTERN = re.compile(r'^[a-z]+_[a-z]+$')

REQUIRED_CONFIG_FILES = [
    "config_meta.py",
    "config_deployment.py",
    "config_hyperparameters.py",
    "config_partitions.py",
    "config_sweep.py",
    "config_queryset.py",
]

# Subdirectories that ModelPathManager validates at runtime
# (views-pipeline-core/.../model_path.py:440-458). A missing directory here
# causes FileNotFoundError on fresh clones. Checked against the git index, not
# the filesystem, so a dev who ran a model locally can't mask a broken scaffold.
REQUIRED_SUBDIRS = [
    "artifacts",
    "data/raw",
    "data/generated",
    "data/processed",
    "logs",
    "notebooks",
    "reports",
]


def _git_tracks_path(rel_path: Path) -> bool:
    """True iff `rel_path` (relative to REPO_ROOT) has any tracked file beneath it."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", str(rel_path)],
        capture_output=True,
    )
    if result.returncode == 0:
        return True
    # ls-files --error-unmatch only matches files directly; check for any tracked
    # descendants via plain ls-files on the directory.
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", str(rel_path)],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


class TestModelNaming:
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_model_name_follows_convention(self, name):
        """Model directory names must be adjective_noun in lowercase."""
        assert MODEL_NAME_PATTERN.match(name), (
            f"Model name '{name}' does not match required pattern 'adjective_noun'"
        )


class TestModelFiles:
    def test_main_py_exists(self, model_dir):
        assert (model_dir / "main.py").exists()

    def test_run_sh_exists(self, model_dir):
        assert (model_dir / "run.sh").exists()

    def test_readme_md_tracked(self, model_dir):
        """README.md must be tracked in git. ModelPathManager._initialize_scripts
        (views-pipeline-core/.../model_path.py:475) validates its existence at
        runtime; a missing file crashes on fresh clone. Checked against the git
        index, not the filesystem, so a local-only file can't mask a broken
        scaffold (same reason TestModelDirectoryStructure uses git ls-files).
        """
        rel_path = (model_dir / "README.md").relative_to(REPO_ROOT)
        assert _git_tracks_path(rel_path), (
            f"{model_dir.name} has no tracked README.md — the file is validated "
            f"at runtime by ModelPathManager and its absence crashes on fresh "
            f"clone. Create an (even empty) README.md and git-add it."
        )

    def test_configs_directory_exists(self, model_dir):
        assert (model_dir / "configs").is_dir()

    @pytest.mark.parametrize("config_file", REQUIRED_CONFIG_FILES)
    def test_required_config_file_exists(self, model_dir, config_file):
        cfg_path = model_dir / "configs" / config_file
        assert cfg_path.exists(), (
            f"{model_dir.name} missing config file: {config_file}"
        )


class TestModelDirectoryStructure:
    @pytest.mark.parametrize("subdir", REQUIRED_SUBDIRS)
    def test_required_subdirectory_tracked(self, model_dir, subdir):
        """The subdirectory must be tracked in git (not just present on the
        local filesystem). Filesystem checks give false green after a local run
        populates the directory; only the git index reflects fresh-clone state.
        """
        rel_path = (model_dir / subdir).relative_to(REPO_ROOT)
        assert _git_tracks_path(rel_path), (
            f"{model_dir.name} has no tracked files under {subdir}/ — "
            f"the directory will be absent on fresh clone and crash "
            f"ModelPathManager validation. Add a .gitkeep file."
        )
