"""Tests that model directories follow the required structure and naming conventions."""
import re

import pytest

from tests.conftest import MODEL_NAMES

MODEL_NAME_PATTERN = re.compile(r'^[a-z]+_[a-z]+$')

REQUIRED_CONFIG_FILES = [
    "config_meta.py",
    "config_deployment.py",
    "config_hyperparameters.py",
    "config_partitions.py",
    "config_sweep.py",
    "config_queryset.py",
]

# Subdirectories that ModelPathManager resolves at runtime in baseline/stepshifter
# managers. A missing directory here causes a TypeError on fresh clones.
REQUIRED_SUBDIRS = ["artifacts", "data/raw", "data/generated", "logs"]


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
    def test_required_subdirectory_exists(self, model_dir, subdir):
        path = model_dir / subdir
        assert path.is_dir(), (
            f"{model_dir.name} missing required subdirectory: {subdir}. "
            f"Add a .gitkeep file to track empty directories in git."
        )
