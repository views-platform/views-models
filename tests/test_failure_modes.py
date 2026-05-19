"""Red-team tests: verify correct error behavior for malformed configs.

These tests use temporary files to inject failures and verify that
the config loading infrastructure fails loudly rather than silently.
Addresses CIC failure modes for CatalogExtractor and config loading.
"""
import subprocess

import pytest

from tests.conftest import load_config_module, REPO_ROOT


class TestConfigLoadingSyntaxError:
    def test_syntax_error_raises(self, tmp_path):
        """A config file with a syntax error must raise SyntaxError."""
        bad_config = tmp_path / "config_meta.py"
        bad_config.write_text("def get_meta_config(\n    # missing closing paren")
        with pytest.raises(SyntaxError):
            load_config_module(bad_config)

    def test_missing_function_is_detectable(self, tmp_path):
        """A config file that loads but lacks the expected function."""
        empty_config = tmp_path / "config_meta.py"
        empty_config.write_text("x = 42\n")
        module = load_config_module(empty_config)
        assert not hasattr(module, "get_meta_config"), (
            "Module should not have get_meta_config if the file doesn't define it"
        )

    def test_nonexistent_file_raises(self, tmp_path):
        """Loading a non-existent config file must raise an error."""
        missing = tmp_path / "does_not_exist.py"
        with pytest.raises((FileNotFoundError, OSError)):
            load_config_module(missing)

    def test_function_returning_non_dict_is_loadable(self, tmp_path):
        """A config that returns a non-dict loads successfully —
        the caller is responsible for type checking."""
        bad_return = tmp_path / "config_meta.py"
        bad_return.write_text(
            'def get_meta_config():\n'
            '    return "not a dict"\n'
        )
        module = load_config_module(bad_return)
        result = module.get_meta_config()
        assert not isinstance(result, dict)

    def test_empty_config_file_loads_without_functions(self, tmp_path):
        """An empty config file loads as a module but has no callable functions."""
        empty = tmp_path / "config_meta.py"
        empty.write_text("")
        module = load_config_module(empty)
        assert not hasattr(module, "get_meta_config")
        assert not hasattr(module, "get_hp_config")
        assert not hasattr(module, "generate")

    def test_config_with_import_error_raises(self, tmp_path):
        """A config that imports a non-existent module must raise ImportError."""
        bad_import = tmp_path / "config_meta.py"
        bad_import.write_text("from nonexistent_package import something\n")
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_config_module(bad_import)

    def test_config_with_runtime_error_raises(self, tmp_path):
        """A config with top-level code that raises must propagate the error."""
        bad_runtime = tmp_path / "config_meta.py"
        bad_runtime.write_text("raise ValueError('bad config')\n")
        with pytest.raises(ValueError, match="bad config"):
            load_config_module(bad_runtime)


class TestIntegrationTestRunnerFailureModes:
    """Red-team tests for run_integration_tests.sh (CIC: IntegrationTestRunner).

    Unlike the config loading tests above (which use tmp_path fixtures),
    these tests invoke the shell script via subprocess to verify CLI
    argument handling and exit code behavior.
    """

    def test_nonexistent_model_exits_with_error(self):
        """Requesting a non-existent model must exit with code 1."""
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "run_integration_tests.sh"),
             "--models", "nonexistent_model_xyz_12345"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1

    def test_unknown_flag_exits_with_error(self):
        """An unknown CLI flag must exit with code 1."""
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "run_integration_tests.sh"),
             "--bogus-flag"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 1
