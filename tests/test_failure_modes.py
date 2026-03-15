"""Red-team tests: verify correct error behavior for malformed configs.

These tests use temporary files to inject failures and verify that
the config loading infrastructure fails loudly rather than silently.
"""
import pytest

from tests.conftest import load_config_module


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
