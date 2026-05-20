"""Red-team tests: verify correct error behavior for malformed configs.

These tests use temporary files to inject failures and verify that
the config loading infrastructure fails loudly rather than silently.
Addresses CIC failure modes for CatalogExtractor and config loading.
"""
import subprocess

import pytest

from tests.conftest import (
    load_config_module,
    REPO_ROOT,
    MODELS_DIR,
    ALL_MODEL_DIRS,
    ALL_ENSEMBLE_DIRS,
)


@pytest.mark.red
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


@pytest.mark.red
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

    def test_help_flag_exits_zero(self):
        """--help must exit 0 and print usage text."""
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "run_integration_tests.sh"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout

    def test_h_flag_exits_zero(self):
        """-h must behave identically to --help."""
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "run_integration_tests.sh"), "-h"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0

    def test_nonexistent_model_produces_warning(self):
        """A nonexistent model with --models should produce a warning message."""
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "run_integration_tests.sh"),
             "--models", "nonexistent_model_xyz_12345"],
            capture_output=True, text=True, timeout=30,
        )
        assert "not found" in result.stdout or "No models found" in result.stdout


@pytest.mark.red
class TestPartitionBoundaryValidation:
    """Red tests for degenerate partition step values.

    generate(steps=0) produces a zero-length test window.
    generate(steps=-1) produces an inverted test window.
    Both are structurally invalid but not guarded.
    """

    @staticmethod
    def _load_partition_module(model_dir):
        cfg = model_dir / "configs" / "config_partitions.py"
        try:
            return load_config_module(cfg)
        except (ImportError, ModuleNotFoundError):
            pytest.skip(f"{model_dir.name}: config_partitions.py has uninstalled deps")

    @pytest.mark.parametrize(
        "model_dir",
        ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS,
        ids=[d.name for d in ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS],
    )
    def test_generate_with_zero_steps(self, model_dir):
        """generate(steps=0) must not crash, but produces a degenerate range."""
        module = self._load_partition_module(model_dir)
        result = module.generate(steps=0)
        forecast_test = result["forecasting"]["test"]
        assert forecast_test[0] == forecast_test[1], (
            f"{model_dir.name}: steps=0 should produce a zero-length test window, "
            f"got {forecast_test}"
        )

    @pytest.mark.parametrize(
        "model_dir",
        ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS,
        ids=[d.name for d in ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS],
    )
    def test_generate_with_negative_steps(self, model_dir):
        """generate(steps=-1) must produce an inverted test range (end < start)."""
        module = self._load_partition_module(model_dir)
        result = module.generate(steps=-1)
        forecast_test = result["forecasting"]["test"]
        assert forecast_test[1] < forecast_test[0], (
            f"{model_dir.name}: steps=-1 should produce inverted range, "
            f"got {forecast_test}"
        )

    @pytest.mark.parametrize(
        "model_dir",
        ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS,
        ids=[d.name for d in ALL_MODEL_DIRS + ALL_ENSEMBLE_DIRS],
    )
    def test_default_steps_produces_valid_range(self, model_dir):
        """Default generate() must produce a valid forecasting test range."""
        module = self._load_partition_module(model_dir)
        result = module.generate()
        forecast_test = result["forecasting"]["test"]
        assert forecast_test[1] > forecast_test[0], (
            f"{model_dir.name}: default steps must produce valid range, "
            f"got {forecast_test}"
        )


@pytest.mark.red
class TestEnsembleConstituentIntegrity:
    """Red tests for ensemble error paths at the config level.

    These test conditions that would cause runtime failures during
    ensemble evaluation: missing models, empty model lists, etc.
    """

    @pytest.mark.parametrize(
        "ensemble_dir",
        ALL_ENSEMBLE_DIRS,
        ids=[d.name for d in ALL_ENSEMBLE_DIRS],
    )
    def test_constituent_model_configs_are_loadable(self, ensemble_dir):
        """Every model listed in an ensemble must have loadable config_meta."""
        meta_cfg = ensemble_dir / "configs" / "config_meta.py"
        module = load_config_module(meta_cfg)
        meta = module.get_meta_config()
        for model_name in meta["models"]:
            model_meta = MODELS_DIR / model_name / "configs" / "config_meta.py"
            if not model_meta.exists():
                pytest.skip(f"{model_name} not present")
            try:
                load_config_module(model_meta).get_meta_config()
            except (ImportError, ModuleNotFoundError):
                pytest.skip(f"{model_name}: config_meta.py has uninstalled deps")

    @pytest.mark.parametrize(
        "ensemble_dir",
        ALL_ENSEMBLE_DIRS,
        ids=[d.name for d in ALL_ENSEMBLE_DIRS],
    )
    def test_constituent_models_have_matching_partitions(self, ensemble_dir):
        """All constituent models must use the same cal/val partition boundaries
        as the ensemble itself."""
        ens_parts_cfg = ensemble_dir / "configs" / "config_partitions.py"
        try:
            ens_parts = load_config_module(ens_parts_cfg).generate()
        except (ImportError, ModuleNotFoundError):
            pytest.skip(f"{ensemble_dir.name}: uninstalled deps in config_partitions")

        meta_cfg = ensemble_dir / "configs" / "config_meta.py"
        meta = load_config_module(meta_cfg).get_meta_config()

        for model_name in meta["models"]:
            model_parts_cfg = MODELS_DIR / model_name / "configs" / "config_partitions.py"
            if not model_parts_cfg.exists():
                pytest.skip(f"{model_name} not present")
            try:
                model_parts = load_config_module(model_parts_cfg).generate()
            except (ImportError, ModuleNotFoundError):
                pytest.skip(f"{model_name}: uninstalled deps in config_partitions")
            for section in ("calibration", "validation"):
                assert model_parts[section] == ens_parts[section], (
                    f"{ensemble_dir.name}: constituent {model_name} has "
                    f"mismatched {section} partitions"
                )

    def test_malformed_model_list_type_detectable(self, tmp_path):
        """A config_meta where models is a string (not list) should be caught."""
        bad_meta = tmp_path / "config_meta.py"
        bad_meta.write_text(
            "def get_meta_config():\n"
            "    return {'name': 'bad', 'models': 'single_model', "
            "'regression_targets': ['t'], 'level': 'pgm', 'aggregation': 'mean'}\n"
        )
        module = load_config_module(bad_meta)
        meta = module.get_meta_config()
        assert not isinstance(meta["models"], list), (
            "This test documents that a string models value loads without error"
        )

    def test_empty_model_list_is_detectable(self, tmp_path):
        """A config_meta with an empty models list should be caught."""
        bad_meta = tmp_path / "config_meta.py"
        bad_meta.write_text(
            "def get_meta_config():\n"
            "    return {'name': 'bad', 'models': [], "
            "'regression_targets': ['t'], 'level': 'pgm', 'aggregation': 'mean'}\n"
        )
        module = load_config_module(bad_meta)
        meta = module.get_meta_config()
        assert meta["models"] == []


@pytest.mark.red
class TestMalformedQuerysetDescriptor:
    """Red tests for malformed queryset config files.

    These verify that the config loading infrastructure handles
    broken queryset configurations correctly.
    """

    def test_queryset_missing_name_key(self, tmp_path):
        """A queryset config without the expected function loads but is detectable."""
        bad_qs = tmp_path / "config_queryset.py"
        bad_qs.write_text(
            "def get_queryset():\n"
            "    return {'theme': 'fatalities', 'loa': 'priogrid_month'}\n"
        )
        module = load_config_module(bad_qs)
        qs = module.get_queryset()
        assert "name" not in qs

    def test_queryset_returning_none(self, tmp_path):
        """A queryset function that returns None is loadable but detectable."""
        bad_qs = tmp_path / "config_queryset.py"
        bad_qs.write_text(
            "def get_queryset():\n"
            "    return None\n"
        )
        module = load_config_module(bad_qs)
        assert module.get_queryset() is None

    def test_queryset_with_circular_import(self, tmp_path):
        """A queryset that triggers circular import must propagate the error."""
        bad_qs = tmp_path / "config_queryset.py"
        bad_qs.write_text(
            "from config_queryset import get_queryset as _self\n"
            "def get_queryset():\n"
            "    return _self()\n"
        )
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_config_module(bad_qs)
