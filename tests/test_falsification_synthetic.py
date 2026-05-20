"""Falsification test stubs from audit of PR #56 (synthetic test models).

These tests encode findings from the falsification audit. They are
intentionally written to FAIL until the underlying issue is addressed.

F3: Ensemble model-order dependency — the ensemble's ground truth comes
from whichever model is listed first in config_meta.models. No test
guards this ordering, so a reorder silently changes the expected MSE.
"""
import pytest
from tests.conftest import load_config_module, ENSEMBLES_DIR

SYNTHETIC_ENSEMBLES = ["synthetic_chorus", "synthetic_choir"]


class TestEnsembleModelOrderDependency:
    """F3: The ensemble MSE depends on which model is first in the model list.

    The README documents that ground truth comes from vertical_dream
    (the first model). This test asserts that vertical_dream IS first,
    so a reorder is caught before the expected MSE silently changes.
    """

    @pytest.mark.red
    @pytest.mark.parametrize("ensemble_name", SYNTHETIC_ENSEMBLES)
    def test_first_model_is_vertical_dream(self, ensemble_name):
        cfg = ENSEMBLES_DIR / ensemble_name / "configs" / "config_meta.py"
        if not cfg.exists():
            pytest.skip(f"{ensemble_name} not present")
        module = load_config_module(cfg)
        meta = module.get_meta_config()
        assert meta["models"][0] == "vertical_dream", (
            f"{ensemble_name} model order changed: first model is "
            f"'{meta['models'][0]}', expected 'vertical_dream'. "
            f"This changes the ground truth and expected MSE (4.34444). "
            f"Update the README derivation if this is intentional."
        )


class TestSyntheticEnsembleParity:
    """Verify synthetic_chorus and synthetic_choir have identical configs.

    synthetic_choir is a parity test for DataFrameEnsembleManager.
    All configs except name and manager class must match synthetic_chorus.
    """

    @pytest.fixture()
    def both_metas(self):
        chorus_cfg = ENSEMBLES_DIR / "synthetic_chorus" / "configs" / "config_meta.py"
        choir_cfg = ENSEMBLES_DIR / "synthetic_choir" / "configs" / "config_meta.py"
        if not chorus_cfg.exists() or not choir_cfg.exists():
            pytest.skip("both synthetic ensembles must be present")
        return (
            load_config_module(chorus_cfg).get_meta_config(),
            load_config_module(choir_cfg).get_meta_config(),
        )

    @pytest.mark.green
    def test_same_constituent_models(self, both_metas):
        chorus_meta, choir_meta = both_metas
        assert chorus_meta["models"] == choir_meta["models"]

    @pytest.mark.green
    def test_same_regression_targets(self, both_metas):
        chorus_meta, choir_meta = both_metas
        assert chorus_meta["regression_targets"] == choir_meta["regression_targets"]

    @pytest.mark.green
    def test_same_aggregation(self, both_metas):
        chorus_meta, choir_meta = both_metas
        assert chorus_meta["aggregation"] == choir_meta["aggregation"]

    @pytest.mark.green
    def test_same_level(self, both_metas):
        chorus_meta, choir_meta = both_metas
        assert chorus_meta["level"] == choir_meta["level"]

    @pytest.mark.green
    def test_names_differ(self, both_metas):
        chorus_meta, choir_meta = both_metas
        assert chorus_meta["name"] != choir_meta["name"]

    @pytest.mark.green
    def test_same_partitions(self):
        chorus_cfg = ENSEMBLES_DIR / "synthetic_chorus" / "configs" / "config_partitions.py"
        choir_cfg = ENSEMBLES_DIR / "synthetic_choir" / "configs" / "config_partitions.py"
        if not chorus_cfg.exists() or not choir_cfg.exists():
            pytest.skip("both synthetic ensembles must be present")
        chorus_parts = load_config_module(chorus_cfg).generate()
        choir_parts = load_config_module(choir_cfg).generate()
        assert chorus_parts == choir_parts

    @pytest.mark.green
    def test_same_hyperparameters(self):
        chorus_cfg = ENSEMBLES_DIR / "synthetic_chorus" / "configs" / "config_hyperparameters.py"
        choir_cfg = ENSEMBLES_DIR / "synthetic_choir" / "configs" / "config_hyperparameters.py"
        if not chorus_cfg.exists() or not choir_cfg.exists():
            pytest.skip("both synthetic ensembles must be present")
        chorus_hp = load_config_module(chorus_cfg).get_hp_config()
        choir_hp = load_config_module(choir_cfg).get_hp_config()
        assert chorus_hp == choir_hp

    @pytest.mark.green
    def test_choir_uses_dataframe_manager(self):
        """synthetic_choir must import DataFrameEnsembleManager, not EnsembleManager."""
        main_py = ENSEMBLES_DIR / "synthetic_choir" / "main.py"
        if not main_py.exists():
            pytest.skip("synthetic_choir not present")
        source = main_py.read_text()
        assert "DataFrameEnsembleManager" in source, (
            "synthetic_choir/main.py must use DataFrameEnsembleManager"
        )

    @pytest.mark.green
    def test_chorus_uses_inheritance_manager(self):
        """synthetic_chorus must import EnsembleManager (inheritance-based)."""
        main_py = ENSEMBLES_DIR / "synthetic_chorus" / "main.py"
        if not main_py.exists():
            pytest.skip("synthetic_chorus not present")
        source = main_py.read_text()
        assert "EnsembleManager" in source
        assert "DataFrameEnsembleManager" not in source, (
            "synthetic_chorus/main.py must use EnsembleManager, not DataFrameEnsembleManager"
        )
