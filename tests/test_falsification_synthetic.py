"""Falsification test stubs from audit of PR #56 (synthetic test models).

These tests encode findings from the falsification audit. They are
intentionally written to FAIL until the underlying issue is addressed.

F3: Ensemble model-order dependency — the ensemble's ground truth comes
from whichever model is listed first in config_meta.models. No test
guards this ordering, so a reorder silently changes the expected MSE.
"""
import pytest
from tests.conftest import load_config_module, ENSEMBLES_DIR


class TestEnsembleModelOrderDependency:
    """F3: The ensemble MSE depends on which model is first in the model list.

    The README documents that ground truth comes from vertical_dream
    (the first model). This test asserts that vertical_dream IS first,
    so a reorder is caught before the expected MSE silently changes.
    """

    def test_synthetic_chorus_first_model_is_vertical_dream(self):
        cfg = ENSEMBLES_DIR / "synthetic_chorus" / "configs" / "config_meta.py"
        if not cfg.exists():
            pytest.skip("synthetic_chorus not present")
        module = load_config_module(cfg)
        meta = module.get_meta_config()
        assert meta["models"][0] == "vertical_dream", (
            f"synthetic_chorus model order changed: first model is "
            f"'{meta['models'][0]}', expected 'vertical_dream'. "
            f"This changes the ground truth and expected MSE (4.34444). "
            f"Update the README derivation if this is intentional."
        )
