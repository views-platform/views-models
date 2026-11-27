"""S3 (sprint: kill silent sample-count failures) — agnostic reader + decoy guard.

Posterior sample count is named differently by each model family's runtime
(baseline `n_samples`, hydranet `n_posterior_samples`, r2darts `num_samples`,
stepshifter `pred_samples` — register C-104). `get_n_posterior_samples` now
reads whichever key a config declares (no forced rename), and fails loud when a
config declares several of them with DIFFERENT values — the decoy divergence
that silently discarded a sample-count change during the 2026-07-20 FAO delivery
(register C-85/C-104).
"""
import textwrap

import pytest

from tests.conftest import SAMPLE_COUNT_CONFIG_KEYS, get_n_posterior_samples

pytestmark = pytest.mark.beige  # convention/structural compliance (ADR-005)


def _model_with_hp(tmp_path, hp_body: str):
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "config_hyperparameters.py").write_text(
        textwrap.dedent(
            f"""
            def get_hp_config():
                return {{{hp_body}}}
            """
        )
    )
    return tmp_path


@pytest.mark.parametrize("key", SAMPLE_COUNT_CONFIG_KEYS)
def test_reads_whichever_family_key_is_present(tmp_path, key):
    m = _model_with_hp(tmp_path, f'"{key}": 48')
    assert get_n_posterior_samples(m) == 48


def test_multi_key_equal_is_accepted(tmp_path):
    # the baseline convention: n_samples + n_posterior_samples, kept equal.
    m = _model_with_hp(tmp_path, '"n_samples": 64, "n_posterior_samples": 64')
    assert get_n_posterior_samples(m) == 64


def test_multi_key_divergent_fails_loud(tmp_path):
    # tonight's exact trap: edit one key, not the other.
    m = _model_with_hp(tmp_path, '"n_samples": 128, "n_posterior_samples": 16')
    with pytest.raises(ValueError, match="disagree"):
        get_n_posterior_samples(m)


def test_absent_returns_none(tmp_path):
    m = _model_with_hp(tmp_path, '"steps": [1, 2, 3]')
    assert get_n_posterior_samples(m) is None
