"""Non-blocking visibility report for the posterior sample-count standard (ADR-015).

The integration-period standard for ``n_posterior_samples`` in views-models is 128.
This test NEVER fails — it emits a warning listing sample-producing models that
declare a different count, so drift is visible in the CI log without crying wolf
(a hard gate would block every deliberate fast-iteration run). The hard contract
lives in test_ensemble_configs.py (per-ensemble, opt-in); this is the soft report.
"""
import warnings

import pytest

from tests.conftest import ALL_MODEL_DIRS, get_n_posterior_samples

pytestmark = pytest.mark.beige

# The integration-period standard (ADR-015). A named constant, not a scattered
# literal — revisit here (and the ADR) when the standard changes.
STANDARD_N_POSTERIOR_SAMPLES = 128


def test_report_off_standard_sample_counts():
    off_standard = {}
    for model_dir in ALL_MODEL_DIRS:
        n = get_n_posterior_samples(model_dir)
        if n is not None and n != STANDARD_N_POSTERIOR_SAMPLES:
            off_standard[model_dir.name] = n

    if off_standard:
        listing = ", ".join(f"{name}={n}" for name, n in sorted(off_standard.items()))
        warnings.warn(
            f"{len(off_standard)} sample-producing model(s) declare "
            f"n_posterior_samples != {STANDARD_N_POSTERIOR_SAMPLES} (ADR-015 standard): "
            f"{listing}. This is a non-blocking report — intentional during model "
            f"integration; normalize before production delivery.",
            UserWarning,
            stacklevel=2,
        )
    # Always passes — visibility only.
    assert True
