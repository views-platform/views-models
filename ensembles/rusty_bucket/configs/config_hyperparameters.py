def get_hp_config():
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        # Explicit belt-and-suspenders declarations (ADR-015). The config-time
        # contract (tests/test_ensemble_configs.py) asserts these match reality:
        #   expected_models == len(config_modelset["models"])
        #   every constituent's n_posterior_samples == expected_samples_per_model
        # PFE concat concatenates draws on the sample axis (pipeline-core
        # prediction_frame_ensemble.py:99), so the pooled total is
        # expected_models × expected_samples_per_model = 8 × 128 = 1024 draws.
        # Equal per-model counts give each constituent equal weight in the pooled
        # mixture and a predictable pooled dimension; a mismatch fails loud at CI.
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
