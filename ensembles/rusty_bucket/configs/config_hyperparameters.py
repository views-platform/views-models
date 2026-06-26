def get_hp_config():
    hp_config = {
        "steps": [*range(1, 36 + 1, 1)],
        # Explicit belt-and-suspenders declarations (ADR-015). The config-time
        # contract (tests/test_ensemble_configs.py) asserts these match reality:
        #   expected_models == len(config_modelset["models"])
        #   every constituent's n_posterior_samples == expected_samples_per_model
        # concat pools by resampling to a FIXED size (= each constituent's count),
        # and pipeline-core's aggregator hard-requires equal counts — so a single
        # number is correct here, and a mismatch fails loud at CI, not mid-run.
        "expected_models": 8,
        "expected_samples_per_model": 128,
    }
    return hp_config
