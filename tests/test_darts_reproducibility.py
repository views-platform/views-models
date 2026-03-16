"""Tests that all darts models have complete ReproducibilityGate parameters.

The views-r2darts2 ReproducibilityGate requires 19 core params for ALL darts
models plus architecture-specific params per algorithm. This test prevents
regression of the missing-params bug fixed in 6 models on 2026-03-16.
"""
import ast

import pytest

from tests.conftest import ALL_MODEL_DIRS, MODEL_NAMES, load_config_module


# Parameters that run_type/name/algorithm are set at runtime — skip them
DARTS_CORE_PARAMS = {
    "random_state", "steps", "loss_function", "lr", "weight_decay",
    "batch_size", "n_epochs", "optimizer_cls", "lr_scheduler_factor",
    "lr_scheduler_patience", "lr_scheduler_min_lr", "early_stopping_patience",
    "early_stopping_min_delta", "gradient_clip_val", "num_samples", "mc_dropout",
}

ARCH_PARAMS = {
    "TFTModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "hidden_size", "lstm_layers", "num_attention_heads", "dropout",
        "feed_forward", "add_relative_index", "use_static_covariates",
        "full_attention", "use_reversible_instance_norm", "norm_type",
        "skip_interpolation", "hidden_continuous_size",
    },
    "TransformerModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "d_model", "nhead", "num_encoder_layers", "num_decoder_layers",
        "dim_feedforward", "dropout", "activation", "norm_type",
        "use_reversible_instance_norm", "detect_anomaly",
    },
    "TiDEModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "num_encoder_layers", "num_decoder_layers", "decoder_output_dim",
        "hidden_size", "temporal_width_past", "temporal_width_future",
        "temporal_decoder_hidden", "use_layer_norm", "dropout",
        "use_static_covariates", "use_reversible_instance_norm",
    },
    "TSMixerModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "num_blocks", "ff_size", "hidden_size", "activation", "dropout",
        "norm_type", "normalize_before", "use_static_covariates",
        "use_reversible_instance_norm",
    },
    "TCNModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "kernel_size", "num_filters", "dilation_base", "dropout",
        "use_reversible_instance_norm",
    },
    "BlockRNNModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "rnn_type", "hidden_dim", "n_rnn_layers", "dropout",
        "use_reversible_instance_norm",
    },
    "NBEATSModel": {
        "input_chunk_length", "output_chunk_length", "output_chunk_shift",
        "num_stacks", "num_blocks", "num_layers", "layer_widths",
        "activation", "dropout", "generic_architecture",
    },
}


def _is_darts_model(model_dir):
    """Check if a model imports from views_r2darts2."""
    source = (model_dir / "main.py").read_text()
    return "views_r2darts2" in source


def _get_algorithm(model_dir):
    """Get algorithm name from config_meta.py."""
    module = load_config_module(model_dir / "configs" / "config_meta.py")
    return module.get_meta_config().get("algorithm")


DARTS_MODELS = [
    d for d in ALL_MODEL_DIRS if _is_darts_model(d)
]
DARTS_MODEL_NAMES = [d.name for d in DARTS_MODELS]


class TestDartsReproducibilityGate:
    @pytest.mark.parametrize("model_dir", DARTS_MODELS, ids=DARTS_MODEL_NAMES)
    def test_has_core_params(self, model_dir):
        """Every darts model must have all 16 core ReproducibilityGate params."""
        hp = load_config_module(
            model_dir / "configs" / "config_hyperparameters.py"
        ).get_hp_config()
        missing = DARTS_CORE_PARAMS - set(hp.keys())
        assert not missing, (
            f"{model_dir.name} missing core ReproducibilityGate params: "
            f"{sorted(missing)}"
        )

    @pytest.mark.parametrize("model_dir", DARTS_MODELS, ids=DARTS_MODEL_NAMES)
    def test_has_architecture_params(self, model_dir):
        """Every darts model must have all params required by its algorithm."""
        algorithm = _get_algorithm(model_dir)
        if algorithm not in ARCH_PARAMS:
            pytest.skip(f"Unknown algorithm '{algorithm}' — no genome defined")

        hp = load_config_module(
            model_dir / "configs" / "config_hyperparameters.py"
        ).get_hp_config()
        required = ARCH_PARAMS[algorithm]
        missing = required - set(hp.keys())
        assert not missing, (
            f"{model_dir.name} ({algorithm}) missing architecture params: "
            f"{sorted(missing)}"
        )
