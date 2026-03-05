
import logging
import torch
from darts.models import NBEATSModel
from darts.timeseries import TimeSeries
import pandas as pd

# Known Pitfall: A local logging.py can shadow the standard library.
# This import helps avoid AttributeError.
from importlib import reload
reload(logging)

def verify_nbeats_architecture():
    """
    Instantiates and prints the architecture of two NBEATS models
    with different num_blocks to verify Darts library behavior.
    """
    print("--- Verifying Darts NBEATSModel architecture ---")

    # Create a minimal dummy TimeSeries for fitting
    dummy_data = pd.DataFrame({
        'time': pd.to_datetime(pd.date_range('2023-01-01', periods=30, freq='D')),
        'value': range(30)
    })
    dummy_ts = TimeSeries.from_dataframe(dummy_data, 'time', 'value')

    # --- Model 1: num_blocks = 1 ---
    print("\n--- Architecture for num_blocks=1 ---")
    model_1 = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,
        generic_architecture=True,
        num_blocks=1,
        num_stacks=2,
        num_layers=2,
        layer_widths=64,
        n_epochs=1,
        random_state=42,
        model_name="nbeats_blocks_1",
    )
    # Fitting the model triggers the internal PyTorch model creation
    model_1.fit(dummy_ts, verbose=False)
    print(model_1.model)


    # --- Model 2: num_blocks = 2 ---
    print("\n--- Architecture for num_blocks=2 ---")
    model_2 = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,
        generic_architecture=True,
        num_blocks=2,
        num_stacks=2,
        num_layers=2,
        layer_widths=64,
        n_epochs=1,
        random_state=42,
        model_name="nbeats_blocks_2",
    )
    # Fitting the model triggers the internal PyTorch model creation
    model_2.fit(dummy_ts, verbose=False)
    print(model_2.model)

    # --- Comparison ---
    arch1_str = str(model_1.model)
    arch2_str = str(model_2.model)

    print("\n--- Conclusion ---")
    if arch1_str == arch2_str:
        print("Architectures are IDENTICAL. This suggests a potential issue within the Darts library itself.")
    else:
        print("Architectures are DIFFERENT. The Darts library is functioning correctly.")
        print("The root cause is very likely in the views-r2darts2 wrapper code, which is probably not passing the 'num_blocks' parameter correctly to the model.")

if __name__ == "__main__":
    # Set a higher log level to suppress unnecessary Darts/PyTorch Lightning info
    logging.basicConfig(level=logging.ERROR)
    verify_nbeats_architecture()
