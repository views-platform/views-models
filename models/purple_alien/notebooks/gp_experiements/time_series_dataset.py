import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from time_series_generator import SyntheticTimeSeriesGenerator


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for loading time series from a DataFrame.

    Supports optional slicing of each time series into fixed-length windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ['series_id', 'timestep', 'value']
    window_length : int or None
        If provided, slice each series into windows of this length.
        If None, return the full time series per sample.
    """

    def __init__(self, df: pd.DataFrame, window_length: int = None):
        self.df = df.copy()
        self.window_length = window_length

        # Group into series
        self.series_groups = list(df.groupby("series_id"))

        # Precompute sliced windows
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []

        for series_id, group in self.series_groups:
            x = torch.tensor(group["timestep"].values, dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(group["value"].values, dtype=torch.float32)

            if self.window_length is None:
                samples.append((x, y))
            else:
                T = len(x)
                for start in range(0, T - self.window_length + 1):
                    x_slice = x[start:start + self.window_length]
                    y_slice = y[start:start + self.window_length]
                    samples.append((x_slice, y_slice))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]




















#
#
#class TimeSeriesDataset(Dataset):
#    """
#    A PyTorch-compatible dataset for time series stored in long-form DataFrames.
#    
#    Each item in the dataset corresponds to a full time series (based on 'series_id').
#    Useful for models like VariationalGP and ApproximateGP that benefit from mini-batching
#    multiple independent time series.
#
#    Parameters
#    ----------
#    df : pd.DataFrame
#        DataFrame containing columns: ['series_id', 'timestep', 'value']
#        (optional trend components may exist but are not returned by default).
#    return_components : bool
#        If True, also returns a dictionary of decomposed components per series.
#    """
#
#    def __init__(self, df: pd.DataFrame, return_components: bool = False):
#        self.df = df
#        self.return_components = return_components
#
#        # Group by series_id, pre-aggregate for faster indexing
#        self.series_groups = [g for _, g in df.groupby("series_id")]
#        self.series_ids = df["series_id"].unique()
#
#    def __len__(self):
#        return len(self.series_groups)
#
#    def __getitem__(self, idx):
#        group = self.series_groups[idx]
#        x = torch.tensor(group["timestep"].values, dtype=torch.float32).unsqueeze(-1)  # Shape: [T, 1]
#        y = torch.tensor(group["value"].values, dtype=torch.float32)  # Shape: [T]
#
#        if self.return_components:
#            components = {
#                "long_term": torch.tensor(group["long_term"].values, dtype=torch.float32),
#                "short_term": torch.tensor(group["short_term"].values, dtype=torch.float32),
#                "seasonal": torch.tensor(group["seasonal"].values, dtype=torch.float32),
#                "noise": torch.tensor(group["noise"].values, dtype=torch.float32),
#                "signal_strength": group["signal_strength"].iloc[0],  # str label
#            }
#            return x, y, components
#
#        return x, y
#
#    def split(self, train_ratio=0.7, val_ratio=0.15):
#        """
#        Placeholder for dataset splitting logic.
#        Returns (train_dataset, val_dataset, test_dataset) with the same structure.
#        """
#        total = len(self)
#        train_end = int(total * train_ratio)
#        val_end = int(total * (train_ratio + val_ratio))
#
#        train_ds = torch.utils.data.Subset(self, range(0, train_end))
#        val_ds = torch.utils.data.Subset(self, range(train_end, val_end))
#        test_ds = torch.utils.data.Subset(self, range(val_end, total))
#        return train_ds, val_ds, test_ds
#
#
#def __main__():
#    # Example usage:
#    generator = SyntheticTimeSeriesGenerator(
#    num_series=10,
#    series_length=100,
#    trend_types=["long_term", "short_term", "seasonal"],
#    noise_level=0.1,
#    min_value=0.0,
#    low_signal_ratio=0.2,
#    low_signal_scale=0.02,
#    seed=42
#    )
#
#    # Generate data
#    df = generator.generate()
#
#    # Create dataset (return trend components later if needed)
#    dataset = TimeSeriesDataset(df)
#
#    # Mini-batch 4 time series at a time
#    loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#    # Example training loop
#    for batch_x, batch_y in loader:
#        # batch_x: [batch_size, T, 1]
#        # batch_y: [batch_size, T]
#        pass
#