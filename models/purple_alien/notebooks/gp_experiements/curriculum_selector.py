import pandas as pd
import torch
import matplotlib.pyplot as plt


class CurriculumSelector:
    def __init__(
        self,
        df: pd.DataFrame,
        window_length: int = 30,
        stride: int = 10,
        signal_metric: str = "var",  # or "std", "mean"
        top_k: int = 50
    ):
        self.df = df.copy()
        self.window_length = window_length
        self.stride = stride
        self.signal_metric = signal_metric
        self.top_k = top_k

    def _generate_windows(self):
        windows = []

        for series_id in self.df["series_id"].unique():
            ts_df = self.df[self.df["series_id"] == series_id].sort_values("timestep")
            values = ts_df["value"].values
            timesteps = ts_df["timestep"].values

            for start in range(0, len(values) - self.window_length + 1, self.stride):
                end = start + self.window_length
                slice_values = values[start:end]
                slice_timesteps = timesteps[start:end]

                window = {
                    "series_id": series_id,
                    "start": start,
                    "end": end,
                    "timestep": slice_timesteps,
                    "value": slice_values,
                    "var": float(pd.Series(slice_values).var()),
                    "std": float(pd.Series(slice_values).std()),
                    "mean": float(pd.Series(slice_values).mean())
                }
                windows.append(window)

        return windows

    def select_top_windows(self):
        all_windows = self._generate_windows()
        ranked = sorted(all_windows, key=lambda w: -w[self.signal_metric])
        return ranked[:self.top_k]

    def to_tensor_dataset(self):
        top_slices = self.select_top_windows()

        x_list, y_list = [], []
        for w in top_slices:
            x = torch.tensor(w["timestep"], dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(w["value"], dtype=torch.float32)
            x_list.append(x)
            y_list.append(y)

        return x_list, y_list

    def plot_sample_slices(self, num_samples: int = 9):
        """
        Plot a 3x3 grid of the top signal slices.
        """
        slices = self.select_top_windows()[:num_samples]
        num_rows = num_cols = int(num_samples**0.5)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        axs = axs.flatten()

        for i, (w, ax) in enumerate(zip(slices, axs)):
            ax.plot(w["timestep"], w["value"], color="black")
            ax.set_title(f"Series {w['series_id']} | {self.signal_metric}={w[self.signal_metric]:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.suptitle("Top Signal-Rich Windows", fontsize=14, y=1.02)
        plt.show()

