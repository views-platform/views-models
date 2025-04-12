import torch

class InducingPointSampler:
    def __init__(self, dataset, sampler=None, strategy: str = "high_signal_only"):
        """
        Parameters
        ----------
        dataset : TimeSeriesDataset
            Dataset with access to x-values per series.
        sampler : CurriculumSampler, optional
            Sampler used to choose which series to sample from (if using curriculum).
        strategy : str
            One of ['random', 'high_signal_only', 'grid'].
        """
        self.dataset = dataset
        self.sampler = sampler
        self.strategy = strategy

    def sample(self, num_points: int) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            Inducing point locations (shape: [num_points, 1])
        """
        xs = []

        if self.strategy == "random":
            for i in range(len(self.dataset)):
                x, _ = self.dataset[i]
                xs.append(x)

        elif self.strategy == "high_signal_only":
            for i in range(len(self.dataset)):
                group = self.dataset.series_groups[i]
                if group["signal_strength"].iloc[0] == "high":
                    x = torch.tensor(group["timestep"].values, dtype=torch.float32).unsqueeze(-1)
                    xs.append(x)

        elif self.strategy == "grid":
            # Use a fixed evenly spaced grid over time domain
            t_min = self.dataset.df["timestep"].min()
            t_max = self.dataset.df["timestep"].max()
            return torch.linspace(t_min, t_max, num_points).unsqueeze(-1)

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        # Stack and flatten all x's, then sample from them
        x_all = torch.cat(xs, dim=0)
        indices = torch.randperm(len(x_all))[:num_points]
        return x_all[indices]
