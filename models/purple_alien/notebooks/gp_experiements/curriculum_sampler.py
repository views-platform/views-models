import numpy as np
import torch
from torch.utils.data import Sampler


class CurriculumSampler(Sampler):
    """
    A curriculum-aware sampler for PyTorch DataLoader.

    Starts training on high signal-to-noise (SNR) series, and gradually mixes in
    low-SNR series over time, following a sigmoid curriculum schedule.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Assumed to return samples with `signal_strength` metadata per item (either 'high' or 'low').
    curriculum_strength : float
        How strongly to enforce curriculum. 0 = random sampling, 1 = full curriculum.
    max_epochs : int
        Curriculum will gradually flatten by this epoch.
    batch_size : int
        Fixed batch size.
    seed : int, optional
        Reproducibility seed.
    """

    def __init__(
        self,
        dataset,
        curriculum_strength: float = 1.0,
        max_epochs: int = 50,
        batch_size: int = 8,
        seed: int = None
    ):
        self.dataset = dataset
        self.curriculum_strength = curriculum_strength
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.epoch = 0
        self.random_state = np.random.RandomState(seed)

        # Pre-index all series by signal strength
        self.high_indices = [
            i for i in range(len(dataset))
            if dataset.series_groups[i]["signal_strength"].iloc[0] == "high"
        ]
        self.low_indices = [
            i for i in range(len(dataset))
            if dataset.series_groups[i]["signal_strength"].iloc[0] == "low"
        ]

    def set_epoch(self, epoch: int):
        """Update internal state for current epoch."""
        self.epoch = epoch

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _calculate_sampling_weights(self):
        """
        Computes probability of sampling from high/low groups using a
        sigmoid decay curve, modulated by curriculum_strength.
        """
        progress = self.epoch / self.max_epochs
        alpha = 10 * self.curriculum_strength  # Sharpening factor
        high_prob = self._sigmoid((1 - progress) * alpha)
        low_prob = 1 - high_prob
        return high_prob, low_prob

    def __iter__(self):
        high_prob, low_prob = self._calculate_sampling_weights()

        num_batches = len(self) // self.batch_size
        batches = []

        for _ in range(num_batches):
            batch = []
            for _ in range(self.batch_size):
                if self.random_state.rand() < high_prob and self.high_indices:
                    idx = self.random_state.choice(self.high_indices)
                elif self.low_indices:
                    idx = self.random_state.choice(self.low_indices)
                else:  # fallback
                    idx = self.random_state.randint(0, len(self.dataset))
                batch.append(idx)
            batches.append(batch)

        # Flatten batches for DataLoader (which expects sequential indices unless using batch_sampler)
        return iter(batches)

    def __len__(self):
        return len(self.dataset)
