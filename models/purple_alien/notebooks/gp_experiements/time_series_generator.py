import numpy as np
import pandas as pd


class SyntheticTimeSeriesGenerator:
    """
    Generates synthetic time series data with configurable trends and noise.

    This class supports the generation of time series data that can include long-term,
    short-term, and seasonal trends, along with Gaussian noise. It can also simulate
    low-signal scenarios by scaling the signal strength for a subset of the series.

    Attributes
    ----------
    num_series : int
        Number of distinct time series to generate.
    series_length : int
        Number of time steps in each series.
    trend_types : list of str
        Trend components to include in each time series. Options: 'long_term', 'short_term', 'seasonal'.
    noise_level : float, optional
        Standard deviation of the Gaussian noise to add. Default is 0.1.
    min_value : float, optional
        Minimum clipped value for the output. Default is 0.0.
    low_signal_ratio : float, optional
        Proportion (0.0 to 1.0) of time series that should have their signal strength reduced. Default is 0.0.
    low_signal_scale : float, optional
        Scale factor (e.g., 0.2) for low-signal series. Default is 0.2.
    seed : int, optional
        Seed for random number generator to ensure reproducibility. Default is None.
    """

    def __init__(
        self,
        num_series: int,
        series_length: int,
        trend_types: list[str],
        noise_level: float = 0.1,
        min_value: float = 0.0,
        low_signal_ratio: float = 0.0,
        low_signal_scale: float = 0.2,
        seed: int = None
    ):
        self.num_series = num_series
        self.series_length = series_length
        self.trend_types = trend_types
        self.noise_level = noise_level
        self.min_value = min_value
        self.low_signal_ratio = low_signal_ratio
        self.low_signal_scale = low_signal_scale
        self.random_state = np.random.RandomState(seed)

    def _generate_time_vector(self) -> np.ndarray:
        """Generates a normalized time vector from 0 to 1."""
        return np.linspace(0, 1, self.series_length)

    def _generate_long_term_trend(self, t: np.ndarray) -> np.ndarray:
        """Generates a smooth long-term sinusoidal trend with random frequency and phase."""
        freq = self.random_state.uniform(0.1, 0.5)
        phase = self.random_state.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * t + phase)

    def _generate_short_term_trend(self, t: np.ndarray) -> np.ndarray:
        """Generates a smoothed short-term trend from random noise via convolution."""
        noise = self.random_state.normal(0, 1, len(t))
        window = np.hanning(15)  # Smooth window to apply convolution
        window /= window.sum()
        smooth_noise = np.convolve(noise, window, mode='same')
        return smooth_noise

    def _generate_seasonal_trend(self, t: np.ndarray) -> np.ndarray:
        """Generates a periodic seasonal trend with random amplitude and phase."""
        amplitude = self.random_state.uniform(0.5, 1.5)
        frequency = self.random_state.randint(2, 6)  # Number of cycles over the series
        phase = self.random_state.uniform(0, 2 * np.pi)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    def _generate_noise(self) -> np.ndarray:
        """Generates Gaussian noise based on the configured noise level."""
        return self.random_state.normal(0, self.noise_level, self.series_length)

    def generate(self) -> pd.DataFrame:
        """
        Generates the synthetic time series dataset.
    
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the generated series with the following columns:
            - series_id: Unique identifier for each time series
            - timestep: Time index from 0 to series_length - 1
            - value: Final value including all trends and noise
            - long_term, short_term, seasonal, noise: Individual signal components
            - signal_strength: 'high' or 'low' depending on signal suppression
        """
        t = self._generate_time_vector()
        data_records = []
    
        for series_id in range(self.num_series):
            # Conditionally generate each component based on trend_types
            long_term = self._generate_long_term_trend(t) if "long_term" in self.trend_types else np.zeros_like(t)
            short_term = self._generate_short_term_trend(t) if "short_term" in self.trend_types else np.zeros_like(t)
            seasonal = self._generate_seasonal_trend(t) if "seasonal" in self.trend_types else np.zeros_like(t)
    
            signal = long_term + short_term + seasonal
            noise = self._generate_noise()
    
            # Determine signal strength and possibly downscale
            is_low_signal = self.random_state.rand() < self.low_signal_ratio
            signal_strength = "low" if is_low_signal else "high"
    
            if is_low_signal:
                signal *= self.low_signal_scale
    
            # Combine signal and noise, and clip to minimum value
            values = signal + noise
            values = np.clip(values, self.min_value, None)
    
            # Record all components per time step
            for i in range(self.series_length):
                data_records.append({
                    "series_id": series_id,
                    "timestep": i,
                    "value": values[i],
                    "long_term": long_term[i],
                    "short_term": short_term[i],
                    "seasonal": seasonal[i],
                    "noise": noise[i],
                    "signal_strength": signal_strength
                })
    
        return pd.DataFrame(data_records)
    

if __name__ == "__main__":
    # Example usage:
    generator = SyntheticTimeSeriesGenerator(
    num_series=10,
    series_length=100,
    trend_types=["long_term", "short_term", "seasonal"],
    noise_level=0.1,
    min_value=0.0,
    low_signal_ratio=0.2,
    low_signal_scale=0.2,
    seed=42
)

    df = generator.generate()
    print(df.head())