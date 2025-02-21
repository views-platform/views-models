
import numpy as np
import scipy.stats as stats
import pytest
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PosteriorAnalyzer:
    """A class to analyze posterior distributions with HDI and MAP estimation."""

    def __init__(self):
        
        pass

    @staticmethod
    def compute_hdi(samples, credible_mass=0.95, enforce_non_negative=True):
        """
        Compute the Highest Density Interval (HDI) for a given set of samples 
        (1D). This returns the single contiguous interval of minimal width 
        that contains `credible_mass` fraction of the samples.

        NOTE: For multi-modal distributions, the true HDI could be 
        a union of intervals. This function returns only one interval 
        spanning that fraction of points.

        Parameters
        ----------
        samples : array-like
            Samples from a probability distribution.
        credible_mass : float
            The credible mass for the HDI (e.g., 0.95 for 95% HDI).

        Returns
        -------
        (float, float)
            Lower and upper bounds of the (single) HDI interval.
        """
        logger.debug("😎 Starting HDI computation with credible_mass=%.3f and enforce_non_negative=%s", credible_mass, enforce_non_negative)

        samples = np.asarray(samples)
        samples = samples[np.isfinite(samples)]  # Drop NaNs or inf
        n_samples = len(samples)

        if n_samples == 0:
            logger.error("❌ No valid samples provided. Cannot compute HDI.")
            raise ValueError("No valid samples provided.")

        if n_samples == 1:
            logger.warning(f"📢 Only one sample provided: {samples[0]}. HDI is trivially that point.")
            return samples[0], samples[0]

        samples.sort()
        logger.debug("Sorted samples for HDI calculation.")

        if credible_mass <= 0.0 or credible_mass > 1.0:
            logger.error(f"❌ Invalid credible_mass value: {credible_mass}. Must be in (0, 1].")
            raise ValueError("credible_mass must be in (0, 1].")

        # If credible_mass == 1.0, the HDI is the entire range
        if np.isclose(credible_mass, 1.0):
            logger.info("📢 Credible mass is 1.0. Returning full range as HDI.")
            return samples[0], samples[-1]

        # For typical case: 0 < credible_mass < 1
        interval_idx_inc = int(np.floor(credible_mass * n_samples))
        interval_idx_inc = min(interval_idx_inc, n_samples - 1)  # Ensure at least one sample in the interval

        n_intervals = n_samples - interval_idx_inc
        if n_intervals < 1:
            logger.warning("📢 Unable to compute HDI with given credible_mass. Returning full sample range.")
            return samples[0], samples[-1]

        logger.debug("🧠 Computing sliding window for HDI estimation...")

        # Slide window over sorted data
        intervals = []
        for i in range(n_intervals):
            low = samples[i]
            high = samples[i + interval_idx_inc]
            intervals.append((low, high))

        # Pick the narrowest
        hdi_min, hdi_max = min(intervals, key=lambda x: x[1] - x[0])
        logger.debug(f"✅ Computed HDI ({credible_mass*100}%): [{hdi_min:.3f}, {hdi_max:.3f}]")

        # Enforce non-negativity if requested
        if enforce_non_negative == True:
            logger.info("📢 Applying non-negativity constraint to HDI values.")
            hdi_min = max(0.0, hdi_min)
            hdi_max = max(0.0, hdi_max)

            logger.debug(f"📝 Final HDI ({credible_mass*100}%) values: [{hdi_min:.3f}, {hdi_max:.3f}]")

        return (hdi_min, hdi_max)


    @staticmethod
    def compute_map(samples, enforce_non_negative=False):
        """
        Compute the Maximum A Posteriori (MAP) estimate using an HDI-based histogram and KDE refinement.

        Parameters:
        ----------
        samples : array-like
            Posterior samples.
        enforce_non_negative : bool
            If True, forces MAP estimate to be non-negative.

        Returns:
        -------
        float
            The estimated MAP.
        """
        samples = np.asarray(samples)

        if len(samples) == 0:
            logger.error("❌ No valid samples. Returning MAP = 0.0")
            return 0.0
        
        # **Compute HDI**
        credible_mass = 0.10 if len(samples) > 5000 else 0.25
        hdi_min, hdi_max = PosteriorAnalyzer.compute_hdi(samples, credible_mass=credible_mass, enforce_non_negative=False)

        # **Debugging: Ensure HDI bounds are reasonable**
        logger.debug(f"📢 HDI Computed: [{hdi_min:.5f}, {hdi_max:.5f}]")

        # **Ensure HDI is valid**
        if hdi_min == hdi_max:
            logger.info(f"✅ HDI contains only one value ({hdi_min}). Setting MAP = {hdi_min}")
            return float(hdi_min)

        # **Select Only the HDI Region**
        subset = samples[(samples >= hdi_min) & (samples <= hdi_max)]
        logger.debug(f"📢 Subset selected for MAP: min={np.min(subset):.5f}, max={np.max(subset):.5f}")

        if len(subset) == 0:
            logger.error(f"❌ No valid samples inside HDI range! Returning hdi_min = {hdi_min:.5f}")
            return float(hdi_min)
        

        # **Adaptive Histogram Binning**
        iqr_value = stats.iqr(subset)
        bin_width = 2 * iqr_value / (len(subset) ** (1/3))
        num_bins = max(20 if stats.skew(subset) > 5 else 10, int((subset.max() - subset.min()) / bin_width))
        hist, bin_edges = np.histogram(subset, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # **Find Histogram Mode**
        hist_mode = bin_centers[np.argmax(hist)]
        logger.debug(f"📢 Histogram Mode Estimate: {hist_mode:.5f}")

        # **Ensure MAP is inside HDI**
        if not (hdi_min <= hist_mode <= hdi_max):
            logger.error(f"❌ MAP estimate {hist_mode:.5f} is OUTSIDE HDI range [{hdi_min:.5f}, {hdi_max:.5f}]! Clamping.")
            hist_mode = np.clip(hist_mode, hdi_min, hdi_max)
            logger.info(f"📢 MAP estimate clamped to {hist_mode:.5f}")

        # **Enforce Non-Negativity if Requested**
        if enforce_non_negative and hist_mode < 0:
            logger.warning(f"📢 Negative MAP estimate detected ({hist_mode:.5f}). Setting to 0.")
            hist_mode = max(0, hist_mode)

        return float(hist_mode)



    @staticmethod
    def plot_posterior_distribution(samples, hdi_levels=[50, 95, 99], grid=True):
        """
        Modern posterior distribution plot with:
        - MAP (most likely value)
        - Customizable HDI levels (50%, 95%, 99% or user-defined)
        - Max observed value
        - Full shading to the top of the y-axis
        - Legend placed outside the plot
        - Grid as a toggle option

        Parameters
        ----------
        samples : array-like
            The posterior samples.
        hdi_levels : list, optional
            The HDI percentages to plot (e.g., [50, 95, 99]). Defaults to [50, 95, 99].
        grid : bool, optional
            If True, shows grid. Defaults to True.
        """

        # Compute MAP, HDIs, and Max
        map_estimate = PosteriorAnalyzer.compute_map(samples)
        hdi_values = [PosteriorAnalyzer.compute_hdi(samples, hdi / 100) for hdi in hdi_levels]
        max_estimate = np.max(samples)

        # Modern color scheme (soft, friendly, readable)
        map_color = "#E74C3C"  # Warm red
        max_color = "#3498DB"  # Cool blue
        hdi_colors = sns.color_palette("husl", len(hdi_levels))  # Dynamically generate distinct colors

        # Trim X-axis range to avoid irrelevant empty space
        x_min = max(0, np.min(samples))
        x_max = np.max(samples) * 1.05  # Slight padding for readability

        # Modern seaborn styling
        sns.set_context("notebook", font_scale=1.5)
        sns.set_style("whitegrid" if grid else "white")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # **Density Plot (KDE)**

        # hitogram
        sns.histplot(samples, bins=50, color="black", kde=True, ax=ax)

        # Get the Y-limit after plotting KDE to ensure HDI shading fills correctly
        y_max = ax.get_ylim()[1]

        # **MAP Line (Most Likely Value)**
        ax.axvline(map_estimate, color=map_color, linestyle="-", linewidth=3, label=f"MAP: {map_estimate:.2f}")

        # **Max Value Line (Worst-case Observation)**
        ax.axvline(max_estimate, color=max_color, linestyle="--", linewidth=2, label=f"Max: {max_estimate:.2f}")

        # **HDI Regions (Full Height to Y-Axis)**
        for ci, (low, high), color in zip(hdi_levels, hdi_values, hdi_colors):
            ax.fill_betweenx(
                y=[0, y_max],  # Now fully fills up to max Y
                x1=low, x2=high,
                color=color, alpha=0.25, label=f"HDI {ci}%: [{low:.2f}, {high:.2f}]"
            )
            ax.axvline(low, color=color, linestyle=":", linewidth=1.5, alpha=0.9)
            ax.axvline(high, color=color, linestyle=":", linewidth=1.5, alpha=0.9)

        # **Set X-axis to meaningful range only**
        ax.set_xlim(x_min, x_max)

        # **Set Y-axis to full range for HDI shading**
        ax.set_ylim(0, y_max)

        # **Axis Labels**
        ax.set_xlabel("Value", fontsize=14, weight="bold")
        ax.set_ylabel("Density", fontsize=14, weight="bold")

        # **Legend Outside Plot**
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12)

        # **Title**
        plt.title("Posterior Distribution: Most Likely & Uncertainty Ranges", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.show()



def test_hdi_function():
    """Test HDI computation on various extreme distributions."""
    np.random.seed(42)  # For reproducibility

    test_cases = {
        "Normal": stats.norm.rvs(loc=5, scale=2, size=10000),
        "Half-Normal": stats.halfnorm.rvs(loc=0, scale=2, size=10000),
        "Cauchy": stats.cauchy.rvs(loc=0, scale=1, size=10000),
        "Laplace": stats.laplace.rvs(loc=0, scale=1, size=10000),
        "Power-Law": np.random.pareto(a=3, size=10000) + 1,
        "Bimodal": np.concatenate([
            stats.norm.rvs(loc=-3, scale=1, size=5000), 
            stats.norm.rvs(loc=3, scale=1, size=5000)
        ]),

        # **🚀 EXTREME DISTRIBUTIONS**
        "Student-t (df=1, Cauchy-like)": stats.t.rvs(df=1, loc=0, scale=1, size=10000),
        "Beta(0.5, 0.5) (U-shaped)": stats.beta.rvs(0.5, 0.5, size=10000),
        "Skewed Normal (alpha=10)": stats.skewnorm.rvs(a=10, loc=0, scale=2, size=10000),
        "Triangular (mode=2)": stats.triang.rvs(c=0.5, loc=0, scale=4, size=10000),
        "Trimodal": np.concatenate([
            stats.norm.rvs(loc=-5, scale=1, size=3000),
            stats.norm.rvs(loc=0, scale=1, size=4000),
            stats.norm.rvs(loc=5, scale=1, size=3000)
        ]),
        "Extreme-Value (Gumbel)": stats.gumbel_r.rvs(loc=0, scale=2, size=10000),
    }

    credible_mass = 0.95
    failed_tests = []  # Track failed cases

    for name, samples in test_cases.items():
        hdi_min, hdi_max = PosteriorAnalyzer.compute_hdi(samples, credible_mass, enforce_non_negative=False)
        in_hdi = (samples >= hdi_min) & (samples <= hdi_max)
        coverage = np.mean(in_hdi)

        if np.isclose(coverage, credible_mass, atol=0.01):
            logger.info(f"✅ HDI test passed for {name}: Interval = [{hdi_min:.3f}, {hdi_max:.3f}], Coverage = {coverage:.3f}")
        else:
            logger.error(f"❌ HDI test failed for {name}: Expected {credible_mass:.3f}, got {coverage:.3f}")
            failed_tests.append(name)

    # Final failure summary
    if failed_tests:
        logger.error(f"❌ HDI tests failed for: {', '.join(failed_tests)}")
    else:
        logger.info("✅ All HDI tests passed successfully!")


def test_map_function():
    """Test MAP computation on various extreme distributions."""
    np.random.seed(42)
    
    test_cases = {
        "Normal": (stats.norm.rvs(loc=5, scale=2, size=10000), 5),
        "Half-Normal": (stats.halfnorm.rvs(loc=0, scale=2, size=10000), 0),
        "Cauchy": (stats.cauchy.rvs(loc=0, scale=1, size=10000), 0),
        "Laplace": (stats.laplace.rvs(loc=0, scale=1, size=10000), 0),
        "Power-Law": (np.random.pareto(a=3, size=10000) + 1, 1),
        "Bimodal": (np.concatenate([
            stats.norm.rvs(loc=-3, scale=1, size=5000), 
            stats.norm.rvs(loc=3, scale=1, size=5000)
        ]), None),  # No single mode

        # **🚀 EXTREME DISTRIBUTIONS**
        "Student-t (df=1, Cauchy-like)": (stats.t.rvs(df=1, loc=0, scale=1, size=10000), 0),
        "Beta(0.5, 0.5) (U-shaped)": (stats.beta.rvs(0.5, 0.5, size=10000), None),  # No single mode
        "Skewed Normal (alpha=10)": (stats.skewnorm.rvs(a=10, loc=0, scale=2, size=10000), 0),
        "Triangular (mode=2)": (stats.triang.rvs(c=0.5, loc=0, scale=4, size=10000), 2),
        "Trimodal": (np.concatenate([
            stats.norm.rvs(loc=-5, scale=1, size=3000),
            stats.norm.rvs(loc=0, scale=1, size=4000),
            stats.norm.rvs(loc=5, scale=1, size=3000)
        ]), None),  # No single mode
        "Extreme-Value (Gumbel)": (stats.gumbel_r.rvs(loc=0, scale=2, size=10000), 0),
    }

    failed_tests = []  # Track failures

    for name, (samples, true_mode) in test_cases.items():
        map_estimate = PosteriorAnalyzer.compute_map(samples, enforce_non_negative=False)

        if true_mode is not None:
            if np.isclose(map_estimate, true_mode, atol=0.5):
                logger.info(f"✅ MAP test passed for {name}: Expected mode ≈ {true_mode:.3f}, got estimated mode = {map_estimate:.3f}")
            else:
                logger.error(f"❌ MAP test failed for {name}: Expected mode ≈ {true_mode:.3f}, but got {map_estimate:.3f}")
                failed_tests.append(name)
        else:
            logger.info(f"✅ MAP test passed for {name}: Estimated mode = {map_estimate:.3f} (expected multimodal)")

    # Final failure summary
    if failed_tests:
        logger.error(f"❌ MAP tests failed for: {', '.join(failed_tests)}")
    else:
        logger.info("✅ All MAP tests passed successfully!")

if __name__ == "__main__":
    test_hdi_function()
    test_map_function()
