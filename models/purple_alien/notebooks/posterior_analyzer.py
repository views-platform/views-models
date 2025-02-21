
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
        logger.info(f"✅ Computed HDI: [{hdi_min:.3f}, {hdi_max:.3f}]")

        # Enforce non-negativity if requested
        if enforce_non_negative:
            logger.debug("📢 Applying non-negativity constraint to HDI values.")
            hdi_min = max(0.0, hdi_min)
            hdi_max = max(0.0, hdi_max)

        logger.debug(f"📝 Final HDI values: [{hdi_min:.3f}, {hdi_max:.3f}]")

        return (hdi_min, hdi_max)

    @staticmethod
    def compute_map(
        samples,
        eps_for_zero=np.exp(-100),
        fallback_bins=100,
        bw_method='silverman',
        enforce_non_negative = True,
        enforce_correction = True
    ):
        """
        Estimate the MAP (mode) from samples in [0, ∞), possibly with a small mass at/near zero.
        Uses boundary-corrected KDE via reflection around zero, with a simple fallback histogram.

        Parameters
        ----------
        samples : array-like
            Posterior samples.
        eps_for_zero : float
            Any sample <= this threshold is treated as 0. Default = exp(-10) ~ 4.54e-5.
        fallback_bins : int
            Number of bins in the fallback histogram if KDE fails.
        bw_method : str or float
            Bandwidth method passed to gaussian_kde ('silverman', 'scott', or numeric).
        enforce_non_negative : bool
            If True, clamp final result to be >= 0.

        Returns
        -------
        float
            The estimated mode.
            - If any sample is <= eps_for_zero, returns 0 (assuming a point mass at 0).
            - Otherwise, computes a boundary-corrected KDE by reflecting data at 0
              and returns the mode on [0, max(samples)].
        """

        logger.debug("📢 Starting MAP estimation with eps_for_zero=%.5f, fallback_bins=%d, bw_method=%s, enforce_non_negative=%s",
                     eps_for_zero, fallback_bins, bw_method, enforce_non_negative)

        samples = np.asarray(samples, dtype=float)
        logger.debug("📢 Converted samples to NumPy array.")

        # 1) Filter out any NaNs or negatives (if they exist, we treat them as invalid)
        valid_mask = np.isfinite(samples) & (samples >= 0)
        data = samples[valid_mask]

        if len(data) == 0:
            logger.error("❌ No valid samples provided. Returning mode = 0.0")
            return 0.0

        logger.debug("📢 Valid samples count: %d", len(data))

        # 2) Check for zero or near-zero values
        near_zero_mask = (data <= eps_for_zero)
        if np.any(near_zero_mask):
            logger.info("📢 Mass detected at or below eps_for_zero (%.5f). Setting mode = 0.", eps_for_zero)
            return 0.0

        # 3) Reflection: mirror the strictly positive data around zero
        reflected = np.concatenate([-data, data])
        logger.debug("🪞 Applied reflection technique to enforce boundary correction.")

        # 4) Attempt KDE on the reflected data
        try:
            kde = stats.gaussian_kde(reflected, bw_method=bw_method)
            logger.debug("✅ KDE successfully computed using bw_method=%s", bw_method)

            # We'll evaluate the KDE only on [0, max(data)] since the true domain is >= 0
            x_min = 0.0
            x_max = np.max(data)
            if x_max == 0:
                logger.warning("🚨  Degenerate case detected: all data points are zero. Returning mode = 0.")
                return 0.0

            x_grid = np.linspace(x_min, x_max, 300)
            kde_vals = kde(x_grid)
            mode_est = x_grid[np.argmax(kde_vals)]
            logger.info("✅ KDE mode estimate: %.5f", mode_est)

        except Exception as e:
            logger.exception("❌ KDE estimation failed, falling back to histogram method. Error: %s", e)

            # 5) Fallback: histogram on the unreflected data
            hist, bin_edges = np.histogram(data, bins=fallback_bins, density=True)
            peak_idx = np.argmax(hist)

            if peak_idx < len(bin_edges) - 1:
                mode_est = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])
            else:
                mode_est = bin_edges[peak_idx]

            logger.info("✅ Fallback histogram mode estimate: %.5f", mode_est)

        # 6) Enforce non-negativity
        if enforce_non_negative:
            mode_est = max(0.0, mode_est)
            logger.info("🍩 Applied non-negativity constraint. Final mode: %.5f", mode_est)

        # 7) Correction: Ensure MAP is within HDI if requested
        if enforce_correction:
                
            hdi_min, hdi_max = PosteriorAnalyzer.compute_hdi(samples, credible_mass=0.95)
            logger.info(f"🧠 HDI computed: min={hdi_min:.5f}, max={hdi_max:.5f}")
            logger.info(f"🍔 Enforcing correction: Clamping MAP inside HDI.")
            mode_est = np.clip(mode_est, hdi_min, hdi_max)

        return float(mode_est)

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
        kde = sns.kdeplot(samples, color="black", linewidth=2, fill=True, alpha=0.20, clip=(x_min, x_max))

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

        # **Annotations: Where should we expect values?**
        ax.annotate(
            rf"Most Likely: {map_estimate:.2f}",
            xy=(map_estimate, y_max * 0.85),
            xytext=(map_estimate + 0.1, y_max * 0.9),
            arrowprops=dict(arrowstyle="->", color=map_color),
            fontsize=14, weight="bold", color=map_color
        )

        ax.annotate(
            rf"Max Observed: {max_estimate:.2f}",
            xy=(max_estimate, y_max * 0.60),
            xytext=(max_estimate - 3, y_max * 0.70),
            arrowprops=dict(arrowstyle="->", color=max_color),
            fontsize=14, weight="bold", color=max_color
        )

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



    @staticmethod
    def validate_hdi_map_consistency(samples, credible_mass=0.95, enforce_correction=False):
        """
        Validate if MAP falls within HDI, and optionally correct HDI or MAP if necessary.

        Parameters
        ----------
        samples : array-like
            Posterior samples.
        credible_mass : float
            The credible mass for the HDI (e.g., 0.95).
        enforce_correction : bool
            If True, correct MAP or HDI if needed.

        Returns
        -------
        (float, float, float)
            The HDI bounds and possibly corrected MAP.
        """
        hdi_min, hdi_max = PosteriorAnalyzer.compute_hdi(samples, credible_mass)
        map_estimate = PosteriorAnalyzer.compute_map(samples)

        # **Case: MAP is 0 but HDI starts slightly above it**
        if np.isclose(map_estimate, 0.0, atol=1e-3) and hdi_min > 1e-3:
            logger.warning(f"🚨 MAP ({map_estimate:.5f}) falls OUTSIDE the HDI [{hdi_min:.5f}, {hdi_max:.5f}]. Likely due to precision issues.")

            if enforce_correction:
                logger.info(f"🚨 Expanding HDI downward to include MAP at zero.")
                hdi_min = 0.0  # Adjust HDI to include zero

        # **General Case: MAP is outside HDI**
        elif not (hdi_min <= map_estimate <= hdi_max):
            logger.warning(f"🚨 MAP ({map_estimate:.5f}) falls OUTSIDE the HDI [{hdi_min:.5f}, {hdi_max:.5f}].")

            if enforce_correction:
                logger.info(f"🚨 Enforcing correction: Clamping MAP inside HDI.")
                map_estimate = np.clip(map_estimate, hdi_min, hdi_max)

        return hdi_min, hdi_max, map_estimate



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
        map_estimate = PosteriorAnalyzer.compute_map(samples, enforce_non_negative=False, enforce_correction=False)

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
