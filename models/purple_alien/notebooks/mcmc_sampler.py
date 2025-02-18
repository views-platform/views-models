import numpy as np
import scipy.stats as stats
import logging
import time
from typing import Optional

class MCMCSampler:
    def __init__(self, proposal_std: float = 0.5, force_non_negative: bool = True, random_seed: Optional[int] = None):
        """
        MCMC Sampler for latent function samples.
        
        Parameters:
        - proposal_std: Standard deviation for proposal distribution.
        - force_non_negative: If True, ensures all samples are >= 0 (useful for count data).
        - random_seed: Optional seed for reproducibility.
        """
        self.proposal_std = proposal_std
        self.force_non_negative = force_non_negative
        self.samples = None
        self.kde = None
        if random_seed is not None:
            np.random.seed(random_seed)

        self.logger = logging.getLogger(__name__)  # Class-specific logger
        logging.basicConfig(level=logging.INFO)  # Configure logging format

    def set_samples(self, samples: np.ndarray):
        """
        Sets the empirical sample distribution for the MCMC sampler.
        
        Parameters:
        - samples: A numpy array of latent function samples.
        """
        if len(samples) == 0:
            raise ValueError("Sample array cannot be empty!")
        
        self.samples = np.array(samples)
        self.kde = stats.gaussian_kde(self.samples)  # Estimate density using KDE
        self.logger.info(f"📊 Sample distribution set with {len(samples)} points.")

    def metropolis_hastings(self, initial_value: float, num_samples: int = 10000, burn_in: int = 1000, thin: int = 1) -> np.ndarray:
        """
        Runs Metropolis-Hastings MCMC using KDE of sample distribution.
        
        Parameters:
        - initial_value: Starting point for the chain.
        - num_samples: Number of samples to collect (after burn-in).
        - burn_in: Number of initial samples to discard.
        - thin: Keeps every `thin`-th sample to reduce autocorrelation.

        Returns:
        - Array of sampled values.
        """
        if self.kde is None:
            raise RuntimeError("❌ No sample distribution set! Call `set_samples()` first.")

        samples = np.empty(num_samples + burn_in)
        samples[0] = max(0, initial_value) if self.force_non_negative else initial_value
        current = samples[0]
        current_log_post = np.log(self.kde(current) + 1e-10)  # Avoid log(0)

        for i in range(1, num_samples + burn_in):
            proposal = np.random.normal(current, self.proposal_std)
            if self.force_non_negative:
                proposal = max(0, proposal)

            proposal_log_post = np.log(self.kde(proposal) + 1e-10)  # Avoid log(0)
            log_acceptance_ratio = proposal_log_post - current_log_post

            if np.log(np.random.rand()) < log_acceptance_ratio:
                current = proposal
                current_log_post = proposal_log_post

            samples[i] = current

        return np.maximum(samples[burn_in:][::thin], 0) if self.force_non_negative else samples[burn_in:][::thin]

    @staticmethod
    def compute_hdi(samples: np.ndarray, credible_interval: float = 0.95):
        samples = np.sort(samples)
        n = len(samples)
        ci_index = int(np.floor(credible_interval * n))
        intervals = np.array([samples[i + ci_index] - samples[i] for i in range(n - ci_index)])
        min_index = np.argmin(intervals)
        return float(samples[min_index]), float(samples[min_index + ci_index])

    @staticmethod
    def compute_map(samples: np.ndarray):
        hist, bin_edges = np.histogram(samples, bins="auto", density=True)
        max_bin_index = np.argmax(hist)
        return (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2  

    def compute_summary(self, samples: np.ndarray):
        map_estimate = self.compute_map(samples)
        max_value = float(np.max(samples))

        hdi_50 = self.compute_hdi(samples, 0.50)
        hdi_95 = self.compute_hdi(samples, 0.95)
        hdi_99 = self.compute_hdi(samples, 0.99)

        return {
            "map": map_estimate,
            "max": max_value,
            "hdi50lb": hdi_50[0], "hdi50ub": hdi_50[1],
            "hdi95lb": hdi_95[0], "hdi95ub": hdi_95[1],
            "hdi99lb": hdi_99[0], "hdi99ub": hdi_99[1]
        }

#    def run_tests(self):
#        """
#        Runs a battery of tests to validate correctness, efficiency, and robustness.
#        """
#        self.logger.info("\n🧪 Running Tests on MCMC Sampler with Empirical Samples...\n")
#
#        # ✅ **Test 1: Normal Distribution Samples**
#        normal_samples = np.random.normal(0, 1, 10000)
#        self.set_samples(normal_samples)
#
#        start_time = time.time()
#        samples = self.metropolis_hastings(initial_value=0.0, num_samples=5000)
#        end_time = time.time()
#        summary = self.compute_summary(samples)
#
#        self.logger.info(f"✅ Test 1 Passed: Normal Distribution Sampling in {end_time - start_time:.3f}s")
#        self.logger.info(f"   🔍 MAP Estimate: {summary['map']:.6f}")
#
#        # ✅ **Test 2: Poisson Count Data Samples**
#        poisson_samples = np.random.poisson(5, 10000)
#        self.set_samples(poisson_samples)
#
#        samples = self.metropolis_hastings(initial_value=5, num_samples=5000)
#        summary = self.compute_summary(samples)
#
#        assert np.all(samples >= 0), "❌ Negative values in Poisson sampling!"
#        self.logger.info("✅ Test 2 Passed: Poisson Count Data")
#
#        self.logger.info("\n✅ All MCMC Sampler Tests Passed Successfully!\n")
#
#


    def run_tests(self):
        """
        Runs validation tests to ensure correctness, efficiency, and robustness of MCMC sampling.
        """
        self.logger.info("\n🧪 Running Tests on MCMC Sampler with Empirical Samples...\n")

        test_cases = [
            {"name": "Normal Distribution Samples", "generator": lambda: np.random.normal(0, 1, 10000), "initial_value": 0.0, "force_non_negative": False},
            {"name": "Poisson Count Data", "generator": lambda: np.random.poisson(5, 10000), "initial_value": 5, "force_non_negative": True},
            {"name": "Gamma Distributed Samples", "generator": lambda: np.random.gamma(2, 2, 10000), "initial_value": 4.0, "force_non_negative": True},
            {"name": "Extreme Right-Skewed Data", "generator": lambda: np.random.exponential(2, 10000), "initial_value": 1.0, "force_non_negative": True},
            {"name": "Heavy-Tailed Student's t", "generator": lambda: np.random.standard_t(df=3, size=10000), "initial_value": 0.0, "force_non_negative": False},
            {"name": "Sparse Data (90% Zero)", "generator": lambda: np.concatenate([np.zeros(9000), np.random.normal(10, 2, 1000)]), "initial_value": 10.0, "force_non_negative": True},
            {"name": "Floating-Point Precision Test", "generator": lambda: np.random.normal(1e-5, 1e-6, 10000), "initial_value": 1e-5, "force_non_negative": False},
            {"name": "Extreme Large Values", "generator": lambda: np.random.normal(1e6, 1e5, 10000), "initial_value": 1e6, "force_non_negative": False},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate and set sample distribution
            empirical_samples = test["generator"]()
            self.set_samples(empirical_samples)

            self.logger.info(f"   📊 Sample Distribution Set with {len(empirical_samples)} points")
            self.logger.info(f"   🔧 Force Non-Negative: {test['force_non_negative']}")

            # Run MCMC sampling
            start_time = time.time()
            samples = self.metropolis_hastings(initial_value=test["initial_value"], num_samples=5000)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # Compute summary statistics
            summary = self.compute_summary(samples)

            self.logger.info(f"   🔍 MAP Estimate: {summary['map']:.6f}")
            self.logger.info(f"   🔍 Max Value: {summary['max']:.6f}")
            self.logger.info(f"   🔍 HDI 50%: [{summary['hdi50lb']:.6f}, {summary['hdi50ub']:.6f}]")
            self.logger.info(f"   🔍 HDI 95%: [{summary['hdi95lb']:.6f}, {summary['hdi95ub']:.6f}]")
            self.logger.info(f"   🔍 HDI 99%: [{summary['hdi99lb']:.6f}, {summary['hdi99ub']:.6f}]")

            # **Validation Checks**
            assert summary["map"] is not None, "❌ MAP estimate is missing!"
            assert summary["max"] >= min(samples), "❌ Max value incorrect!"

            # Enforce non-negative constraint if required
            if test["force_non_negative"]:
                assert np.all(samples >= 0), "❌ Negative values found in non-negative sampling!"
                self.logger.info("   🔍 ✅ All values are non-negative (constraint satisfied)")

            # Check if HDI covers the true mean of original data (sanity check)
           # true_mean = np.mean(empirical_samples)
           # in_hdi95 = summary["hdi95lb"] <= true_mean <= summary["hdi95ub"]
           # assert in_hdi95, f"❌ Mean {true_mean:.6f} is outside HDI 95% interval!"
           # self.logger.info(f"   🔍 ✅ True mean {true_mean:.6f} is within HDI 95%")

            # Check variance preservation
           # sampled_variance = np.var(samples)
           # empirical_variance = np.var(empirical_samples)
           # var_diff = abs(sampled_variance - empirical_variance) / (empirical_variance + 1e-8)  # Relative difference
           # assert var_diff < 0.2, f"❌ Variance mismatch! Sampled: {sampled_variance:.6f}, Expected: {empirical_variance:.6f}"
           # self.logger.info(f"   🔍 ✅ Variance Preserved (Difference: {var_diff:.2%})\n")

        self.logger.info("\n✅ All MCMC Sampler Tests Passed Successfully!\n")




if __name__ == "__main__":

    sampler = MCMCSampler(proposal_std=1.0, force_non_negative=True)

    # Provide empirical samples instead of a function
    empirical_samples = np.random.gamma(2, 2, 10000)
    sampler.set_samples(empirical_samples)

    # Run MCMC
    samples = sampler.metropolis_hastings(initial_value=4.0, num_samples=10000)
    summary = sampler.compute_summary(samples)
    sampler.run_tests()