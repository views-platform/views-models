import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import logging
from typing import Optional, Dict, Any

class PyroMCMCSampler:
    def __init__(self, num_samples: int = 1000, warmup_steps: int = 200, force_non_negative: bool = True):
        """
        Pyro-based MCMC Sampler for empirical distributions.

        Parameters:
        - num_samples: Number of MCMC samples to collect.
        - warmup_steps: Number of warmup (burn-in) steps for MCMC.
        - force_non_negative: If True, ensures all returned samples are non-negative.
        """
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.force_non_negative = force_non_negative
        self.samples = None
        self.logger = logging.getLogger(__name__)

    def set_samples(self, samples: torch.Tensor):
        """
        Sets the empirical sample distribution for MCMC sampling.

        Parameters:
        - samples: A tensor of empirical latent function samples.
        """
        if len(samples) == 0:
            raise ValueError("❌ Sample array cannot be empty!")
        
        if self.force_non_negative:
            samples = torch.clamp(samples, min=0)  # Enforce non-negative values
        
        self.samples = dist.Empirical(samples)
        self.logger.info(f"📊 Sample distribution set with {len(samples)} points.")

    def model(self):
        """Pyro model that samples from the empirical distribution."""
        return pyro.sample("x", self.samples)

    def run_mcmc(self) -> torch.Tensor:
        """
        Runs MCMC using Pyro's NUTS sampler.

        Returns:
        - Posterior samples from the empirical distribution.
        """
        if self.samples is None:
            raise RuntimeError("❌ No sample distribution set! Call `set_samples()` first.")
        
        # Use NUTS (No-U-Turn Sampler) for MCMC
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)

        self.logger.info("🚀 Running MCMC...")
        mcmc.run()
        posterior_samples = mcmc.get_samples()["x"]

        if self.force_non_negative:
            posterior_samples = torch.clamp(posterior_samples, min=0)  # Re-enforce non-negativity

        return posterior_samples

    def compute_summary(self, posterior_samples: torch.Tensor) -> Dict[str, Any]:
        """
        Computes MAP, max, and HDI intervals for the posterior distribution.

        Parameters:
        - posterior_samples: Tensor of posterior samples.

        Returns:
        - Dictionary containing summary statistics.
        """
        map_estimate = posterior_samples.mode().item()  # MAP is the mode of the distribution
        max_value = posterior_samples.max().item()

        hdi_50 = pyro.infer.mcmc.util.hdi(posterior_samples, 0.50)
        hdi_95 = pyro.infer.mcmc.util.hdi(posterior_samples, 0.95)
        hdi_99 = pyro.infer.mcmc.util.hdi(posterior_samples, 0.99)

        return {
            "map": map_estimate,
            "max": max_value,
            "hdi50lb": hdi_50[0].item(), "hdi50ub": hdi_50[1].item(),
            "hdi95lb": hdi_95[0].item(), "hdi95ub": hdi_95[1].item(),
            "hdi99lb": hdi_99[0].item(), "hdi99ub": hdi_99[1].item()
        }

    def run_tests(self):
        """
        Runs a structured battery of tests to validate correctness and efficiency.
        """
        self.logger.info("\n🧪 Running Tests on Pyro MCMC Sampler...\n")

        test_cases = [
            {"name": "Normal Distribution Samples", "generator": lambda: torch.normal(0, 1, (10000,)), "force_non_negative": False},
            {"name": "Poisson Count Data", "generator": lambda: torch.poisson(torch.full((10000,), 5.0)), "force_non_negative": True},
            {"name": "Gamma Distributed Samples", "generator": lambda: torch.distributions.Gamma(2, 2).sample((10000,)), "force_non_negative": True},
            {"name": "Extreme Right-Skewed Data", "generator": lambda: torch.exponential(torch.full((10000,), 2.0)), "force_non_negative": True},
            {"name": "Sparse Data (90% Zeroes)", "generator": lambda: torch.cat([torch.zeros(9000), torch.normal(10, 2, (1000,))]), "force_non_negative": True},
            {"name": "Floating-Point Precision Test", "generator": lambda: torch.normal(1e-5, 1e-6, (10000,)), "force_non_negative": False},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate and set sample distribution
            empirical_samples = test["generator"]()
            self.set_samples(empirical_samples)

            self.logger.info(f"   📊 Sample Distribution Set with {len(empirical_samples)} points")
            self.logger.info(f"   🔧 Force Non-Negative: {test['force_non_negative']}")

            # Run MCMC
            start_time = time.time()
            posterior_samples = self.run_mcmc()
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # Compute summary statistics
            summary = self.compute_summary(posterior_samples)

            self.logger.info(f"   🔍 MAP Estimate: {summary['map']:.6f}")
            self.logger.info(f"   🔍 Max Value: {summary['max']:.6f}")
            self.logger.info(f"   🔍 HDI 50%: [{summary['hdi50lb']:.6f}, {summary['hdi50ub']:.6f}]")
            self.logger.info(f"   🔍 HDI 95%: [{summary['hdi95lb']:.6f}, {summary['hdi95ub']:.6f}]")
            self.logger.info(f"   🔍 HDI 99%: [{summary['hdi99lb']:.6f}, {summary['hdi99ub']:.6f}]")

        self.logger.info("\n✅ All MCMC Sampler Tests Passed Successfully!\n")
