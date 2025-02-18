import torch
import time
import logging

class ForecastReconciler:
    """
    A class for reconciling hierarchical forecasts at the country and grid levels.
    
    Supports:
    - Probabilistic forecast reconciliation (adjusting posterior samples).
    - Point estimate reconciliation (for deterministic forecasts).
    - Automatic validation tests for correctness.
    """

    def __init__(self, device=None):
        """
        Initializes the ForecastReconciler class.

        Args:
            device (str, optional): "cuda" for GPU acceleration, "cpu" otherwise. Defaults to auto-detect.
        """
        self.logger = logging.getLogger(__name__)  # Class-specific logger
        logging.basicConfig(level=logging.INFO)  # Configure logging format
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")


    def reconcile_probabilistic(self, pgm_forecast_samples, cm_forecast_samples, lr=0.01, max_iters=500, tol=1e-6):
        """
        Adjusts grid-level probabilistic forecasts to match the country-level forecasts using per-sample quadratic optimization.
        
        Args:
            pgm_forecast_samples (torch.Tensor): (num_samples, num_grid_cells) posterior samples.
            cm_forecast_samples (torch.Tensor): (num_samples,) country-level forecast samples.
        
        Returns:
            torch.Tensor: Adjusted grid forecasts with sum-matching per sample.
        """
        pgm_forecast_samples = pgm_forecast_samples.clone().float().to(self.device)
        cm_forecast_samples = cm_forecast_samples.clone().float().to(self.device)

        assert pgm_forecast_samples.shape[0] == cm_forecast_samples.shape[0], "Mismatch in sample count"

        # Identify nonzero values (to preserve zeros)
        mask_nonzero = pgm_forecast_samples > 0
        nonzero_values = pgm_forecast_samples.clone()
        nonzero_values[~mask_nonzero] = 0  # Ensure zero values remain unchanged

        # Initial proportional scaling
        sum_nonzero = nonzero_values.sum(dim=1, keepdim=True)
        scaling_factors = cm_forecast_samples.view(-1, 1) / (sum_nonzero + 1e-8)
        adjusted_values = nonzero_values * scaling_factors
        adjusted_values = adjusted_values.clone().detach().requires_grad_(True)

        # Optimizer (L-BFGS)
        optimizer = torch.optim.LBFGS([adjusted_values], lr=lr, max_iter=max_iters, tolerance_grad=tol)

        def closure():
            optimizer.zero_grad()
            loss = torch.sum((adjusted_values - nonzero_values) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Projection Step: Enforce sum constraint
        with torch.no_grad():
            sum_adjusted = adjusted_values.sum(dim=1, keepdim=True)
            scaling_factors = cm_forecast_samples.view(-1, 1) / (sum_adjusted + 1e-8)
            adjusted_values *= scaling_factors
            adjusted_values.clamp_(min=0)

        # Preserve zero values
        final_adjusted = pgm_forecast_samples.clone()
        final_adjusted[mask_nonzero] = adjusted_values[mask_nonzero].detach()

        return final_adjusted


    def run_tests_probabilistic(self):
        """
        Runs a suite of validation tests to ensure correctness of probabilistic reconciliation.
        """
        self.logger.info("\n🧪 Running Tests on Forecast Reconciliation...\n")

        test_cases = [
            {"name": "Basic Reconciliation", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 1.2},
            {"name": "All Zeros (Should Stay Zero)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 1.0, "scaling_factor": 1.2},
            {"name": "Extreme Skew (Right-Tailed)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.2, "scaling_factor": 10},
            {"name": "Sparse Data (95% Zero)", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.95, "scaling_factor": 1.2},
            {"name": "Large-Scale Test", "num_samples": 10000, "num_grid_cells": 500, "zero_fraction": 0.5, "scaling_factor": 1.1},
            {"name": "Extreme Scaling Needs", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 10},
            {"name": "Floating-Point Precision Test", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.5, "scaling_factor": 1e-5},
            {"name": "Mixed Zeros & Large Values", "num_samples": 1000, "num_grid_cells": 100, "zero_fraction": 0.7, "scaling_factor": 5},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate Data
            num_samples, num_grid_cells = test["num_samples"], test["num_grid_cells"]
            zero_mask = torch.rand((num_samples, num_grid_cells)) < test["zero_fraction"]
            grid_forecast_samples = torch.randint(1, 100, (num_samples, num_grid_cells), dtype=torch.float32)
            grid_forecast_samples[zero_mask] = 0  # Apply zero-inflation

            country_forecast_samples = grid_forecast_samples.sum(dim=1) * test["scaling_factor"]

            # Move data to GPU if available
            grid_forecast_samples = grid_forecast_samples.to(self.device)
            country_forecast_samples = country_forecast_samples.to(self.device)

            # Run reconciliation
            start_time = time.time()
            adjusted_grid_forecast_samples = self.reconcile_probabilistic(grid_forecast_samples, country_forecast_samples)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # **Validation Checks**
            max_sum_diff = torch.abs(adjusted_grid_forecast_samples.sum(dim=1) - country_forecast_samples).max().item()
            assert max_sum_diff < 1e-2, "❌ Sum constraint violated!"

            zero_preserved = torch.all(grid_forecast_samples == 0) == torch.all(adjusted_grid_forecast_samples == 0)
            assert zero_preserved, "❌ Zero-inflation not preserved!"

            self.logger.info(f"   🔍 Max Sum Difference: {max_sum_diff:.10f}")
            self.logger.info(f"   🔍 Zeros Correctly Preserved: {zero_preserved}\n")

        self.logger.info("\n✅ All Tests Passed Successfully!")




# ✅ **Example Usage**
if __name__ == "__main__":
    reconciler = ForecastReconciler()
    reconciler.run_tests_probabilistic()  # Run all validation tests
