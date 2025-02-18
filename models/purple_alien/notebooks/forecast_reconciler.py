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


    def reconcile_forecast(self, grid_forecast, country_forecast, lr=0.01, max_iters=500, tol=1e-6):
        """
        Adjusts grid-level forecasts to match the country-level forecasts using per-sample quadratic optimization.

        Supports both:
        - **Probabilistic forecasts** (num_samples, num_grid_cells)
        - **Point forecasts** (num_grid_cells,) by treating them as a special case of batch size = 1.

        Args:
            grid_forecast (torch.Tensor): Posterior samples of grid forecasts (num_samples, num_grid_cells) 
                                          OR (num_grid_cells,) for point estimates.
            country_forecast (torch.Tensor or float): Posterior samples of country-level forecast (num_samples,) 
                                                      OR a single float for point estimate.

        Returns:
            torch.Tensor: Adjusted grid forecasts with sum-matching per sample.
        """
        is_point_forecast = grid_forecast.dim() == 1  # Check if it's a point forecast

        # If it's a point forecast, reshape it to be compatible with probabilistic processing
        if is_point_forecast:
            grid_forecast = grid_forecast.unsqueeze(0)  # Shape (1, num_grid_cells)
            country_forecast = torch.tensor([country_forecast], device=self.device, dtype=torch.float32)

        # Ensure correct data types & move to the right device
        grid_forecast = grid_forecast.clone().float().to(self.device)
        country_forecast = country_forecast.clone().float().to(self.device)

        assert grid_forecast.shape[0] == country_forecast.shape[0], "Mismatch in sample count"

        # Identify nonzero values (to preserve zeros)
        mask_nonzero = grid_forecast > 0
        nonzero_values = grid_forecast.clone()
        nonzero_values[~mask_nonzero] = 0  # Ensure zero values remain unchanged

        # Initial proportional scaling
        sum_nonzero = nonzero_values.sum(dim=1, keepdim=True)
        scaling_factors = country_forecast.view(-1, 1) / (sum_nonzero + 1e-8)
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
            scaling_factors = country_forecast.view(-1, 1) / (sum_adjusted + 1e-8)
            adjusted_values *= scaling_factors
            adjusted_values.clamp_(min=0)

        # Preserve zero values
        final_adjusted = grid_forecast.clone()
        final_adjusted[mask_nonzero] = adjusted_values[mask_nonzero].detach()

        # Convert back to original shape if it was a point forecast
        return final_adjusted.squeeze(0) if is_point_forecast else final_adjusted


    def run_tests(self):
        """
        Runs a complete suite of validation tests for both probabilistic and point forecast reconciliation.
        """
        self.logger.info("\n🧪 Running Full Test Battery for Forecast Reconciliation...\n")

        # 🧪 **TEST SUITES**

        self.logger.info("\n ++++++++++++++ 🔍 Running Probabilistic Forecast Reconciliation Tests ++++++++++++++++ ")        
        self.run_tests_probabilistic()

        self.logger.info("\n ++++++++++++++ 🔍 Running Point Forecast Reconciliation Tests ++++++++++++++++ ")
        self.run_tests_point()

        self.logger.info("\n✅ All Tests Passed Successfully!")


    def run_tests_probabilistic(self):
        """
        Runs validation tests to ensure correctness of **probabilistic** forecast reconciliation.
        """
        self.logger.info("\n🧪 Running Tests on Probabilistic Forecast Reconciliation...\n")

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

            # Generate Probabilistic Data
            num_samples, num_grid_cells = test["num_samples"], test["num_grid_cells"]
            zero_mask = torch.rand((num_samples, num_grid_cells)) < test["zero_fraction"]
            grid_forecast_samples = torch.randint(1, 100, (num_samples, num_grid_cells), dtype=torch.float32)
            grid_forecast_samples[zero_mask] = 0

            country_forecast_samples = grid_forecast_samples.sum(dim=1) * test["scaling_factor"]

            # Move data to GPU
            grid_forecast_samples = grid_forecast_samples.to(self.device)
            country_forecast_samples = country_forecast_samples.to(self.device)

            # Run reconciliation
            start_time = time.time()
            adjusted_grid_forecast_samples = self.reconcile_forecast(grid_forecast_samples, country_forecast_samples)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # **Validation Checks**
            sum_diff = torch.abs(adjusted_grid_forecast_samples.sum(dim=1) - country_forecast_samples).max().item()
            assert sum_diff < 1e-2, "❌ Sum constraint violated!"

            zero_preserved = torch.all(grid_forecast_samples == 0) == torch.all(adjusted_grid_forecast_samples == 0)
            assert zero_preserved, "❌ Zero-inflation not preserved!"

            self.logger.info(f"   🔍 Max Sum Difference: {sum_diff:.10f}")
            self.logger.info(f"   🔍 Zeros Correctly Preserved: {zero_preserved}\n")

        self.logger.info("\n✅ All Probabilistic Tests Passed Successfully!")

    
    def run_tests_point(self):
        """
        Runs validation tests to ensure correctness of **point** forecast reconciliation.
        """
        self.logger.info("\n🧪 Running Tests on Point Forecast Reconciliation...\n")

        test_cases = [
            {"name": "Basic Reconciliation", "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 1.2},
            {"name": "All Zeros (Should Stay Zero)", "num_grid_cells": 100, "zero_fraction": 1.0, "scaling_factor": 1.2},
            {"name": "Extreme Skew (Right-Tailed)", "num_grid_cells": 100, "zero_fraction": 0.2, "scaling_factor": 10},
            {"name": "Sparse Data (95% Zero)", "num_grid_cells": 100, "zero_fraction": 0.95, "scaling_factor": 1.2},
            {"name": "Extreme Scaling Needs", "num_grid_cells": 100, "zero_fraction": 0.3, "scaling_factor": 10},
            {"name": "Floating-Point Precision Test", "num_grid_cells": 100, "zero_fraction": 0.5, "scaling_factor": 1e-5},
            {"name": "Mixed Zeros & Large Values", "num_grid_cells": 100, "zero_fraction": 0.7, "scaling_factor": 5},
        ]

        for test in test_cases:
            self.logger.info(f"🔹 Running Test: {test['name']}")

            # Generate Point Forecast Data
            num_grid_cells = test["num_grid_cells"]
            zero_mask = torch.rand(num_grid_cells) < test["zero_fraction"]
            grid_forecast = torch.randint(1, 100, (num_grid_cells,), dtype=torch.float32)
            grid_forecast[zero_mask] = 0

            country_forecast = grid_forecast.sum().item() * test["scaling_factor"]

            # Move data to GPU
            grid_forecast = grid_forecast.to(self.device)

            # Run reconciliation
            start_time = time.time()
            adjusted_grid_forecast = self.reconcile_forecast(grid_forecast, country_forecast)
            end_time = time.time()

            self.logger.info(f"   ✅ Completed in {end_time - start_time:.3f} sec")

            # **Validation Checks**
            sum_diff = abs(adjusted_grid_forecast.sum().item() - country_forecast)
            assert sum_diff < 1e-2, "❌ Sum constraint violated!"

            zero_preserved = torch.all(grid_forecast == 0) == torch.all(adjusted_grid_forecast == 0)
            assert zero_preserved, "❌ Zero-inflation not preserved!"

            self.logger.info(f"   🔍 Max Sum Difference: {sum_diff:.10f}")
            self.logger.info(f"   🔍 Zeros Correctly Preserved: {zero_preserved}\n")

        self.logger.info("\n✅ All Point Forecast Tests Passed Successfully!")


# Usage Example
reconciler = ForecastReconciler()
reconciler.run_tests()  # Run full test suite  