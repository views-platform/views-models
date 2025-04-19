from typing import Union

import pandas as pd

import torch
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import Kernel
from typing import Optional
import matplotlib.pyplot as plt
import copy
from copy import deepcopy

import logging
logger = logging.getLogger(__name__)

class FrozenKernelGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood, kernel: Kernel, mean_type="constant"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean() if mean_type == "constant" else ZeroMean()
        self.covar_module = kernel

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)


class FrozenKernelPredictor:
    def __init__(
        self,
        kernel,
        mean_type: str = "constant",
        likelihood: Optional[GaussianLikelihood] = None,
        component_names: Optional[list] = None,
    ):
        self.kernel = kernel
        self.mean_type = mean_type
        self.likelihood = likelihood or GaussianLikelihood()
        self.component_names = component_names or [] 

        # Validate component naming
        if hasattr(self.kernel, "kernels"):
            num_components = len(self.kernel.kernels)
            if component_names is not None and len(component_names) != num_components:
                raise ValueError(
                    f"Provided {len(component_names)} component names, but kernel has {num_components} components."
                )
            if component_names is None:
                # Default names if none provided
                self.component_names = [f"component_{i}" for i in range(num_components)]


    def infer(self, x: Tensor, y: Tensor):
        self.model = FrozenKernelGP(x, y, self.likelihood, self.kernel, mean_type=self.mean_type)
        self.model.eval()
        self.likelihood.eval()
        self.x = x
        self.y = y
        return self

    def posterior(self, x_test: Tensor):
        with torch.no_grad():
            return self.likelihood(self.model(x_test))

    def posterior_mean_and_ci(self, x_test: Tensor):
        dist = self.posterior(x_test)
        return dist.mean, *dist.confidence_region()

    def sample_posterior(self, x_test: Tensor, num_samples: int = 10):
        dist = self.posterior(x_test)
        return dist.rsample(torch.Size([num_samples]))








    def posterior_decomposition(
        self,
        x_test: Tensor,
        num_samples: int = 0,
        return_dict: bool = True,
        adjust: bool = False,
    ) -> Union[dict, list]:
        """
        Decompose the GP posterior into additive kernel components.

        Parameters
        ----------
        x_test : torch.Tensor
            Input points to evaluate the GP on. Shape: [N, D]
        num_samples : int
            Number of Monte Carlo samples to draw from each component.
        return_dict : bool
            If True, returns a dict keyed by component name. Else, returns list.
        adjust : bool
            If True, applies mean-centering and orthogonalization adjustments
            to seasonal and short-term trends for interpretability.

        Returns
        -------
        Dict[str, Dict] or List[Dict]
            Each entry contains:
                - mean : Tensor [N]
                - lower: Tensor [N]
                - upper: Tensor [N]
                - samples (optional): Tensor [num_samples, N]
        """
        assert self.model is not None, "Must call .infer(x, y) before decomposition."

        if not hasattr(self.kernel, "kernels"):
            raise ValueError("Decomposition requires an additive kernel (with `.kernels`)")

        components = []
        for i, k in enumerate(self.kernel.kernels):
            sub_model = FrozenKernelGP(
                train_x=self.x,
                train_y=self.y,
                likelihood=self.likelihood,
                kernel=k,
                mean_type=self.mean_type
            )
            sub_model.eval()

            with torch.no_grad():
                dist = self.likelihood(sub_model(x_test))
                mean = dist.mean.clone()
                lower, upper = dist.confidence_region()

                comp = {
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                }

                if num_samples > 0:
                    comp["samples"] = dist.rsample(torch.Size([num_samples]))

                components.append(comp)

        # === Optional Adjustment for Identifiability ===
        if adjust:
            logger.info("[Decomposition] Adjustment enabled (adjust=True)")

            if len(components) >= 3:
                long = components[0]["mean"]
                short = components[1]["mean"]
                seas = components[2]["mean"]

                seas_centered = seas - seas.mean()
                full_adjustment = long + seas_centered

                components[1]["mean"] -= full_adjustment
                components[1]["lower"] -= full_adjustment
                components[1]["upper"] -= full_adjustment
                if "samples" in components[1]:
                    components[1]["samples"] -= full_adjustment.unsqueeze(0)

                components[2]["mean"] = seas_centered
                components[2]["lower"] -= seas.mean()
                components[2]["upper"] -= seas.mean()
                if "samples" in components[2]:
                    components[2]["samples"] -= seas.mean()

            elif len(components) == 2:
                long = components[0]["mean"]
                components[1]["mean"] -= long
                components[1]["lower"] -= long
                components[1]["upper"] -= long
                if "samples" in components[1]:
                    components[1]["samples"] -= long.unsqueeze(0)

        else:
            logger.info("[Decomposition] No adjustment (adjust=False)")

        # Return with names or raw list
        names = self.component_names or [f"component_{i}" for i in range(len(components))]
        return {name: comp for name, comp in zip(names, components)} if return_dict else components






#    def posterior_decomposition(
#        self,
#        x_test: Tensor,
#        num_samples: int = 0,
#        return_dict: bool = True,
#        adjust: bool = False,
#    ) -> Union[dict, list]:
#        """
#        Decompose the GP posterior into additive kernel components.
#    
#        Parameters
#        ----------
#        x_test : torch.Tensor
#            Input points to evaluate the GP on. Shape: [N, D]
#        num_samples : int
#            Number of Monte Carlo samples to draw from each component.
#        return_dict : bool
#            If True, returns a dict keyed by component name. Else, returns list.
#        adjust : bool
#            If True, applies mean-centering and orthogonalization adjustments
#            to seasonal and short-term trends for interpretability.
#    
#        Returns
#        -------
#        Dict[str, Dict] or List[Dict]
#            Each entry contains:
#                - mean : Tensor [N]
#                - lower: Tensor [N]
#                - upper: Tensor [N]
#                - samples (optional): Tensor [num_samples, N]
#        """
#        assert self.model is not None, "Must call .infer(x, y) before decomposition."
#    
#        if not hasattr(self.kernel, "kernels"):
#            raise ValueError("Decomposition requires an additive kernel (with `.kernels`)")
#    
#        components = []
#        for i, k in enumerate(self.kernel.kernels):
#            sub_model = FrozenKernelGP(
#                train_x=self.x,
#                train_y=self.y,
#                likelihood=self.likelihood,
#                kernel=k,
#                mean_type=self.mean_type
#            )
#            sub_model.eval()
#    
#            with torch.no_grad():
#                dist = self.likelihood(sub_model(x_test))
#                mean = dist.mean.clone()
#                lower, upper = dist.confidence_region()
#    
#                comp = {
#                    "mean": mean,
#                    "lower": lower,
#                    "upper": upper,
#                }
#    
#                if num_samples > 0:
#                    comp["samples"] = dist.rsample(torch.Size([num_samples]))  # shape [S, N]
#    
#                components.append(comp)
#    
#        # === Optional Adjustment ===
#        if adjust:
#            logger.info("[Decomposition] Adjustment enabled (adjust=True)")
#    
#            if len(components) >= 2:
#                long_term = components[0]["mean"]
#                components[1]["mean"] -= long_term
#                components[1]["lower"] -= long_term
#                components[1]["upper"] -= long_term
#                if "samples" in components[1]:
#                    components[1]["samples"] -= long_term.unsqueeze(0)
#    
#            if len(components) >= 3:
#                seasonal = components[2]["mean"]
#                components[1]["mean"] -= seasonal
#                components[1]["lower"] -= seasonal
#                components[1]["upper"] -= seasonal
#                if "samples" in components[1]:
#                    components[1]["samples"] -= seasonal.unsqueeze(0)
#    
#                # Finally, center seasonal around zero
#                seasonal_mean = seasonal.mean()
#                components[2]["mean"] -= seasonal_mean
#                components[2]["lower"] -= seasonal_mean
#                components[2]["upper"] -= seasonal_mean
#                if "samples" in components[2]:
#                    components[2]["samples"] -= seasonal_mean
#    
#        else:
#            logger.info("[Decomposition] No adjustment (adjust=False)")
#    
#        names = self.component_names or [f"component_{i}" for i in range(len(components))]
#        return {name: comp for name, comp in zip(names, components)} if return_dict else components
    































#
#    def posterior_decomposition(
#        self,
#        x_test: Tensor,
#        num_samples: int = 0,
#        return_dict: bool = True,
#        adjust: bool = False,
#    ) -> Union[dict, list]:
#        """
#        Decompose the GP posterior into additive kernel components.
#
#        Parameters
#        ----------
#        x_test : torch.Tensor
#            Input points to evaluate the GP on. Shape: [N, D]
#        num_samples : int
#            Number of Monte Carlo samples to draw from each component.
#        return_dict : bool
#            If True, returns a dict keyed by component name. Else, returns list.
#        adjust : bool
#            If True, applies mean-centering and orthogonalization adjustments
#            to seasonal and short-term trends for interpretability.
#
#        Returns
#        -------
#        Dict[str, Dict] or List[Dict]
#            Each entry contains:
#                - mean : Tensor [N]
#                - lower: Tensor [N]
#                - upper: Tensor [N]
#                - samples (optional): Tensor [num_samples, N]
#        """
#        assert self.model is not None, "Must call .infer(x, y) before decomposition."
#
#        if not hasattr(self.kernel, "kernels"):
#            raise ValueError("Decomposition requires an additive kernel (with `.kernels`)")
#
#        components = []
#        for i, k in enumerate(self.kernel.kernels):
#            sub_model = FrozenKernelGP(
#                train_x=self.x,
#                train_y=self.y,
#                likelihood=self.likelihood,
#                kernel=k,
#                mean_type=self.mean_type
#            )
#            sub_model.eval()
#
#            with torch.no_grad():
#                dist = self.likelihood(sub_model(x_test))
#                mean = dist.mean.clone()
#                lower, upper = dist.confidence_region()
#
#                comp = {
#                    "mean": mean,
#                    "lower": lower,
#                    "upper": upper,
#                }
#
#                if num_samples > 0:
#                    comp["samples"] = dist.rsample(torch.Size([num_samples]))
#
#                components.append(comp)
#
#        # === Optional Adjustment ===
#        if adjust:
#            logger.info("[Decomposition] Adjustment enabled (adjust=True)")
#
#            if len(components) >= 2:
#                base = components[0]["mean"]
#                components[1]["mean"] -= base
#                components[1]["lower"] -= base
#                components[1]["upper"] -= base
#
#                if "samples" in components[1]:
#                    components[1]["samples"] -= base.unsqueeze(0)
#
#            if len(components) >= 3:
#                seasonal_shift = components[2]["mean"].mean()
#                components[2]["mean"] -= seasonal_shift
#                components[2]["lower"] -= seasonal_shift
#                components[2]["upper"] -= seasonal_shift
#
#                if "samples" in components[2]:
#                    components[2]["samples"] -= seasonal_shift
#
#        else:
#            logger.info("[Decomposition] No adjustment (adjust=False)")
#
#        # === Return structure
#        names = self.component_names or [f"component_{i}" for i in range(len(components))]
#        return {name: comp for name, comp in zip(names, components)} if return_dict else components
#



    def freeze(self):
        """
        Return a detached, deepcopy-safe version of the predictor.
        Useful for multiprocessing or parallel inference.
        """
        # Detach kernel and likelihood safely
        frozen = FrozenKernelPredictor(
            kernel=deepcopy(self.kernel),
            mean_type=self.mean_type,
            likelihood=deepcopy(self.likelihood)
        )

        if hasattr(self, "x") and hasattr(self, "y"):
            frozen.infer(self.x.detach().clone(), self.y.detach().clone())

        return frozen


    
    def to_dataframe(self, x_test: Tensor, include_ci: bool = True, adjust: bool = False) -> pd.DataFrame:
        """
        Returns a long-form DataFrame of posterior means (and optionally CI) for full trend and each component.
    
        Parameters
        ----------
        x_test : Tensor
            Input grid for evaluation.
        include_ci : bool
            If True, includes lower and upper confidence intervals.
        adjust : bool
            If True, applies adjustment to components for identifiability.
    
        Returns
        -------
        pd.DataFrame
            Columns include:
                - timestep
                - full_mean, (full_lower, full_upper)
                - {component}_mean [, _lower, _upper]
                - observed (if available)
        """
        assert self.model is not None, "Must call `.infer(x, y)` before using `.to_dataframe()`"
    
        logger.info(f"[to_dataframe] adjust={adjust}")
    
        # Full posterior
        full_mean, full_lower, full_upper = self.posterior_mean_and_ci(x_test)
    
        df = pd.DataFrame({
            "timestep": x_test.squeeze().numpy(),
            "full_mean": full_mean.numpy(),
        })
    
        if include_ci:
            df["full_lower"] = full_lower.numpy()
            df["full_upper"] = full_upper.numpy()
    
        # Components (with optional adjustment)
        comp_dict = self.posterior_decomposition(x_test, return_dict=True, adjust=adjust)
    
        for name, stats in comp_dict.items():
            df[f"{name}_mean"] = stats["mean"].numpy()
            if include_ci:
                df[f"{name}_lower"] = stats["lower"].numpy()
                df[f"{name}_upper"] = stats["upper"].numpy()
    
        # Observed values (merged for convenience)
        if self.x is not None and self.y is not None:
            obs_df = pd.DataFrame({
                "timestep": self.x.squeeze().numpy(),
                "observed": self.y.numpy()
            })
            df = pd.merge(df, obs_df, on="timestep", how="left")
    
        return df
    

#
#
#    def to_dataframe(self, x_test: Tensor, include_ci: bool = True) -> pd.DataFrame:
#        """
#        Returns a long-form DataFrame of posterior means (and optionally CI) for full trend and each component.
#
#        Parameters
#        ----------
#        x_test : Tensor
#            Input grid for evaluation.
#        include_ci : bool
#            If True, includes lower and upper confidence intervals.
#
#        Returns
#        -------
#        pd.DataFrame
#        """
#        assert self.model is not None, "Must call `.infer(x, y)` before using `.to_dataframe()`"
#
#        # Full posterior
#        full_mean, full_lower, full_upper = self.posterior_mean_and_ci(x_test)
#
#        # Start frame
#        df = pd.DataFrame({
#            "timestep": x_test.squeeze().numpy(),
#            "full_mean": full_mean.numpy(),
#        })
#
#        if include_ci:
#            df["full_lower"] = full_lower.numpy()
#            df["full_upper"] = full_upper.numpy()
#
#        # Components
#        comp_dict = self.posterior_decomposition(x_test, return_dict=True)
#        for name, stats in comp_dict.items():
#            df[f"{name}_mean"] = stats["mean"].numpy()
#            if include_ci:
#                df[f"{name}_lower"] = stats["lower"].numpy()
#                df[f"{name}_upper"] = stats["upper"].numpy()
#
#        # Optionally merge observed data
#        if self.x is not None and self.y is not None:
#            observed_df = pd.DataFrame({
#                "timestep": self.x.squeeze().numpy(),
#                "observed": self.y.numpy()
#            })
#            df = pd.merge(df, observed_df, on="timestep", how="left")
#
#        return df
#






    def check_decomposition_integrity(self, x_grid: Tensor, adjust: bool = False, tol: float = 1e-2):
        """
        Checks if the sum of additive components matches the full posterior mean.

        Parameters
        ----------
        x_grid : torch.Tensor
            Input grid to evaluate the decomposition.
        adjust : bool
            Whether to apply identifiability adjustment.
        tol : float
            Tolerance for max absolute difference.
        """
        full_mean, _, _ = self.posterior_mean_and_ci(x_grid)
        decomp = self.posterior_decomposition(x_grid, adjust=adjust)

        summed = sum(comp["mean"] for comp in decomp.values())
        max_diff = torch.max(torch.abs(summed - full_mean)).item()
        mean_diff = torch.mean(torch.abs(summed - full_mean)).item()

        print(f"[Integrity Check] Max abs diff: {max_diff:.5f} | Mean abs diff: {mean_diff:.5f}")
        if max_diff > tol:
            print("❌ Decomposition mismatch — investigate kernels or scaling ❌")
        else:
            print("✅ Decomposition sum matches full posterior ✅")


    def plot_decomposition(
        self,
        x_test: Tensor,
        num_samples: int = 0,
        adjust: bool = False,
    ):
        """
        Visualize the posterior decomposition with CI and optional samples.

        Parameters
        ----------
        x_test : Tensor
            Input points to evaluate the GP on.
        num_samples : int
            Number of posterior samples to plot per component.
        adjust : bool
            Whether to apply identifiability correction to components.
        """
    
        # Get decomposition
        decomp = self.posterior_decomposition(
            x_test,
            num_samples=num_samples,
            adjust=adjust,
            return_dict=True,
        )

        title_suffix = " (Adjusted)" if adjust else ""
        plt.figure(figsize=(12, 6))

        for i, (name, stats) in enumerate(decomp.items()):
            mean = stats["mean"].detach().cpu().numpy()
            lower = stats["lower"].detach().cpu().numpy()
            upper = stats["upper"].detach().cpu().numpy()
            x_np = x_test.squeeze().detach().cpu().numpy()

            plt.plot(x_np, mean, label=f"{name} mean")
            plt.fill_between(x_np, lower, upper, alpha=0.2, label=f"{name} 95% CI")

            if num_samples > 0 and "samples" in stats:
                samples = stats["samples"].detach().cpu().numpy()
                for j in range(num_samples):
                    plt.plot(x_np, samples[j], linestyle="--", linewidth=1, alpha=0.3,
                             label=f"{name} sample" if j == 0 else None)

        plt.title(f"GP Trend Decomposition{title_suffix}")
        plt.xlabel("Timestep")
        plt.ylabel("Trend Contribution")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def describe(self):
        return {
            "kernel": self.kernel.__class__.__name__,
            "mean_type": self.mean_type,
            "noise": self.likelihood.noise.item()
        }

    def plot_fit(self, x_plot: Tensor, show_samples: bool = False, num_samples: int = 10):
        mean, lower, upper = self.posterior_mean_and_ci(x_plot)
        plt.figure(figsize=(10, 5))
        plt.plot(x_plot.numpy(), mean.numpy(), label="Posterior Mean", color="black")
        plt.fill_between(x_plot.squeeze(), lower.numpy(), upper.numpy(), alpha=0.3, label="95% CI")
        plt.scatter(self.x.numpy(), self.y.numpy(), color='red', label="Observed", s=20)

        if show_samples:
            samples = self.sample_posterior(x_plot, num_samples=num_samples)
            for i in range(num_samples):
                plt.plot(x_plot.numpy(), samples[i].numpy(), alpha=0.3, linewidth=1)

        plt.title("GP Posterior Fit (Frozen Kernel)")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
