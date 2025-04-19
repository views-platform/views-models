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

#    def sample_posterior(self, x_test: Tensor, num_samples: int = 10):
#        dist = self.posterior(x_test)
#        return dist.rsample(torch.Size([num_samples]))


    def sample_posterior(
        self,
        x_test: Tensor,
        num_samples: int = 10,
        return_dataframe: bool = False
    ) -> Union[Tensor, pd.DataFrame]:
        """
        Draw samples from the full GP posterior.

        Parameters
        ----------
        x_test : Tensor
            Input points to evaluate [N, D].
        num_samples : int
            Number of Monte Carlo samples.
        return_dataframe : bool
            If True, returns a long-form DataFrame instead of tensor.

        Returns
        -------
        Tensor [S, N] or pd.DataFrame
        """
        assert self.model is not None, "Must call `.infer(x, y)` before sampling."

        dist = self.posterior(x_test)
        samples = dist.rsample(torch.Size([num_samples]))  # [S, N]

        if not return_dataframe:
            return samples  # <- Needed for internal use

        # Convert to long-form dataframe
        samples_np = samples.detach().cpu().numpy()
        timesteps = x_test.squeeze().cpu().numpy()

        df = pd.DataFrame(samples_np)
        df["sample_id"] = df.index
        df = df.melt(id_vars="sample_id", var_name="timestep_idx", value_name="value")
        df["timestep"] = df["timestep_idx"].map(lambda i: timesteps[i])
        df["component"] = "full"

        return df[["timestep", "sample_id", "component", "value"]]



    def sample_decomposition(
        self,
        x_test: Tensor,
        num_samples: int = 10,
        adjust: bool = False,
        return_dataframe: bool = False
    ) -> Union[dict, pd.DataFrame]:
        """
        Draw samples from each additive component of the GP posterior.
    
        Parameters
        ----------
        x_test : Tensor
            Input points to evaluate [N, D].
        num_samples : int
            Number of Monte Carlo samples to draw.
        adjust : bool
            Apply identifiability correction to components.
        return_dataframe : bool
            If True, returns a long-form DataFrame. Else, dict of tensors.
    
        Returns
        -------
        Dict[str, Tensor] or pd.DataFrame
        """
        assert self.model is not None, "Must call `.infer()` first."
    
        # Run posterior_decomposition with sampling
        comps = self.posterior_decomposition(
            x_test,
            num_samples=num_samples,
            return_dict=True,
            adjust=adjust
        )
    
        if not return_dataframe:
            return {
                name: comp["samples"]
                for name, comp in comps.items()
                if "samples" in comp
            }
    
        # Construct dataframe
        dfs = []
        timesteps = x_test.squeeze().detach().cpu().numpy()
    
        for name, comp in comps.items():
            if "samples" not in comp:
                continue
            samples_np = comp["samples"].detach().cpu().numpy()  # shape: [S, N]
    
            df = pd.DataFrame(samples_np)
            df["sample_id"] = df.index
            df = df.melt(id_vars="sample_id", var_name="timestep_idx", value_name="value")
            df["timestep"] = df["timestep_idx"].map(lambda i: timesteps[i])
            df["component"] = name
            dfs.append(df)
    
        return pd.concat(dfs, ignore_index=True)[["timestep", "sample_id", "component", "value"]]
    

#    def sample_decomposition(
#        self,
#        x_test: Tensor,
#        num_samples: int = 10,
#        adjust: bool = False,
#        return_dict: bool = True,
#    ) -> Union[dict, pd.DataFrame]:
#        """
#        Draw samples from the posterior of each additive component.
#
#        Parameters
#        ----------
#        x_test : Tensor
#            Input grid to evaluate the GP on.
#        num_samples : int
#            Number of samples to draw per component.
#        adjust : bool
#            Whether to apply decomposition adjustment.
#        return_dict : bool
#            If False, returns a long-form dataframe (with unit/timestep/sample_id).
#
#        Returns
#        -------
#        dict or pd.DataFrame
#            Samples per component [S, N] or melted DataFrame.
#        """
#        assert self.model is not None, "Must call `.infer(x, y)` first."
#
#        logger.info(f"[sample_decomposition] Sampling {num_samples} draws per component (adjust={adjust})")
#
#        # Reuse posterior_decomposition logic to get component-wise samples
#        comps = self.posterior_decomposition(
#            x_test,
#            num_samples=num_samples,
#            adjust=adjust,
#            return_dict=True
#        )
#
#        # Extract just the samples
#        sample_dict = {
#            name: stats["samples"] for name, stats in comps.items() if "samples" in stats
#        }
#
#        if return_dict:
#            return sample_dict
#
#        # Convert to long-form dataframe
#        df_list = []
#        timesteps = x_test.squeeze().cpu().numpy()
#
#        for name, tensor in sample_dict.items():
#            samples_np = tensor.cpu().numpy()  # [S, N]
#            df_comp = pd.DataFrame(samples_np)
#            df_comp["sample_id"] = df_comp.index
#            df_comp = df_comp.melt(id_vars="sample_id", var_name="timestep_idx", value_name="value")
#            df_comp["timestep"] = df_comp["timestep_idx"].map(lambda i: timesteps[i])
#            df_comp["component"] = name
#            df_list.append(df_comp[["timestep", "sample_id", "component", "value"]])
#
#        return pd.concat(df_list, ignore_index=True)
#
#

    def posterior_decomposition(
        self,
        x_test: Tensor,
        num_samples: int = 0,
        return_dict: bool = True,
        adjust: bool = False,
    ) -> Union[dict, list]:
        """
        Decompose the GP posterior into additive kernel components.

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

        # === Handle non-additive kernels ===
        if not hasattr(self.kernel, "kernels"):
            logger.warning("[Decomposition] Kernel is not additive — returning full posterior as single component.")

            mean, lower, upper = self.posterior_mean_and_ci(x_test)
            comp = {
                "mean": mean,
                "lower": lower,
                "upper": upper,
            }
            if num_samples > 0:
                comp["samples"] = self.sample_posterior(x_test, num_samples=num_samples)

            return {"full": comp} if return_dict else [comp]

        # === Additive decomposition ===
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

        # === Return with names or raw list
        names = self.component_names or [f"component_{i}" for i in range(len(components))]
        return {name: comp for name, comp in zip(names, components)} if return_dict else components


    def posterior_derivatives(
        self,
        x_test: Tensor,
        order: int = 1,
        decompose: bool = False,
        adjust: bool = False,
    ) -> Union[Tensor, dict]:
        """
        Compute derivatives of the GP posterior (full or decomposed).

        Parameters
        ----------
        x_test : Tensor
            Input points with `requires_grad=True`.
        order : int
            Derivative order (1 or 2).
        decompose : bool
            If True, compute derivative for each additive component.
        adjust : bool
            Apply identifiability adjustment to components (if decomposed).

        Returns
        -------
        Tensor (full) or Dict[str, Tensor] (per component).
        """
        assert self.model is not None, "Must call .infer(x, y) first."
        assert order >= 1, "Only order >= 1 supported."

        x_test = x_test.clone().detach().requires_grad_(True)

        def compute_derivative(model):
            with torch.enable_grad():
                output = self.likelihood(model(x_test))
                mean = output.mean
                deriv = torch.autograd.grad(
                    outputs=mean,
                    inputs=x_test,
                    grad_outputs=torch.ones_like(mean),
                    create_graph=(order > 1)
                )[0]

                # Second order
                if order == 2:
                    deriv2 = torch.autograd.grad(
                        deriv,
                        x_test,
                        grad_outputs=torch.ones_like(deriv),
                        create_graph=False
                    )[0]
                    return deriv2.squeeze(-1)
                return deriv.squeeze(-1)

        # === Full (non-decomposed)
        if not decompose or not hasattr(self.kernel, "kernels"):
            return compute_derivative(self.model)

        # === Decomposed derivatives
        components = []
        for k in self.kernel.kernels:
            sub_model = FrozenKernelGP(
                train_x=self.x,
                train_y=self.y,
                likelihood=self.likelihood,
                kernel=k,
                mean_type=self.mean_type,
            )
            sub_model.eval()
            d = compute_derivative(sub_model)
            components.append(d)

        if adjust:
            logger.info("[Derivatives] Adjustment enabled (adjust=True)")
            if len(components) >= 3:
                long = components[0]
                seas = components[2] - components[2].mean()
                components[1] = components[1] - long - seas
                components[2] = seas
            elif len(components) == 2:
                components[1] = components[1] - components[0]
        else:
            logger.info("[Derivatives] No adjustment (adjust=False)")

        names = self.component_names or [f"component_{i}" for i in range(len(components))]
        return {name: d for name, d in zip(names, components)}




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




    def to_dataframe(
        self,
        x_test: Tensor,
        include_ci: bool = True,
        adjust: bool = False,
        include_derivatives: bool = False,
    ) -> pd.DataFrame:
        """
        Returns long-form DataFrame with posterior mean, CI, and optionally derivatives.

        Parameters
        ----------
        x_test : Tensor
            Grid of input points [N, D]
        include_ci : bool
            Whether to include 95% CI columns
        adjust : bool
            Whether to apply identifiability adjustment
        include_derivatives : bool
            Whether to include first and second order derivatives

        Returns
        -------
        pd.DataFrame
        """
        assert self.model is not None, "Must call `.infer()` before using `.to_dataframe()`"

        logger.info(f"[to_dataframe] adjust={adjust}, derivatives={include_derivatives}")

        # === Full posterior mean and CI ===
        full_mean, full_lower, full_upper = self.posterior_mean_and_ci(x_test)
        df = pd.DataFrame({
            "timestep": x_test.squeeze().cpu().numpy(),
            "full_mean": full_mean.cpu().numpy(),
        })

        if include_ci:
            df["full_lower"] = full_lower.cpu().numpy()
            df["full_upper"] = full_upper.cpu().numpy()

        # === Full derivatives
        if include_derivatives:
            d1_full = self.posterior_derivatives(x_test, order=1)
            d2_full = self.posterior_derivatives(x_test, order=2)
            df["full_d1"] = d1_full.cpu().numpy()
            df["full_d2"] = d2_full.cpu().numpy()

        # === Components (mean, CI, and optionally d1, d2)
        comp_dict = self.posterior_decomposition(x_test, return_dict=True, adjust=adjust)
        if include_derivatives:
            d1_dict = self.posterior_derivatives(x_test, order=1, decompose=True, adjust=adjust)
            d2_dict = self.posterior_derivatives(x_test, order=2, decompose=True, adjust=adjust)

        for name, stats in comp_dict.items():
            df[f"{name}_mean"] = stats["mean"].cpu().numpy()
            if include_ci:
                df[f"{name}_lower"] = stats["lower"].cpu().numpy()
                df[f"{name}_upper"] = stats["upper"].cpu().numpy()
            if include_derivatives:
                df[f"{name}_d1"] = d1_dict[name].cpu().numpy()
                df[f"{name}_d2"] = d2_dict[name].cpu().numpy()

        # === Merge observed values (optional)
        if self.x is not None and self.y is not None:
            obs_df = pd.DataFrame({
                "timestep": self.x.squeeze().cpu().numpy(),
                "observed": self.y.cpu().numpy(),
            })
            df = pd.merge(df, obs_df, on="timestep", how="left")

        return df


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

        print(f"[Integrity Check]") 
        print(f"Max abs diff: {max_diff:.5f}")
        print(f"Mean abs diff: {mean_diff:.5f}")
        if max_diff > tol:
            print("❌ Decomposition mismatch — investigate kernels or scaling ❌")
        else:
            print("✅ Decomposition sum matches full posterior ✅")



    def check_derivative_consistency(
        self,
        x_test: Tensor,
        order: int = 1,
        adjust: bool = False,
        tol: float = 1e-2,
    ):
        """
        Checks if the sum of component-wise derivatives matches the full posterior derivative.

        Parameters
        ----------
        x_test : Tensor
            Input grid with requires_grad=True.
        order : int
            Derivative order to check (1 or 2).
        adjust : bool
            Apply adjustment logic if enabled.
        tol : float
            Tolerance for max absolute difference.
        """
        x_test = x_test.clone().detach().requires_grad_(True)

        # Full GP derivative
        d_full = self.posterior_derivatives(x_test, order=order, decompose=False)

        # Decomposed
        d_comps = self.posterior_derivatives(x_test, order=order, decompose=True, adjust=adjust)
        d_sum = sum(d_comps.values())

        max_diff = torch.max(torch.abs(d_full - d_sum)).item()
        mean_diff = torch.mean(torch.abs(d_full - d_sum)).item()

        print(f"[Derivative Consistency] Order={order} | Adjust={adjust}")
        print(f"  Max abs diff : {max_diff:.6f}")
        print(f"  Mean abs diff: {mean_diff:.6f}")

        if max_diff > tol:
            print("❌ Derivative mismatch — investigate kernels or decomposition ❌")
        else:
            print("✅ Derivative sum matches full posterior ✅")




    def plot_decomposition(
        self,
        x_test: Tensor,
        num_samples: int = 0,
        adjust: bool = False,
        show_d1: bool = False,
        show_d2: bool = False,
    ):
        """
        Visualize the posterior decomposition with CI, optional samples, and optionally 1st/2nd derivatives.

        Parameters
        ----------
        x_test : Tensor
            Input points to evaluate the GP on.
        num_samples : int
            Number of posterior samples to plot per component.
        adjust : bool
            Whether to apply identifiability correction to components.
        show_d1 : bool
            Whether to plot first-order derivatives.
        show_d2 : bool
            Whether to plot second-order derivatives.
        """

        # === Get decomposed components
        decomp = self.posterior_decomposition(
            x_test,
            num_samples=num_samples,
            adjust=adjust,
            return_dict=True,
        )

        # === Optional derivatives
        df = self.to_dataframe(x_test, include_derivatives=True, adjust=adjust)

        n_panels = 1 + int(show_d1) + int(show_d2)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        x_np = x_test.squeeze().detach().cpu().numpy()

        # === Main trend decomposition plot
        ax = axes[0]
        for name, stats in decomp.items():
            ax.plot(x_np, stats["mean"].detach().cpu().numpy(), label=f"{name} mean")
            ax.fill_between(
                x_np,
                stats["lower"].detach().cpu().numpy(),
                stats["upper"].detach().cpu().numpy(),
                alpha=0.2,
                label=f"{name} 95% CI"
            )

            if num_samples > 0 and "samples" in stats:
                for j in range(num_samples):
                    ax.plot(
                        x_np,
                        stats["samples"][j].detach().cpu().numpy(),
                        linestyle="--",
                        linewidth=1,
                        alpha=0.3,
                        label=f"{name} sample" if j == 0 else None,
                    )

        ax.set_title("GP Trend Decomposition" + (" (Adjusted)" if adjust else ""))
        ax.set_ylabel("Value")
        ax.legend()

        # === d1: First derivative
        if show_d1:
            ax = axes[1]
            for name in self.component_names:
                if f"{name}_d1" in df:
                    ax.plot(df["timestep"], df[f"{name}_d1"], label=f"{name} (d1)")
            ax.set_ylabel("First Derivative (d1)")
            ax.legend()

        # === d2: Second derivative
        if show_d2:
            ax = axes[-1]
            for name in self.component_names:
                if f"{name}_d2" in df:
                    ax.plot(df["timestep"], df[f"{name}_d2"], label=f"{name} (d2)")
            ax.set_ylabel("Second Derivative (d2)")
            ax.legend()

        axes[-1].set_xlabel("Timestep")
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
