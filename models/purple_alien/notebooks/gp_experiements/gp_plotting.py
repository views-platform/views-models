# gp_plotting.py
import torch
import numpy as np
import pandas as pd

from collections import Counter

from gpytorch.kernels import Kernel
import gpytorch.kernels as kernels
from gpytorch.utils.cholesky import psd_safe_cholesky


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def plot_single_series(df: pd.DataFrame, series_id: int):
    series_df = df[df["series_id"] == series_id]

    plt.figure(figsize=(12, 6))
    plt.plot(series_df["timestep"], series_df["value"], label="Value", linewidth=2)
    plt.plot(series_df["timestep"], series_df["long_term"], label="Long-Term Trend", linestyle="--")
    plt.plot(series_df["timestep"], series_df["short_term"], label="Short-Term Trend", linestyle="--")
    plt.plot(series_df["timestep"], series_df["seasonal"], label="Seasonal", linestyle="--")
    plt.plot(series_df["timestep"], series_df["noise"], label="Noise", linestyle=":", alpha=0.7)
    
    plt.title(f"Time Series Decomposition (Series ID = {series_id})")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_series(df: pd.DataFrame, n: int = 4):
    unique_ids = df["series_id"].unique()[:n]
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    
    for i, series_id in enumerate(unique_ids):
        series_df = df[df["series_id"] == series_id]
        axes[i].plot(series_df["timestep"], series_df["value"], label=f"Series {series_id}")
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()


def set_plot_style(style="whitegrid"):
    sns.set(style=style)


def plot_signal_vs_noise_comparison(df: pd.DataFrame):
    """
    Compares a high-signal and low-signal time series using all decomposed components.
    """
    high_ids = df[df["signal_strength"] == "high"]["series_id"].unique()
    low_ids = df[df["signal_strength"] == "low"]["series_id"].unique()

    if len(high_ids) == 0 or len(low_ids) == 0:
        print("Not enough data to plot both high and low signal series.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for idx, (label, sid) in enumerate([("High", high_ids[0]), ("Low", low_ids[0])]):
        subdf = df[df["series_id"] == sid]
        axes[idx].plot(subdf["timestep"], subdf["value"], label="Value", linewidth=2)
        axes[idx].plot(subdf["timestep"], subdf["long_term"], label="Long-Term", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["short_term"], label="Short-Term", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["seasonal"], label="Seasonal", linestyle="--")
        axes[idx].plot(subdf["timestep"], subdf["noise"], label="Noise", linestyle=":", alpha=0.6)
        axes[idx].set_title(f"{label}-Signal Series (ID: {sid})")
        axes[idx].legend(loc="upper right")

    plt.xlabel("Timestep")
    plt.tight_layout()
    plt.show()



def plot_component_distributions(df: pd.DataFrame, component: str):
    """
    Plots the distribution (KDE) of a single trend component across all series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from SyntheticTimeSeriesGenerator with decomposed components.
    component : str
        One of 'long_term', 'short_term', 'seasonal', or 'noise'.
    """
    if component not in ["long_term", "short_term", "seasonal", "noise"]:
        raise ValueError("Invalid component. Must be one of: 'long_term', 'short_term', 'seasonal', 'noise'.")

    plt.figure(figsize=(10, 4))
    sns.kdeplot(df[component], fill=True, linewidth=2, color="skyblue")
    plt.title(f"Distribution of {component.replace('_', ' ').title()} Component")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()



def plot_kernel_prior_samples(
    kernel_config: dict,
    num_samples: int = 5,
    x_range: tuple = (0, 1),
    num_points: int = 100,
    seed: int = None
):
    """
    Plots sample functions drawn from a zero-mean GP with a kernel defined by a config,
    where kernel hyperparameters are randomly sampled from their priors.

    Parameters
    ----------
    kernel_config : dict
        Configuration dictionary compatible with `build_kernel`.
    num_samples : int
        Number of function samples to draw.
    x_range : tuple
        Range of input values (e.g., (0, 1)).
    num_points : int
        Number of input points.
    seed : int
        Optional random seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(-1)  # [N, 1]
    x_np = x.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        # Sample a kernel with sampled hyperparameters from priors
        kernel = build_kernel_with_sampled_priors(kernel_config)

        # Get covariance matrix and sample from MVN
        cov = kernel(x).evaluate() + 1e-2 * torch.eye(num_points)  # numerical stability
        samples = torch.distributions.MultivariateNormal(
            loc=torch.zeros(num_points), covariance_matrix=cov
        ).sample()

        plt.plot(x_np, samples.numpy(), label=f"Sample {i+1}")

    plt.title("Sample Functions from GP Prior (Sampled Hyperparameters)")
    plt.xlabel("Input x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.legend()
    plt.show()


def build_kernel_with_sampled_priors(config: dict) -> Kernel:
    """
    Helper to build a GPyTorch kernel with hyperparameters sampled from defined priors.
    Assumes all priors are Normal.
    """
    import copy
    kernel_type = config.get("type")

    if kernel_type in ["add", "product"]:
        subconfigs = [build_kernel_with_sampled_priors(sc) for sc in config["components"]]
        cls = {"add": kernels.AdditiveKernel, "product": kernels.ProductKernel}[kernel_type]
        return cls(*subconfigs)

    if kernel_type == "scale":
        base_kernel = build_kernel_with_sampled_priors(config["base_kernel"])
        scale_kernel = kernels.ScaleKernel(base_kernel)

        prior_cfg = config.get("outputscale_prior")
        if prior_cfg:
            mu, sigma = prior_cfg["mean"], prior_cfg["stddev"]
            scale_kernel.outputscale = torch.tensor(np.random.normal(mu, sigma), dtype=torch.float32)

        return scale_kernel

    # Base kernels
    if kernel_type == "RBF":
        kernel = kernels.RBFKernel()

    elif kernel_type == "Matern":
        kernel = kernels.MaternKernel(nu=config.get("nu", 2.5))

    elif kernel_type == "Periodic":
        kernel = kernels.PeriodicKernel()
        if "period_length" in config:
            kernel.period_length = torch.tensor(config["period_length"], dtype=torch.float32)

    elif kernel_type == "Linear":
        kernel = kernels.LinearKernel()

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # Sample lengthscale
    prior_cfg = config.get("lengthscale_prior")
    if prior_cfg:
        mu, sigma = prior_cfg["mean"], prior_cfg["stddev"]
        sampled_lengthscale = torch.tensor(np.random.normal(mu, sigma), dtype=torch.float32)
        kernel.lengthscale = sampled_lengthscale

    return kernel


def plot_curriculum_progression_over_epochs(
    sampler,
    dataset,
    num_batches: int = 100,
    num_epochs: int = 30
):
    """
    Plots how the CurriculumSampler transitions from high to low signal series over training epochs.

    Shows a grouped bar chart with counts of high vs low signal samples per epoch.
    """
    records = []

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        it = iter(sampler)

        high_count = 0
        low_count = 0

        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            for idx in batch:
                strength = dataset.series_groups[idx]["signal_strength"].iloc[0]
                if strength == "high":
                    high_count += 1
                elif strength == "low":
                    low_count += 1

        records.append({"epoch": epoch, "signal": "high", "count": high_count})
        records.append({"epoch": epoch, "signal": "low", "count": low_count})

    df = pd.DataFrame(records)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=df, x="epoch", y="count", hue="signal", palette="viridis")

    plt.title("Curriculum Sampling Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Sample Count (over sampled batches)")
    plt.legend(title="Signal Strength")
    plt.tight_layout()
    plt.show()


def plot_inducing_points_vs_inputs(train_x, train_y, inducing_points, title="Inducing Points vs Training Inputs"):
    """
    Plots training series (train_x, train_y) with inducing points overlaid.

    Parameters
    ----------
    train_x : torch.Tensor
        Shape [B, T, 1] — input time points for a batch of series.
    train_y : torch.Tensor
        Shape [B, T] — values for each series.
    inducing_points : torch.Tensor
        Shape [M, 1] — inducing point locations.
    title : str
        Plot title.
    """
    B, T, _ = train_x.shape
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))

    # Plot training series
    for i in range(B):
        plt.plot(train_x[i].squeeze(), train_y[i], alpha=0.5, label=f"Series {i}" if i < 5 else None)

    # Overlay inducing points as vertical lines
    for ip in inducing_points.squeeze():
        plt.axvline(ip.item(), color="red", linestyle="--", alpha=0.7)

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()









def plot_gp_diagnostics(
    model,
    likelihood,
    num_samples=5,
    include_data=None,
    x_range=(0, 1),
    num_points=200,
    seed=None,
    posterior=True,
):
    """
    Plot GP posterior/prior samples and show learned kernel hyperparameters.

    Parameters
    ----------
    model : ApproximateGP
        Trained GP model.
    likelihood : GaussianLikelihood
        Trained likelihood (used for posterior).
    num_samples : int
        Number of samples to draw.
    include_data : tuple[torch.Tensor, torch.Tensor], optional
        (x_train, y_train) to overlay original data.
    x_range : tuple
        Input range to evaluate the GP over.
    num_points : int
        Number of evaluation points.
    seed : int, optional
        Random seed for reproducibility.
    posterior : bool
        If True, samples from the posterior; otherwise from the prior.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    likelihood.eval()

    x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(-1)
    x_np = x.squeeze().cpu().numpy()

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    try:
        if posterior:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                dist = likelihood(model(x))

                for i in range(num_samples):
                    # Handles both batch + non-batch
                    sample = dist.rsample()[i] if dist.batch_shape else dist.rsample()
                    plt.plot(x_np, sample.detach().cpu().numpy(), label=f"Posterior Sample {i+1}")
        else:
            # Fallback to prior sampling
            with torch.no_grad():
                cov = model.covar_module(x).evaluate()
                cov += 1e-2 * torch.eye(num_points)
                L = psd_safe_cholesky(cov)
                for i in range(num_samples):
                    z = torch.randn(num_points)
                    sample = L @ z
                    plt.plot(x_np, sample.detach().cpu().numpy(), label=f"Prior Sample {i+1}")
    except Exception as e:
        print("[WARNING] Failed to sample posterior, falling back to prior:", str(e))
        with torch.no_grad():
            cov = model.covar_module(x).evaluate()
            cov += 1e-2 * torch.eye(num_points)
            L = psd_safe_cholesky(cov)
            for i in range(num_samples):
                z = torch.randn(num_points)
                sample = L @ z
                plt.plot(x_np, sample.detach().cpu().numpy(), label=f"Prior Sample {i+1}")

    # --- Optional training data overlay ---
    if include_data is not None:
        x_data, y_data = include_data
        x_data = x_data.detach().cpu().squeeze()
        y_data = y_data.detach().cpu()
        plt.scatter(x_data, y_data, color="black", label="Training Data", alpha=0.6, s=30)

    # --- Hyperparameter summary ---
    def extract_params(k):
        lines = []
        if hasattr(k, "lengthscale") and k.lengthscale is not None:
            val = k.lengthscale.item() if torch.is_tensor(k.lengthscale) else k.lengthscale
            lines.append(f"lengthscale: {val:.4f}")
        if hasattr(k, "outputscale") and k.outputscale is not None:
            val = k.outputscale.item() if torch.is_tensor(k.outputscale) else k.outputscale
            lines.append(f"outputscale: {val:.4f}")
        if hasattr(k, "period_length") and k.period_length is not None:
            val = k.period_length.item() if torch.is_tensor(k.period_length) else k.period_length
            lines.append(f"period: {val:.4f}")
        return lines

    def get_all_kernel_params(kernel):
        lines = []
        if hasattr(kernel, "kernels"):  # Additive/Product
            for i, subk in enumerate(kernel.kernels):
                lines.append(f"[Kernel {i+1}]")
                lines += extract_params(subk)
        elif hasattr(kernel, "base_kernel"):
            lines += extract_params(kernel)
            lines += extract_params(kernel.base_kernel)
        else:
            lines += extract_params(kernel)
        return lines

    hyper_lines = get_all_kernel_params(model.covar_module)
    param_text = "\n".join(hyper_lines)

    plt.title("GP Diagnostic Plot", fontsize=14)
    plt.suptitle(param_text, fontsize=10, y=1.02)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()




















#
#def plot_curriculum_sampling_distribution(sampler, dataset, num_batches=100, epoch=None):
#    """
#    Visualizes the proportion of high vs low signal series sampled by the CurriculumSampler.
#
#    Parameters
#    ----------
#    sampler : CurriculumSampler
#        The curriculum sampler to inspect.
#    dataset : TimeSeriesDataset
#        The dataset (needed to query signal_strength metadata).
#    num_batches : int
#        How many batches to simulate.
#    epoch : int, optional
#        If provided, sets sampler to this epoch before sampling.
#    """
#    if epoch is not None:
#        sampler.set_epoch(epoch)
#
#    high_count = 0
#    low_count = 0
#    total = 0
#
#    it = iter(sampler)
#    for _ in range(num_batches):
#        try:
#            batch = next(it)
#        except StopIteration:
#            break
#        for idx in batch:
#            series_df = dataset.series_groups[idx]
#            strength = series_df["signal_strength"].iloc[0]
#            if strength == "high":
#                high_count += 1
#            elif strength == "low":
#                low_count += 1
#            total += 1
#
#    # Plotting
#    sns.set(style="whitegrid")
#    plt.figure(figsize=(6, 4))
#    sns.barplot(x=["High Signal", "Low Signal"], y=[high_count, low_count], palette="viridis")
#    plt.title(f"Curriculum Sampling Distribution at Epoch {sampler.epoch}")
#    plt.ylabel("Sample Count")
#    plt.tight_layout()
#    plt.show()
#
#
#
#