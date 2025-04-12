import gpytorch.kernels as kernels
import torch
from gpytorch.priors import NormalPrior


def build_kernel(config: dict):
    """
    Recursively builds a GPyTorch kernel from a nested config dictionary.

    Supports: RBF, Matern (with nu), Periodic, Linear, Scale, Add, Product
    Allows setting priors on lengthscale, outputscale, and period_length.

    Parameters
    ----------
    config : dict
        Configuration dictionary defining the kernel structure.

    Returns
    -------
    gpytorch.kernels.Kernel
        A fully constructed GPyTorch kernel instance.
    """

    kernel_type = config.get("type")

    # Composite kernels
    if kernel_type == "add":
        components = [build_kernel(subconfig) for subconfig in config["components"]]
        return kernels.AdditiveKernel(*components)

    if kernel_type == "product":
        components = [build_kernel(subconfig) for subconfig in config["components"]]
        return kernels.ProductKernel(*components)

    # ScaleKernel wrapper (for outputscale control)
    if kernel_type == "scale":
        base_kernel = build_kernel(config["base_kernel"])
        scale_kernel = kernels.ScaleKernel(base_kernel)

        if "outputscale_prior" in config:
            prior_cfg = config["outputscale_prior"]
            prior = NormalPrior(prior_cfg["mean"], prior_cfg["stddev"])
            scale_kernel.register_prior("outputscale_prior", prior, "outputscale")

        return scale_kernel

    # Base kernels
    if kernel_type == "RBF":
        kernel = kernels.RBFKernel()

    elif kernel_type == "Matern":
        nu = config.get("nu", 2.5)
        kernel = kernels.MaternKernel(nu=nu)

    elif kernel_type == "Periodic":
        kernel = kernels.PeriodicKernel()

        # Optional: period_length prior
        if "period_length" in config:
            kernel.period_length = torch.tensor(config["period_length"])

    elif kernel_type == "Linear":
        kernel = kernels.LinearKernel()

    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # Common priors (lengthscale)
    if "lengthscale_prior" in config:
        prior_cfg = config["lengthscale_prior"]
        prior = NormalPrior(prior_cfg["mean"], prior_cfg["stddev"])
        kernel.register_prior("lengthscale_prior", prior, "lengthscale")

    return kernel
