import torch
import gpytorch.kernels as kernels
from gpytorch.priors import NormalPrior


class KernelBuilder:
    def __init__(self, config: dict):
        self.config = config

    def build(self):
        return self._build_kernel(self.config)

    def _build_kernel(self, config: dict):
        kernel_type = config.get("type")

        if kernel_type == "add":
            return self._build_add(config)
        elif kernel_type == "product":
            return self._build_product(config)
        elif kernel_type == "scale":
            return self._build_scale(config)
        elif kernel_type == "RBF":
            return self._build_rbf(config)
        elif kernel_type == "Matern":
            return self._build_matern(config)
        elif kernel_type == "Periodic":
            return self._build_periodic(config)
        elif kernel_type == "Linear":
            return self._build_linear(config)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # ----------- Composite Kernels -----------

    def _build_add(self, config):
        components = [self._build_kernel(c) for c in config["components"]]
        return kernels.AdditiveKernel(*components)

    def _build_product(self, config):
        components = [self._build_kernel(c) for c in config["components"]]
        return kernels.ProductKernel(*components)

    def _build_scale(self, config):
        base_kernel = self._build_kernel(config["base_kernel"])
        scale_kernel = kernels.ScaleKernel(base_kernel)

        if "outputscale_prior" in config:
            p = config["outputscale_prior"]
            prior = NormalPrior(p["mean"], p["stddev"])
            scale_kernel.register_prior("outputscale_prior", prior, "outputscale")

        if "outputscale" in config:
            scale_kernel.outputscale = torch.tensor(config["outputscale"], dtype=torch.float32)

        return scale_kernel

    # ----------- Base Kernels -----------

    def _build_rbf(self, config):
        kernel = kernels.RBFKernel()
        self._maybe_add_lengthscale(kernel, config)
        return kernel

    def _build_matern(self, config):
        kernel = kernels.MaternKernel(nu=config.get("nu", 2.5))
        self._maybe_add_lengthscale(kernel, config)
        return kernel

    def _build_periodic(self, config):
        kernel = kernels.PeriodicKernel()
        self._maybe_add_lengthscale(kernel, config)

        if "period_length" in config:
            kernel.period_length = torch.tensor(config["period_length"], dtype=torch.float32)

        return kernel

    def _build_linear(self, config):
        return kernels.LinearKernel()


    # ----------- Utility -----------

    def _maybe_add_lengthscale(self, kernel, config):
        if "lengthscale_prior" in config:
            p = config["lengthscale_prior"]
            prior = NormalPrior(p["mean"], p["stddev"])
            kernel.register_prior("lengthscale_prior", prior, "lengthscale")
        if "lengthscale" in config:
            kernel.lengthscale = torch.tensor(config["lengthscale"], dtype=torch.float32)
