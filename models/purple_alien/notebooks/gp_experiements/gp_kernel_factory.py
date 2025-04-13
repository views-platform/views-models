import torch
import gpytorch.kernels as kernels
from gpytorch.priors import NormalPrior


class KernelBuilder:
    """
    Factory for building GPyTorch kernels from a config dict.

    Supports:
    - RBF, Matern, Periodic, Linear
    - ScaleKernel for outputscale
    - Additive, Product compositions
    - Hyperparameter priors
    """

    def __init__(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("KernelBuilder expects a config dictionary.")
        if "type" not in config:
            raise ValueError("Kernel config must include a 'type' field.")
        self.config = config

    def build(self):
        return self._build_kernel(self.config)

    def _build_kernel(self, config):
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

    def _build_add(self, config):
        components = config.get("components", [])
        if not components:
            raise ValueError("Additive kernel requires a non-empty 'components' list.")
        return kernels.AdditiveKernel(*[self._build_kernel(c) for c in components])

    def _build_product(self, config):
        components = config.get("components", [])
        if not components:
            raise ValueError("Product kernel requires a non-empty 'components' list.")
        return kernels.ProductKernel(*[self._build_kernel(c) for c in components])

    def _build_scale(self, config):
        if "base_kernel" not in config:
            raise ValueError("Scale kernel requires a 'base_kernel' config.")
        base_kernel = self._build_kernel(config["base_kernel"])
        kernel = kernels.ScaleKernel(base_kernel)

        if "outputscale" in config:
            kernel.outputscale = torch.tensor(config["outputscale"], dtype=torch.float32)

        if "outputscale_prior" in config:
            self._validate_prior(config["outputscale_prior"], "outputscale_prior")
            p = config["outputscale_prior"]
            prior = NormalPrior(p["mean"], p["stddev"])
            kernel.register_prior("outputscale_prior", prior, "outputscale")

        return kernel

    def _build_rbf(self, config):
        kernel = kernels.RBFKernel()
        self._maybe_add_lengthscale(kernel, config)
        return kernel

    def _build_matern(self, config):
        nu = config.get("nu", 2.5)
        kernel = kernels.MaternKernel(nu=nu)
        self._maybe_add_lengthscale(kernel, config)
        return kernel

    def _build_periodic(self, config):
        kernel = kernels.PeriodicKernel()
        self._maybe_add_lengthscale(kernel, config)

        if "period_length" in config:
            kernel.period_length = torch.tensor(config["period_length"], dtype=torch.float32)

        if "period_length_prior" in config:
            self._validate_prior(config["period_length_prior"], "period_length_prior")
            p = config["period_length_prior"]
            prior = NormalPrior(p["mean"], p["stddev"])
            kernel.register_prior("period_length_prior", prior, "period_length")

        return kernel

    def _build_linear(self, config):
        return kernels.LinearKernel()

    def _maybe_add_lengthscale(self, kernel, config):
        if "lengthscale" in config:
            kernel.lengthscale = torch.tensor(config["lengthscale"], dtype=torch.float32)

        if "lengthscale_prior" in config:
            self._validate_prior(config["lengthscale_prior"], "lengthscale_prior")
            p = config["lengthscale_prior"]
            prior = NormalPrior(p["mean"], p["stddev"])
            kernel.register_prior("lengthscale_prior", prior, "lengthscale")

    def _validate_prior(self, prior_cfg, name):
        if not isinstance(prior_cfg, dict):
            raise ValueError(f"{name} must be a dict with 'mean' and 'stddev'")
        if "mean" not in prior_cfg or "stddev" not in prior_cfg:
            raise ValueError(f"{name} must contain 'mean' and 'stddev' keys.")

    def log_kernel_structure(self, kernel=None, indent=0):
        """Recursively print kernel structure for debugging."""
        if kernel is None:
            kernel = self.build()
    
        space = "  " * indent
        print(f"{space}{kernel.__class__.__name__}")
    
        # Composite structures
        if hasattr(kernel, "kernels"):
            for i, sub in enumerate(kernel.kernels):
                print(f"{space}[Component {i}]")
                self.log_kernel_structure(sub, indent + 1)
        elif hasattr(kernel, "base_kernel"):
            self.log_kernel_structure(kernel.base_kernel, indent + 1)
    
        # Safe helper
        def try_print(attr, label):
            value = getattr(kernel, attr, None)
            if value is None:
                return
            try:
                print(f"{space}{label}: {value.item():.4f}")
            except:
                try:
                    print(f"{space}{label}: {value.detach().cpu().numpy()}")
                except:
                    print(f"{space}{label}: <unreadable>")
    
        try_print("lengthscale", "lengthscale")
        try_print("outputscale", "outputscale")
        try_print("period_length", "period")
    
    







































