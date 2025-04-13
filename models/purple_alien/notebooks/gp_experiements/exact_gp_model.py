import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal


from gpytorch.kernels import (
    RBFKernel, MaternKernel, PeriodicKernel,
    LinearKernel, ScaleKernel,
    AdditiveKernel, ProductKernel
)


try:
    from gp_kernel_factory import KernelBuilder
except ImportError:
    KernelBuilder = None


class ExactGPModel(ExactGP):
    """
    Production-ready Exact GP model with modular kernel support and MVP fallback.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs [N, D].
    train_y : torch.Tensor
        Training targets [N].
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood object (e.g., GaussianLikelihood).
    kernel_config : dict, optional
        Kernel config dictionary (requires `KernelBuilder`).
    use_mvp_kernel : bool
        Use Scale(RBF) if True. Overrides kernel_config.
    mean_type : str
        One of ["zero", "constant"].
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_config: dict = None,
        use_mvp_kernel: bool = False,
        mean_type: str = "constant",
    ):
        super().__init__(train_x, train_y, likelihood)

        # === Mean Module ===
        if mean_type == "zero":
            self.mean_module = ZeroMean()
        elif mean_type == "constant":
            self.mean_module = ConstantMean()
        else:
            raise ValueError(f"Invalid mean_type: '{mean_type}'. Use 'zero' or 'constant'.")

        # === Kernel / Covariance Module ===
        if use_mvp_kernel:
            self.covar_module = ScaleKernel(RBFKernel())
        else:
            if KernelBuilder is None:
                raise ImportError("`KernelBuilder` not found and `use_mvp_kernel` is False.")
            if kernel_config is None:
                raise ValueError("kernel_config must be provided if not using MVP kernel.")
            self.covar_module = KernelBuilder(kernel_config).build()

        # === Constraints (optional but useful) ===
        if hasattr(self.covar_module, "outputscale"):
            self.covar_module.register_constraint("raw_outputscale", GreaterThan(1e-4))

        # === Print model summary ===
        self.log_model_config()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def log_model_config(self):
        print("🔧 ExactGPModel configuration:")
        print(f"  • Mean       : {self.mean_module.__class__.__name__}")
        print(f"  • Kernel     : {self.covar_module.__class__.__name__}")
        if hasattr(self.covar_module, "base_kernel"):
            print(f"    - Base     : {self.covar_module.base_kernel.__class__.__name__}")
        if hasattr(self.covar_module, "kernels"):
            print(f"    - Composite of : {[k.__class__.__name__ for k in self.covar_module.kernels]}")
        print("\n")

    def describe(self):
        return {
            "mean_type": self.mean_module.__class__.__name__,
            "kernel": str(self.covar_module),
            "kernel_class": self.covar_module.__class__.__name__,
        }


    def print_hyperparams(self):
        """Recursively prints kernel hyperparameters and likelihood noise."""

        def recurse(kernel, prefix=""):
            if isinstance(kernel, ScaleKernel):
                recurse(kernel.base_kernel, prefix + "  ")
                print(f"{prefix}Outputscale: {kernel.outputscale.item():.4f}")

            elif isinstance(kernel, RBFKernel):
                print(f"{prefix}Lengthscale (RBF): {kernel.lengthscale.item():.4f}")

            elif isinstance(kernel, MaternKernel):
                print(f"{prefix}Lengthscale (Matern): {kernel.lengthscale.item():.4f}")

            elif isinstance(kernel, PeriodicKernel):
                print(f"{prefix}Lengthscale (Periodic): {kernel.lengthscale.item():.4f}")
                print(f"{prefix}Period Length: {kernel.period_length.item():.4f}")

            elif isinstance(kernel, LinearKernel):
                print(f"{prefix}Linear kernel — no lengthscale")

            elif isinstance(kernel, (AdditiveKernel, ProductKernel)):
                for i, k in enumerate(kernel.kernels):
                    print(f"{prefix}Component {i}:")
                    recurse(k, prefix + "  ")
            else:
                print(f"{prefix}Unknown kernel type: {type(kernel)}")

        print("\n✅ Model Hyperparameters:")
        recurse(self.covar_module)
        print(f"Likelihood noise: {self.likelihood.noise.item():.4f}")