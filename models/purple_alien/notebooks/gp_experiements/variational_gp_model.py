import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean
from gp_kernel_factory import build_kernel


class VariationalGPModel(ApproximateGP):
    """
    A variational Gaussian process model with modular kernel configuration and
    pre-initialized inducing points.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Tensor of shape [M, D] used to initialize the variational strategy.
    kernel_config : dict
        Configuration dictionary to build a kernel using `build_kernel`.
    learn_inducing_points : bool
        If False, keeps inducing points fixed throughout training.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_config: dict,
        learn_inducing_points: bool = True
    ):
        # Variational distribution (shared across strategy)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        # Variational strategy (wraps the inducing points)
        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules
        self.mean_module = ZeroMean()
        self.covar_module = build_kernel(kernel_config)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
