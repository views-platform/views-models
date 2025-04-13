import torch
from torch import Tensor
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from typing import Optional
import matplotlib.pyplot as plt

class FrozenKernelGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: GaussianLikelihood, kernel, mean_type="constant"):
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
        likelihood: Optional[GaussianLikelihood] = None
    ):
        self.kernel = kernel
        self.mean_type = mean_type
        self.likelihood = likelihood or GaussianLikelihood()

    def infer(self, x: Tensor, y: Tensor):
        """
        Fit the frozen-kernel GP model to new data for posterior inference.
        """
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

    def describe(self):
        return {
            "kernel": self.kernel.__class__.__name__,
            "mean_type": self.mean_type,
            "noise": self.likelihood.noise.item()
        }


    def plot_fit(self, x_plot: Tensor, show_samples: bool = False, num_samples: int = 10):
        """
        Plots posterior mean, confidence intervals, and optionally samples.

        Parameters
        ----------
        x_plot : Tensor
            Input points to evaluate the GP on (e.g., a dense linspace).
        show_samples : bool
            Whether to plot posterior samples.
        num_samples : int
            Number of samples to draw.
        """
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