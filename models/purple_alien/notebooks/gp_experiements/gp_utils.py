import torch
from gpytorch.utils.cholesky import psd_safe_cholesky

import torch
from gpytorch.utils.cholesky import psd_safe_cholesky


def extract_latent_components(model, x_test, use_mean=True, seed=None):
    """
    Extract latent trend components from an AdditiveKernel GP model.

    Parameters
    ----------
    model : VariationalGPModel
        A trained GP model using AdditiveKernel.
    x_test : torch.Tensor
        Inputs to evaluate the components on. Shape: [T, 1]
    use_mean : bool
        If True, returns mean function per component; else samples.
    seed : int, optional
        Seed for reproducibility when sampling.

    Returns
    -------
    components : dict[int, torch.Tensor]
        Dictionary mapping kernel component index to its output over x_test.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    x_test = x_test.detach()

    covar_module = model.covar_module

    if not hasattr(covar_module, "kernels"):
        raise ValueError("Model kernel is not additive — cannot extract components.")

    components = {}
    num_points = x_test.shape[0]

    # Get inducing points and posterior mean of q(u)
    inducing_points = model.variational_strategy.inducing_points
    variational_dist = model.variational_strategy._variational_distribution

    # This is the actual variational mean (the learned posterior mean of inducing values)
    m_u = variational_dist.variational_mean  # shape: [num_inducing]

    # Approximate α = K_zz⁻¹ m_u
    K_zz = model.covar_module(inducing_points).evaluate()
    K_zz += 1e-4 * torch.eye(inducing_points.size(0))  # jitter
    alpha = torch.linalg.solve(K_zz, m_u)

    for i, k_i in enumerate(covar_module.kernels):
        # Compute K(x, z) for the i-th kernel component
        K_xz = k_i(x_test, inducing_points).evaluate()

        if use_mean:
            # Predictive mean for this component
            f_i = K_xz @ alpha
        else:
            # Sample from prior using this component
            K_xx = k_i(x_test).evaluate()
            K_xx += 1e-2 * torch.eye(num_points)
            L = psd_safe_cholesky(K_xx)
            f_i = L @ torch.randn(num_points)

        components[i] = f_i.detach()

    return components
