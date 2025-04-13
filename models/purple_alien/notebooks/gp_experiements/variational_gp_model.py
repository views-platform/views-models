import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ZeroMean, ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky

try:
    from gp_kernel_factory import KernelBuilder
except ImportError:
    KernelBuilder = None


class VariationalGPModel(ApproximateGP):
    """
    Modular variational GP with optional MVP fallback mode.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Tensor of shape [M, D] used for the variational strategy.
    kernel_config : dict, optional
        Kernel config for KernelBuilder. Ignored if use_mvp_kernel=True.
    learn_inducing_points : bool
        Whether to learn inducing point locations.
    mean_type : str
        "zero" or "constant".
    use_mvp_kernel : bool
        If True, use fixed Scale(RBF) kernel like the MVP.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        kernel_config: dict = None,
        learn_inducing_points: bool = True,
        mean_type: str = "zero",
        use_mvp_kernel: bool = False
    ):
        # Sanity checks
        assert inducing_points.ndim == 2, "Inducing points must be [M, D] shape."

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points
        )

        super().__init__(variational_strategy)

        # === Inducing Points ===
        self._learn_inducing_points = learn_inducing_points

        # === Mean Module ===
        if mean_type == "zero":
            self.mean_module = ZeroMean()
        elif mean_type == "constant":
            self.mean_module = ConstantMean()
        else:
            raise ValueError(f"Invalid mean_type '{mean_type}'. Use 'zero' or 'constant'.")

        # === Covariance Module ===
        if use_mvp_kernel:
            self.covar_module = ScaleKernel(RBFKernel())
        else:
            if KernelBuilder is None:
                raise ImportError("KernelBuilder is not available, but use_mvp_kernel=False.")
            if kernel_config is None:
                raise ValueError("kernel_config must be provided unless use_mvp_kernel=True.")
            self.covar_module = KernelBuilder(kernel_config).build()

        # === Print model summary ===
        self.log_model_config()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def check_K_zz(model):
        """Diagnostic check for PSD-ness of K_zz matrix."""
        K = model.covar_module(model.variational_strategy.inducing_points).evaluate()
        try:
            psd_safe_cholesky(K + 1e-4 * torch.eye(K.size(0)))
            print("✅ K_zz is stable")
        except Exception as e:
            print("❌ K_zz is not positive definite:", str(e))

    def log_model_config(self):
        print("🔧 VariationalGPModel configuration:")
        print(f"  • Mean type      : {self.mean_module.__class__.__name__}")
        print(f"  • Kernel         : {self.covar_module.__class__.__name__}")
        if hasattr(self.covar_module, "base_kernel"):
            print(f"    - Base kernel  : {self.covar_module.base_kernel.__class__.__name__}")
        if hasattr(self.covar_module, "kernels"):
            print(f"    - Composite of : {[k.__class__.__name__ for k in self.covar_module.kernels]}")
        print(f"  • Inducing points: {tuple(self.variational_strategy.inducing_points.shape)}")
        print(f"  • Learnable IPs  : {self._learn_inducing_points}")
        print("\n")

    def describe(self) -> dict:
        """Return structured description of model config (for logging or model cards)."""
        return {
            "mean_type": self.mean_module.__class__.__name__,
            "kernel": str(self.covar_module),
            "inducing_points_shape": tuple(self.variational_strategy.inducing_points.shape),
            "learn_inducing_points": self._learn_inducing_points,
            "kernel_class": self.covar_module.__class__.__name__,
        }









#import torch
#import gpytorch
#from gpytorch.models import ApproximateGP
#from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
#from gpytorch.distributions import MultivariateNormal
#from gpytorch.means import ZeroMean
#from gpytorch.utils.cholesky import psd_safe_cholesky
#
##from gp_kernel_factory import KernelBuilder
#
#
#
#import torch
#from gpytorch.models import ApproximateGP
#from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
#from gpytorch.means import ZeroMean, ConstantMean
#from gpytorch.kernels import RBFKernel, ScaleKernel
#from gpytorch.distributions import MultivariateNormal
#from gpytorch.utils.cholesky import psd_safe_cholesky
#
#try:
#    from gp_kernel_factory import KernelBuilder
#except ImportError:
#    KernelBuilder = None
#
#
#class VariationalGPModel(ApproximateGP):
#    """
#    Modular variational GP with optional MVP fallback mode.
#
#    Parameters
#    ----------
#    inducing_points : torch.Tensor
#        Tensor of shape [M, D] used for the variational strategy.
#    kernel_config : dict, optional
#        Kernel config for KernelBuilder. Ignored if use_mvp_kernel=True.
#    learn_inducing_points : bool
#        Whether to learn inducing point locations.
#    mean_type : str
#        "zero" or "constant".
#    use_mvp_kernel : bool
#        If True, use fixed Scale(RBF) kernel like the MVP.
#    """
#
#    def __init__(
#        self,
#        inducing_points: torch.Tensor,
#        kernel_config: dict = None,
#        learn_inducing_points: bool = True,
#        mean_type: str = "zero",
#        use_mvp_kernel: bool = False
#    ):
#        variational_distribution = CholeskyVariationalDistribution(
#            num_inducing_points=inducing_points.size(0)
#        )
#
#        variational_strategy = VariationalStrategy(
#            model=self,
#            inducing_points=inducing_points,
#            variational_distribution=variational_distribution,
#            learn_inducing_locations=learn_inducing_points
#        )
#
#        super().__init__(variational_strategy)
#
#        # === Mean Module ===
#        if mean_type == "zero":
#            self.mean_module = ZeroMean()
#        elif mean_type == "constant":
#            self.mean_module = ConstantMean()
#        else:
#            raise ValueError(f"Invalid mean_type '{mean_type}'. Use 'zero' or 'constant'.")
#
#        # === Covariance Module ===
#        if use_mvp_kernel:
#            self.covar_module = ScaleKernel(RBFKernel())
#        else:
#            if KernelBuilder is None:
#                raise ImportError("KernelBuilder is not available, but use_mvp_kernel=False.")
#            if kernel_config is None:
#                raise ValueError("kernel_config must be provided unless use_mvp_kernel=True.")
#            self.covar_module = KernelBuilder(kernel_config).build()
#
#    def forward(self, x):
#        mean_x = self.mean_module(x)
#        covar_x = self.covar_module(x)
#        return MultivariateNormal(mean_x, covar_x)
#
#    @staticmethod
#    def check_K_zz(model):
#        K = model.covar_module(model.variational_strategy.inducing_points).evaluate()
#        try:
#            psd_safe_cholesky(K + 1e-4 * torch.eye(K.size(0)))
#            print("✅ K_zz is stable")
#        except Exception as e:
#            print("❌ K_zz is not positive definite:", str(e))
#
#
#
#
#
#
#
















#
#
#
#class VariationalGPModel(ApproximateGP):
#    """
#    A variational Gaussian process model with modular kernel configuration and
#    pre-initialized inducing points.
#
#    Parameters
#    ----------
#    inducing_points : torch.Tensor
#        Tensor of shape [M, D] used to initialize the variational strategy.
#    kernel_config : dict
#        Configuration dictionary to build a kernel using `build_kernel`.
#    learn_inducing_points : bool
#        If False, keeps inducing points fixed throughout training.
#    """
#
#    def __init__(
#        self,
#        inducing_points: torch.Tensor,
#        kernel_config: dict,
#        learn_inducing_points: bool = True
#    ):
#        # Variational distribution (shared across strategy)
#        variational_distribution = CholeskyVariationalDistribution(
#            num_inducing_points=inducing_points.size(0)
#        )
#
#        # Variational strategy (wraps the inducing points)
#        variational_strategy = VariationalStrategy(
#            model=self,
#            inducing_points=inducing_points,
#            variational_distribution=variational_distribution,
#            learn_inducing_locations=learn_inducing_points
#        )
#
#        super().__init__(variational_strategy)
#
#        # Mean and covariance modules
#        self.mean_module = ZeroMean()
#        self.covar_module = KernelBuilder(kernel_config).build()
#
#    def forward(self, x):
#        mean_x = self.mean_module(x)
#        covar_x = self.covar_module(x)
#        return MultivariateNormal(mean_x, covar_x)
#
#
#    @staticmethod
#    def check_K_zz(model):
#        K = model.covar_module(model.variational_strategy.inducing_points).evaluate()
#        try:
#            psd_safe_cholesky(K + 1e-4 * torch.eye(K.size(0)))
#            print("✅ K_zz is stable")
#        except Exception as e:
#            print("❌ K_zz is not positive definite:", str(e))