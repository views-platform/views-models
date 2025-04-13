import os
import torch
import logging
from torch.nn.utils import clip_grad_norm_
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

logger = logging.getLogger(__name__)

class ExactGPTrainer:
    def __init__(
        self,
        model: ExactGP,
        likelihood: GaussianLikelihood,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lr: float = 0.01,
        max_grad_norm: float = None,
        loss_reduction: str = "sum",  # "mean" or "sum"
        print_every: int = 25,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.likelihood = likelihood.to(device)
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.device = device

        if loss_reduction not in ["sum", "mean"]:
            raise ValueError("loss_reduction must be 'sum' or 'mean'")

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=lr
        )
        self.mll = ExactMarginalLogLikelihood(likelihood, model)
        self.loss_reduction = loss_reduction
        self.max_grad_norm = max_grad_norm
        self.print_every = print_every

    def train(self, num_epochs: int = 300):
        self.model.train()
        self.likelihood.train()

        for epoch in range(1, num_epochs + 1):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)

            if self.loss_reduction == "mean":
                loss = loss.mean()
            else:
                loss = loss.sum()

            loss.backward()

            if self.max_grad_norm is not None:
                clip_grad_norm_(
                    list(self.model.parameters()) + list(self.likelihood.parameters()),
                    max_norm=self.max_grad_norm
                )

            self.optimizer.step()

            if epoch % self.print_every == 0 or epoch == 1:
                logger.info(f"[Epoch {epoch:03d}] Loss: {loss.item():.6f}")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.likelihood.state_dict(), os.path.join(path, "likelihood.pt"))
        logger.info(f"Saved model and likelihood to: {path}")

    @staticmethod
    def load(path: str, model: ExactGP, likelihood: GaussianLikelihood):
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        likelihood.load_state_dict(torch.load(os.path.join(path, "likelihood.pt")))
        logger.info(f"Loaded model and likelihood from: {path}")
        return model, likelihood
