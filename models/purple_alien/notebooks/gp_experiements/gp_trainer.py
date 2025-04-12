import os
import torch
import logging
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GPTrainer:
    def __init__(
        self,
        model: ApproximateGP,
        likelihood: GaussianLikelihood,
        train_loader: torch.utils.data.DataLoader,
        optimizer_config: dict,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.likelihood = likelihood.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=optimizer_config.get("lr", 0.01),
            weight_decay=optimizer_config.get("weight_decay", 0.0)
        )

        self.mll = VariationalELBO(
            likelihood, model, num_data=len(train_loader.dataset)
        )

    def train(self, num_epochs: int, checkpoint_path: str = None):
        self.model.train()
        self.likelihood.train()

        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -self.mll(output, y_batch).sum()  # <- ensure scalar loss for backward()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"[Epoch {epoch:3d}] Avg Loss: {avg_loss:.4f}")

            if checkpoint_path:
                self.save(checkpoint_path)
                logger.debug(f"Model checkpoint saved to: {checkpoint_path}")

    def save(self, path: str):
        """Saves model and likelihood state_dicts."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.likelihood.state_dict(), os.path.join(path, "likelihood.pt"))
        logger.debug(f"Saved model and likelihood to {path}")

    @staticmethod
    def load(path: str, model: ApproximateGP, likelihood: GaussianLikelihood):
        """Loads model and likelihood state_dicts into provided model objects."""
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        likelihood.load_state_dict(torch.load(os.path.join(path, "likelihood.pt")))
        logger.info(f"Loaded model and likelihood from {path}")
        return model, likelihood



#trainer = GPTrainer(
#    model=model,
#    likelihood=likelihood,
#    train_loader=dataloader,
#    optimizer_config={"lr": 0.01}
#)
#
#trainer.train(num_epochs=30, checkpoint_path="checkpoints/gp_model/")
#
#
#
#
#
#model = VariationalGPModel(inducing_points, kernel_config)
#likelihood = GaussianLikelihood()
#model, likelihood = GPTrainer.load("checkpoints/gp_model/", model, likelihood)
#model.eval()
#likelihood.eval()
#