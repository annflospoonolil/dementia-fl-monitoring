from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional, Union
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class AutoencoderTrainConfig:
    learning_rate: float = 1e-3
    local_epochs: int = 1
    seed: int = 42
    weight_decay: float = 0.0


class AutoencoderAdapter:
    """Federated-compatible model adapter for MLPAutoencoder training and scoring.

    Adheres strictly to the federated adapter contract expected by FederatedServer:
    - get_weights() -> OrderedDict[str, np.ndarray]
    - set_weights(weights: Mapping[str, np.ndarray]) -> None
    - train_local(train_loader: DataLoader, config) -> {"num_samples": int, "train_loss": float}

    Trains using Mean Squared Error (MSE) reconstruction loss: L(x, x_hat) = ||x - x_hat||^2 / D.
    Does not require or use agitation labels for training.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss(reduction="mean")

    def get_weights(self) -> OrderedDict[str, np.ndarray]:
        """Return model parameters as an OrderedDict of NumPy arrays."""
        out = OrderedDict()
        for k, v in self.model.state_dict().items():
            try:
                arr = v.detach().cpu().numpy().copy()
            except RuntimeError:
                arr = np.array(v.detach().cpu().tolist(), dtype=np.float32)
            out[k] = arr
        return out

    def set_weights(self, weights: Mapping[str, np.ndarray]) -> None:
        """Update model parameters after validating keys and shapes."""
        current = self.model.state_dict()
        if list(weights.keys()) != list(current.keys()):
            raise ValueError("Weight keys/order do not match model")
        replacement = OrderedDict()
        for name, tensor in current.items():
            arr = np.asarray(weights[name])
            if tuple(arr.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: got {arr.shape}, expected {tuple(tensor.shape)}"
                )
            try:
                t = torch.as_tensor(arr, dtype=tensor.dtype, device=self.device)
            except RuntimeError:
                t = torch.tensor(arr.tolist(), dtype=tensor.dtype, device=self.device)
            replacement[name] = t
        self.model.load_state_dict(replacement, strict=True)

    @staticmethod
    def _unpack_features(batch) -> torch.Tensor:
        """Extract input features without requiring or consuming classification labels."""
        if isinstance(batch, dict):
            x = batch.get("features", batch.get("x"))
            if x is None:
                raise ValueError("Batch dict missing 'features' or 'x' key")
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        elif isinstance(batch, torch.Tensor):
            x = batch
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        return x

    def train_local(
        self,
        train_loader: DataLoader,
        config: Optional[Union[AutoencoderTrainConfig, dict]] = None,
    ) -> dict:
        """Run local training on eligible activity windows using reconstruction MSE loss."""
        if config is None:
            config = AutoencoderTrainConfig()
        elif isinstance(config, dict):
            config = AutoencoderTrainConfig(**config)
        if config.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")

        torch.manual_seed(config.seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.model.train()
        total_loss, seen = 0.0, 0

        for _ in range(config.local_epochs):
            for batch in train_loader:
                x_raw = self._unpack_features(batch)
                try:
                    x = torch.as_tensor(x_raw, dtype=torch.float32, device=self.device)
                except RuntimeError:
                    x = torch.tensor(
                        np.asarray(x_raw).tolist(), dtype=torch.float32, device=self.device
                    )
                if len(x) == 0:
                    continue

                optimizer.zero_grad(set_to_none=True)
                recon = self.model(x)
                if recon.shape != x.shape:
                    raise ValueError(
                        f"Reconstructed shape {tuple(recon.shape)} does not match input {tuple(x.shape)}"
                    )
                loss = self.criterion(recon, x)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu()) * len(x)
                seen += len(x)

        if seen == 0:
            raise ValueError("Training loader contained no samples")

        n = len(train_loader.dataset) if hasattr(train_loader, "dataset") else seen // config.local_epochs
        return {"num_samples": int(n), "train_loss": float(total_loss / seen)}

    @torch.no_grad()
    def compute_reconstruction_error(self, features) -> np.ndarray:
        """Calculate per-window mean squared reconstruction error anomaly scores."""
        self.model.eval()
        try:
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        except RuntimeError:
            x = torch.tensor(
                np.asarray(features).tolist(), dtype=torch.float32, device=self.device
            )
        if x.ndim == 1:
            x = x.unsqueeze(0)
        recon = self.model(x)
        # Per-sample MSE across feature dimension D: (1/D) * sum_d (x_d - recon_d)^2
        err = torch.mean((x - recon) ** 2, dim=-1)
        try:
            return err.cpu().numpy()
        except RuntimeError:
            return np.array(err.cpu().tolist(), dtype=np.float32)

    @torch.no_grad()
    def reconstruct(self, features) -> np.ndarray:
        """Return reconstructed feature vectors."""
        self.model.eval()
        try:
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        except RuntimeError:
            x = torch.tensor(
                np.asarray(features).tolist(), dtype=torch.float32, device=self.device
            )
        if x.ndim == 1:
            x = x.unsqueeze(0)
        recon = self.model(x)
        try:
            return recon.cpu().numpy()
        except RuntimeError:
            return np.array(recon.cpu().tolist(), dtype=np.float32)
