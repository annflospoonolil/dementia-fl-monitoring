from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .metrics import binary_metrics

@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    local_epochs: int = 1
    threshold: float = 0.5
    seed: int = 42
    weight_decay: float = 0.0

class ModelAdapter:
    """Stable adapter contract for the federated-learning engine."""
    def __init__(self, model: nn.Module, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_weights(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict((k, v.detach().cpu().numpy().copy()) for k, v in self.model.state_dict().items())

    def set_weights(self, weights: Mapping[str, np.ndarray]) -> None:
        current = self.model.state_dict()
        if list(weights.keys()) != list(current.keys()):
            raise ValueError("Weight keys/order do not match model")
        replacement = OrderedDict()
        for name, tensor in current.items():
            arr = np.asarray(weights[name])
            if tuple(arr.shape) != tuple(tensor.shape):
                raise ValueError(f"Shape mismatch for {name}: got {arr.shape}, expected {tuple(tensor.shape)}")
            replacement[name] = torch.as_tensor(arr, dtype=tensor.dtype, device=self.device)
        self.model.load_state_dict(replacement, strict=True)

    @staticmethod
    def _unpack(batch):
        if isinstance(batch, dict):
            x, y = batch.get("features", batch.get("x")), batch.get("labels", batch.get("y"))
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Batch must be (features, labels) or dict")
        if x is None or y is None:
            raise ValueError("Batch missing features/labels")
        return x, y

    def train_local(self, train_loader: DataLoader, config=None) -> dict:
        if config is None:
            config = TrainConfig()
        elif isinstance(config, dict):
            config = TrainConfig(**config)
        if config.local_epochs < 1:
            raise ValueError("local_epochs must be >= 1")
        torch.manual_seed(config.seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.model.train()
        total_loss, seen = 0.0, 0
        for _ in range(config.local_epochs):
            for batch in train_loader:
                x, y = self._unpack(batch)
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
                if not len(x):
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                if logits.shape != y.shape:
                    raise ValueError(f"Output {tuple(logits.shape)} does not match targets {tuple(y.shape)}")
                loss = self.criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(x)
                seen += len(x)
        if seen == 0:
            raise ValueError("Training loader contained no samples")
        n = len(train_loader.dataset) if hasattr(train_loader, "dataset") else seen // config.local_epochs
        return {"num_samples": int(n), "train_loss": float(total_loss / seen)}

    @torch.no_grad()
    def predict_proba(self, features) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return torch.sigmoid(self.model(x).reshape(-1)).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader, threshold=0.5) -> dict:
        self.model.eval()
        ys, ps, loss_sum, n = [], [], 0.0, 0
        for batch in eval_loader:
            x, y = self._unpack(batch)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss_sum += float(loss.cpu()) * len(x)
            n += len(x)
            ys.extend(y.cpu().numpy().reshape(-1).astype(int).tolist())
            ps.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
        if n == 0:
            raise ValueError("Evaluation loader contained no samples")
        out = binary_metrics(ys, ps, threshold)
        out["loss"] = loss_sum / n
        return out
