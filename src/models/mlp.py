from collections.abc import Sequence
import torch
from torch import nn

class MLPClassifier(nn.Module):
    """Binary MLP returning logits with shape [batch_size, 1]."""
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(int(n) < 1 for n in hidden_dims):
            raise ValueError("input_dim and hidden_dims must be positive")
        layers, prev = [], input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, int(width)), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, features], got {tuple(x.shape)}")
        return self.network(x)
