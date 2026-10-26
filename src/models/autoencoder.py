from collections.abc import Sequence
import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    """Symmetric MLP Autoencoder for activity feature vector reconstruction.

    Compresses input feature vector x in R^D into a lower-dimensional latent bottleneck
    representation z in R^L via an encoder, and reconstructs the feature vector x_hat in R^D
    via a decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        latent_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if any(int(h) < 1 for h in hidden_dims):
            raise ValueError(f"hidden_dims must all be positive, got {hidden_dims}")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        # Build Encoder: input_dim -> hidden_dims -> latent_dim
        encoder_layers = []
        prev = self.input_dim
        for width in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev, width))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(float(dropout)))
            prev = width
        encoder_layers.append(nn.Linear(prev, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder: latent_dim -> reversed(hidden_dims) -> input_dim
        decoder_layers = []
        prev = self.latent_dim
        for width in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev, width))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(float(dropout)))
            prev = width
        decoder_layers.append(nn.Linear(prev, self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input features into the latent bottleneck space."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected shape [batch_size, {self.input_dim}], got {tuple(x.shape)}"
            )
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct feature vectors from latent bottleneck representations."""
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected shape [batch_size, {self.latent_dim}], got {tuple(z.shape)}"
            )
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input feature vector with identical output shape."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected shape [batch_size, {self.input_dim}], got {tuple(x.shape)}"
            )
        return self.decoder(self.encoder(x))
