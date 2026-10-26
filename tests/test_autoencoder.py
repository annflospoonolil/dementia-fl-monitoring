import pytest
import torch
from src.models.autoencoder import MLPAutoencoder


def test_autoencoder_output_shape():
    """Verify encoder, decoder, and end-to-end reconstruction shapes."""
    batch_size = 8
    input_dim = 12
    latent_dim = 4
    hidden_dims = (24, 16)

    model = MLPAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )

    x = torch.randn(batch_size, input_dim)
    reconstruction = model(x)
    assert reconstruction.shape == (batch_size, input_dim), (
        f"Expected output shape {(batch_size, input_dim)}, got {reconstruction.shape}"
    )

    latent = model.encode(x)
    assert latent.shape == (batch_size, latent_dim), (
        f"Expected latent shape {(batch_size, latent_dim)}, got {latent.shape}"
    )

    decoded = model.decode(latent)
    assert decoded.shape == (batch_size, input_dim), (
        f"Expected decoded shape {(batch_size, input_dim)}, got {decoded.shape}"
    )


def test_autoencoder_invalid_dimensions():
    """Verify proper validation errors for invalid dimensions."""
    with pytest.raises(ValueError, match="input_dim must be positive"):
        MLPAutoencoder(input_dim=0)

    with pytest.raises(ValueError, match="latent_dim must be positive"):
        MLPAutoencoder(input_dim=10, latent_dim=-1)

    with pytest.raises(ValueError, match="hidden_dims must all be positive"):
        MLPAutoencoder(input_dim=10, hidden_dims=(16, 0))

    model = MLPAutoencoder(input_dim=5, latent_dim=2)

    with pytest.raises(ValueError, match="Expected shape"):
        model(torch.randn(4, 6))  # Wrong feature count

    with pytest.raises(ValueError, match="Expected shape"):
        model(torch.randn(5))  # 1D instead of 2D


def test_autoencoder_reconstruction_loss_reduction():
    """Verify that training with MSE loss reduces reconstruction error on a fixed batch."""
    torch.manual_seed(42)
    model = MLPAutoencoder(input_dim=6, hidden_dims=(12,), latent_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    x = torch.randn(16, 6)

    initial_loss = criterion(model(x), x).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), x).item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
