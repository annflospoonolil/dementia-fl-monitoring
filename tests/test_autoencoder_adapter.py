import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.autoencoder import MLPAutoencoder
from src.models.autoencoder_adapter import AutoencoderAdapter, AutoencoderTrainConfig
from src.training.anomaly_dataset import ActivityWindowDataset


def test_adapter_weights_roundtrip():
    """Verify get_weights and set_weights work accurately and validate shapes."""
    model = MLPAutoencoder(input_dim=6, hidden_dims=(12,), latent_dim=3)
    adapter = AutoencoderAdapter(model)

    weights = adapter.get_weights()
    assert isinstance(weights, dict)
    assert len(weights) > 0

    # Clean set_weights
    adapter.set_weights(weights)

    # Shape mismatch detection
    first_key = next(iter(weights))
    bad_weights = dict(weights)
    bad_weights[first_key] = np.zeros((999, 999), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        adapter.set_weights(bad_weights)

    # Key mismatch detection
    with pytest.raises(ValueError, match="Weight keys/order"):
        adapter.set_weights({"unknown_layer.weight": np.zeros((2, 2))})


def test_adapter_train_local_unsupervised():
    """Verify train_local operates purely on reconstruction error without labels."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    x = rng.normal(size=(48, 8)).astype(np.float32)
    dataset = ActivityWindowDataset(features=x)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    adapter = AutoencoderAdapter(MLPAutoencoder(input_dim=8, latent_dim=3))
    config = AutoencoderTrainConfig(local_epochs=2, learning_rate=1e-2, seed=42)

    result = adapter.train_local(loader, config)

    assert "num_samples" in result
    assert result["num_samples"] == 48
    assert "train_loss" in result
    assert np.isfinite(result["train_loss"])
    assert result["train_loss"] >= 0.0


def test_adapter_compute_reconstruction_error():
    """Verify per-sample reconstruction error anomaly score calculation."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    x = rng.normal(size=(10, 5)).astype(np.float32)
    adapter = AutoencoderAdapter(MLPAutoencoder(input_dim=5, latent_dim=2))

    scores = adapter.compute_reconstruction_error(x)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)

    # Single-sample 1D vector test
    single_score = adapter.compute_reconstruction_error(x[0])
    assert single_score.shape == (1,)
    assert np.isclose(single_score[0], scores[0], atol=1e-5)
