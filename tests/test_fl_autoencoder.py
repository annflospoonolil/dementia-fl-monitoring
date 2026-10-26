import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.fl_engine import ClientNode, FederatedServer, fedavg
from src.models.autoencoder import MLPAutoencoder
from src.models.autoencoder_adapter import AutoencoderAdapter, AutoencoderTrainConfig
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
from src.training.anomaly_dataset import ActivityWindowDataset
from src.training.dataset import WindowDataset


def test_federated_autoencoder_aggregation():
    """Verify multi-client federated training of MLPAutoencoders with FederatedServer."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    feature_dim = 8
    hidden_dims = (16, 8)
    latent_dim = 4

    # Create synthetic baseline data for 2 clients
    x1 = rng.normal(size=(32, feature_dim)).astype(np.float32)
    x2 = rng.normal(size=(24, feature_dim)).astype(np.float32)

    loader1 = DataLoader(ActivityWindowDataset(x1), batch_size=8, shuffle=True)
    loader2 = DataLoader(ActivityWindowDataset(x2), batch_size=8, shuffle=True)

    adapter1 = AutoencoderAdapter(
        MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    )
    adapter2 = AutoencoderAdapter(
        MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    )

    server = FederatedServer(adapter1.get_weights())

    clients = [
        ClientNode("client_1", adapter1, loader1),
        ClientNode("client_2", adapter2, loader2),
    ]

    config = AutoencoderTrainConfig(local_epochs=1, learning_rate=1e-3, seed=42)

    # Run two federated rounds
    for expected_round in (1, 2):
        result = server.run_round(clients, config)

        assert result["round"] == expected_round
        assert result["num_successful_clients"] == 2
        assert result["num_failed_clients"] == 0
        assert result["total_samples"] == 56
        assert np.isfinite(result["weighted_train_loss"])
        assert result["weighted_train_loss"] >= 0.0

    # Ensure global weights were updated
    assert server.round_number == 2
    assert len(server.history) == 2


def test_federated_server_rejects_incompatible_architectures():
    """Verify that clients with incompatible architectures or shapes cannot aggregate."""
    # Client 1 uses input_dim=8, Client 2 uses input_dim=10
    adapter1 = AutoencoderAdapter(MLPAutoencoder(input_dim=8, latent_dim=4))
    adapter2 = AutoencoderAdapter(MLPAutoencoder(input_dim=10, latent_dim=4))

    server = FederatedServer(adapter1.get_weights())

    # Mock client with mismatched shape
    loader = DataLoader(ActivityWindowDataset(np.zeros((8, 10), dtype=np.float32)))
    clients = [
        ClientNode("client_bad", adapter2, loader)
    ]

    with pytest.raises(RuntimeError, match="no valid client updates"):
        server.run_round(clients, AutoencoderTrainConfig(local_epochs=1))


def test_existing_classifier_and_autoencoder_coexistence():
    """Verify that the existing MLP classifier and new autoencoder both function independently."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # 1. Existing classifier pipeline
    xc = rng.normal(size=(20, 5)).astype(np.float32)
    yc = (xc[:, 0] > 0).astype(np.float32)
    c_loader = DataLoader(WindowDataset(xc, yc), batch_size=5)
    c_adapter = ModelAdapter(MLPClassifier(input_dim=5))
    c_res = c_adapter.train_local(c_loader, TrainConfig(local_epochs=1))
    assert c_res["num_samples"] == 20

    # 2. New autoencoder pipeline
    xa = rng.normal(size=(20, 5)).astype(np.float32)
    a_loader = DataLoader(ActivityWindowDataset(xa), batch_size=5)
    a_adapter = AutoencoderAdapter(MLPAutoencoder(input_dim=5, latent_dim=2))
    a_res = a_adapter.train_local(a_loader, AutoencoderTrainConfig(local_epochs=1))
    assert a_res["num_samples"] == 20
    assert np.isfinite(a_res["train_loss"])
