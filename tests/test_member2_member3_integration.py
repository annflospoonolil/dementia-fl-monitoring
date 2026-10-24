
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.fl_engine import ClientNode, FederatedServer
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
from src.training.dataset import WindowDataset


def test_member2_model_works_with_member3_fl_engine():
    # Make the test repeatable.
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Synthetic data for two simulated clients.
    # These are NOT real TIHM data.
    x1 = rng.normal(size=(32, 10)).astype(np.float32)
    y1 = (x1[:, 0] > 0).astype(np.float32)

    x2 = rng.normal(size=(24, 10)).astype(np.float32)
    y2 = (x2[:, 0] > 0).astype(np.float32)

    loader1 = DataLoader(
        WindowDataset(x1, y1), batch_size=8, shuffle=True
    )
    loader2 = DataLoader(
        WindowDataset(x2, y2), batch_size=8, shuffle=True
    )

    # Each client has its own model adapter.
    adapter1 = ModelAdapter(MLPClassifier(input_dim=10))
    adapter2 = ModelAdapter(MLPClassifier(input_dim=10))

    # Initialize the server with the first model's weights.
    server = FederatedServer(adapter1.get_weights())

    clients = [
        ClientNode("client_1", adapter1, loader1),
        ClientNode("client_2", adapter2, loader2),
    ]

    config = TrainConfig(local_epochs=1, seed=42)

    # Run two federated rounds.
    for expected_round in (1, 2):
        result = server.run_round(clients, config)

        assert result["round"] == expected_round
        assert result["num_successful_clients"] == 2
        assert result["num_failed_clients"] == 0
        assert result["total_samples"] == 56

    print("Federated integration test passed!")
