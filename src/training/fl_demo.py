
import numpy as np
from torch.utils.data import DataLoader

from src.fl_engine import ClientNode, FederatedServer
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
from src.training.dataset import WindowDataset


def make_client_data(seed, num_samples=128):
    """Create synthetic data for one simulated client."""
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(num_samples, 12)).astype("float32")
    y = ((x[:, 0] + 0.6 * x[:, 1]) > 0).astype("float32")

    dataset = WindowDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def main():
    # Create two simulated clients with different training data.
    client1_loader = make_client_data(seed=1)
    client2_loader = make_client_data(seed=2)

    # Create the initial model and server.
    initial_adapter = ModelAdapter(
        MLPClassifier(input_dim=12)
    )
    server = FederatedServer(initial_adapter.get_weights())

    # Each client has its own model adapter.
    client1 = ClientNode(
        client_id="client_1",
        adapter=ModelAdapter(MLPClassifier(input_dim=12)),
        train_loader=client1_loader,
    )
    client2 = ClientNode(
        client_id="client_2",
        adapter=ModelAdapter(MLPClassifier(input_dim=12)),
        train_loader=client2_loader,
    )

    # Run one round of local training followed by FedAvg.
    config = TrainConfig(local_epochs=2)

    result = server.run_round(
        clients=[client1, client2],
        config=config,
    )

    print("Federated round result:")
    print(result)
    print("Server round number:", server.round_number)
    print("Global weight tensors:", len(server.get_global_weights()))
    print("Federated round demo passed.")


if __name__ == "__main__":
    main()
