
import numpy as np
import pytest

from src.fl_engine import (
    ClientUpdate,
    fedavg,
    ClientNode,
    FederatedServer,
)


def test_fedavg_uses_sample_counts():
    client1 = ClientUpdate(
        client_id="client_1",
        weights={
            "layer.weight": np.array([1.0, 3.0])
        },
        num_samples=100,
    )

    client2 = ClientUpdate(
        client_id="client_2",
        weights={
            "layer.weight": np.array([5.0, 7.0])
        },
        num_samples=50,
    )

    result = fedavg([client1, client2])

    expected = np.array([7.0 / 3.0, 13.0 / 3.0])

    np.testing.assert_allclose(
        result["layer.weight"],
        expected,
    )


def test_fedavg_rejects_shape_mismatch():
    client1 = ClientUpdate(
        client_id="client_1",
        weights={"layer.weight": np.array([1.0, 2.0])},
        num_samples=10,
    )

    client2 = ClientUpdate(
        client_id="client_2",
        weights={"layer.weight": np.array([3.0])},
        num_samples=10,
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        fedavg([client1, client2])


def test_fedavg_rejects_empty_updates():
    with pytest.raises(ValueError, match="no client updates"):
        fedavg([])

class DummyAdapter:
    """A fake model adapter for testing without PyTorch."""

    def __init__(self, delta, num_samples, fail=False):
        self.weights = None
        self.delta = delta
        self.num_samples = num_samples
        self.fail = fail

    def set_weights(self, weights):
        self.weights = weights

    def train_local(self, train_loader, config):
        if self.fail:
            raise RuntimeError("Simulated client failure")

        for name in self.weights:
            self.weights[name] = self.weights[name] + self.delta

        return {
            "num_samples": self.num_samples,
            "train_loss": 0.5,
        }

    def get_weights(self):
        return self.weights


def make_client(client_id, delta, num_samples, fail=False):
    adapter = DummyAdapter(delta, num_samples, fail)

    return ClientNode(
        client_id=client_id,
        adapter=adapter,
        train_loader=None,
    )


def test_server_completes_round():
    server = FederatedServer({"weight": np.array([0.0])})

    clients = [
        make_client("A", delta=1.0, num_samples=100),
        make_client("B", delta=3.0, num_samples=50),
    ]

    result = server.run_round(clients, config={})

    # Weighted average: (100/150)*1 + (50/150)*3 = 5/3
    np.testing.assert_allclose(
        server.global_weights["weight"],
        np.array([5.0 / 3.0]),
    )

    assert result["round"] == 1
    assert result["num_successful_clients"] == 2
    assert result["total_samples"] == 150


def test_server_continues_if_one_client_fails():
    server = FederatedServer({"weight": np.array([0.0])})

    clients = [
        make_client("A", delta=2.0, num_samples=20),
        make_client("B", delta=9.0, num_samples=30, fail=True),
    ]

    result = server.run_round(clients, config={})

    np.testing.assert_allclose(
        server.global_weights["weight"],
        np.array([2.0]),
    )

    assert result["num_successful_clients"] == 1
    assert result["num_failed_clients"] == 1
    assert result["failed_clients"][0]["client_id"] == "B"


def test_server_preserves_weights_if_all_clients_fail():
    server = FederatedServer({"weight": np.array([7.0])})

    client = make_client(
        "A",
        delta=2.0,
        num_samples=20,
        fail=True,
    )

    with pytest.raises(RuntimeError, match="no valid client updates"):
        server.run_round([client], config={})

    np.testing.assert_array_equal(
        server.global_weights["weight"],
        np.array([7.0]),
    )

    assert server.round_number == 0

