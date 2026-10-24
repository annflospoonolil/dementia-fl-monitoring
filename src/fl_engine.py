from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


Weights = OrderedDict[str, npt.NDArray[np.floating]]


@dataclass
class ClientUpdate:
    client_id: str
    weights: Mapping[str, np.ndarray]
    num_samples: int


def validate_updates(updates: list[ClientUpdate]) -> None:
    """Validate client updates before aggregation."""

    if not updates:
        raise ValueError("Cannot aggregate: no client updates received.")

    reference_keys = list(updates[0].weights.keys())
    reference_shapes = {
        name: np.asarray(updates[0].weights[name]).shape
        for name in reference_keys
    }

    for update in updates:
        if (
            not isinstance(update.num_samples, (int, np.integer))
            or isinstance(update.num_samples, bool)
            or update.num_samples <= 0
        ):
            raise ValueError(
                f"Client {update.client_id} has invalid sample count."
            )

        if list(update.weights.keys()) != reference_keys:
            raise ValueError(
                f"Client {update.client_id} has different weight keys/order."
            )

        for name in reference_keys:
            array = np.asarray(update.weights[name])

            if array.shape != reference_shapes[name]:
                raise ValueError(
                    f"Shape mismatch for {name} from "
                    f"client {update.client_id}."
                )

            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(
                    f"Weight {name} must have a floating-point dtype."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Weight {name} contains NaN or infinity."
                )


def fedavg(updates: list[ClientUpdate]) -> Weights:
    """Compute sample-weighted Federated Averaging."""

    validate_updates(updates)

    total_samples = sum(u.num_samples for u in updates)
    reference = updates[0].weights
    aggregated = OrderedDict()

    for name in reference:
        total = np.zeros(
            np.asarray(reference[name]).shape,
            dtype=np.float64,
        )

        for update in updates:
            weight = update.num_samples / total_samples
            total += weight * np.asarray(
                update.weights[name], dtype=np.float64
            )

        aggregated[name] = total.astype(
            np.asarray(reference[name]).dtype
        )

    return aggregated


def copy_weights(weights):
    """Return independent copies of model parameters."""
    return OrderedDict(
        (name, np.array(value, copy=True))
        for name, value in weights.items()
    )


@dataclass
class ClientNode:
    """A simulated client and its local training data."""
    client_id: str
    adapter: Any
    train_loader: Any


class FederatedServer:
    """Coordinates federated training rounds and global aggregation."""

    def __init__(self, initial_weights):
        if not initial_weights:
            raise ValueError("Initial global weights cannot be empty.")

        self.global_weights = copy_weights(initial_weights)
        self.round_number = 0
        self.history = []

    def get_global_weights(self):
        """Return a copy of the current global weights."""
        return copy_weights(self.global_weights)

    def run_round(self, clients, config):
        """Run one federated training round."""

        clients = list(clients)

        if not clients:
            raise ValueError("No clients selected for this round.")

        updates = []
        failed_clients = []
        train_loss_sum = 0.0
        train_loss_samples = 0

        # All clients start from the same global weights.
        round_start_weights = self.get_global_weights()

        for client in clients:
            try:
                client.adapter.set_weights(
                    copy_weights(round_start_weights)
                )

                train_result = client.adapter.train_local(
                    client.train_loader, config
                )

                if not isinstance(train_result, dict):
                    raise TypeError(
                        "train_local() must return a dictionary."
                    )

                num_samples = train_result.get("num_samples")
                train_loss = train_result.get("train_loss")

                # 1. Validate sample count.
                if (
                    not isinstance(num_samples, (int, np.integer))
                    or isinstance(num_samples, bool)
                    or num_samples <= 0
                ):
                    raise ValueError(
                        "num_samples must be a positive integer."
                    )

                # 2. Validate loss before accepting the update.
                if train_loss is not None:
                    train_loss = float(train_loss)

                    if not np.isfinite(train_loss):
                        raise ValueError(
                            "train_loss must be finite."
                        )

                # 3. Get and validate the local model weights.
                client_weights = client.adapter.get_weights()

                update = ClientUpdate(
                    client_id=client.client_id,
                    weights=copy_weights(client_weights),
                    num_samples=int(num_samples),
                )

                validate_updates(updates + [update])

                # 4. Accept the update only after validation.
                updates.append(update)

                # 5. Record the loss of the accepted update.
                if train_loss is not None:
                    train_loss_sum += train_loss * num_samples
                    train_loss_samples += num_samples

            except Exception as exc:
                failed_clients.append({
                    "client_id": client.client_id,
                    "error": str(exc),
                })

        # Preserve global weights if all clients fail.
        if not updates:
            raise RuntimeError(
                "Federated round failed: no valid client updates. "
                f"Failures: {failed_clients}"
            )

        # Aggregate successful updates.
        new_global_weights = fedavg(updates)

        self.global_weights = copy_weights(new_global_weights)
        self.round_number += 1

        total_samples = sum(u.num_samples for u in updates)

        result = {
            "round": self.round_number,
            "successful_clients": [u.client_id for u in updates],
            "failed_clients": failed_clients,
            "num_successful_clients": len(updates),
            "num_failed_clients": len(failed_clients),
            "total_samples": total_samples,
            "weighted_train_loss": (
                train_loss_sum / train_loss_samples
                if train_loss_samples > 0
                else None
            ),
        }

        self.history.append(result)
        return deepcopy(result)