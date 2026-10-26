"""Federated Learning demonstration for personalized activity anomaly detection.

Coordinates multiple simulated clients training local MLPAutoencoders using reconstruction
MSE loss, aggregating model parameters via the central FederatedServer and FedAvg.
Follows with local personalized anomaly threshold calibration.

Note: This demo uses synthetic activity feature data for multi-client simulation.
Real TIHM feature evaluation is demonstrated when processed dataset tables are provided.
"""

from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.fl_engine import ClientNode, FederatedServer
from src.models.autoencoder import MLPAutoencoder
from src.models.autoencoder_adapter import AutoencoderAdapter, AutoencoderTrainConfig
from src.models.anomaly_detector import AnomalyThresholdCalibrator
from src.training.anomaly_dataset import ActivityWindowDataset


def generate_synthetic_client_data(
    client_id: str,
    num_samples: int = 120,
    feature_dim: int = 10,
    seed: int = 42,
) -> ActivityWindowDataset:
    """Generate synthetic activity features representing presumed-normal baseline windows."""
    rng = np.random.default_rng(seed)
    # Synthetic room trigger distributions with client-specific activity profile
    client_offset = rng.uniform(-0.5, 0.5, size=(1, feature_dim)).astype(np.float32)
    features = np.abs(rng.normal(loc=1.0, scale=0.5, size=(num_samples, feature_dim)).astype(np.float32) + client_offset)
    
    return ActivityWindowDataset(
        features=features,
        participant_ids=[client_id] * num_samples,
        label_status=["unknown"] * num_samples,  # Presumed normal training baseline
    )


def main():
    print("=" * 60)
    print("FEDERATED MLP AUTOENCODER TRAINING DEMO (FedAvg)")
    print("=" * 60)

    feature_dim = 10
    hidden_dims = (16, 8)
    latent_dim = 4

    # Setup simulated clients
    client_ids = ["participant_101", "participant_102", "participant_103"]
    client_nodes: List[ClientNode] = []
    client_datasets = {}

    for i, cid in enumerate(client_ids):
        ds = generate_synthetic_client_data(cid, num_samples=96 + i * 24, feature_dim=feature_dim, seed=100 + i)
        client_datasets[cid] = ds
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        adapter = AutoencoderAdapter(
            MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        )
        client_nodes.append(ClientNode(client_id=cid, adapter=adapter, train_loader=loader))

    # Initialize central server with global autoencoder weights
    initial_server_adapter = AutoencoderAdapter(
        MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    )
    server = FederatedServer(initial_server_adapter.get_weights())
    print(f"Initialized central FederatedServer with global autoencoder ({len(server.get_global_weights())} parameter tensors).")

    # Run federated rounds
    num_rounds = 3
    config = AutoencoderTrainConfig(local_epochs=2, learning_rate=1e-3, seed=42)

    for r in range(1, num_rounds + 1):
        result = server.run_round(client_nodes, config)
        print(
            f"Round {result['round']}: "
            f"Successful clients = {result['num_successful_clients']}/{len(client_nodes)}, "
            f"Total samples = {result['total_samples']}, "
            f"Weighted Reconstruction MSE = {result['weighted_train_loss']:.6f}"
        )

    print("\nFederated parameter aggregation complete.")

    # Personalized Threshold Calibration on historical baseline data
    print("\nPersonalized Anomaly Threshold Calibration:")
    calibrator = AnomalyThresholdCalibrator(
        method="quantile", quantile=0.95, min_participant_samples=20
    )

    participant_errors = {}
    for node in client_nodes:
        # Client loads the final aggregated global model
        node.adapter.set_weights(server.get_global_weights())
        ds = client_datasets[node.client_id]
        errs = node.adapter.compute_reconstruction_error(ds.features)
        participant_errors[node.client_id] = errs

    calibrator.fit_personalized(participant_errors)
    print(f"Global Cohort Threshold: {calibrator.global_threshold:.6f}")
    for cid, thresh in calibrator.personalized_thresholds.items():
        t_type = "fallback" if cid in calibrator.fallback_participants else "personalized"
        print(f"  {cid}: Threshold = {thresh:.6f} ({t_type}, n={calibrator.participant_sample_counts[cid]})")

    # Example test anomaly scoring
    test_normal = client_datasets["participant_101"].features[:5]
    test_anomalous = test_normal * 10.0  # Simulated severe spike in activity counts
    test_batch = torch.cat([test_normal, test_anomalous], dim=0)
    test_pids = ["participant_101"] * len(test_batch)

    adapter = client_nodes[0].adapter
    errs = adapter.compute_reconstruction_error(test_batch)
    scoring = calibrator.score(errs, participant_ids=test_pids)

    print("\nTest Anomaly Scoring on Client participant_101:")
    for idx, (err, is_anom) in enumerate(zip(scoring.reconstruction_errors, scoring.is_anomaly)):
        category = "Simulated Normal" if idx < 5 else "Simulated Anomaly"
        print(f"  Sample {idx+1} ({category}): MSE = {err:.6f} -> Anomaly = {is_anom}")

    print("\nFederated autoencoder demo finished successfully.")


if __name__ == "__main__":
    main()
