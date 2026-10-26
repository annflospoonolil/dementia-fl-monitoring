"""Real-data federated training and held-out evaluation on the TIHM dataset.

Pipeline steps:
1. Load preprocessed TIHM feature table with 6-hour windows and agitation labels.
2. Filter eligible activity windows (`has_activity_data == 1`) and exclude sensor gaps.
   Explicitly acknowledge that presence of activity records does not prove complete coverage.
3. Perform leakage-safe participant-aware chronological splitting (70% train, 15% val, 15% test).
4. Fit feature standardizer strictly on training presumed-normal baseline features.
5. Coordinate multi-client federated training using FederatedServer and FedAvg with MLPAutoencoder.
6. Calibrate global and personalized anomaly thresholds strictly on validation data.
7. Evaluate held-out test windows, reporting reconstruction metrics and comparing scores
   with recorded agitation events where ground truth exists.
8. Save results, evaluation metrics, and final model weights.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.fl_engine import ClientNode, FederatedServer
from src.models.autoencoder import MLPAutoencoder
from src.models.autoencoder_adapter import AutoencoderAdapter, AutoencoderTrainConfig
from src.models.anomaly_detector import AnomalyThresholdCalibrator
from src.training.anomaly_dataset import (
    ActivityWindowDataset,
    filter_eligible_windows,
    split_participant_data,
)
from src.models.metrics import reconstruction_metrics

# 15 Core Activity and Cyclical Time Features
FEATURE_COLS = [
    "activity_count_back_door",
    "activity_count_bathroom",
    "activity_count_bedroom",
    "activity_count_front_door",
    "activity_count_fridge_door",
    "activity_count_hallway",
    "activity_count_kitchen",
    "activity_count_lounge",
    "activity_count_other",
    "total_activity_records",
    "active_location_count",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    num_rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    latent_dim: int = 8,
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL TIHM FEDERATED AUTOENCODER TRAINING & ANOMALY DETECTION PIPELINE")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1/7] Loading preprocessed data from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Feature table not found at {data_path}")

    raw_df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded raw table: {len(raw_df)} rows, {raw_df['patient_id'].nunique()} participants.")

    # 2. Filter eligible activity windows
    print("\n[2/7] Filtering eligible activity windows (excluding sensor recording gaps)...")
    eligible_df, audit = filter_eligible_windows(
        raw_df,
        feature_columns=FEATURE_COLS,
        require_activity_data=True,
        exclude_positive_labels=False,
    )
    print(f"  Total candidate windows: {audit['total_candidate_windows']}")
    print(f"  Eligible activity windows: {audit['eligible_windows']}")
    print(f"  Excluded sensor gaps (has_activity_data=False): {audit['excluded_sensor_gaps']}")
    print(f"  Presumed-normal unlabelled windows: {audit['presumed_normal_unlabelled']}")
    print("  Note: Presence of activity records proves triggers occurred, but does not prove complete sensor coverage.")

    # 3. Leakage-safe participant-aware chronological split
    print("\n[3/7] Performing participant-aware chronological train/val/test splits (70/15/15)...")
    train_df, val_df, test_df = split_participant_data(eligible_df, train_frac=0.70, val_frac=0.15)

    # Training baseline uses ONLY presumed-normal unlabelled windows
    train_normal_df = train_df[train_df["label_status"] != "positive"].copy()
    val_normal_df = val_df[val_df["label_status"] != "positive"].copy()

    print(f"  Train split: {len(train_df)} total ({len(train_normal_df)} presumed-normal, {len(train_df) - len(train_normal_df)} positive held out)")
    print(f"  Val split:   {len(val_df)} total ({len(val_normal_df)} presumed-normal, {len(val_df) - len(val_normal_df)} positive)")
    print(f"  Test split:  {len(test_df)} total ({(test_df['label_status'] == 'unknown').sum()} unlabelled, {(test_df['label_status'] == 'positive').sum()} positive)")

    # 4. Training-only feature standardizer
    print("\n[4/7] Fitting feature standardizer strictly on training presumed-normal baseline...")
    scaler = StandardScaler()
    scaler.fit(train_normal_df[FEATURE_COLS].to_numpy(dtype=np.float32))

    # Transform splits
    x_train_scaled = scaler.transform(train_normal_df[FEATURE_COLS].to_numpy(dtype=np.float32))
    x_val_scaled = scaler.transform(val_df[FEATURE_COLS].to_numpy(dtype=np.float32))
    x_val_normal_scaled = scaler.transform(val_normal_df[FEATURE_COLS].to_numpy(dtype=np.float32))
    x_test_scaled = scaler.transform(test_df[FEATURE_COLS].to_numpy(dtype=np.float32))

    train_normal_df["scaled_features"] = list(x_train_scaled)
    val_df["scaled_features"] = list(x_val_scaled)
    val_normal_df["scaled_features"] = list(x_val_normal_scaled)
    test_df["scaled_features"] = list(x_test_scaled)

    # 5. Build federated client nodes from participants
    print("\n[5/7] Partitioning data into federated client nodes...")
    feature_dim = len(FEATURE_COLS)
    hidden_dims = (32, 16)

    client_nodes: List[ClientNode] = []
    participant_train_counts = {}

    for pid, group in train_normal_df.groupby("patient_id"):
        n_samples = len(group)
        participant_train_counts[pid] = n_samples
        if n_samples < 5:
            continue  # Exclude participants with fewer than 5 baseline windows

        features = np.stack(group["scaled_features"].values).astype(np.float32)
        dataset = ActivityWindowDataset(
            features=features,
            participant_ids=group["patient_id"].values,
            timestamps=group["window_start"].values,
            label_status=group["label_status"].values,
        )
        loader = DataLoader(dataset, batch_size=min(batch_size, n_samples), shuffle=True)
        adapter = AutoencoderAdapter(
            MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
        )
        client_nodes.append(ClientNode(client_id=str(pid), adapter=adapter, train_loader=loader))

    print(f"  Constructed {len(client_nodes)} participating federated client nodes.")

    # Initialize central server
    init_adapter = AutoencoderAdapter(
        MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    )
    server = FederatedServer(init_adapter.get_weights())
    train_config = AutoencoderTrainConfig(
        local_epochs=local_epochs, learning_rate=learning_rate, seed=seed
    )

    print(f"\n[5/7 continued] Running {num_rounds} federated FedAvg rounds...")
    round_history = []
    for r in range(1, num_rounds + 1):
        round_res = server.run_round(client_nodes, train_config)
        round_history.append({
            "round": round_res["round"],
            "successful_clients": round_res["num_successful_clients"],
            "total_samples": round_res["total_samples"],
            "weighted_train_loss": round_res["weighted_train_loss"],
        })
        print(
            f"  Round {round_res['round']}/{num_rounds}: "
            f"Clients = {round_res['num_successful_clients']}/{len(client_nodes)}, "
            f"Samples = {round_res['total_samples']}, "
            f"Weighted Reconstruction MSE = {round_res['weighted_train_loss']:.6f}"
        )

    # 6. Validation-only threshold calibration
    print("\n[6/7] Calibrating anomaly thresholds strictly on validation baseline...")
    # Update global model with aggregated weights
    global_model = MLPAutoencoder(input_dim=feature_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    eval_adapter = AutoencoderAdapter(global_model)
    eval_adapter.set_weights(server.get_global_weights())

    calibrator = AnomalyThresholdCalibrator(
        method="quantile", quantile=0.95, min_participant_samples=10
    )

    # Compute validation errors per participant on presumed-normal validation windows
    val_errors_by_participant = {}
    for pid, group in val_normal_df.groupby("patient_id"):
        features = np.stack(group["scaled_features"].values).astype(np.float32)
        errs = eval_adapter.compute_reconstruction_error(features)
        val_errors_by_participant[str(pid)] = errs

    calibrator.fit_personalized(val_errors_by_participant)
    print(f"  Global Cohort Validation Threshold (95th pct): {calibrator.global_threshold:.6f}")
    print(f"  Personalized thresholds calibrated: {len(calibrator.personalized_thresholds) - len(calibrator.fallback_participants)}")
    print(f"  Participants using global fallback (n < 10): {len(calibrator.fallback_participants)}")

    # 7. Held-out test evaluation
    print("\n[7/7] Evaluating held-out test split...")
    test_features = np.stack(test_df["scaled_features"].values).astype(np.float32)
    test_pids = test_df["patient_id"].astype(str).values
    test_labels = test_df["label_status"].astype(str).values

    test_reconstruction_errors = eval_adapter.compute_reconstruction_error(test_features)
    test_scoring = calibrator.score(test_reconstruction_errors, participant_ids=test_pids)

    test_recon_metrics = reconstruction_metrics(
        test_features, eval_adapter.reconstruct(test_features)
    )

    # Analysis on recorded agitation ground truth vs unlabelled windows
    is_positive_event = test_labels == "positive"
    is_unlabelled = test_labels == "unknown"

    pos_count = int(np.sum(is_positive_event))
    unlabelled_count = int(np.sum(is_unlabelled))

    pos_errors = test_reconstruction_errors[is_positive_event] if pos_count > 0 else np.array([])
    unlabelled_errors = test_reconstruction_errors[is_unlabelled] if unlabelled_count > 0 else np.array([])

    pos_flagged_as_anomaly = int(np.sum(test_scoring.is_anomaly[is_positive_event])) if pos_count > 0 else 0
    unlabelled_flagged_as_anomaly = int(np.sum(test_scoring.is_anomaly[is_unlabelled])) if unlabelled_count > 0 else 0

    agitation_sensitivity = (pos_flagged_as_anomaly / pos_count) if pos_count > 0 else 0.0
    unlabelled_anomaly_rate = (unlabelled_flagged_as_anomaly / unlabelled_count) if unlabelled_count > 0 else 0.0

    print(f"  Held-out Test Windows: {len(test_df)}")
    print(f"    - Recorded agitation positive events: {pos_count}")
    print(f"    - Preserved unlabelled windows: {unlabelled_count}")
    print(f"  Test Reconstruction Metrics: Mean MSE = {test_recon_metrics['mean_mse']:.6f}, Median MSE = {test_recon_metrics['median_mse']:.6f}")
    if pos_count > 0:
        print(f"  Agitation Events Anomaly Detection: {pos_flagged_as_anomaly}/{pos_count} flagged as anomalous (Recall/Sensitivity = {agitation_sensitivity:.1%})")
        print(f"    - Mean MSE on recorded agitation events: {np.mean(pos_errors):.6f}")
        print(f"    - Mean MSE on unlabelled windows: {np.mean(unlabelled_errors):.6f}")
    print(f"  Unlabelled Windows Flagged as Anomalous: {unlabelled_flagged_as_anomaly}/{unlabelled_count} ({unlabelled_anomaly_rate:.1%})")
    print("  Crucial note: Unlabelled windows are unknown, not confirmed negatives. High anomaly scores on unlabelled windows represent genuine behavioural deviations from baseline, not guaranteed false positives.")

    # 8. Save outputs and summary
    results_summary = {
        "dataset_statistics": {
            "total_candidate_windows": audit["total_candidate_windows"],
            "eligible_windows": audit["eligible_windows"],
            "excluded_sensor_gaps": audit["excluded_sensor_gaps"],
            "num_participants": int(raw_df["patient_id"].nunique()),
            "train_samples": len(train_df),
            "train_presumed_normal_samples": len(train_normal_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "test_positive_events": pos_count,
            "test_unlabelled_windows": unlabelled_count,
        },
        "training_configuration": {
            "feature_columns": FEATURE_COLS,
            "input_dim": feature_dim,
            "hidden_dims": list(hidden_dims),
            "latent_dim": latent_dim,
            "num_clients": len(client_nodes),
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "federated_rounds": round_history,
        "validation_calibration": {
            "global_threshold": calibrator.global_threshold,
            "num_personalized_thresholds": len(calibrator.personalized_thresholds) - len(calibrator.fallback_participants),
            "num_fallback_participants": len(calibrator.fallback_participants),
        },
        "test_evaluation": {
            "reconstruction_metrics": test_recon_metrics,
            "agitation_event_recall": float(agitation_sensitivity),
            "agitation_events_detected": pos_flagged_as_anomaly,
            "agitation_events_total": pos_count,
            "mean_mse_agitation": float(np.mean(pos_errors)) if pos_count > 0 else None,
            "mean_mse_unlabelled": float(np.mean(unlabelled_errors)) if unlabelled_count > 0 else None,
            "unlabelled_flagged_count": unlabelled_flagged_as_anomaly,
            "unlabelled_flagged_rate": float(unlabelled_anomaly_rate),
        },
    }

    # Save artifacts
    results_json_path = output_dir / "real_tihm_fl_results.json"
    weights_path = output_dir / "global_autoencoder_weights.pt"
    test_predictions_csv = output_dir / "held_out_test_predictions.csv"

    results_json_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    torch.save(global_model.state_dict(), weights_path)

    test_pred_df = test_df[["patient_id", "window_start", "label_status"]].copy()
    test_pred_df["reconstruction_error"] = test_reconstruction_errors
    test_pred_df["applied_threshold"] = test_scoring.thresholds
    test_pred_df["is_anomaly"] = test_scoring.is_anomaly
    test_pred_df["threshold_type"] = test_scoring.threshold_types
    test_pred_df.to_csv(test_predictions_csv, index=False)

    print(f"\nSaved results summary: {results_json_path}")
    print(f"Saved global weights:  {weights_path}")
    print(f"Saved test predictions: {test_predictions_csv}")
    print("\nReal TIHM federated training pipeline completed successfully!")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/member2_features_labels.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/tihm_fl_experiment"),
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        learning_rate=args.lr,
    )
