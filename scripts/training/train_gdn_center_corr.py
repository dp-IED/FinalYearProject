#!/usr/bin/env python3
"""
Phase 2: Training script for MultiLabelGDN with Center Loss + Correlation-Consistency Loss.
Loads Phase 1 checkpoint and adds correlation-consistency loss for improved separation.

This script:
1. Loads a Phase 1 checkpoint (must have stable center_dist ≥ 0.15)
2. Adds correlation-consistency loss (normal windows only)
3. Trains to push separation ratio from 2× → 2.5–3×
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.gdn_model import MultiLabelGDN
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    DATA_PATH,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
    FORECAST_HORIZON,
)

torch.set_default_dtype(torch.float32)

# ============================================================================
# Constants
# ============================================================================

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_GLOBAL = 0.3
LAMBDA_SEPARATION = 1.0  # Weight for separation loss
LAMBDA_CORR = 0.2  # Start conservative, tune to 0.3-0.5 if needed

# Model architecture (must match Phase 1)
EMBED_DIM = 32
TOP_K = 3
HIDDEN_DIM = 32


# ============================================================================
# Training Function with Correlation Loss
# ============================================================================


def train_model_with_correlation_loss(
    model,
    normal_center,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    lambda_separation=LAMBDA_SEPARATION,
    lambda_global=LAMBDA_GLOBAL,
    lambda_corr=LAMBDA_CORR,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    model_save_path="best_multilabel_gdn_center_corr.pt",
):
    """
    Train MultiLabelGDN with Separation Loss + Correlation-Consistency Loss (Phase 2).

    Args:
        model: Pre-trained model from Phase 1
        normal_center: Pre-trained normal center parameter from Phase 1
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        device: Device to train on
        lambda_separation: Weight for separation loss
        lambda_global: Weight for global loss
        lambda_corr: Weight for correlation loss
        learning_rate: Learning rate for model optimizer
        weight_decay: Weight decay for model optimizer
        model_save_path: Path to save best checkpoint

    Returns:
        model: Trained model
        normal_center: Updated normal center parameter
    """
    # Pre-compute learned adjacency (once before training loop)
    print("\nPre-computing learned adjacency from sensor embeddings...")
    with torch.no_grad():
        emb = model.sensor_embeddings.to(device)  # (N, 32)
        emb_norm = F.normalize(emb, dim=1)
        sim = emb_norm @ emb_norm.t()  # (N, N) cosine similarity
        sim.fill_diagonal_(-1e9)  # Exclude self-loops

        k = model.top_k
        topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)  # (N, k)

    print(f"Learned adjacency computed: {topk_vals.shape}")
    print(
        f"Top-k similarity range: [{topk_vals.min().item():.4f}, {topk_vals.max().item():.4f}]"
    )

    # Single optimizer (includes normal center)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [normal_center],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Loss functions
    sensor_criterion = nn.BCELoss(reduction="none")
    global_criterion = nn.BCELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")

    print(f"\n{'=' * 80}")
    print("Training Multi-Label GDN with Separation Loss + Correlation Loss (Phase 2)")
    print(f"{'=' * 80}")
    print(
        f"Lambda_global: {lambda_global}, Lambda_separation: {lambda_separation}, Lambda_corr: {lambda_corr}"
    )
    print(f"Model LR: {learning_rate}")
    print(f"Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()

        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_separation = 0.0
        train_loss_corr = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            for X_batch, _, sensor_labels_batch, window_labels_batch in pbar:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                B, W, N = X_batch.shape

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)  # (B, hidden_dim)

                # Classification losses
                loss_sensor = sensor_criterion(sensor_probs, sensor_labels_batch).mean()
                loss_global = global_criterion(global_prob, window_labels_batch.float())

                # Separation loss: minimize normal distance, maximize anomalous distance
                distances = torch.norm(
                    embeddings - normal_center.unsqueeze(0), dim=1
                )  # (B,)
                normal_mask = window_labels_batch == 0
                loss_normal = (
                    distances[normal_mask].mean()
                    if normal_mask.any()
                    else torch.tensor(0.0, device=device)
                )
                anomalous_mask = window_labels_batch == 1
                loss_anomalous = (
                    -distances[anomalous_mask].mean()
                    if anomalous_mask.any()
                    else torch.tensor(0.0, device=device)
                )
                loss_separation_val = loss_normal + loss_anomalous

                # Correlation-consistency loss (normal windows only)
                normal_mask = (
                    window_labels_batch == 0
                )  # Boolean mask for normal windows

                if normal_mask.any():
                    X_norm = X_batch[normal_mask]  # (B_norm, W, N)
                    B_norm = X_norm.shape[0]

                    # Compute per-window empirical correlation matrices (no grad for stability)
                    with torch.no_grad():
                        # Center each sensor
                        Xc = X_norm - X_norm.mean(dim=1, keepdim=True)  # (B_norm, W, N)

                        # Covariance: Xc^T @ Xc / (W-1)
                        cov = (Xc.transpose(1, 2) @ Xc) / (
                            W - 1 + 1e-8
                        )  # (B_norm, N, N)

                        # Std per sensor
                        var = torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(
                            -1
                        )  # (B_norm, N, 1)
                        std = torch.sqrt(torch.clamp(var, min=1e-8))

                        # Correlation = cov / (std_i * std_j)
                        denom = std @ std.transpose(1, 2)  # (B_norm, N, N)
                        corr = cov / (denom + 1e-8)
                        corr = corr.clamp(-1.0, 1.0)  # (B_norm, N, N)

                    # Extract correlations on model's top-k edges
                    # For each node i, k neighbours j = topk_idx[i, :]
                    i_idx = (
                        torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
                    )  # (N, k)
                    j_idx = topk_idx.to(device)  # (N, k)

                    actual_corr_edges = corr[:, i_idx, j_idx]  # (B_norm, N, k)
                    target_corr_edges = topk_vals.unsqueeze(0).to(device)  # (1, N, k)

                    # MSE loss between actual and learned correlations
                    loss_corr_val = (
                        (actual_corr_edges - target_corr_edges) ** 2
                    ).mean()
                else:
                    loss_corr_val = torch.tensor(0.0, device=device)

                # Combined loss
                loss = (
                    loss_sensor
                    + lambda_global * loss_global
                    + lambda_separation * loss_separation_val
                    + lambda_corr * loss_corr_val
                )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update model (normal center updates through model gradients)
                optimizer.step()

                train_loss_sensor += loss_sensor.item() * X_batch.size(0)
                train_loss_global += loss_global.item() * X_batch.size(0)
                train_loss_separation += loss_separation_val.item() * X_batch.size(0)
                train_loss_corr += loss_corr_val.item() * X_batch.size(0)

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        train_loss_separation /= len(train_loader.dataset)
        train_loss_corr /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_corr = 0.0

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                B, W, N = X_batch.shape

                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)

                loss_sensor = sensor_criterion(sensor_probs, sensor_labels_batch).mean()
                loss_global = global_criterion(global_prob, window_labels_batch.float())

                # Separation loss (same as training)
                distances = torch.norm(embeddings - normal_center.unsqueeze(0), dim=1)
                normal_mask = window_labels_batch == 0
                loss_normal = (
                    distances[normal_mask].mean()
                    if normal_mask.any()
                    else torch.tensor(0.0, device=device)
                )
                anomalous_mask = window_labels_batch == 1
                loss_anomalous = (
                    -distances[anomalous_mask].mean()
                    if anomalous_mask.any()
                    else torch.tensor(0.0, device=device)
                )
                loss_separation_val = loss_normal + loss_anomalous

                # Correlation loss for validation
                normal_mask = window_labels_batch == 0
                if normal_mask.any():
                    X_norm = X_batch[normal_mask]
                    Xc = X_norm - X_norm.mean(dim=1, keepdim=True)
                    cov = (Xc.transpose(1, 2) @ Xc) / (W - 1 + 1e-8)
                    var = torch.diagonal(cov, dim1=1, dim2=2).unsqueeze(-1)
                    std = torch.sqrt(torch.clamp(var, min=1e-8))
                    denom = std @ std.transpose(1, 2)
                    corr = cov / (denom + 1e-8).clamp(-1.0, 1.0)

                    i_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
                    j_idx = topk_idx.to(device)
                    actual_corr_edges = corr[:, i_idx, j_idx]
                    target_corr_edges = topk_vals.unsqueeze(0).to(device)
                    loss_corr_val = (
                        (actual_corr_edges - target_corr_edges) ** 2
                    ).mean()
                else:
                    loss_corr_val = torch.tensor(0.0, device=device)

                loss = (
                    loss_sensor
                    + lambda_global * loss_global
                    + lambda_separation * loss_separation_val
                    + lambda_corr * loss_corr_val
                )
                val_loss += loss.item() * X_batch.size(0)
                val_loss_corr += loss_corr_val.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_corr /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # Compute separation metrics
        with torch.no_grad():
            sample_embeddings = []
            sample_labels = []
            for X_batch, _, _, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                embeddings = model.get_embeddings(X_batch)
                distances = torch.norm(embeddings - normal_center.unsqueeze(0), dim=1)
                sample_embeddings.append(distances.cpu())
                sample_labels.append(window_labels_batch)

            all_distances = torch.cat(sample_embeddings)
            all_labels = torch.cat(sample_labels)
            normal_distances = all_distances[all_labels == 0]
            anomalous_distances = all_distances[all_labels == 1]

            normal_mean = (
                normal_distances.mean().item() if len(normal_distances) > 0 else 0.0
            )
            anomalous_mean = (
                anomalous_distances.mean().item()
                if len(anomalous_distances) > 0
                else 0.0
            )
            separation_ratio = anomalous_mean / normal_mean if normal_mean > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Sensor: {train_loss_sensor:.4f} | "
            f"Global: {train_loss_global:.4f} | "
            f"Separation: {train_loss_separation:.4f} | "
            f"Corr: {train_loss_corr:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Sep_ratio: {separation_ratio:.2f}×"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normal_center": normal_center.data.cpu(),
                    "sensor_names": SENSOR_COLS,
                    "window_size": WINDOW_SIZE,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": model.sensor_embeddings.data.cpu(),
                    "lambda_separation": lambda_separation,
                    "lambda_global": lambda_global,
                    "lambda_corr": lambda_corr,
                    "final_separation_ratio": separation_ratio,
                    "normal_mean_distance": normal_mean,
                    "anomalous_mean_distance": anomalous_mean,
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "phase": 2,  # Mark as Phase 2 checkpoint
                },
                model_save_path,
            )
            print(f"  ✓ Best model saved to {model_save_path}")

    # Load best checkpoint
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normal_center.data = checkpoint["normal_center"].to(device)

    print(f"\n✓ Phase 2 training complete. Best model saved to: {model_save_path}\n")

    return model, normal_center


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train MultiLabelGDN with Center Loss + Correlation Loss (Phase 2)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase 1 checkpoint (best_multilabel_gdn_center.pt)",
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_separation",
        type=float,
        default=LAMBDA_SEPARATION,
        help="Separation loss weight",
    )
    parser.add_argument(
        "--lambda_global", type=float, default=LAMBDA_GLOBAL, help="Global loss weight"
    )
    parser.add_argument(
        "--lambda_corr", type=float, default=LAMBDA_CORR, help="Correlation loss weight"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--output",
        type=str,
        default="best_multilabel_gdn_center_corr.pt",
        help="Output checkpoint path",
    )
    args = parser.parse_args()

    # Validate Phase 1 checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {args.checkpoint}")

    # Device detection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load Phase 1 checkpoint
    print(f"\nLoading Phase 1 checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Validate checkpoint has required keys
    required_keys = [
        "model_state_dict",
        "normal_center",
        "sensor_names",
        "window_size",
        "embed_dim",
        "top_k",
        "hidden_dim",
    ]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: {key}")

    # Check separation ratio
    if "final_separation_ratio" in checkpoint:
        sep_ratio = checkpoint["final_separation_ratio"]
        print(f"Phase 1 separation ratio: {sep_ratio:.2f}×")
        if sep_ratio < 1.7:
            print(
                f"WARNING: Phase 1 separation ratio ({sep_ratio:.2f}×) is below recommended threshold (1.7×)"
            )
            print(
                "Phase 2 may not achieve optimal results. Consider retraining Phase 1."
            )
    else:
        print("WARNING: Phase 1 checkpoint does not contain separation ratio info")

    # Initialize model and normal center
    num_sensors = len(checkpoint["sensor_names"])
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=checkpoint["window_size"],
        embed_dim=checkpoint["embed_dim"],
        top_k=checkpoint["top_k"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    normal_center = nn.Parameter(checkpoint["normal_center"].to(device))

    # Load state dicts
    model.load_state_dict(checkpoint["model_state_dict"])

    print("✓ Phase 1 checkpoint loaded successfully")

    # Load and preprocess data (same as Phase 1)
    print(f"\nLoading data from {args.data_path}...")
    df_list = []
    for file in os.listdir(args.data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{args.data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)

    print(f"Loaded {len(df_list)} files")

    # Combine all dataframes
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data[ID_COL].nunique()}")

    # Preprocessing (same as Phase 1)
    print("\nPreprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ]
    )
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=[ID_COL]
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(
        data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=1
    )
    data = filter_long_drives(
        data, id_col=ID_COL, min_length=WINDOW_SIZE + FORECAST_HORIZON
    )
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15) - must match Phase 1 split
    print("\nSplitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)

    train_drives = unique_drives[: int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives) :]

    print(
        f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}"
    )

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    test_data = data[data[ID_COL].isin(test_drives)].copy()

    print(
        f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}"
    )

    # Build clean windows (must use same scaler as Phase 1 - but we'll refit for consistency)
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data,
        checkpoint["sensor_names"],
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=None,
    )
    X_val, y_val, _ = build_clean_windows(
        val_data,
        checkpoint["sensor_names"],
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=scaler_train,
    )
    X_test_clean, y_test_clean, _ = build_clean_windows(
        test_data,
        checkpoint["sensor_names"],
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=scaler_train,
    )

    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")
    print(f"Clean test windows: {len(X_test_clean)}")

    # Inject faults with sensor-level labels (same random seeds as Phase 1)
    print("\nInjecting faults with sensor-level labels (balanced distribution)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = (
        inject_faults_with_sensor_labels(
            X_train,
            y_train,
            checkpoint["sensor_names"],
            fault_percentage=0.15,
            random_state=42,
        )
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels = (
        inject_faults_with_sensor_labels(
            X_val,
            y_val,
            checkpoint["sensor_names"],
            fault_percentage=0.15,
            random_state=43,
        )
    )
    X_test_sensor, _, test_sensor_labels, test_window_labels = (
        inject_faults_with_sensor_labels(
            X_test_clean,
            y_test_clean,
            checkpoint["sensor_names"],
            fault_percentage=0.30,
            random_state=44,
        )
    )

    # Create dataloaders
    train_ds = TensorDataset(
        X_train_sensor, y_train, train_sensor_labels, train_window_labels
    )
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)
    test_ds = TensorDataset(
        X_test_sensor, y_test_clean, test_sensor_labels, test_window_labels
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    # Train model with correlation loss
    model, normal_center = train_model_with_correlation_loss(
        model,
        normal_center,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=device,
        lambda_separation=args.lambda_separation,
        lambda_global=args.lambda_global,
        lambda_corr=args.lambda_corr,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        model_save_path=args.output,
    )

    print("✓ Phase 2 training complete!")


if __name__ == "__main__":
    main()
