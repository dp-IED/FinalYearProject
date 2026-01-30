#!/usr/bin/env python3
"""
Retrain GDN with improved hyperparameters to fix false positive problem.

Key improvements:
1. Weighted loss for imbalanced data (pos_weight based on imbalance ratio)
2. Higher embedding dimension (64 instead of 32)
3. More training epochs with early stopping
4. Learning rate scheduling
5. Better regularization

Usage:
    python retrain_gdn_improved.py --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_improved.pt
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path
import json

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "anomaly-detection"))

from models.gdn_model import MultiLabelGDN
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
)

torch.set_default_dtype(torch.float32)


def train_gdn_improved(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    num_epochs=150,
    device="cpu",
    lambda_separation=1.0,
    lambda_global=0.3,
    learning_rate=0.001,
    weight_decay=1e-4,
    embed_dim=64,  # Increased from 32
    hidden_dim=64,  # Increased from 32
    top_k=3,
    pos_weight=None,  # Will be computed from data if None
    early_stop_patience=20,
    model_save_path="best_multilabel_gdn_improved.pt",
):
    """
    Train MultiLabelGDN with improved hyperparameters.

    Key improvements:
    - Weighted loss for imbalanced data
    - Higher embedding dimension
    - Early stopping
    - Learning rate scheduling

    Returns:
        model: Trained model
        normal_center: Learned normal center parameter
        history: Training history dictionary
    """
    # Initialize model with improved architecture
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=top_k,
        hidden_dim=hidden_dim,
    ).to(device)

    # Learnable normal center for separation loss
    normal_center = nn.Parameter(torch.randn(hidden_dim) * 0.1).to(device)

    # Single optimizer (includes normal center)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [normal_center],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # CRITICAL: Weighted loss for imbalanced data
    if pos_weight is None:
        # Compute pos_weight from training data
        total_samples = 0
        positive_samples = 0
        for _, _, sensor_labels_batch, window_labels_batch in train_loader:
            total_samples += len(window_labels_batch)
            positive_samples += (window_labels_batch == 1).sum().item()

        if positive_samples > 0:
            pos_weight_value = (total_samples - positive_samples) / positive_samples
        else:
            pos_weight_value = 1.0
        print(f"   Computed pos_weight: {pos_weight_value:.2f}")
    else:
        pos_weight_value = pos_weight

    # Weighted BCE loss for sensor classification
    sensor_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value]).to(device)
    )
    global_criterion = nn.BCELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    # Training history
    history = {
        "train_loss": [],
        "train_loss_sensor": [],
        "train_loss_global": [],
        "train_loss_separation": [],
        "val_loss": [],
        "val_f1": [],
        "val_auc": [],
        "separation_ratio": [],
        "learning_rates": [],
    }

    best_val_f1 = 0.0
    patience_counter = 0

    print(f"\n{'=' * 80}")
    print("Training Improved Multi-Label GDN")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  Embedding dim: {embed_dim} (increased from 32)")
    print(f"  Hidden dim: {hidden_dim} (increased from 32)")
    print(f"  Pos weight: {pos_weight_value:.2f} (for imbalanced data)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda_global: {lambda_global}, Lambda_separation: {lambda_separation}")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()

        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_separation = 0.0
        train_loss_total = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            for X_batch, _, sensor_labels_batch, window_labels_batch in pbar:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)  # (B, hidden_dim)

                # Classification losses (use logits for weighted BCE)
                # Convert probabilities to logits safely
                sensor_probs_clamped = sensor_probs.clamp(1e-7, 1 - 1e-7)
                sensor_logits = torch.log(
                    sensor_probs_clamped / (1 - sensor_probs_clamped)
                )
                loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                loss_global = global_criterion(global_prob, window_labels_batch.float())

                # Separation loss: minimize normal distance, maximize anomalous distance
                distances = torch.norm(
                    embeddings - normal_center.unsqueeze(0), dim=1
                )  # (B,)

                # Normal samples: minimize distance (pull closer to center)
                normal_mask = window_labels_batch == 0
                loss_normal = (
                    distances[normal_mask].mean()
                    if normal_mask.any()
                    else torch.tensor(0.0, device=device)
                )

                # Anomalous samples: maximize distance (push away from center)
                anomalous_mask = window_labels_batch == 1
                loss_anomalous = (
                    -distances[anomalous_mask].mean()
                    if anomalous_mask.any()
                    else torch.tensor(0.0, device=device)
                )

                # Separation loss: normal close + anomalous far
                loss_separation = loss_normal + loss_anomalous

                # Combined loss
                loss = (
                    loss_sensor
                    + lambda_global * loss_global
                    + lambda_separation * loss_separation
                )

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update model
                optimizer.step()

                train_loss_sensor += loss_sensor.item() * X_batch.size(0)
                train_loss_global += loss_global.item() * X_batch.size(0)
                train_loss_separation += loss_separation.item() * X_batch.size(0)
                train_loss_total += loss.item() * X_batch.size(0)

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        train_loss_separation /= len(train_loader.dataset)
        train_loss_total /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_probs = []
        all_val_labels = []
        all_val_distances = []
        all_val_window_labels = []

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)

                # Losses
                sensor_probs_clamped = sensor_probs.clamp(1e-7, 1 - 1e-7)
                sensor_logits = torch.log(
                    sensor_probs_clamped / (1 - sensor_probs_clamped)
                )
                loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                loss_global = global_criterion(global_prob, window_labels_batch.float())

                # Separation loss
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
                loss_separation = loss_normal + loss_anomalous

                loss = (
                    loss_sensor
                    + lambda_global * loss_global
                    + lambda_separation * loss_separation
                )
                val_loss += loss.item() * X_batch.size(0)

                # Collect predictions for metrics
                all_val_probs.append(global_prob.cpu().numpy())
                all_val_labels.append(window_labels_batch.numpy())
                all_val_distances.append(distances.cpu().numpy())
                all_val_window_labels.append(window_labels_batch.numpy())

        val_loss /= len(val_loader.dataset)

        # Compute metrics
        val_probs = np.concatenate(all_val_probs)
        val_labels = np.concatenate(all_val_labels)
        val_preds = (val_probs > 0.5).astype(int)

        val_f1 = f1_score(val_labels, val_preds)
        val_auc = (
            roc_auc_score(val_labels, val_probs)
            if len(np.unique(val_labels)) > 1
            else 0.0
        )

        # Compute separation ratio
        all_distances = np.concatenate(all_val_distances)
        all_window_labels = np.concatenate(all_val_window_labels)
        normal_distances = all_distances[all_window_labels == 0]
        anomalous_distances = all_distances[all_window_labels == 1]

        normal_mean = normal_distances.mean() if len(normal_distances) > 0 else 0.0
        anomalous_mean = (
            anomalous_distances.mean() if len(anomalous_distances) > 0 else 0.0
        )
        separation_ratio = anomalous_mean / normal_mean if normal_mean > 0 else 0.0

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss_total)
        history["train_loss_sensor"].append(train_loss_sensor)
        history["train_loss_global"].append(train_loss_global)
        history["train_loss_separation"].append(train_loss_separation)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["separation_ratio"].append(separation_ratio)
        history["learning_rates"].append(current_lr)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss_total:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Sep Ratio: {separation_ratio:.2f}× | "
                f"LR: {current_lr:.6f}"
            )

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "normal_center": normal_center.data.cpu(),
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": embed_dim,
                    "top_k": top_k,
                    "hidden_dim": hidden_dim,
                    "sensor_embeddings": model.sensor_embeddings.data.cpu(),
                    "lambda_separation": lambda_separation,
                    "lambda_global": lambda_global,
                    "pos_weight": pos_weight_value,
                    "final_separation_ratio": separation_ratio,
                    "normal_mean_distance": normal_mean,
                    "anomalous_mean_distance": anomalous_mean,
                    "best_val_f1": best_val_f1,
                    "best_val_auc": val_auc,
                    "epoch": epoch + 1,
                },
                model_save_path,
            )
            print(f"  ✓ New best model saved (F1={best_val_f1:.4f}, AUC={val_auc:.4f})")
        else:
            patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    # Load best checkpoint
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    normal_center.data = checkpoint["normal_center"].to(device)

    print(f"\n✓ Training complete. Best model saved to: {model_save_path}")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Final Separation Ratio: {checkpoint['final_separation_ratio']:.2f}×")

    return model, normal_center, history


def main():
    parser = argparse.ArgumentParser(
        description="Retrain GDN with improved hyperparameters"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/carOBD/obdiidata",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anomaly-detection/best_multilabel_gdn_improved.pt",
        help="Output model path",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--lambda-separation", type=float, default=1.0, help="Separation loss weight"
    )
    parser.add_argument(
        "--lambda-global", type=float, default=0.3, help="Global loss weight"
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive class weight (auto-computed if None)",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--fault-percentage",
        type=float,
        default=0.15,
        help="Fault injection percentage",
    )

    args = parser.parse_args()

    # Device detection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load data
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

    # Preprocessing
    print("\nPreprocessing data...")
    data = data.drop(
        columns=["WARM_UPS_SINCE_CODES_CLEARED ()"]
        if "WARM_UPS_SINCE_CODES_CLEARED ()" in data.columns
        else []
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL, TIME_COL])
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, TIME_COL, id_cols=[ID_COL]
    )
    data = downsample(data, TIME_COL, ID_COL, downsample_factor=2)
    data = filter_long_drives(data, ID_COL, min_length=WINDOW_SIZE)
    data = add_cross_channel_features(data)

    # Train/val/test split (80/10/10)
    drive_ids = data[ID_COL].unique()
    np.random.seed(42)
    np.random.shuffle(drive_ids)

    n_train = int(0.8 * len(drive_ids))
    n_val = int(0.1 * len(drive_ids))

    train_drives = drive_ids[:n_train]
    val_drives = drive_ids[n_train : n_train + n_val]
    test_drives = drive_ids[n_train + n_val :]

    train_data = data[data[ID_COL].isin(train_drives)].reset_index(drop=True)
    val_data = data[data[ID_COL].isin(val_drives)].reset_index(drop=True)
    test_data = data[data[ID_COL].isin(test_drives)].reset_index(drop=True)

    print(f"\nTrain drives: {len(train_drives)}")
    print(f"Val drives: {len(val_drives)}")
    print(f"Test drives: {len(test_drives)}")

    # Build windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )
    X_test, y_test, _ = build_clean_windows(
        test_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Train windows: {len(X_train)}")
    print(f"Val windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)}")

    # Inject faults
    print(f"\nInjecting faults ({args.fault_percentage * 100:.1f}% of windows)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = (
        inject_faults_with_sensor_labels(
            X_train,
            y_train,
            SENSOR_COLS,
            fault_percentage=args.fault_percentage,
            random_state=42,
        )
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels = (
        inject_faults_with_sensor_labels(
            X_val,
            y_val,
            SENSOR_COLS,
            fault_percentage=args.fault_percentage,
            random_state=43,
        )
    )
    X_test_sensor, _, test_sensor_labels, test_window_labels = (
        inject_faults_with_sensor_labels(
            X_test, y_test, SENSOR_COLS, fault_percentage=0.30, random_state=44
        )
    )

    # Statistics
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()
    test_faulty = (test_sensor_labels.sum(dim=1) > 0).sum().item()

    print(f"\nTrain: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")
    print(f"Test:  {test_faulty}/{len(X_test_sensor)} faulty windows")

    # Create dataloaders
    train_ds = TensorDataset(
        X_train_sensor, y_train, train_sensor_labels, train_window_labels
    )
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)
    test_ds = TensorDataset(
        X_test_sensor, y_test, test_sensor_labels, test_window_labels
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Train model
    model, normal_center, history = train_gdn_improved(
        train_loader,
        val_loader,
        num_sensors=len(SENSOR_COLS),
        window_size=WINDOW_SIZE,
        num_epochs=args.epochs,
        device=device,
        lambda_separation=args.lambda_separation,
        lambda_global=args.lambda_global,
        learning_rate=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        pos_weight=args.pos_weight,
        early_stop_patience=args.early_stop_patience,
        model_save_path=args.output,
    )

    # Save training history
    history_path = args.output.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print(f"\n{'=' * 80}")
    print("Training complete!")
    print(f"{'=' * 80}")
    print(f"Model saved to: {args.output}")
    print(f"History saved to: {history_path}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate improved model: python evaluate_gdn.py --model {args.output}")
    print(f"  2. Compare to original: python diagnose_gdn.py")


if __name__ == "__main__":
    main()
