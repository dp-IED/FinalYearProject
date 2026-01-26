#!/usr/bin/env python3
"""
Training script for MultiLabelGDN with Center Loss with Repulsion.

Alternative implementation using CenterLossWithRepulsion which adds explicit
repulsion between centers to prevent collapse.

- Increased embedding dimensions (128 vs 32)
- Higher lambda_center (0.8 vs 0.1)
- Explicit center repulsion (target distance = 4.0)

Target: Center separation >2.0 (vs current 0.097)
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "anomaly-detection"))
from models.gdn_model import MultiLabelGDN, CenterLossWithRepulsion

torch.set_default_dtype(torch.float32)

# ============================================================================
# Constants
# ============================================================================

DATA_PATH = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata"
SENSOR_COLS = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
]
ID_COL = "drive_id"
TIME_COL = "ENGINE_RUN_TINE ()"
WINDOW_SIZE = 300
FORECAST_HORIZON = 1

# Training hyperparameters
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_GLOBAL = 0.3
LAMBDA_CENTER = 0.8  # Increased from 0.1

# Model architecture - INCREASED dimensions
EMBED_DIM = 128  # Increased from 32
TOP_K = 3
HIDDEN_DIM = 128  # Increased from 32

# Center Loss with Repulsion parameters
REPULSION_ALPHA = 0.5
REPULSION_BETA = 1.0  # Repulsion weight

# Warmup epochs (train with only BCE before adding metric losses)
WARMUP_EPOCHS = 0

# ============================================================================
# Data Preprocessing Functions (same as train_gdn_center_loss.py)
# ============================================================================

# Import all data preprocessing functions from train_gdn_triplet_center.py
# For brevity, we'll copy the essential ones here

def remove_zero_variance_columns(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
    """Remove columns with zero variance."""
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]

    std_df = df[cols_to_check].std()
    zero_variance_cols = std_df[std_df == 0].index.tolist()

    print(f"{len(zero_variance_cols)} columns with zero variance: {zero_variance_cols}")

    if len(zero_variance_cols) > 0:
        df = df.drop(columns=zero_variance_cols)

    return df


def mean_fill_missing_timestamps_and_remove_duplicates(
    df: pd.DataFrame, time_col: str, id_cols: list[str] = None
) -> pd.DataFrame:
    """Remove duplicate timestamps by averaging all numeric columns."""
    if id_cols is None:
        id_cols = []

    existing_id_cols = [col for col in id_cols if col in df.columns]
    group_cols = [time_col] + existing_id_cols

    agg_dict = {}
    for col in df.columns:
        if col not in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

    df_clean = df.groupby(group_cols, as_index=False).agg(agg_dict)
    return df_clean


def downsample(df, time_col, source_file_col, downsample_factor=2):
    """Downsample data by factor."""
    result_dfs = []

    for source_file in df[source_file_col].unique():
        file_df = df[df[source_file_col] == source_file].copy()

        if len(file_df) < downsample_factor * 2:
            continue

        file_df = file_df.sort_values(time_col).reset_index(drop=True)
        downsampled = file_df.iloc[::downsample_factor].copy()
        downsampled[time_col] = np.arange(len(downsampled)) * downsample_factor

        result_dfs.append(downsampled.reset_index(drop=True))

    return pd.concat(result_dfs, ignore_index=True)


def filter_long_drives(df, id_col="drive_id", min_length=608):
    """Keep only drives long enough for context window."""
    drive_lengths = df.groupby(id_col).size()
    valid_drives = drive_lengths[drive_lengths >= min_length].index

    print(f"Keeping {len(valid_drives)}/{df[id_col].nunique()} drives")
    print(f"Dropped {len(df) - df[df[id_col].isin(valid_drives)].shape[0]} timesteps")

    return df[df[id_col].isin(valid_drives)].reset_index(drop=True)


def add_cross_channel_features(data):
    """Engineer features that capture cross-channel relationships."""
    if "ENGINE_RPM ()" in data.columns and "VEHICLE_SPEED ()" in data.columns:
        data["RPM_SPEED_RATIO"] = data["ENGINE_RPM ()"] / (data["VEHICLE_SPEED ()"] + 1)

    if "THROTTLE ()" in data.columns and "ENGINE_LOAD ()" in data.columns:
        data["THROTTLE_LOAD_RATIO"] = data["THROTTLE ()"] / (data["ENGINE_LOAD ()"] + 1)

    if "VEHICLE_SPEED ()" in data.columns:
        data["IS_IDLE"] = (data["VEHICLE_SPEED ()"] < 5).astype(float)
        data["IS_HIGHWAY"] = (data["VEHICLE_SPEED ()"] > 60).astype(float)

    if "ENGINE_RPM ()" in data.columns:
        data["RPM_ACCEL"] = data.groupby("drive_id")["ENGINE_RPM ()"].diff().fillna(0)

    return data


def build_clean_windows(df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None):
    """Build windows from CLEAN data only. Returns normalized windows."""
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_list = [], []

    for drive_id, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, num_sensors = values.shape
        if T_ <= window_size + horizon:
            continue

        for t in range(T_ - window_size - horizon + 1):
            X_window = values[t : t + window_size]
            y_target = values[t + window_size + horizon - 1]
            X_list.append(X_window)
            y_list.append(y_target)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    return X, y, scaler


def inject_faults_with_sensor_labels(
    X_windows, y_windows, sensor_cols, fault_percentage=0.30, random_state=42
):
    """Inject faults and return SENSOR-LEVEL labels."""
    np.random.seed(random_state)

    N, W, D = X_windows.shape
    n_fault = max(1, int(N * fault_percentage))

    X_faulty = X_windows.clone()
    sensor_labels = torch.zeros(N, D, dtype=torch.float32)
    window_labels = torch.zeros(N, dtype=torch.long)

    fault_indices = np.random.choice(N, n_fault, replace=False)
    pid_idx = {name: i for i, name in enumerate(sensor_cols)}

    for idx in fault_indices:
        win = X_faulty[idx].numpy()

        fault_type = np.random.choice(
            ["vss_dropout", "maf_scale_low", "coolant_dropout", "tps_stuck", "rpm_speed_decouple"],
            p=[0.20, 0.20, 0.20, 0.20, 0.20],
        )

        affected_sensors = []

        if fault_type == "vss_dropout" and "VEHICLE_SPEED ()" in pid_idx:
            speed_i = pid_idx["VEHICLE_SPEED ()"]
            if win[:, speed_i].mean() > 0.15:
                start = int(W * 0.30)
                end = int(W * 0.70)
                win[start:end, speed_i] = 0.0
                win[start:end, speed_i] += np.random.uniform(0, 0.02, end - start)
                affected_sensors.append(speed_i)

        elif fault_type == "maf_scale_low" and "INTAKE_MANIFOLD_PRESSURE ()" in pid_idx:
            map_i = pid_idx["INTAKE_MANIFOLD_PRESSURE ()"]
            scale_factor = np.random.uniform(0.75, 0.80)
            win[:, map_i] = win[:, map_i] * scale_factor
            affected_sensors.append(map_i)

            if "SHORT_TERM_FUEL_TRIM_BANK_1 ()" in pid_idx:
                stft_i = pid_idx["SHORT_TERM_FUEL_TRIM_BANK_1 ()"]
                win[:, stft_i] = np.clip(win[:, stft_i] + 0.15, 0.0, 1.0)
                affected_sensors.append(stft_i)

        elif fault_type == "coolant_dropout" and "COOLANT_TEMPERATURE ()" in pid_idx:
            cool_i = pid_idx["COOLANT_TEMPERATURE ()"]
            if win[:, cool_i].mean() > 0.5:
                n_dropouts = np.random.randint(2, 4)
                for _ in range(n_dropouts):
                    drop_start = np.random.randint(0, W - 60)
                    drop_len = np.random.randint(30, 60)
                    win[drop_start : drop_start + drop_len, cool_i] = np.random.uniform(0.05, 0.15)
                affected_sensors.append(cool_i)

        elif fault_type == "tps_stuck" and "THROTTLE ()" in pid_idx:
            thr_i = pid_idx["THROTTLE ()"]
            freeze_point = W // 2
            stuck_value = win[freeze_point, thr_i]
            if stuck_value > 0.15 and win[:freeze_point, thr_i].std() > 0.05:
                win[freeze_point:, thr_i] = stuck_value
                affected_sensors.append(thr_i)

        elif fault_type == "rpm_speed_decouple":
            if "ENGINE_RPM ()" in pid_idx and "VEHICLE_SPEED ()" in pid_idx:
                speed_i = pid_idx["VEHICLE_SPEED ()"]
                rpm_i = pid_idx["ENGINE_RPM ()"]
                if win[:, speed_i].mean() > 0.20 and win[:, rpm_i].mean() > 0.30:
                    start = int(W * 0.25)
                    end = int(W * 0.75)
                    win[start:end, speed_i] = win[start:end, speed_i] * np.random.uniform(0.3, 0.5)
                    affected_sensors.append(speed_i)

        if len(affected_sensors) > 0:
            X_faulty[idx] = torch.tensor(win, dtype=torch.float32)
            window_labels[idx] = 1
            for sensor_i in affected_sensors:
                sensor_labels[idx, sensor_i] = 1.0

    return X_faulty, y_windows, sensor_labels, window_labels


# ============================================================================
# Training Function
# ============================================================================


def train_model(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    lambda_center=LAMBDA_CENTER,
    lambda_global=LAMBDA_GLOBAL,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    model_save_path="best_multilabel_gdn_center_repulsion.pt",
):
    """
    Train MultiLabelGDN with Center Loss with Repulsion.
    
    Returns:
        model: Trained model
        center_loss: Trained CenterLossWithRepulsion module
    """
    # Initialize model with increased dimensions
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=EMBED_DIM,
        top_k=TOP_K,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Initialize Center Loss with Repulsion
    center_loss = CenterLossWithRepulsion(
        embed_dim=HIDDEN_DIM,  # Use hidden_dim for embedding dimension
        num_classes=2,
        alpha=REPULSION_ALPHA,
        beta=REPULSION_BETA,
    ).to(device)

    # Separate optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_center = torch.optim.SGD(
        center_loss.parameters(), lr=0.1
    )  # Reduced from 5.0 to 0.1 for stability

    # Loss functions - use BCEWithLogitsLoss for numerical stability
    sensor_criterion = nn.BCEWithLogitsLoss(reduction="none")
    global_criterion = nn.BCEWithLogitsLoss()

    # Learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scheduler_center = torch.optim.lr_scheduler.StepLR(
        optimizer_center, step_size=10, gamma=0.5
    )

    best_val_loss = float("inf")

    print(f"\n{'='*80}")
    print("Training Multi-Label GDN with Center Loss with Repulsion")
    print(f"{'='*80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Lambda_global: {lambda_global}, Lambda_center: {lambda_center}")
    print(f"Repulsion alpha: {REPULSION_ALPHA}, Repulsion beta: {REPULSION_BETA}")
    print(f"Target center distance: 4.0")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()
        center_loss.train()

        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_center = 0.0

        # Progressive loss weighting
        def get_loss_weights(epoch, total_epochs):
            """Progressive loss weight ramping for stability"""
            if epoch < WARMUP_EPOCHS:
                return {"global": lambda_global, "center": 0.0}
            elif epoch < WARMUP_EPOCHS + 10:
                # Gradual introduction over 10 epochs
                progress = (epoch - WARMUP_EPOCHS) / 10.0
                return {
                    "global": lambda_global,
                    "center": lambda_center * progress,
                }
            else:
                # Full weights after warmup + ramp
                return {
                    "global": lambda_global,
                    "center": lambda_center * 0.625,  # Reduced from 0.8 to 0.5
                }

        weights = get_loss_weights(epoch, num_epochs)
        use_metric_losses = weights["center"] > 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
            for X_batch, _, sensor_labels_batch, window_labels_batch in pbar:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                # Zero gradients
                optimizer.zero_grad()
                optimizer_center.zero_grad()

                # Forward pass (returns logits, not probabilities)
                sensor_logits, global_logits = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)  # (B, hidden_dim)

                # Check for NaN in forward pass
                if (
                    torch.isnan(sensor_logits).any()
                    or torch.isnan(embeddings).any()
                    or torch.isnan(global_logits).any()
                ):
                    print(f"NaN detected in forward pass at epoch {epoch+1}, skipping batch")
                    continue

                # Classification losses (logits input - no clamping needed)
                loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch).mean()
                loss_global = global_criterion(global_logits, window_labels_batch.float())

                # Center loss with repulsion (with stability checks)
                if use_metric_losses:
                    loss_center_val = center_loss(embeddings, window_labels_batch)
                    
                    # Check for NaN
                    if torch.isnan(loss_center_val):
                        print(f"NaN in center loss at epoch {epoch+1}, skipping batch")
                        continue
                else:
                    loss_center_val = torch.tensor(0.0, device=device)

                # Combined loss with progressive weights
                loss = (
                    loss_sensor
                    + weights["global"] * loss_global
                    + weights["center"] * loss_center_val
                )

                # Check for NaN in total loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf in total loss at epoch {epoch+1}, skipping batch")
                    continue

                loss.backward()

                # CRITICAL: Gradient clipping for both model and centers
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_metric_losses:
                    torch.nn.utils.clip_grad_norm_(
                        center_loss.parameters(), max_norm=0.5
                    )  # Stricter for centers

                # Update model and centers
                optimizer.step()
                if use_metric_losses:
                    optimizer_center.step()

                train_loss_sensor += loss_sensor.item() * X_batch.size(0)
                train_loss_global += loss_global.item() * X_batch.size(0)
                if use_metric_losses:
                    train_loss_center += loss_center_val.item() * X_batch.size(0)

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        if use_metric_losses:
            train_loss_center /= len(train_loader.dataset)

        # Validation
        model.eval()
        center_loss.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                sensor_logits, global_logits = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)

                loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch).mean()
                loss_global = global_criterion(global_logits, window_labels_batch.float())

                if use_metric_losses:
                    loss_center_val = center_loss(embeddings, window_labels_batch)
                else:
                    loss_center_val = torch.tensor(0.0, device=device)

                loss = (
                    loss_sensor
                    + weights["global"] * loss_global
                    + weights["center"] * loss_center_val
                )
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if use_metric_losses:
            scheduler_center.step()

        # Compute center separation
        center_separation = center_loss.get_center_separation()

        # Compute embedding distance statistics
        with torch.no_grad():
            sample_embeddings = []
            sample_labels = []
            for X_batch, _, _, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                embeddings = model.get_embeddings(X_batch)
                # Distance to normal center (class 0)
                normal_center = center_loss.centers[0]
                distances = torch.norm(embeddings - normal_center.unsqueeze(0), dim=1)
                sample_embeddings.append(distances.cpu())
                sample_labels.append(window_labels_batch)

            all_distances = torch.cat(sample_embeddings)
            all_labels = torch.cat(sample_labels)
            normal_distances = all_distances[all_labels == 0]
            anomalous_distances = all_distances[all_labels == 1]

            normal_mean = normal_distances.mean().item() if len(normal_distances) > 0 else 0.0
            anomalous_mean = (
                anomalous_distances.mean().item() if len(anomalous_distances) > 0 else 0.0
            )

        phase_str = "[WARMUP]" if not use_metric_losses else ""
        print(
            f"Epoch {epoch+1}/{num_epochs} {phase_str} | "
            f"Sensor: {train_loss_sensor:.4f} | "
            f"Global: {train_loss_global:.4f} | "
            f"Center: {train_loss_center:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Center_dist: {center_separation:.4f} | "
            f"Normal_mean: {normal_mean:.4f} | "
            f"Anomalous_mean: {anomalous_mean:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "center_loss_state_dict": center_loss.state_dict(),
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": model.sensor_embeddings.data.cpu(),
                    "lambda_center": lambda_center,
                    "lambda_global": lambda_global,
                    "center_separation": center_separation,
                    "normal_mean_distance": normal_mean,
                    "anomalous_mean_distance": anomalous_mean,
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                },
                model_save_path,
            )
            print(f"  ✓ Best model saved (center_separation: {center_separation:.4f})")

    # Load best checkpoint
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    center_loss.load_state_dict(checkpoint["center_loss_state_dict"])

    print(f"\n✓ Training complete. Best model saved to: {model_save_path}")
    print(f"  Final center separation: {checkpoint['center_separation']:.4f}")
    print(f"  Normal mean distance: {checkpoint['normal_mean_distance']:.4f}")
    print(f"  Anomalous mean distance: {checkpoint['anomalous_mean_distance']:.4f}\n")

    return model, center_loss


# ============================================================================
# Main Function (same structure as train_gdn_triplet_center.py)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train MultiLabelGDN with Center Loss with Repulsion"
    )
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_center", type=float, default=LAMBDA_CENTER, help="Center loss weight"
    )
    parser.add_argument("--lambda_global", type=float, default=LAMBDA_GLOBAL, help="Global loss weight")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--output", type=str, default="best_multilabel_gdn_center_repulsion.pt", help="Output checkpoint path"
    )
    args = parser.parse_args()

    # Device detection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load data (same as train_gdn_triplet_center.py)
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
        columns=["WARM_UPS_SINCE_CODES_CLEARED ()", "TIME_SINCE_TROUBLE_CODES_CLEARED ()"]
    )
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=[ID_COL]
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=1)
    data = filter_long_drives(data, id_col=ID_COL, min_length=WINDOW_SIZE + FORECAST_HORIZON)
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15)
    print("\nSplitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)

    train_drives = unique_drives[: int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives) :]

    print(f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}")

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    test_data = data[data[ID_COL].isin(test_drives)].copy()

    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")

    # Build clean windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )
    X_test_clean, y_test_clean, _ = build_clean_windows(
        test_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")
    print(f"Clean test windows: {len(X_test_clean)}")

    # Inject faults with sensor-level labels
    print("\nInjecting faults with sensor-level labels (balanced distribution)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = inject_faults_with_sensor_labels(
        X_train, y_train, SENSOR_COLS, fault_percentage=0.15, random_state=42
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels = inject_faults_with_sensor_labels(
        X_val, y_val, SENSOR_COLS, fault_percentage=0.15, random_state=43
    )
    X_test_sensor, _, test_sensor_labels, test_window_labels = inject_faults_with_sensor_labels(
        X_test_clean, y_test_clean, SENSOR_COLS, fault_percentage=0.30, random_state=44
    )

    # Statistics
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()
    test_faulty = (test_sensor_labels.sum(dim=1) > 0).sum().item()

    print(f"\nTrain: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"  Avg sensors per fault: {train_sensor_labels[train_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")
    print(f"  Avg sensors per fault: {val_sensor_labels[val_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}")
    print(f"Test:  {test_faulty}/{len(X_test_sensor)} faulty windows")
    print(f"  Avg sensors per fault: {test_sensor_labels[test_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}")

    # Create dataloaders
    train_ds = TensorDataset(X_train_sensor, y_train, train_sensor_labels, train_window_labels)
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)
    test_ds = TensorDataset(X_test_sensor, y_test_clean, test_sensor_labels, test_window_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    num_sensors = len(SENSOR_COLS)
    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    # Train model
    model, center_loss = train_model(
        train_loader,
        val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        num_epochs=args.epochs,
        device=device,
        lambda_center=args.lambda_center,
        lambda_global=args.lambda_global,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        model_save_path=args.output,
    )

    print("✓ Training complete!")


if __name__ == "__main__":
    main()
