#!/usr/bin/env python3
"""
Stage 3: Fine-tuning (Task-Specific)
Training script for GDN with classification loss + small forecast regularization.

Objective: Calibrate for final classification task while maintaining graph structure
with small forecast weight (0.1 * L_forecast).
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
from models.gdn_model import MultiLabelGDN

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
NUM_EPOCHS = 25  # Default: 20-30 epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
LAMBDA_GLOBAL = 0.3
LAMBDA_FORECAST = 0.1  # Small forecast weight to maintain structure

# Model architecture (must match previous stages)
EMBED_DIM = 64
TOP_K = 3
HIDDEN_DIM = 64

# ============================================================================
# Model with Forecasting Head (reused from Stage 1)
# ============================================================================


class GDNWithForecasting(nn.Module):
    """
    MultiLabelGDN extended with forecasting head.
    Predicts next timestep sensor values from current window.
    """

    def __init__(self, base_model: MultiLabelGDN):
        super().__init__()
        self.base_model = base_model
        num_nodes = base_model.num_nodes
        hidden_dim = base_model.hidden_dim

        # Forecasting head: predicts next timestep sensor values
        self.forecasting_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_nodes),  # Predict each sensor's next value
        )

    def forward(self, x, return_forecast=False, return_global=False):
        """
        Forward pass through base model.

        Args:
            x: (B, W, N) input tensor
            return_forecast: If True, also return forecasted next timestep
            return_global: If True, also return global window logits

        Returns:
            - sensor_logits: (B, N) logits for each sensor
            - global_logits: (B,) logits for window (optional)
            - forecast: (B, N) predicted next timestep values (optional)
        """
        # Get base model outputs
        if return_global:
            sensor_logits, global_logits = self.base_model(x, return_global=True)
        else:
            sensor_logits = self.base_model(x)
            global_logits = None

        if return_forecast:
            # Get embeddings and predict next timestep
            embeddings = self.base_model.get_embeddings(x)  # (B, hidden_dim)
            forecast = self.forecasting_head(embeddings)  # (B, N)
            if return_global:
                return sensor_logits, global_logits, forecast
            return sensor_logits, forecast

        if return_global:
            return sensor_logits, global_logits
        return sensor_logits

    def get_embeddings(self, x):
        """Get embeddings from base model."""
        return self.base_model.get_embeddings(x)


# ============================================================================
# Data Preprocessing Functions (reused from previous stages)
# ============================================================================


def remove_zero_variance_columns(
    df: pd.DataFrame, exclude_cols: list[str] = None
) -> pd.DataFrame:
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


def build_clean_windows(
    df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None
):
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


def build_forecast_windows(
    df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None
):
    """
    Build windows for forecasting: window[t] predicts window[t+1]'s last timestep.
    """
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_forecast_list = [], []

    for drive_id, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, num_sensors = values.shape
        if T_ <= window_size + horizon:
            continue

        for t in range(T_ - window_size - horizon):
            X_window = values[t : t + window_size]
            y_target = values[t + window_size]
            X_list.append(X_window)
            y_forecast_list.append(y_target)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y_forecast = torch.tensor(np.stack(y_forecast_list), dtype=torch.float32)
    return X, y_forecast, scaler


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
            [
                "vss_dropout",
                "maf_scale_low",
                "coolant_dropout",
                "tps_stuck",
                "rpm_speed_decouple",
            ],
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
                    win[drop_start : drop_start + drop_len, cool_i] = np.random.uniform(
                        0.05, 0.15
                    )
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
                    win[start:end, speed_i] = win[
                        start:end, speed_i
                    ] * np.random.uniform(0.3, 0.5)
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


def train_stage3(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    stage2_checkpoint_path,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    lambda_global=LAMBDA_GLOBAL,
    lambda_forecast=LAMBDA_FORECAST,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    checkpoint_dir="checkpoints",
):
    """
    Fine-tune GDN for classification task with small forecast regularization.

    Args:
        stage2_checkpoint_path: Path to Stage 2 checkpoint to load

    Returns:
        model: Fine-tuned model
    """
    # Load Stage 2 checkpoint
    print(f"\nLoading Stage 2 checkpoint from {stage2_checkpoint_path}...")
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location=device)

    # Initialize base model
    base_model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=EMBED_DIM,
        top_k=TOP_K,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Load Stage 2 model state
    base_model.load_state_dict(stage2_checkpoint["model_state_dict"])

    # Wrap with forecasting head (for forecast regularization)
    model = GDNWithForecasting(base_model).to(device)

    # CRITICAL: Unfreeze all parameters for fine-tuning
    print("\nUnfreezing all parameters for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    print(f"  All parameters trainable: {all(p.requires_grad for p in model.parameters())}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Loss functions
    sensor_criterion = nn.BCEWithLogitsLoss(reduction="none")
    global_criterion = nn.BCEWithLogitsLoss()
    forecast_criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    best_val_clf_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "stage3_best_finetune.pt")

    print(f"\n{'=' * 80}")
    print("Stage 3: Fine-tuning (Task-Specific)")
    print(f"{'=' * 80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Lambda_global: {lambda_global}, Lambda_forecast: {lambda_forecast}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_forecast = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            for batch in pbar:
                # Handle different batch formats
                if len(batch) == 5:
                    # (X, y, sensor_labels, window_labels, y_forecast)
                    X_batch, _, sensor_labels_batch, window_labels_batch, y_forecast_batch = batch
                else:
                    # Fallback: assume no forecast target
                    X_batch, _, sensor_labels_batch, window_labels_batch = batch
                    y_forecast_batch = None

                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                if y_forecast_batch is not None:
                    sensor_logits, global_logits, forecast = model(
                        X_batch, return_forecast=True, return_global=True
                    )
                    y_forecast_batch = y_forecast_batch.to(device)
                else:
                    sensor_logits, global_logits = model(
                        X_batch, return_forecast=False, return_global=True
                    )
                    forecast = None

                # Classification losses (main objective)
                loss_sensor = sensor_criterion(
                    sensor_logits, sensor_labels_batch
                ).mean()
                loss_global = global_criterion(
                    global_logits, window_labels_batch.float()
                )

                # Forecasting loss (small regularization)
                if forecast is not None and y_forecast_batch is not None:
                    loss_forecast = forecast_criterion(forecast, y_forecast_batch)
                else:
                    loss_forecast = torch.tensor(0.0, device=device)

                # Combined loss: L_BCE + 0.1*L_forecast
                loss = (
                    loss_sensor
                    + lambda_global * loss_global
                    + lambda_forecast * loss_forecast
                )

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf in loss at epoch {epoch + 1}, skipping batch")
                    continue

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update
                optimizer.step()

                train_loss_sensor += loss_sensor.item() * X_batch.size(0)
                train_loss_global += loss_global.item() * X_batch.size(0)
                if forecast is not None:
                    train_loss_forecast += loss_forecast.item() * X_batch.size(0)

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        if forecast is not None:
            train_loss_forecast /= len(train_loader.dataset)
        else:
            train_loss_forecast = 0.0

        # Validation
        model.eval()
        val_loss_sensor = 0.0
        val_loss_global = 0.0
        val_loss_forecast = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 5:
                    X_batch, _, sensor_labels_batch, window_labels_batch, y_forecast_batch = batch
                else:
                    X_batch, _, sensor_labels_batch, window_labels_batch = batch
                    y_forecast_batch = None

                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                if y_forecast_batch is not None:
                    sensor_logits, global_logits, forecast = model(
                        X_batch, return_forecast=True, return_global=True
                    )
                    y_forecast_batch = y_forecast_batch.to(device)
                else:
                    sensor_logits, global_logits = model(
                        X_batch, return_forecast=False, return_global=True
                    )
                    forecast = None

                loss_sensor = sensor_criterion(
                    sensor_logits, sensor_labels_batch
                ).mean()
                loss_global = global_criterion(
                    global_logits, window_labels_batch.float()
                )

                if forecast is not None and y_forecast_batch is not None:
                    loss_forecast = forecast_criterion(forecast, y_forecast_batch)
                else:
                    loss_forecast = torch.tensor(0.0, device=device)

                val_loss_sensor += loss_sensor.item() * X_batch.size(0)
                val_loss_global += loss_global.item() * X_batch.size(0)
                if forecast is not None:
                    val_loss_forecast += loss_forecast.item() * X_batch.size(0)

        val_loss_sensor /= len(val_loader.dataset)
        val_loss_global /= len(val_loader.dataset)
        if forecast is not None:
            val_loss_forecast /= len(val_loader.dataset)
        else:
            val_loss_forecast = 0.0

        # Classification loss (main metric)
        val_clf_loss = val_loss_sensor + lambda_global * val_loss_global

        # Update scheduler
        scheduler.step(val_clf_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Sensor: {train_loss_sensor:.4f} | "
            f"Global: {train_loss_global:.4f} | "
            f"Forecast: {train_loss_forecast:.6f} | "
            f"Val CLF: {val_clf_loss:.4f} | "
            f"Val Forecast: {val_loss_forecast:.6f}"
        )

        # Save best model (based on classification loss)
        if val_clf_loss < best_val_clf_loss:
            best_val_clf_loss = val_clf_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": base_model.sensor_embeddings.data.cpu(),
                    "lambda_global": lambda_global,
                    "lambda_forecast": lambda_forecast,
                    "epoch": epoch + 1,
                    "best_val_clf_loss": val_clf_loss,
                    "stage": 3,
                },
                best_checkpoint_path,
            )
            print(f"  ✓ Best model saved (Val CLF Loss: {val_clf_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no improvement for {max_patience} epochs)"
            )
            break

    # Load best checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n{'=' * 80}")
    print(f"Stage 3 training complete!")
    print(f"Best validation CLF loss: {checkpoint['best_val_clf_loss']:.4f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"{'=' * 80}\n")

    return model


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Task-Specific Fine-tuning"
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 2 checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_global", type=float, default=LAMBDA_GLOBAL, help="Global loss weight"
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=LAMBDA_FORECAST,
        help="Forecast loss weight",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--use_forecast",
        action="store_true",
        help="Use forecast regularization (requires forecast targets in data)",
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
        data, id_col=ID_COL, min_length=WINDOW_SIZE + FORECAST_HORIZON + 1
    )
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15) - same as previous stages
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

    # Build clean windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")

    # Inject faults with sensor-level labels
    print("\nInjecting faults with sensor-level labels...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = (
        inject_faults_with_sensor_labels(
            X_train, y_train, SENSOR_COLS, fault_percentage=0.15, random_state=42
        )
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels = (
        inject_faults_with_sensor_labels(
            X_val, y_val, SENSOR_COLS, fault_percentage=0.15, random_state=43
        )
    )

    # Build forecast targets if needed
    if args.use_forecast:
        print("\nBuilding forecast targets...")
        _, y_train_forecast, _ = build_forecast_windows(
            train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
        )
        _, y_val_forecast, _ = build_forecast_windows(
            val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
        )
        # Align forecast targets with fault-injected windows (take first N)
        min_len = min(len(X_train_sensor), len(y_train_forecast))
        y_train_forecast = y_train_forecast[:min_len]
        X_train_sensor = X_train_sensor[:min_len]
        train_sensor_labels = train_sensor_labels[:min_len]
        train_window_labels = train_window_labels[:min_len]

        min_len_val = min(len(X_val_sensor), len(y_val_forecast))
        y_val_forecast = y_val_forecast[:min_len_val]
        X_val_sensor = X_val_sensor[:min_len_val]
        val_sensor_labels = val_sensor_labels[:min_len_val]
        val_window_labels = val_window_labels[:min_len_val]
    else:
        y_train_forecast = None
        y_val_forecast = None

    # Statistics
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()

    print(f"\nTrain: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")

    # Create dataloaders
    if args.use_forecast and y_train_forecast is not None:
        train_ds = TensorDataset(
            X_train_sensor,
            y_train,
            train_sensor_labels,
            train_window_labels,
            y_train_forecast,
        )
        val_ds = TensorDataset(
            X_val_sensor, y_val, val_sensor_labels, val_window_labels, y_val_forecast
        )
    else:
        train_ds = TensorDataset(
            X_train_sensor, y_train, train_sensor_labels, train_window_labels
        )
        val_ds = TensorDataset(
            X_val_sensor, y_val, val_sensor_labels, val_window_labels
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_sensors = len(SENSOR_COLS)
    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    # Train model
    model = train_stage3(
        train_loader,
        val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        stage2_checkpoint_path=args.stage2_checkpoint,
        num_epochs=args.epochs,
        device=device,
        lambda_global=args.lambda_global,
        lambda_forecast=args.lambda_forecast if args.use_forecast else 0.0,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("✓ Stage 3 training complete!")


if __name__ == "__main__":
    main()
