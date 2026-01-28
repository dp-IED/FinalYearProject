#!/usr/bin/env python3
"""
Diagnostic script to assess Stage 2 multi-level center loss training.

Checks:
1. Embedding compactness (distance to assigned centers)
2. Separation / compactness ratio
3. Per-sensor separations (with high precision)
4. Whether clusters are actually forming
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
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "anomaly-detection"))
from models.gdn_model import MultiLabelGDN
from models.multi_level_center_loss import MultiLevelCenterLoss

# Constants (must match training script)
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
EMBED_DIM = 64
TOP_K = 3
HIDDEN_DIM = 64


def remove_zero_variance_columns(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
    """Remove columns with zero variance."""
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
    std_df = df[cols_to_check].std()
    zero_variance_cols = std_df[std_df == 0].index.tolist()
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


# Import fault injection
training_dir = Path(__file__).parent.parent / "training"
sys.path.insert(0, str(training_dir))
from fault_injection import inject_faults_with_sensor_labels


def diagnose_compactness(
    checkpoint_path,
    data_path,
    device="cpu",
    batch_size=32,
    num_samples=1000,
):
    """
    Diagnose embedding compactness and separation on validation set.
    """
    print(f"\n{'=' * 80}")
    print("Stage 2 Multi-Level Center Loss Diagnostic")
    print(f"{'=' * 80}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    num_sensors = len(checkpoint.get("sensor_names", SENSOR_COLS))
    window_size = checkpoint.get("window_size", WINDOW_SIZE)
    embed_dim = checkpoint.get("embed_dim", EMBED_DIM)
    hidden_dim = checkpoint.get("hidden_dim", HIDDEN_DIM)
    top_k = checkpoint.get("top_k", TOP_K)
    
    print(f"  Model config: {num_sensors} sensors, window_size={window_size}")
    print(f"  Embed dim: {embed_dim}, Hidden dim: {hidden_dim}, Top-K: {top_k}")
    
    # Load model
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=top_k,
        hidden_dim=hidden_dim,
    ).to(device)
    # Filter out GAT parameters that don't match (due to architecture change)
    filtered_state = {k: v for k, v in checkpoint["model_state_dict"].items() 
                     if not k.startswith('gat.')}
    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    
    # Load center loss
    center_loss = MultiLevelCenterLoss(
        embed_dim=hidden_dim,
        num_sensors=num_sensors,
        num_classes=2,
        margin=2.0,  # Default, will be overwritten by state
        lambda_intra=1.5,
        lambda_sensor=0.8,
    ).to(device)
    center_loss.load_state_dict(checkpoint["center_loss_state_dict"])
    center_loss.eval()
    
    print("  ✓ Model and center loss loaded\n")
    
    # Load validation data
    print(f"Loading validation data from {data_path}...")
    df_list = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)
    
    data = pd.concat(df_list, ignore_index=True)
    
    # Preprocessing (same as training)
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ],
        errors="ignore"
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
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    
    # Split by drive (same as training: 70/15/15)
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    
    print(f"  Validation drives: {len(val_drives)}")
    
    # Build clean windows
    print("\nBuilding validation windows...")
    X_val, y_val, scaler = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    
    # Inject faults
    print("Injecting faults...")
    X_val_faulty, _, val_sensor_labels, val_window_labels = (
        inject_faults_with_sensor_labels(
            X_val,
            y_val,
            SENSOR_COLS,
            fault_percentage=0.15,
            random_state=43,
            use_stratified=True,
        )
    )
    
    # Subsample for faster computation
    if len(X_val_faulty) > num_samples:
        indices = np.random.choice(len(X_val_faulty), num_samples, replace=False)
        X_val_faulty = X_val_faulty[indices]
        val_sensor_labels = val_sensor_labels[indices]
        val_window_labels = val_window_labels[indices]
    
    print(f"  Using {len(X_val_faulty)} validation samples\n")
    
    # Create dataloader
    val_ds = TensorDataset(
        X_val_faulty, y_val[:len(X_val_faulty)], val_sensor_labels, val_window_labels
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Get separations
    separations = center_loss.get_separations()
    window_centers = center_loss.get_window_centers()
    sensor_centers = center_loss.get_sensor_centers()
    
    print(f"{'=' * 80}")
    print("SEPARATION METRICS")
    print(f"{'=' * 80}")
    print(f"Window separation: {separations['window_separation']:.6f}")
    print(f"Sensor mean separation: {separations['sensor_mean_separation']:.6f}")
    print(f"Sensor range: [{separations['sensor_min_separation']:.6f}, "
          f"{separations['sensor_max_separation']:.6f}]")
    
    print(f"\nPer-sensor separations (6 decimal precision):")
    sensor_names = checkpoint.get("sensor_names", SENSOR_COLS)
    for i, sep in enumerate(separations['sensor_separations']):
        print(f"  {sensor_names[i]:40s}: {sep:.6f}")
    
    # Compute compactness
    print(f"\n{'=' * 80}")
    print("COMPACTNESS METRICS")
    print(f"{'=' * 80}")
    
    all_window_embs = []
    all_sensor_embs = []
    all_window_labels = []
    all_sensor_labels = []
    
    with torch.no_grad():
        for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
            X_batch = X_batch.to(device)
            window_labels_batch = window_labels_batch.long().to(device)
            sensor_labels_batch = sensor_labels_batch.long().to(device)
            
            window_embs = model.get_embeddings(X_batch)
            sensor_embs = model.get_sensor_embeddings(X_batch)
            
            all_window_embs.append(window_embs.cpu())
            all_sensor_embs.append(sensor_embs.cpu())
            all_window_labels.append(window_labels_batch.cpu())
            all_sensor_labels.append(sensor_labels_batch.cpu())
    
    window_embs = torch.cat(all_window_embs, dim=0)
    sensor_embs = torch.cat(all_sensor_embs, dim=0)
    window_labels = torch.cat(all_window_labels, dim=0)
    sensor_labels = torch.cat(all_sensor_labels, dim=0)
    
    # Normalize embeddings
    window_embs = F.normalize(window_embs, p=2, dim=1)
    
    # Window-level compactness
    normal_mask = (window_labels == 0)
    anomaly_mask = (window_labels == 1)
    
    normal_embs = window_embs[normal_mask]
    anomaly_embs = window_embs[anomaly_mask]
    
    window_center_normal = window_centers[0].cpu()
    window_center_anomaly = window_centers[1].cpu()
    
    normal_compactness = torch.norm(
        normal_embs - window_center_normal, dim=1
    ).mean().item()
    
    anomaly_compactness = torch.norm(
        anomaly_embs - window_center_anomaly, dim=1
    ).mean().item()
    
    avg_compactness = (normal_compactness + anomaly_compactness) / 2
    separation_ratio = separations['window_separation'] / avg_compactness
    
    print(f"\nWindow-level:")
    print(f"  Normal samples: {normal_mask.sum().item()}")
    print(f"  Anomaly samples: {anomaly_mask.sum().item()}")
    print(f"  Normal compactness: {normal_compactness:.6f}")
    print(f"  Anomaly compactness: {anomaly_compactness:.6f}")
    print(f"  Average compactness: {avg_compactness:.6f}")
    print(f"  Separation / avg compactness ratio: {separation_ratio:.2f}")
    
    # Sensor-level compactness
    print(f"\nSensor-level compactness:")
    sensor_names = checkpoint.get("sensor_names", SENSOR_COLS)
    sensor_compactness_normal = []
    sensor_compactness_anomaly = []
    
    for sensor_idx in range(num_sensors):
        sensor_emb = sensor_embs[:, sensor_idx, :]  # (N, embed_dim)
        sensor_label = sensor_labels[:, sensor_idx]  # (N,)
        sensor_center = sensor_centers[sensor_idx].cpu()  # (2, embed_dim)
        
        normal_mask_sensor = (sensor_label == 0)
        anomaly_mask_sensor = (sensor_label == 1)
        
        if normal_mask_sensor.sum() > 0:
            normal_emb_sensor = sensor_emb[normal_mask_sensor]
            normal_center = sensor_center[0]
            compact_normal = torch.norm(
                normal_emb_sensor - normal_center, dim=1
            ).mean().item()
            sensor_compactness_normal.append(compact_normal)
        
        if anomaly_mask_sensor.sum() > 0:
            anomaly_emb_sensor = sensor_emb[anomaly_mask_sensor]
            anomaly_center = sensor_center[1]
            compact_anomaly = torch.norm(
                anomaly_emb_sensor - anomaly_center, dim=1
            ).mean().item()
            sensor_compactness_anomaly.append(compact_anomaly)
    
    if sensor_compactness_normal:
        avg_sensor_normal = np.mean(sensor_compactness_normal)
        print(f"  Average normal sensor compactness: {avg_sensor_normal:.6f}")
    
    if sensor_compactness_anomaly:
        avg_sensor_anomaly = np.mean(sensor_compactness_anomaly)
        print(f"  Average anomaly sensor compactness: {avg_sensor_anomaly:.6f}")
    
    # Summary assessment
    print(f"\n{'=' * 80}")
    print("ASSESSMENT")
    print(f"{'=' * 80}")
    
    if avg_compactness < 0.3 and separation_ratio > 5.0:
        status = "✅ EXCELLENT"
        recommendation = "Continue training to epoch 30. Clusters are tight and well-separated."
    elif avg_compactness < 0.5 and separation_ratio > 3.0:
        status = "⚠️  GOOD"
        recommendation = "Training is working but could be better. Consider continuing or fixing repulsion loss."
    else:
        status = "❌ POOR"
        recommendation = "Clusters are not forming well. Restart with fixed exponential repulsion loss."
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    print(f"\nKey metrics:")
    print(f"  - Compactness: {avg_compactness:.4f} (target: < 0.3)")
    print(f"  - Separation ratio: {separation_ratio:.2f} (target: > 5.0)")
    print(f"  - Window separation: {separations['window_separation']:.6f} (margin: 2.0)")
    
    # Check if all separations are exactly 2.0
    all_exactly_2 = (
        abs(separations['window_separation'] - 2.0) < 1e-5 and
        all(abs(s - 2.0) < 1e-5 for s in separations['sensor_separations'])
    )
    
    if all_exactly_2:
        print(f"\n⚠️  WARNING: All separations are exactly 2.0 (margin value).")
        print(f"  This suggests centers are stuck at the constraint boundary.")
        print(f"  The ReLU repulsion loss (F.relu(margin - sep)) stops pushing once sep >= margin.")
        print(f"  Consider switching to exponential repulsion: torch.exp(-0.5 * sep)")
    
    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Stage 2 multi-level center loss compactness"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Stage 2 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of validation samples to use",
    )
    
    args = parser.parse_args()
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    diagnose_compactness(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
