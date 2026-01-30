#!/usr/bin/env python3
"""
Visualize per-sensor separation and compactness metrics.

This script loads a trained GDN model checkpoint, extracts sensor centers
from the center loss module, computes per-sensor separation and compactness
metrics, and creates comprehensive visualizations.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

# Import model and data preprocessing
from models.gdn_model import MultiLabelGDN, KAGOptimizedGDN
from models.multi_level_center_loss import SensorOnlyCenterLoss

# Import data preprocessing functions from training script
sys.path.insert(0, str(project_root / "scripts" / "training"))
from train_gdn_stage2_multilevel import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
)
from fault_injection import inject_faults_with_sensor_labels

# Constants
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
NUM_SENSORS = 8
HIDDEN_DIM = 64
EMBED_DIM = 32
TOP_K = 5

SENSOR_NAMES = [
    'RPM', 'SPEED', 'THROTTLE', 'LOAD',
    'COOLANT', 'MANIFOLD', 'STFT', 'LTFT'
]


def load_model_and_centers(checkpoint_path, device='cpu'):
    """Load GDN model and center loss from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type
    checkpoint_model_type = checkpoint.get("model_type", None)
    if checkpoint_model_type not in ["enhanced", "kag_optimized"]:
        if "base_model_state_dict" in checkpoint:
            base_state = checkpoint["base_model_state_dict"]
            has_temporal_pooling = any("temporal_pooling" in k for k in base_state.keys())
            checkpoint_model_type = "enhanced" if has_temporal_pooling else "kag_optimized"
        elif "model_state_dict" in checkpoint:
            base_state = checkpoint["model_state_dict"]
            has_temporal_pooling = any("temporal_pooling" in k for k in base_state.keys())
            checkpoint_model_type = "enhanced" if has_temporal_pooling else "kag_optimized"
        else:
            checkpoint_model_type = "enhanced"
    
    # Initialize model
    if checkpoint_model_type == "kag_optimized":
        print("Using KAG-Optimized model")
        model = KAGOptimizedGDN(
            num_nodes=NUM_SENSORS,
            window_size=WINDOW_SIZE,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
    else:
        print("Using Enhanced MultiLabelGDN model")
        model = MultiLabelGDN(
            num_nodes=NUM_SENSORS,
            window_size=WINDOW_SIZE,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
    
    # Load model state
    if "base_model_state_dict" in checkpoint:
        base_state = checkpoint["base_model_state_dict"]
        if checkpoint_model_type == "kag_optimized":
            filtered_state = {
                k: v for k, v in base_state.items()
                if not any(x in k for x in ['temporal_pooling', 'multi_scale_gat'])
            }
        else:
            filtered_state = base_state
        model.load_state_dict(filtered_state, strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    model.eval()
    
    # Load center loss
    center_loss = SensorOnlyCenterLoss(
        embed_dim=HIDDEN_DIM,
        num_sensors=NUM_SENSORS,
        num_classes=2,
    ).to(device)
    
    if "center_loss_state_dict" in checkpoint:
        center_loss.load_state_dict(checkpoint["center_loss_state_dict"])
        print("✓ Center loss loaded")
    else:
        print("⚠ Warning: No center loss state found in checkpoint")
    
    print("✓ Model and centers loaded successfully")
    return model, center_loss, checkpoint_model_type


def compute_per_sensor_separation(center_loss):
    """Compute per-sensor separation from centers."""
    sensor_centers = center_loss.get_sensor_centers()  # (N, 2, D)
    sensor_centers = F.normalize(sensor_centers, p=2, dim=2)
    
    separations = []
    for i in range(NUM_SENSORS):
        sep = torch.norm(sensor_centers[i, 0] - sensor_centers[i, 1], p=2).item()
        separations.append(sep)
    
    return separations


def compute_per_sensor_compactness(model, X, y, center_loss, device='cpu', 
                                   fault_percentage=0.3, batch_size=32, max_samples=5000):
    """Compute per-sensor compactness from test data."""
    # Inject faults
    print(f"Injecting faults ({fault_percentage*100:.0f}% rate)...")
    # Convert to torch tensors if needed
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
    else:
        X_tensor = X
    
    if isinstance(y, np.ndarray):
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        y_tensor = y
    
    X_fault, y_window, y_sensor, window_labels = inject_faults_with_sensor_labels(
        X_tensor, y_tensor, SENSOR_COLS,
        fault_percentage=fault_percentage, 
        random_state=42
    )
    y_sensor = y_sensor.long()
    
    # Limit samples if needed
    if len(X_fault) > max_samples:
        indices = np.random.choice(len(X_fault), max_samples, replace=False)
        X_fault = X_fault[indices]
        y_sensor = y_sensor[indices]
    
    # Create data loader
    dataset = TensorDataset(X_fault, y_sensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get sensor centers
    sensor_centers = center_loss.get_sensor_centers()  # (N, 2, D)
    sensor_centers = F.normalize(sensor_centers, p=2, dim=2)
    
    # Accumulate distances per sensor and class
    normal_distances = [[] for _ in range(NUM_SENSORS)]
    anomaly_distances = [[] for _ in range(NUM_SENSORS)]
    
    model.eval()
    print("Computing compactness metrics...")
    with torch.no_grad():
        for X_batch, y_sensor_batch in tqdm(data_loader, desc="Processing batches"):
            X_batch = X_batch.to(device)
            
            # Get sensor embeddings
            try:
                sensor_embs = model.get_sensor_embeddings(X_batch)  # (B, N, D)
            except:
                _, _, sensor_embs = model(X_batch, return_sensor_embeddings=True)
            
            # Normalize embeddings
            sensor_embs = F.normalize(sensor_embs, p=2, dim=2)
            
            # Compute distances to centers for each sensor
            B, N, D = sensor_embs.shape
            for sensor_idx in range(N):
                sensor_emb = sensor_embs[:, sensor_idx, :]  # (B, D)
                sensor_labels = y_sensor_batch[:, sensor_idx]  # (B,)
                
                # Normal center
                normal_center = sensor_centers[sensor_idx, 0]  # (D,)
                normal_dist = torch.norm(sensor_emb - normal_center, p=2, dim=1)  # (B,)
                normal_mask = (sensor_labels == 0)
                if normal_mask.sum() > 0:
                    normal_distances[sensor_idx].extend(normal_dist[normal_mask].cpu().numpy().tolist())
                
                # Anomaly center
                anomaly_center = sensor_centers[sensor_idx, 1]  # (D,)
                anomaly_dist = torch.norm(sensor_emb - anomaly_center, p=2, dim=1)  # (B,)
                anomaly_mask = (sensor_labels == 1)
                if anomaly_mask.sum() > 0:
                    anomaly_distances[sensor_idx].extend(anomaly_dist[anomaly_mask].cpu().numpy().tolist())
    
    # Compute mean compactness (mean distance to center)
    normal_compactness = [np.mean(dists) if len(dists) > 0 else 0.0 for dists in normal_distances]
    anomaly_compactness = [np.mean(dists) if len(dists) > 0 else 0.0 for dists in anomaly_distances]
    
    return normal_compactness, anomaly_compactness


def load_and_preprocess_data(data_path, scaler=None, max_windows=None):
    """Load and preprocess OBD data."""
    print(f"Loading data from {data_path}...")
    
    # Load CSV files
    df_list = []
    csv_files = list(Path(data_path).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files[:10]:  # Limit to first 10 files for speed
        df = pd.read_csv(file, index_col=False)
        df[ID_COL] = file.stem
        df_list.append(df)
    
    if len(df_list) == 0:
        raise ValueError(f"No CSV files found in {data_path}")
    
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    
    # Preprocessing (same as training)
    print("Preprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ],
        errors='ignore'
    )
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=[ID_COL]
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(
        data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=1
    )
    data = filter_long_drives(data, id_col=ID_COL, min_length=WINDOW_SIZE + 1)
    data = add_cross_channel_features(data)
    
    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    
    # Build windows
    print("Building windows...")
    X, y, scaler = build_clean_windows(
        data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, horizon=1, scaler=scaler
    )
    
    print(f"Built {len(X)} windows")
    
    # Limit windows if requested
    if max_windows is not None and len(X) > max_windows:
        indices = np.random.choice(len(X), max_windows, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Limited to {max_windows} windows")
    
    return X, y, scaler


def plot_sensor_metrics(separations, normal_comp, anomaly_comp, sensor_names, save_path='figures/sensor_metrics.png'):
    """Plot per-sensor metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Sensor Separation & Compactness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Separation bar chart
    ax = axes[0, 0]
    colors = ['#06A77D' if s > 1.85 else '#F18F01' if s > 1.70 else '#C73E1D' for s in separations]
    bars = ax.barh(sensor_names, separations, color=colors, edgecolor='black', linewidth=1.5)
    ax.axvline(x=1.85, color='green', linestyle='--', linewidth=2, label='Target (1.85)')
    ax.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Maximum (2.0)')
    ax.set_xlabel('Separation', fontsize=12)
    ax.set_title('Sensor Separation (Higher is Better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, sep) in enumerate(zip(bars, separations)):
        ax.text(sep + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{sep:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 2. Compactness comparison
    ax = axes[0, 1]
    x = np.arange(len(sensor_names))
    width = 0.35
    ax.bar(x - width/2, normal_comp, width, label='Normal', color='#2E86AB', 
           edgecolor='black', linewidth=1.5)
    ax.bar(x + width/2, anomaly_comp, width, label='Anomaly', color='#C73E1D', 
           edgecolor='black', linewidth=1.5)
    ax.axhline(y=0.15, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (Normal)')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (Anomaly)')
    ax.set_ylabel('Compactness', fontsize=12)
    ax.set_title('Cluster Compactness (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Separation-Compactness scatter
    ax = axes[1, 0]
    # Compute ratio: separation / mean_compactness
    mean_compactness = [(nc + ac) / 2 for nc, ac in zip(normal_comp, anomaly_comp)]
    ratios = [sep / (mc + 1e-6) for sep, mc in zip(separations, mean_compactness)]
    
    scatter = ax.scatter(separations, ratios, s=200, c=ratios, cmap='RdYlGn', 
                        edgecolors='black', linewidths=2, alpha=0.8, vmin=2.0, vmax=5.0)
    for i, name in enumerate(sensor_names):
        ax.annotate(name, (separations[i], ratios[i]), fontsize=9, 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Separation', fontsize=12)
    ax.set_ylabel('Separation Ratio', fontsize=12)
    ax.set_title('Separation vs Ratio (Color = Ratio)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Ratio', fontsize=10)
    
    # 4. Heatmap of metrics
    ax = axes[1, 1]
    metrics_matrix = np.array([
        separations,
        normal_comp,
        anomaly_comp,
        ratios
    ])
    
    # Normalize each row for better visualization
    metrics_matrix_norm = metrics_matrix.copy()
    for i in range(len(metrics_matrix_norm)):
        row = metrics_matrix_norm[i]
        if row.max() > row.min():
            metrics_matrix_norm[i] = (row - row.min()) / (row.max() - row.min())
    
    sns.heatmap(metrics_matrix_norm, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=sensor_names, 
                yticklabels=['Separation', 'Normal Comp', 'Anomaly Comp', 'Ratio'],
                ax=ax, cbar_kws={'label': 'Normalized Value'}, linewidths=1, linecolor='black')
    ax.set_title('Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sensor metrics to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize per-sensor separation and compactness')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata',
                       help='Path to OBD data directory')
    parser.add_argument('--output', type=str, default='figures/sensor_metrics.png',
                       help='Output path for visualization')
    parser.add_argument('--max_windows', type=int, default=3000,
                       help='Maximum number of windows to process for compactness')
    parser.add_argument('--fault_percentage', type=float, default=0.3,
                       help='Percentage of windows to inject faults (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda). Auto-detects if not specified')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--skip_compactness', action='store_true',
                       help='Skip compactness computation (faster, uses placeholder values)')
    
    args = parser.parse_args()
    
    # Device detection
    if args.cpu_only:
        device = 'cpu'
    elif args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model and centers
    model, center_loss, model_type = load_model_and_centers(args.checkpoint, device)
    
    # Compute separations from centers
    print("\nComputing per-sensor separations...")
    separations = compute_per_sensor_separation(center_loss)
    print(f"Separations: {[f'{s:.4f}' for s in separations]}")
    print(f"Mean separation: {np.mean(separations):.4f}")
    print(f"Range: [{np.min(separations):.4f}, {np.max(separations):.4f}]")
    
    # Compute compactness from test data
    if args.skip_compactness:
        print("\n⚠ Skipping compactness computation (using placeholder values)")
        normal_compactness = [0.09, 0.08, 0.10, 0.11, 0.07, 0.09, 0.12, 0.10]
        anomaly_compactness = [0.85, 0.92, 0.98, 0.88, 1.05, 0.89, 0.95, 0.91]
    else:
        print("\nComputing per-sensor compactness from test data...")
        X, y, scaler = load_and_preprocess_data(args.data_path, max_windows=args.max_windows)
        normal_compactness, anomaly_compactness = compute_per_sensor_compactness(
            model, X, y, center_loss, device=device,
            fault_percentage=args.fault_percentage,
            batch_size=args.batch_size,
            max_samples=args.max_windows
        )
        print(f"Normal compactness: {[f'{c:.4f}' for c in normal_compactness]}")
        print(f"Anomaly compactness: {[f'{c:.4f}' for c in anomaly_compactness]}")
    
    # Plot metrics
    plot_sensor_metrics(separations, normal_compactness, anomaly_compactness, 
                       SENSOR_NAMES, save_path=args.output)
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
