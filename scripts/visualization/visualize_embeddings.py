#!/usr/bin/env python3
"""
Visualize sensor embeddings in 2D using t-SNE.

This script loads a trained GDN model checkpoint, extracts sensor embeddings
from test data, and visualizes them using t-SNE to show:
1. Normal vs anomaly clustering
2. Sensor-type clustering
3. Combined view (sensor + anomaly markers)
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

# Import model and data preprocessing
from models.gdn_model import MultiLabelGDN, KAGOptimizedGDN

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
    "RPM",
    "SPEED",
    "THROTTLE",
    "LOAD",
    "COOLANT",
    "MANIFOLD",
    "STFT",
    "LTFT",
]


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Load GDN model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine model type
    checkpoint_model_type = checkpoint.get("model_type", None)
    if checkpoint_model_type not in ["enhanced", "kag_optimized"]:
        # Try to infer from checkpoint structure
        if "base_model_state_dict" in checkpoint:
            base_state = checkpoint["base_model_state_dict"]
            has_temporal_pooling = any(
                "temporal_pooling" in k for k in base_state.keys()
            )
            checkpoint_model_type = (
                "enhanced" if has_temporal_pooling else "kag_optimized"
            )
        elif "model_state_dict" in checkpoint:
            base_state = checkpoint["model_state_dict"]
            has_temporal_pooling = any(
                "temporal_pooling" in k for k in base_state.keys()
            )
            checkpoint_model_type = (
                "enhanced" if has_temporal_pooling else "kag_optimized"
            )
        else:
            checkpoint_model_type = "enhanced"  # Default

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
        # Filter out incompatible layers
        if checkpoint_model_type == "kag_optimized":
            filtered_state = {
                k: v
                for k, v in base_state.items()
                if not any(x in k for x in ["temporal_pooling", "multi_scale_gat"])
            }
        else:
            filtered_state = base_state
        model.load_state_dict(filtered_state, strict=False)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    print("✓ Model loaded successfully")
    return model, checkpoint_model_type


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
        errors="ignore",
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


def extract_embeddings(model, data_loader, device="cpu"):
    """Extract sensor embeddings from model."""
    model.eval()
    embeddings = []
    labels = []
    sensor_ids = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(data_loader, desc="Processing batches")
        ):
            X_batch = X_batch.to(device)

            # Get sensor embeddings
            try:
                sensor_embs = model.get_sensor_embeddings(X_batch)  # (B, N, D)
            except (AttributeError, RuntimeError):
                # Fallback: use forward with return_sensor_embeddings
                _, _, sensor_embs = model(X_batch, return_sensor_embeddings=True)

            # Normalize embeddings
            sensor_embs = F.normalize(sensor_embs, p=2, dim=2)

            # Flatten: (B*N, D)
            B, N, D = sensor_embs.shape
            sensor_embs_flat = sensor_embs.reshape(-1, D)

            # Create sensor IDs: 0-7 for each batch
            sensor_id_batch = (
                torch.arange(N, device=device).unsqueeze(0).repeat(B, 1).reshape(-1)
            )

            # For labels, we'll use a placeholder (0 = normal) since we don't have fault injection here
            # In a real scenario, you'd inject faults and get true labels
            label_batch = torch.zeros(B * N, device=device, dtype=torch.long)

            embeddings.append(sensor_embs_flat.cpu().numpy())
            labels.append(label_batch.cpu().numpy())
            sensor_ids.append(sensor_id_batch.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensor_ids = np.concatenate(sensor_ids, axis=0)

    print(f"Extracted {len(embeddings)} embeddings")
    return embeddings, labels, sensor_ids


def extract_embeddings_with_faults(
    model, X, y, device="cpu", fault_percentage=0.3, batch_size=32
):
    """Extract embeddings with fault injection for proper labels."""
    # Inject faults
    print(f"Injecting faults ({fault_percentage * 100:.0f}% rate)...")
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
        X_tensor,
        y_tensor,
        SENSOR_COLS,
        fault_percentage=fault_percentage,
        random_state=42,
    )
    y_sensor = y_sensor.long()

    # Create data loader
    dataset = TensorDataset(X_fault, y_sensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    embeddings = []
    labels = []
    sensor_ids = []

    print("Extracting embeddings with fault labels...")
    with torch.no_grad():
        for X_batch, y_sensor_batch in tqdm(data_loader, desc="Processing batches"):
            X_batch = X_batch.to(device)

            # Get sensor embeddings
            try:
                sensor_embs = model.get_sensor_embeddings(X_batch)  # (B, N, D)
            except (AttributeError, RuntimeError):
                _, _, sensor_embs = model(X_batch, return_sensor_embeddings=True)

            # Normalize embeddings
            sensor_embs = F.normalize(sensor_embs, p=2, dim=2)

            # Flatten: (B*N, D)
            B, N, D = sensor_embs.shape
            sensor_embs_flat = sensor_embs.reshape(-1, D)

            # Flatten labels: (B*N,)
            y_sensor_flat = y_sensor_batch.reshape(-1)

            # Create sensor IDs: 0-7 for each batch
            sensor_id_batch = (
                torch.arange(N, device=device).unsqueeze(0).repeat(B, 1).reshape(-1)
            )

            embeddings.append(sensor_embs_flat.cpu().numpy())
            labels.append(y_sensor_flat.cpu().numpy())
            sensor_ids.append(sensor_id_batch.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    sensor_ids = np.concatenate(sensor_ids, axis=0)

    print(f"Extracted {len(embeddings)} embeddings ({np.sum(labels)} anomalies)")
    return embeddings, labels, sensor_ids


def plot_tsne_embeddings(
    embeddings, labels, sensor_ids, sensor_names, save_path="figures/embedding_tsne.png"
):
    """Plot t-SNE visualization of embeddings."""
    print("Computing t-SNE (this may take a few minutes)...")

    # Subsample for faster t-SNE if too many points
    if len(embeddings) > 20000:
        print(f"Subsampling from {len(embeddings)} to 20000 points for t-SNE...")
        indices = np.random.choice(len(embeddings), 20000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        sensor_ids = sensor_ids[indices]

    # Compute t-SNE
    # Use max_iter instead of n_iter (newer scikit-learn versions)
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "Sensor Embedding Visualization (t-SNE)", fontsize=16, fontweight="bold"
    )

    # 1. Color by normal/anomaly
    ax = axes[0]
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    if np.sum(normal_mask) > 0:
        ax.scatter(
            embeddings_2d[normal_mask, 0],
            embeddings_2d[normal_mask, 1],
            c="#2E86AB",
            alpha=0.3,
            s=10,
            label="Normal",
            edgecolors="none",
        )

    if np.sum(anomaly_mask) > 0:
        ax.scatter(
            embeddings_2d[anomaly_mask, 0],
            embeddings_2d[anomaly_mask, 1],
            c="#C73E1D",
            alpha=0.5,
            s=20,
            label="Anomaly",
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_title("Normal vs Anomaly", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.3)

    # 2. Color by sensor type
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_names)))
    for i, sensor_name in enumerate(sensor_names):
        mask = sensor_ids == i
        if np.sum(mask) > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                alpha=0.4,
                s=15,
                label=sensor_name,
                edgecolors="none",
            )
    ax.set_title("By Sensor Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=9, markerscale=1.5, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Combined view (sensor + anomaly)
    ax = axes[2]
    for i, sensor_name in enumerate(sensor_names):
        # Normal for this sensor
        mask = (sensor_ids == i) & (labels == 0)
        if np.sum(mask) > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                alpha=0.2,
                s=10,
                edgecolors="none",
            )

        # Anomaly for this sensor
        mask = (sensor_ids == i) & (labels == 1)
        if np.sum(mask) > 0:
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                alpha=0.7,
                s=30,
                marker="X",
                edgecolors="black",
                linewidths=0.5,
                label=f"{sensor_name} (anomaly)",
            )

    ax.set_title("Sensor Types + Anomalies", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(fontsize=8, markerscale=1, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved t-SNE plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize sensor embeddings using t-SNE"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata",
        help="Path to OBD data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/embedding_tsne.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=5000,
        help="Maximum number of windows to process",
    )
    parser.add_argument(
        "--fault_percentage",
        type=float,
        default=0.3,
        help="Percentage of windows to inject faults (0.0-1.0)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda). Auto-detects if not specified",
    )
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Device detection
    if args.cpu_only:
        device = "cpu"
    elif args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    model, model_type = load_model_from_checkpoint(args.checkpoint, device)

    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data(
        args.data_path, max_windows=args.max_windows
    )

    # Extract embeddings with fault injection for proper labels
    embeddings, labels, sensor_ids = extract_embeddings_with_faults(
        model,
        X,
        y,
        device=device,
        fault_percentage=args.fault_percentage,
        batch_size=args.batch_size,
    )

    # Plot t-SNE
    plot_tsne_embeddings(
        embeddings, labels, sensor_ids, SENSOR_NAMES, save_path=args.output
    )

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
