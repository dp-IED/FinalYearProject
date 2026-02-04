#!/usr/bin/env python3
"""
Visualize sensor embeddings in 3D using t-SNE.

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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

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


def plot_tsne_embeddings_interactive(
    embeddings,
    labels,
    sensor_ids,
    sensor_names,
    save_path="figures/embedding_tsne.html",
):
    """Plot interactive 3D t-SNE visualization using Plotly. Creates 3 separate HTML files."""
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install it with: pip install plotly"
        )

    print("Computing 3D t-SNE (this may take a few minutes)...")

    # Subsample for faster t-SNE if too many points
    if len(embeddings) > 20000:
        print(f"Subsampling from {len(embeddings)} to 20000 points for t-SNE...")
        indices = np.random.choice(len(embeddings), 20000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        sensor_ids = sensor_ids[indices]

    # Filter out RPM (index 0), SPEED (index 1), and LTFT (index 7)
    print("Filtering out RPM, SPEED, and LTFT sensors...")
    filter_mask = (
        (sensor_ids != 0) & (sensor_ids != 1) & (sensor_ids != 7)
    )  # Exclude RPM (0), SPEED (1), and LTFT (7)
    embeddings_filtered = embeddings[filter_mask]
    labels_filtered = labels[filter_mask]
    sensor_ids_filtered = sensor_ids[filter_mask]

    print(f"  Removed {np.sum(~filter_mask)} points (RPM, SPEED, and LTFT)")
    print(
        f"  Remaining: {len(embeddings_filtered)} points from {len(np.unique(sensor_ids_filtered))} sensors"
    )

    # Compute 3D t-SNE (once, shared across all three visualizations)
    tsne = TSNE(
        n_components=3, random_state=42, perplexity=30, max_iter=1000, verbose=1
    )
    embeddings_3d = tsne.fit_transform(embeddings_filtered)

    # Color palette (only for remaining sensors)
    # Map sensor IDs: original IDs 2-7 become 0-5 for color indexing
    sensor_id_map = {
        orig_id: idx
        for idx, orig_id in enumerate(sorted(np.unique(sensor_ids_filtered)))
    }
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_names)))
    color_hex = [
        f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})" for c in colors
    ]

    # Get remaining sensor indices (exclude RPM=0 and SPEED=1)
    remaining_sensor_indices = sorted(np.unique(sensor_ids_filtered))

    # Determine base path for output files
    save_path = Path(save_path)
    base_path = save_path.parent / save_path.stem

    # Ensure output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Normal vs Anomaly
    print("Creating Normal vs Anomaly visualization...")
    fig1 = go.Figure()
    normal_mask = labels_filtered == 0
    anomaly_mask = labels_filtered == 1

    if np.sum(normal_mask) > 0:
        fig1.add_trace(
            go.Scatter3d(
                x=embeddings_3d[normal_mask, 0],
                y=embeddings_3d[normal_mask, 1],
                z=embeddings_3d[normal_mask, 2],
                mode="markers",
                marker=dict(size=3, color="#2E86AB", opacity=0.4, line=dict(width=0)),
                name="Normal",
                hovertemplate="Normal<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        )

    if np.sum(anomaly_mask) > 0:
        fig1.add_trace(
            go.Scatter3d(
                x=embeddings_3d[anomaly_mask, 0],
                y=embeddings_3d[anomaly_mask, 1],
                z=embeddings_3d[anomaly_mask, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color="#C73E1D",
                    opacity=0.7,
                    line=dict(width=1, color="black"),
                ),
                name="Anomaly",
                hovertemplate="Anomaly<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
            )
        )

    fig1.update_layout(
        title_text="Normal vs Anomaly (Interactive 3D t-SNE)",
        title_x=0.5,
        height=800,
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
            aspectmode="cube",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    path1 = f"{base_path}_normal_vs_anomaly.html"
    fig1.write_html(path1)
    print(f"  ✓ Saved to {path1}")

    # 2. By Sensor Type
    print("Creating By Sensor Type visualization...")
    fig2 = go.Figure()

    # Only iterate over remaining sensors (exclude RPM=0 and SPEED=1)
    for orig_sensor_id in remaining_sensor_indices:
        sensor_name = sensor_names[orig_sensor_id]
        mask = sensor_ids_filtered == orig_sensor_id
        if np.sum(mask) > 0:
            fig2.add_trace(
                go.Scatter3d(
                    x=embeddings_3d[mask, 0],
                    y=embeddings_3d[mask, 1],
                    z=embeddings_3d[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=color_hex[orig_sensor_id],
                        opacity=0.5,
                        line=dict(width=0),
                    ),
                    name=sensor_name,
                    hovertemplate=f"{sensor_name}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>",
                )
            )

    fig2.update_layout(
        title_text="By Sensor Type (Interactive 3D t-SNE)",
        title_x=0.5,
        height=800,
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
            aspectmode="cube",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    path2 = f"{base_path}_by_sensor_type.html"
    fig2.write_html(path2)
    print(f"  ✓ Saved to {path2}")

    # 3. Combined view (sensor + anomaly)
    print("Creating Sensor Types + Anomalies visualization...")
    fig3 = go.Figure()

    # Only iterate over remaining sensors (exclude RPM=0 and SPEED=1)
    for orig_sensor_id in remaining_sensor_indices:
        sensor_name = sensor_names[orig_sensor_id]
        # Normal for this sensor
        mask = (sensor_ids_filtered == orig_sensor_id) & (labels_filtered == 0)
        if np.sum(mask) > 0:
            fig3.add_trace(
                go.Scatter3d(
                    x=embeddings_3d[mask, 0],
                    y=embeddings_3d[mask, 1],
                    z=embeddings_3d[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=color_hex[orig_sensor_id],
                        opacity=0.2,
                        line=dict(width=0),
                    ),
                    name=f"{sensor_name} (normal)",
                    hovertemplate=f"{sensor_name} (Normal)<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>",
                    showlegend=False,
                )
            )

        # Anomaly for this sensor
        mask = (sensor_ids_filtered == orig_sensor_id) & (labels_filtered == 1)
        if np.sum(mask) > 0:
            fig3.add_trace(
                go.Scatter3d(
                    x=embeddings_3d[mask, 0],
                    y=embeddings_3d[mask, 1],
                    z=embeddings_3d[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=color_hex[orig_sensor_id],
                        opacity=0.8,
                        symbol="x",
                        line=dict(width=1, color="black"),
                    ),
                    name=f"{sensor_name} (anomaly)",
                    hovertemplate=f"{sensor_name} (Anomaly)<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>",
                )
            )

    fig3.update_layout(
        title_text="Sensor Types + Anomalies (Interactive 3D t-SNE)",
        title_x=0.5,
        height=800,
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
            aspectmode="cube",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    path3 = f"{base_path}_sensor_types_anomalies.html"
    fig3.write_html(path3)
    print(f"  ✓ Saved to {path3}")

    print(f"\n✓ Created 3 interactive 3D t-SNE visualizations:")
    print(f"  1. {path1}")
    print(f"  2. {path2}")
    print(f"  3. {path3}")
    print(
        f"\n  Open any of these files in your web browser to explore the embedding space!"
    )


def plot_tsne_embeddings(
    embeddings, labels, sensor_ids, sensor_names, save_path="figures/embedding_tsne.png"
):
    """Plot static 3D t-SNE visualization of embeddings using matplotlib."""
    print("Computing 3D t-SNE (this may take a few minutes)...")

    # Filter out RPM (index 0), SPEED (index 1), and LTFT (index 7)
    print("Filtering out RPM, SPEED, and LTFT sensors...")
    filter_mask = (
        (sensor_ids != 0) & (sensor_ids != 1) & (sensor_ids != 7)
    )  # Exclude RPM (0), SPEED (1), and LTFT (7)
    embeddings = embeddings[filter_mask]
    labels = labels[filter_mask]
    sensor_ids = sensor_ids[filter_mask]
    print(f"  Removed {np.sum(~filter_mask)} points (RPM, SPEED, and LTFT)")
    print(
        f"  Remaining: {len(embeddings)} points from {len(np.unique(sensor_ids))} sensors"
    )

    # Subsample for faster t-SNE if too many points
    if len(embeddings) > 20000:
        print(f"Subsampling from {len(embeddings)} to 20000 points for t-SNE...")
        indices = np.random.choice(len(embeddings), 20000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        sensor_ids = sensor_ids[indices]

    # Compute 3D t-SNE
    # Use max_iter instead of n_iter (newer scikit-learn versions)
    tsne = TSNE(
        n_components=3, random_state=42, perplexity=30, max_iter=1000, verbose=1
    )
    embeddings_3d = tsne.fit_transform(embeddings)

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(24, 7))
    fig.suptitle(
        "Sensor Embedding Visualization (3D t-SNE)", fontsize=16, fontweight="bold"
    )

    # 1. Color by normal/anomaly
    ax1 = fig.add_subplot(131, projection="3d")
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    if np.sum(normal_mask) > 0:
        ax1.scatter(
            embeddings_3d[normal_mask, 0],
            embeddings_3d[normal_mask, 1],
            embeddings_3d[normal_mask, 2],
            c="#2E86AB",
            alpha=0.3,
            s=10,
            label="Normal",
            edgecolors="none",
        )

    if np.sum(anomaly_mask) > 0:
        ax1.scatter(
            embeddings_3d[anomaly_mask, 0],
            embeddings_3d[anomaly_mask, 1],
            embeddings_3d[anomaly_mask, 2],
            c="#C73E1D",
            alpha=0.5,
            s=20,
            label="Anomaly",
            edgecolors="black",
            linewidths=0.5,
        )

    ax1.set_title("Normal vs Anomaly", fontsize=14, fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax1.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax1.legend(fontsize=12, markerscale=2)

    # 2. Color by sensor type
    ax2 = fig.add_subplot(132, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_names)))
    remaining_sensor_indices = sorted(np.unique(sensor_ids))
    for orig_sensor_id in remaining_sensor_indices:
        sensor_name = sensor_names[orig_sensor_id]
        mask = sensor_ids == orig_sensor_id
        if np.sum(mask) > 0:
            ax2.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[colors[orig_sensor_id]],
                alpha=0.4,
                s=15,
                label=sensor_name,
                edgecolors="none",
            )
    ax2.set_title("By Sensor Type", fontsize=14, fontweight="bold")
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax2.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax2.legend(fontsize=9, markerscale=1.5, loc="best", ncol=2)

    # 3. Combined view (sensor + anomaly)
    ax3 = fig.add_subplot(133, projection="3d")
    for orig_sensor_id in remaining_sensor_indices:
        sensor_name = sensor_names[orig_sensor_id]
        # Normal for this sensor
        mask = (sensor_ids == orig_sensor_id) & (labels == 0)
        if np.sum(mask) > 0:
            ax3.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[colors[orig_sensor_id]],
                alpha=0.2,
                s=10,
                edgecolors="none",
            )

        # Anomaly for this sensor
        mask = (sensor_ids == orig_sensor_id) & (labels == 1)
        if np.sum(mask) > 0:
            ax3.scatter(
                embeddings_3d[mask, 0],
                embeddings_3d[mask, 1],
                embeddings_3d[mask, 2],
                c=[colors[orig_sensor_id]],
                alpha=0.7,
                s=30,
                marker="X",
                edgecolors="black",
                linewidths=0.5,
                label=f"{sensor_name} (anomaly)",
            )

    ax3.set_title("Sensor Types + Anomalies", fontsize=14, fontweight="bold")
    ax3.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax3.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax3.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax3.legend(fontsize=8, markerscale=1, loc="best", ncol=2)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Validate file extension
    valid_extensions = [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"]
    file_ext = Path(save_path).suffix.lower()
    if file_ext not in valid_extensions:
        # Try to fix common typos
        if file_ext == ".wpng":
            save_path = str(Path(save_path).with_suffix(".png"))
            print(f"⚠ Fixed file extension typo: using {save_path}")
        else:
            raise ValueError(
                f"Unsupported file format '{file_ext}'. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved 3D t-SNE plot to {save_path}")
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
        help="Output path for visualization (use .html for interactive, .png for static)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML visualization (requires plotly). "
        "If output ends with .html, this is automatically enabled.",
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

    # Determine if we should use interactive visualization
    use_interactive = args.interactive or args.output.endswith(".html")

    if use_interactive:
        if not HAS_PLOTLY:
            print(
                "⚠ Warning: Plotly not available. Falling back to static visualization."
            )
            print("  Install plotly with: pip install plotly")
            use_interactive = False

    # Plot t-SNE
    if use_interactive:
        # Ensure output is HTML
        if not args.output.endswith(".html"):
            args.output = str(Path(args.output).with_suffix(".html"))
        plot_tsne_embeddings_interactive(
            embeddings, labels, sensor_ids, SENSOR_NAMES, save_path=args.output
        )
    else:
        # Ensure output is PNG (or other image format)
        if args.output.endswith(".html"):
            args.output = str(Path(args.output).with_suffix(".png"))
        plot_tsne_embeddings(
            embeddings, labels, sensor_ids, SENSOR_NAMES, save_path=args.output
        )

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
