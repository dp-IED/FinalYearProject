#!/usr/bin/env python3
"""
Add subsystem overlays to existing t-SNE plot.
MINIMAL - just hulls and labels, nothing else.
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import ConvexHull

try:
    from scipy.spatial import QhullError
except ImportError:
    try:
        from scipy.spatial.qhull import QhullError
    except ImportError:
        # For older scipy versions
        QhullError = ValueError
from pathlib import Path
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
from scipy.spatial.distance import pdist, cdist

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

# Subsystem definitions
SUBSYSTEM_SENSORS = {
    "Powertrain": ["RPM", "SPEED", "THROTTLE", "LOAD"],
    "Fuel System": ["STFT", "LTFT", "MANIFOLD"],
    "Thermal": ["COOLANT"],
}

SUBSYSTEM_COLORS = {
    "Powertrain": "#E74C3C",
    "Fuel System": "#F39C12",
    "Thermal": "#3498DB",
}

# Legacy SUBSYSTEMS dict for backward compatibility
SUBSYSTEMS = {
    "Powertrain": {
        "sensors": ["RPM", "SPEED", "THROTTLE", "LOAD"],
        "color": "#E74C3C",
        "hull_alpha": 0.08,
        "edge_width": 2.5,
        "label_bg": "#E74C3C",
        "tighten": 0.9,
        "outlier_pct": 97,
        "min_area": 80,
    },
    "Fuel System": {
        "sensors": ["STFT", "LTFT", "MANIFOLD"],
        "color": "#F39C12",
        "hull_alpha": 0.08,
        "edge_width": 2.5,
        "label_bg": "#F39C12",
        "tighten": 1.0,
        "outlier_pct": 97,
        "min_area": 80,
    },
    "Thermal": {
        "sensors": ["COOLANT"],
        "color": "#3498DB",
        "hull_alpha": 0.08,
        "edge_width": 2.5,
        "label_bg": "#3498DB",
        "tighten": 0.8,
        "outlier_pct": 97,
        "min_area": 50,
    },
}

# Sensor names (must match order used in embeddings)
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

# Create sensor to subsystem mapping
SENSOR_TO_SUBSYSTEM = {}
for subsystem, sensors in SUBSYSTEM_SENSORS.items():
    for sensor in sensors:
        SENSOR_TO_SUBSYSTEM[sensor] = subsystem


def draw_subsystem_region(
    ax,
    pts,
    name,
    color,
    tighten=1.0,
    outlier_pct=97,
    min_area=50,
    hull_alpha=0.08,
    edge_width=2.5,
):
    """
    Draw a subsystem region using optimized alpha shapes.

    Args:
        ax: Matplotlib axis
        pts: (N, 2) array of t-SNE points for this subsystem
        name: Subsystem name for label
        color: Color for the region
        tighten: <1.0 = tighter, >1.0 = looser hull (multiplier on auto alpha)
        outlier_pct: Discard furthest X% of points before computing hull
        min_area: Minimum area to draw (skip tiny polygons)
        hull_alpha: Transparency for filled region
        edge_width: Line width for outline
    """
    if len(pts) < 10:
        return  # Not enough points for a sensible region

    # 1) Drop extreme outliers so they don't blow up the region
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    keep = d <= np.percentile(d, outlier_pct)
    pts_clean = pts[keep]

    if len(pts_clean) < 10:
        return

    # 2) Choose a good alpha automatically, then optionally tighten/loosen
    # Remove duplicate points first
    pts_clean_unique = np.unique(pts_clean, axis=0)
    if len(pts_clean_unique) < 3:
        return  # Need at least 3 unique points

    # Convert to list of tuples as expected by alphashape
    points_list = [(float(x), float(y)) for x, y in pts_clean_unique]

    # Use alphashape.optimizealpha to automatically find a good alpha value
    try:
        alpha_auto = alphashape.optimizealpha(points_list)
        # If optimizealpha returns 0 or None, it failed - use fallback
        if alpha_auto is None or alpha_auto == 0:
            raise ValueError("optimizealpha returned invalid value")
    except (ValueError, RuntimeError, TypeError, AttributeError):
        # Fallback: if optimizealpha fails, use a simple heuristic
        spread = np.percentile(np.linalg.norm(pts_clean_unique - center, axis=1), 95)
        alpha_auto = spread * 0.15

    alpha = alpha_auto * tighten

    # Use the same points_list format for alphashape
    shape = alphashape.alphashape(points_list, alpha)

    # We might get a MultiPolygon (disconnected blobs); draw each one
    geoms = shape.geoms if isinstance(shape, MultiPolygon) else [shape]

    for geom in geoms:
        if geom.area < min_area:
            continue  # Skip tiny islands

        x, y = geom.exterior.xy
        ax.fill(x, y, color=color, alpha=hull_alpha, zorder=1)
        ax.plot(x, y, color=color, lw=edge_width, ls="--", alpha=0.9, zorder=2)

    # 3) Put a label at the median of all subsystem points (not per blob)
    cx, cy = center
    ax.text(
        cx,
        cy,
        name,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            alpha=0.95,
        ),
        zorder=1000,
    )


def sensor_polygon(points, outlier_pct=98):
    """
    Create a small convex hull around ONE sensor cluster.

    Args:
        points: (N, 2) array of t-SNE points for this sensor
        outlier_pct: Percentile to trim extreme outliers (default 98)

    Returns:
        Polygon or None if insufficient points
    """
    if len(points) < 10:
        return None

    # Trim extreme outliers
    center = np.median(points, axis=0)
    d = np.linalg.norm(points - center, axis=1)
    keep = d <= np.percentile(d, outlier_pct)
    pts = points[keep]

    if len(pts) < 3:
        return None

    # Remove duplicate points
    pts_unique = np.unique(pts, axis=0)
    if len(pts_unique) < 3:
        return None

    # Compute convex hull
    try:
        hull = ConvexHull(pts_unique)
        return Polygon(pts_unique[hull.vertices])
    except (ValueError, QhullError):
        return None


def draw_subsystem_ellipse(
    ax,
    pts,
    color,
    chi2_val=5.99,  # ≈95% in 2D
    outlier_pct=97,
    fill_alpha=0.08,
    edge_alpha=0.9,
    lw=2.5,
):
    """
    Draw a smooth confidence ellipse for a subsystem.

    Args:
        ax: Matplotlib axis
        pts: (N, 2) t-SNE coordinates for one subsystem
        color: Color for the ellipse
        chi2_val: Controls ellipse size (5.99≈95%, 9.21≈99%, 4.0≈86%)
        outlier_pct: Percentile to trim extreme outliers
        fill_alpha: Alpha for filled background
        edge_alpha: Alpha for border lines
        lw: Line width for border
    """
    if len(pts) < 20:
        return

    # 1) Trim furthest outliers so they don't blow up the ellipse
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    keep = d <= np.percentile(d, outlier_pct)
    pts = pts[keep]
    if len(pts) < 10:
        return

    # 2) Covariance and eigendecomposition
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)  # vals ascending

    # Radii along principal axes
    r1, r2 = np.sqrt(vals * chi2_val)

    # 3) Parametric ellipse
    t = np.linspace(0, 2 * np.pi, 200)
    ellipse = np.column_stack([r1 * np.cos(t), r2 * np.sin(t)])
    ellipse = ellipse @ vecs.T + center

    x, y = ellipse[:, 0], ellipse[:, 1]

    ax.fill(x, y, color=color, alpha=fill_alpha, zorder=1)
    ax.plot(x, y, color=color, lw=lw, ls="--", alpha=edge_alpha, zorder=2)


def compute_separation_metrics(embeddings, sensor_ids, sensor_names):
    """
    Compute quantitative separation metrics for subsystems.

    Args:
        embeddings: (N, D) original embeddings (before t-SNE)
        sensor_ids: (N,) sensor indices 0-7
        sensor_names: List of sensor names

    Returns:
        Dictionary with separation metrics
    """
    # Create subsystem labels
    subsystem_labels = []
    subsystem_names = []
    for i, sensor_id in enumerate(sensor_ids):
        sensor_name = sensor_names[sensor_id]
        for subsystem, sensor_list in SUBSYSTEM_SENSORS.items():
            if sensor_name in sensor_list:
                subsystem_labels.append(subsystem)
                subsystem_names.append(subsystem)
                break

    subsystem_labels = np.array(subsystem_labels)
    unique_subsystems = np.unique(subsystem_labels)

    if len(unique_subsystems) < 2:
        return {"error": "Need at least 2 subsystems for separation metrics"}

    metrics = {}

    # 1) Silhouette score
    try:
        silhouette = silhouette_score(embeddings, subsystem_labels)
        metrics["silhouette_score"] = float(silhouette)
    except Exception as e:
        metrics["silhouette_error"] = str(e)

    # 2) Intra vs inter-subsystem distances
    intra_distances = []
    inter_distances = []

    for subsystem in unique_subsystems:
        mask = subsystem_labels == subsystem
        subsystem_pts = embeddings[mask]

        # Intra-subsystem distances
        if len(subsystem_pts) > 1:
            # Pairwise distances within subsystem
            intra_dists = pdist(subsystem_pts)
            intra_distances.extend(intra_dists)

        # Inter-subsystem distances (to other subsystems)
        other_mask = subsystem_labels != subsystem
        other_pts = embeddings[other_mask]
        if len(other_pts) > 0:
            # Distance from each point in this subsystem to nearest point in other subsystems
            inter_dists = cdist(subsystem_pts, other_pts).min(axis=1)
            inter_distances.extend(inter_dists)

    if intra_distances and inter_distances:
        metrics["mean_intra_distance"] = float(np.mean(intra_distances))
        metrics["mean_inter_distance"] = float(np.mean(inter_distances))
        metrics["separation_ratio"] = float(
            np.mean(inter_distances) / np.mean(intra_distances)
        )

    # 3) Nearest-centroid classifier accuracy
    try:
        clf = NearestCentroid()
        clf.fit(embeddings, subsystem_labels)
        predictions = clf.predict(embeddings)
        accuracy = np.mean(predictions == subsystem_labels)
        metrics["nearest_centroid_accuracy"] = float(accuracy)

        # Per-subsystem accuracy
        per_subsystem_acc = {}
        for subsystem in unique_subsystems:
            mask = subsystem_labels == subsystem
            if mask.sum() > 0:
                acc = np.mean(predictions[mask] == subsystem_labels[mask])
                per_subsystem_acc[subsystem] = float(acc)
        metrics["per_subsystem_accuracy"] = per_subsystem_acc
    except Exception as e:
        metrics["classifier_error"] = str(e)

    return metrics


def draw_subsystem_regions(ax, embeddings_2d, sensor_ids, sensor_names):
    """
    Draw filled background regions per subsystem using confidence ellipses.
    Tighter ellipses for Powertrain and Fuel System, with adjusted alpha.

    Args:
        ax: Matplotlib axis
        embeddings_2d: (N, 2) t-SNE coordinates
        sensor_ids: (N,) sensor indices 0-7
        sensor_names: List of sensor names
    """
    for subsystem, sensor_list in SUBSYSTEM_SENSORS.items():
        idxs = [sensor_names.index(s) for s in sensor_list]
        mask = np.isin(sensor_ids, idxs)
        pts = embeddings_2d[mask]

        if len(pts) < 20:
            continue

        color = SUBSYSTEM_COLORS[subsystem]

        # Tighter ellipses for Powertrain and Fuel System
        if subsystem == "Fuel System":
            chi2 = 3.5  # Tighter
            outlier_pct = 95  # More aggressive outlier trimming
            fill_alpha = 0.04  # Faded
        elif subsystem == "Powertrain":
            chi2 = 4.5  # Tighter
            outlier_pct = 95
            fill_alpha = 0.04  # Faded
        else:  # Thermal
            chi2 = 5.99  # ≈95%
            outlier_pct = 97
            fill_alpha = 0.08  # More prominent

        draw_subsystem_ellipse(
            ax,
            pts,
            color=color,
            chi2_val=chi2,
            outlier_pct=outlier_pct,
            fill_alpha=fill_alpha,
            edge_alpha=0.9,
        )

        # Label at median
        center = np.median(pts, axis=0)
        ax.text(
            center[0],
            center[1],
            subsystem,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
                alpha=0.95,
            ),
            zorder=1000,
        )


def add_subsystem_overlays(embeddings_2d, sensor_ids, sensor_names, ax):
    """
    Add optimized alpha shape subsystem overlays with:
    - Automatic alpha optimization per subsystem
    - Outlier trimming to prevent regions from being too large
    - Support for disconnected regions (MultiPolygon)
    - Clear visual hierarchy
    """

    for subsystem, info in SUBSYSTEMS.items():
        # Collect points for this subsystem
        subsystem_points = []

        for sensor_name in info["sensors"]:
            sensor_idx = sensor_names.index(sensor_name)
            mask = sensor_ids == sensor_idx
            points = embeddings_2d[mask]

            if len(points) > 0:
                subsystem_points.append(points)

        if not subsystem_points:
            continue

        subsystem_points = np.vstack(subsystem_points)

        # Draw subsystem region using optimized alpha shapes
        draw_subsystem_region(
            ax,
            subsystem_points,
            name=subsystem,
            color=info["color"],
            tighten=info.get("tighten", 1.0),
            outlier_pct=info.get("outlier_pct", 97),
            min_area=info.get("min_area", 50),
            hull_alpha=info["hull_alpha"],
            edge_width=info["edge_width"],
        )

        # Add sensor list below main label (at centroid)
        subsystem_centroid = np.median(subsystem_points, axis=0)
        sensor_list = ", ".join(info["sensors"])
        ax.text(
            subsystem_centroid[0],
            subsystem_centroid[1] - 12,
            f"({sensor_list})",
            fontsize=9,
            ha="center",
            va="top",
            color=info["color"],
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=info["color"],
                linewidth=1.5,
                alpha=0.95,
            ),
            zorder=999,
        )


def plot_tsne_with_subsystems_minimal(
    embeddings_2d,
    sensor_ids,
    labels,
    sensor_names,
    save_path="figures/tsne_subsystems.png",
    figsize=(16, 12),
):
    """
    Create t-SNE plot with detailed subsystem overlays.
    Matches your original style exactly, adds clear subsystem boundaries.

    Args:
        embeddings_2d: (N, 2) t-SNE coordinates
        sensor_ids: (N,) sensor indices 0-7
        labels: (N,) anomaly labels (0=normal, 1=anomaly)
        sensor_names: List of 8 sensor names
        save_path: Path to save the figure
        figsize: Figure size tuple
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot sensor points (EXACTLY like original)
    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for i, sensor_name in enumerate(sensor_names):
        mask = sensor_ids == i
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            alpha=0.6,
            s=30,
            label=sensor_name,
            edgecolors="none",
            zorder=100,
        )

    # Add subsystem overlays (BEHIND the points)
    add_subsystem_overlays(embeddings_2d, sensor_ids, sensor_names, ax)

    # Formatting (match original, enhanced)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight="bold")
    ax.set_title(
        "Sensor Embedding Visualization (t-SNE)\nBy Sensor Type",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=11,
        ncol=2,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)  # Grid behind everything

    # Adjust limits slightly for better view
    x_margin = (embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()) * 0.05
    y_margin = (embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()) * 0.05
    ax.set_xlim(
        embeddings_2d[:, 0].min() - x_margin, embeddings_2d[:, 0].max() + x_margin
    )
    ax.set_ylim(
        embeddings_2d[:, 1].min() - y_margin, embeddings_2d[:, 1].max() + y_margin
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved detailed subsystem visualization to {save_path}")
    plt.close()

    return fig


def plot_tsne_with_sensor_hulls(
    embeddings_2d,
    sensor_ids,
    labels,
    sensor_names,
    save_path="figures/tsne_subsystems_fast.png",
    figsize=(16, 10),
    embeddings_original=None,
    two_panel=False,
):
    """
    Create t-SNE plot with subsystem regions using confidence ellipses.

    Args:
        embeddings_2d: (N, 2) t-SNE coordinates
        sensor_ids: (N,) sensor indices 0-7
        labels: (N,) anomaly labels (unused here, but kept for compatibility)
        sensor_names: List of 8 sensor names
        save_path: Path to save the figure
        figsize: Figure size tuple
        embeddings_original: (N, D) original embeddings for computing metrics
        two_panel: If True, create two panels (with and without ellipses)
    """
    if two_panel:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 10))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # 1) Draw background subsystem regions FIRST (so they're really in the back)
    # Use confidence ellipses for all subsystems (smooth, compact, no spiky edges)
    draw_subsystem_regions(ax1, embeddings_2d, sensor_ids, sensor_names)

    # 2) Draw your usual scatter on top (exactly as you had it)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_names)))
    for i, name in enumerate(sensor_names):
        m = sensor_ids == i
        ax1.scatter(
            embeddings_2d[m, 0],
            embeddings_2d[m, 1],
            c=[colors[i]],
            s=20,
            alpha=0.7,
            edgecolors="none",
            label=name,
            zorder=100,
        )

    ax1.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax1.set_title(
        "Sensor Embedding Visualization (t-SNE)\nBy Sensor Type",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # Second panel: just points, no ellipses
    if ax2 is not None:
        for i, name in enumerate(sensor_names):
            m = sensor_ids == i
            # Color by subsystem
            subsystem = SENSOR_TO_SUBSYSTEM.get(name, "Unknown")
            color = SUBSYSTEM_COLORS.get(subsystem, "gray")
            ax2.scatter(
                embeddings_2d[m, 0],
                embeddings_2d[m, 1],
                c=[color],
                s=20,
                alpha=0.7,
                edgecolors="none",
                label=name if i < len(sensor_names) else None,
                zorder=100,
            )

        ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax2.set_title(
            "Sensor Embedding Visualization (t-SNE)\nPoints Only (No Background Regions)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.legend(loc="upper right", ncol=2, framealpha=0.95)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved sensor hull visualization to {save_path}")
    plt.close()

    # Compute and print separation metrics if original embeddings provided
    if embeddings_original is not None:
        print("\n" + "=" * 60)
        print("SUBSYSTEM SEPARATION METRICS")
        print("=" * 60)
        metrics = compute_separation_metrics(
            embeddings_original, sensor_ids, sensor_names
        )
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60 + "\n")

    return fig


# ============================================================================
# Data Loading and Embedding Extraction
# ============================================================================


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """Load GDN model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Read dimensions from checkpoint if available
    embed_dim = checkpoint.get("embed_dim", EMBED_DIM)
    hidden_dim = checkpoint.get("hidden_dim", HIDDEN_DIM)
    top_k = checkpoint.get("top_k", TOP_K)
    window_size = checkpoint.get("window_size", WINDOW_SIZE)

    # Try to infer number of GAT heads from checkpoint
    state_dict = checkpoint.get("base_model_state_dict") or checkpoint.get(
        "model_state_dict", {}
    )
    num_heads = 2  # Default
    if "gat.att_src" in state_dict:
        # Shape is [1, num_heads, hidden_dim]
        num_heads = state_dict["gat.att_src"].shape[1]
        print(f"Inferred {num_heads} GAT attention heads from checkpoint")

    print(
        f"Model config: embed_dim={embed_dim}, hidden_dim={hidden_dim}, top_k={top_k}, window_size={window_size}, num_heads={num_heads}"
    )

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

    # Initialize model with dimensions from checkpoint
    # Note: KAGOptimizedGDN hardcodes heads=2, so if checkpoint has different heads,
    # we'll need to filter those layers or use a compatible checkpoint
    if checkpoint_model_type == "kag_optimized":
        print("Using KAG-Optimized model")
        # If checkpoint has different number of heads, we'll filter incompatible layers
        model = KAGOptimizedGDN(
            num_nodes=NUM_SENSORS,
            window_size=window_size,
            embed_dim=embed_dim,
            top_k=top_k,
            hidden_dim=hidden_dim,
        ).to(device)
    else:
        print("Using Enhanced MultiLabelGDN model")
        model = MultiLabelGDN(
            num_nodes=NUM_SENSORS,
            window_size=window_size,
            embed_dim=embed_dim,
            top_k=top_k,
            hidden_dim=hidden_dim,
        ).to(device)

    # Load model state with filtering for incompatible layers
    if "base_model_state_dict" in checkpoint:
        base_state = checkpoint["base_model_state_dict"]
        # Filter out incompatible layers
        if checkpoint_model_type == "kag_optimized":
            # Filter out layers that don't match the model structure
            filtered_state = {}
            model_state = model.state_dict()
            for k, v in base_state.items():
                if k in model_state:
                    if model_state[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        print(
                            f"Skipping {k} due to shape mismatch: {model_state[k].shape} vs {v.shape}"
                        )
                elif not any(x in k for x in ["temporal_pooling", "multi_scale_gat"]):
                    # Try to load if it's not a known incompatible layer
                    pass
        else:
            filtered_state = base_state
        model.load_state_dict(filtered_state, strict=False)
    elif "model_state_dict" in checkpoint:
        # Filter incompatible layers
        state_to_load = {}
        model_state = model.state_dict()
        for k, v in checkpoint["model_state_dict"].items():
            if k in model_state and model_state[k].shape == v.shape:
                state_to_load[k] = v
            elif k not in model_state:
                # Skip layers not in model
                pass
            else:
                print(
                    f"Skipping {k} due to shape mismatch: {model_state[k].shape} vs {v.shape}"
                )
        model.load_state_dict(state_to_load, strict=False)

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


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE embeddings with subsystem overlays"
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
        default="figures/tsne_subsystems.png",
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
    parser.add_argument(
        "--style",
        type=str,
        choices=["subsystem", "sensor"],
        default="subsystem",
        help="Visualization style: 'subsystem' (large subsystem regions) or 'sensor' (small per-sensor hulls)",
    )
    parser.add_argument(
        "--two_panel",
        action="store_true",
        help="Create two-panel figure (with and without ellipses)",
    )

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

    # Compute t-SNE
    print("Computing t-SNE (this may take a few minutes)...")

    # Store original embeddings before subsampling (for metrics)
    embeddings_original = embeddings.copy()

    # Subsample for faster t-SNE if too many points
    if len(embeddings) > 20000:
        print(f"Subsampling from {len(embeddings)} to 20000 points for t-SNE...")
        indices = np.random.choice(len(embeddings), 20000, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        sensor_ids = sensor_ids[indices]
        # Also subsample original embeddings for metrics (to match)
        embeddings_original = embeddings_original[indices]

    # Compute t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    print("✓ t-SNE completed")

    # Create visualization based on style choice
    if args.style == "sensor":
        plot_tsne_with_sensor_hulls(
            embeddings_2d,
            sensor_ids,
            labels,
            SENSOR_NAMES,
            save_path=args.output,
            embeddings_original=embeddings_original,
            two_panel=args.two_panel,
        )
    else:
        plot_tsne_with_subsystems_minimal(
            embeddings_2d, sensor_ids, labels, SENSOR_NAMES, save_path=args.output
        )

    print(f"\n✓ Visualization saved to {args.output}")


# Usage example
if __name__ == "__main__":
    main()
