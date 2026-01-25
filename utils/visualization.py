"""
Visualization utilities for embedding space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from matplotlib.figure import Figure

# Try to import UMAP, fallback to t-SNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    try:
        from sklearn.manifold import TSNE
        HAS_TSNE = True
    except ImportError:
        HAS_TSNE = False


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    title: str = "Window Embedding Space",
    save_path: Optional[str] = None
) -> Figure:
    """
    Visualize window embeddings in 2D using UMAP or t-SNE projection.
    
    Args:
        embeddings: (N, hidden_dim) array - window embeddings
        labels: (N,) array - binary labels (0=normal, 1=anomalous)
        centers: (2, hidden_dim) array - class centers [normal, anomalous]
        title: Plot title
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Project to 2D
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
        centers_2d = reducer.transform(centers)
        projection_method = "UMAP"
    elif HAS_TSNE:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
        centers_2d = reducer.transform(centers)
        projection_method = "t-SNE"
    else:
        raise ImportError("Neither UMAP nor sklearn.manifold.TSNE available. Please install umap-learn or scikit-learn.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Separate normal and anomalous windows
    normal_mask = labels == 0
    anomalous_mask = labels == 1
    
    # Plot normal windows
    if normal_mask.any():
        ax.scatter(
            embeddings_2d[normal_mask, 0],
            embeddings_2d[normal_mask, 1],
            c='blue',
            alpha=0.6,
            s=30,
            label='Normal Windows',
            edgecolors='darkblue',
            linewidths=0.5
        )
    
    # Plot anomalous windows
    if anomalous_mask.any():
        ax.scatter(
            embeddings_2d[anomalous_mask, 0],
            embeddings_2d[anomalous_mask, 1],
            c='red',
            alpha=0.6,
            s=30,
            label='Anomalous Windows',
            edgecolors='darkred',
            linewidths=0.5
        )
    
    # Plot class centers as stars
    ax.scatter(
        centers_2d[0, 0],
        centers_2d[0, 1],
        c='blue',
        marker='*',
        s=500,
        label='Normal Center',
        edgecolors='darkblue',
        linewidths=2,
        zorder=10
    )
    
    ax.scatter(
        centers_2d[1, 0],
        centers_2d[1, 1],
        c='red',
        marker='*',
        s=500,
        label='Anomalous Center',
        edgecolors='darkred',
        linewidths=2,
        zorder=10
    )
    
    # Compute decision boundary (perpendicular bisector of centers)
    center_midpoint = (centers_2d[0] + centers_2d[1]) / 2
    center_vector = centers_2d[1] - centers_2d[0]
    center_distance = np.linalg.norm(center_vector)
    
    if center_distance > 0:
        # Perpendicular vector
        perp_vector = np.array([-center_vector[1], center_vector[0]]) / center_distance
        
        # Extend line beyond plot bounds
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        plot_size = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        
        # Draw decision boundary line
        line_length = plot_size * 1.5
        boundary_x = [center_midpoint[0] - perp_vector[0] * line_length,
                     center_midpoint[0] + perp_vector[0] * line_length]
        boundary_y = [center_midpoint[1] - perp_vector[1] * line_length,
                     center_midpoint[1] + perp_vector[1] * line_length]
        
        ax.plot(
            boundary_x,
            boundary_y,
            'k--',
            alpha=0.5,
            linewidth=2,
            label='Decision Boundary'
        )
    
    # Compute center distances for legend
    normal_distances = []
    anomalous_distances = []
    for i in range(len(embeddings)):
        if normal_mask[i]:
            dist = np.linalg.norm(embeddings[i] - centers[0])
            normal_distances.append(dist)
        elif anomalous_mask[i]:
            dist = np.linalg.norm(embeddings[i] - centers[1])
            anomalous_distances.append(dist)
    
    mean_normal_dist = np.mean(normal_distances) if normal_distances else 0.0
    mean_anomalous_dist = np.mean(anomalous_distances) if anomalous_distances else 0.0
    
    # Add legend with center distances
    legend = ax.legend(loc='upper right', fontsize=10)
    legend_text = (
        f"\nMean distances:\n"
        f"Normal: {mean_normal_dist:.3f}\n"
        f"Anomalous: {mean_anomalous_dist:.3f}"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'{projection_method} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{projection_method} Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved embedding visualization to {save_path}")
    
    return fig
