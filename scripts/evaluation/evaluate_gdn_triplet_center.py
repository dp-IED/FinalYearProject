#!/usr/bin/env python3
"""
Evaluation script for GDN models trained with Triplet-Center Loss or Center Loss with Repulsion.

Evaluates trained models and analyzes:
- Center separation (should be >2.0)
- Embedding distributions (normal vs anomalous)
- Classification metrics (AUC, F1, precision, recall)
- ROC curves and separation analysis
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
)
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))
from models.gdn_model import (
    MultiLabelGDN,
    TripletCenterLoss,
    CenterLossWithRepulsion,
)

# ============================================================================
# Constants
# ============================================================================

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


# ============================================================================
# Helper Functions
# ============================================================================


def load_model_and_loss(checkpoint_path, device="cpu"):
    """
    Load model and loss function from checkpoint.
    
    Returns:
        model: Loaded MultiLabelGDN model
        center_loss: Loaded TripletCenterLoss or CenterLossWithRepulsion
        loss_type: 'triplet_center' or 'center_repulsion'
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine loss type from checkpoint
    if "triplet_center_loss_state_dict" in checkpoint:
        loss_type = "triplet_center"
        loss_state_key = "triplet_center_loss_state_dict"
    elif "center_loss_state_dict" in checkpoint:
        loss_type = "center_repulsion"
        loss_state_key = "center_loss_state_dict"
    else:
        raise ValueError("Checkpoint does not contain recognized loss function state dict")
    
    # Load model
    embed_dim = checkpoint.get("embed_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    window_size = checkpoint.get("window_size", 300)
    top_k = checkpoint.get("top_k", 3)
    num_sensors = len(checkpoint.get("sensor_names", SENSOR_COLS))
    
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=top_k,
        hidden_dim=hidden_dim,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load loss function
    if loss_type == "triplet_center":
        center_loss = TripletCenterLoss(
            embed_dim=hidden_dim,
            num_classes=2,
            margin=2.0,  # Default, may be in checkpoint
            lambda_c=1.5,  # Default, may be in checkpoint
        ).to(device)
    else:
        center_loss = CenterLossWithRepulsion(
            embed_dim=hidden_dim,
            num_classes=2,
            alpha=0.5,  # Default
            beta=1.0,  # Default
        ).to(device)
    
    center_loss.load_state_dict(checkpoint[loss_state_key])
    center_loss.eval()
    
    return model, center_loss, loss_type, checkpoint


def extract_embeddings_and_distances(model, X_windows, center_loss, device="cpu", batch_size=32):
    """
    Extract embeddings and compute distances to centers.
    
    Returns:
        embeddings: (N, hidden_dim) array
        distances_to_normal: (N,) array
        distances_to_anomalous: (N,) array
        center_separation: float
    """
    X_tensor = torch.from_numpy(X_windows).float().to(device)
    num_windows = len(X_tensor)
    
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_windows, batch_size), desc="Extracting embeddings"):
            batch = X_tensor[i : i + batch_size]
            embeddings = model.get_embeddings(batch)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, hidden_dim)
    
    # Get centers
    normal_center = center_loss.centers[0].detach().cpu().numpy()
    anomalous_center = center_loss.centers[1].detach().cpu().numpy()
    
    # Compute distances
    distances_to_normal = np.linalg.norm(all_embeddings - normal_center, axis=1)
    distances_to_anomalous = np.linalg.norm(all_embeddings - anomalous_center, axis=1)
    
    # Center separation
    center_separation = np.linalg.norm(normal_center - anomalous_center)
    
    return all_embeddings, distances_to_normal, distances_to_anomalous, center_separation


def compute_classification_metrics(y_true, scores, threshold=None):
    """
    Compute classification metrics from anomaly scores.
    
    Args:
        y_true: Binary labels (1 = anomaly, 0 = normal)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Optional threshold. If None, uses F1-optimal threshold.
    
    Returns:
        Dictionary of metrics and optimal threshold
    """
    if threshold is None:
        # Find F1-optimal threshold
        thresholds = np.linspace(np.min(scores), np.max(scores), 1000)
        best_f1 = 0
        best_threshold = np.median(scores)
        
        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            if len(np.unique(y_pred)) > 1:  # Need both classes
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
        threshold = best_threshold
    
    # Compute predictions
    y_pred = (scores >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # AUC metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    
    # ROC-AUC and PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)
            
            metrics["roc_auc"] = float(roc_auc)
            metrics["pr_auc"] = float(pr_auc)
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        else:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    except Exception as e:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        metrics["auc_error"] = str(e)
    
    return metrics


def plot_embedding_distributions(
    distances_to_normal, labels, center_separation, output_path, loss_type="triplet_center"
):
    """
    Plot embedding distance distributions for normal and anomalous samples.
    """
    normal_distances = distances_to_normal[labels == 0]
    anomalous_distances = distances_to_normal[labels == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(
        normal_distances,
        bins=50,
        alpha=0.6,
        label=f"Normal (mean={normal_distances.mean():.4f})",
        color="blue",
        density=True,
    )
    ax1.hist(
        anomalous_distances,
        bins=50,
        alpha=0.6,
        label=f"Anomalous (mean={anomalous_distances.mean():.4f})",
        color="red",
        density=True,
    )
    ax1.axvline(normal_distances.mean(), color="blue", linestyle="--", linewidth=2)
    ax1.axvline(anomalous_distances.mean(), color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Distance to Normal Center")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Embedding Distance Distribution\nCenter Separation: {center_separation:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [normal_distances, anomalous_distances]
    bp = ax2.boxplot(data_to_plot, labels=["Normal", "Anomalous"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax2.set_ylabel("Distance to Normal Center")
    ax2.set_title("Distance Distribution Comparison")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved embedding distribution plot to {output_path}")


def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved ROC curve to {output_path}")


# ============================================================================
# Main Evaluation Function
# ============================================================================


def evaluate_model(checkpoint_path, X_test, y_test, output_dir=None, device="cpu"):
    """
    Evaluate trained model with triplet-center or center-repulsion loss.
    
    Args:
        checkpoint_path: Path to model checkpoint
        X_test: Test windows (N, W, D) numpy array
        y_test: Test labels (N,) numpy array (0=normal, 1=anomalous)
        output_dir: Directory to save results and plots
        device: Device to use
    
    Returns:
        Dictionary of evaluation results
    """
    print(f"\n{'='*80}")
    print("Evaluating GDN Model with Improved Metric Learning")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load model and loss
    print("Loading model and loss function...")
    model, center_loss, loss_type, checkpoint = load_model_and_loss(checkpoint_path, device)
    print(f"  ✓ Loaded model with {loss_type} loss")
    print(f"  Embedding dim: {checkpoint.get('embed_dim', 'N/A')}")
    print(f"  Hidden dim: {checkpoint.get('hidden_dim', 'N/A')}")
    
    # Extract embeddings and compute distances
    print("\nExtracting embeddings and computing distances...")
    embeddings, distances_to_normal, distances_to_anomalous, center_separation = (
        extract_embeddings_and_distances(model, X_test, center_loss, device)
    )
    
    print(f"\n  Center separation: {center_separation:.4f}")
    print(f"  Target: >2.0 (current: {center_separation:.4f})")
    
    # Compute statistics
    normal_mask = y_test == 0
    anomalous_mask = y_test == 1
    
    normal_distances = distances_to_normal[normal_mask]
    anomalous_distances = distances_to_normal[anomalous_mask]
    
    normal_mean = normal_distances.mean()
    anomalous_mean = anomalous_distances.mean()
    normal_std = normal_distances.std()
    anomalous_std = anomalous_distances.std()
    
    print(f"\n  Distance Statistics:")
    print(f"    Normal mean: {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"    Anomalous mean: {anomalous_mean:.4f} ± {anomalous_std:.4f}")
    print(f"    Separation ratio: {anomalous_mean / normal_mean:.2f}×" if normal_mean > 0 else "    N/A")
    
    # Use distance to normal center as anomaly score
    anomaly_scores = distances_to_normal
    
    # Compute classification metrics
    print("\nComputing classification metrics...")
    metrics = compute_classification_metrics(y_test, anomaly_scores)
    
    print(f"\n  Classification Metrics:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1 Score: {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"    Optimal Threshold: {metrics['threshold']:.4f}")
    
    # Prepare results dictionary
    results = {
        "checkpoint_path": str(checkpoint_path),
        "loss_type": loss_type,
        "center_separation": float(center_separation),
        "embedding_stats": {
            "normal_mean": float(normal_mean),
            "normal_std": float(normal_std),
            "anomalous_mean": float(anomalous_mean),
            "anomalous_std": float(anomalous_std),
            "separation_ratio": float(anomalous_mean / normal_mean) if normal_mean > 0 else 0.0,
        },
        "classification_metrics": metrics,
        "model_config": {
            "embed_dim": checkpoint.get("embed_dim", "N/A"),
            "hidden_dim": checkpoint.get("hidden_dim", "N/A"),
            "lambda_center": checkpoint.get("lambda_center", "N/A"),
            "lambda_global": checkpoint.get("lambda_global", "N/A"),
        },
    }
    
    # Generate plots if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating plots...")
        
        # Embedding distribution plot
        dist_plot_path = output_dir / "embedding_distributions.png"
        plot_embedding_distributions(
            distances_to_normal, y_test, center_separation, dist_plot_path, loss_type
        )
        
        # ROC curve
        if "roc_curve" in metrics:
            roc_plot_path = output_dir / "roc_curve.png"
            plot_roc_curve(
                np.array(metrics["roc_curve"]["fpr"]),
                np.array(metrics["roc_curve"]["tpr"]),
                metrics["roc_auc"],
                roc_plot_path,
            )
        
        # Save results JSON
        results_json_path = output_dir / "evaluation_results.json"
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved results to {results_json_path}")
    
    print(f"\n{'='*80}")
    print("Evaluation Complete")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# Main Function
# ============================================================================


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate GDN model trained with Triplet-Center Loss or Center Loss with Repulsion"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data numpy file (.npz)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Device detection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = np.load(args.test_data)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    
    print(f"  Test windows: {len(X_test)}")
    print(f"  Normal: {(y_test == 0).sum()}, Anomalous: {(y_test == 1).sum()}")
    
    # Evaluate
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir,
        device=device,
    )
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
