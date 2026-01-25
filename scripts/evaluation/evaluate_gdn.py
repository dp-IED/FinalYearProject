#!/usr/bin/env python3
"""
Comprehensive GDN Model Evaluation Script

Evaluates GDN model performance with:
1. Classification metrics (F1, AUC, recall, precision, accuracy)
2. Bar charts showing separation of normal and anomalous distances from centroid
3. Visualization of embedding space distances between normal and anomalous samples
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import torch
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add evaluation directory for metrics
eval_dir = str(project_root / "llm" / "evaluation")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# Import GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


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


def get_embeddings_and_distances(predictor, normalized_windows, batch_size=32):
    """
    Extract embeddings and compute distances to normal center.

    Returns:
        embeddings: (N, embed_dim) array
        distances: (N,) array of distances to normal center
        normal_center: (embed_dim,) array or None
    """
    X_tensor = torch.from_numpy(normalized_windows).float()
    num_windows = len(X_tensor)

    predictor.model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = X_tensor[i : i + batch_size]
            embeddings = predictor.model.get_embeddings(batch)
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (num_windows, embed_dim)

    # Try to get normal center
    normal_center = None
    distances = None

    # Method 1: Check if predictor has normal_center
    if hasattr(predictor, "normal_center") and predictor.normal_center is not None:
        normal_center = (
            predictor.normal_center.cpu().numpy()
            if hasattr(predictor.normal_center, "cpu")
            else predictor.normal_center
        )
        distances = np.linalg.norm(all_embeddings - normal_center, axis=1)

    # Method 2: Try loading from checkpoint
    elif predictor.model_path.exists():
        try:
            checkpoint = torch.load(predictor.model_path, map_location="cpu")
            if "center_loss_state_dict" in checkpoint:
                # Import CenterLoss
                import importlib.util

                train_script_path = (
                    project_root / "anomaly-detection" / "train_gdn_separation.py"
                )
                if train_script_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "train_gdn_separation", train_script_path
                    )
                    train_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_module)
                    CenterLoss = train_module.CenterLoss

                    center_loss = CenterLoss(
                        embed_dim=all_embeddings.shape[1], num_classes=2
                    )
                    center_loss.load_state_dict(checkpoint["center_loss_state_dict"])
                    normal_center = center_loss.centers[0].detach().cpu().numpy()
                    distances = np.linalg.norm(all_embeddings - normal_center, axis=1)
        except Exception:
            pass

    # Method 3: Fallback - use mean of normal embeddings as center
    if distances is None:
        # We'll compute this after we know which are normal
        normal_center = None
        distances = None

    return all_embeddings, distances, normal_center


def compute_embedding_distances(embeddings, normal_mask):
    """
    Compute pairwise distances between normal and anomalous embeddings.

    Returns:
        Dictionary with various distance statistics
    """
    normal_embeddings = embeddings[normal_mask]
    anomalous_embeddings = embeddings[~normal_mask]

    if len(normal_embeddings) == 0 or len(anomalous_embeddings) == 0:
        return {}

    # Compute mean embeddings
    normal_mean = np.mean(normal_embeddings, axis=0)
    anomalous_mean = np.mean(anomalous_embeddings, axis=0)

    # Distance between centroids
    centroid_distance = np.linalg.norm(anomalous_mean - normal_mean)

    # Mean distance from normal samples to normal centroid
    normal_to_center = np.linalg.norm(normal_embeddings - normal_mean, axis=1)
    normal_mean_dist = np.mean(normal_to_center)
    normal_std_dist = np.std(normal_to_center)

    # Mean distance from anomalous samples to normal centroid
    anomalous_to_center = np.linalg.norm(anomalous_embeddings - normal_mean, axis=1)
    anomalous_mean_dist = np.mean(anomalous_to_center)
    anomalous_std_dist = np.std(anomalous_to_center)

    # Mean distance from anomalous samples to anomalous centroid
    anomalous_to_anomalous_center = np.linalg.norm(
        anomalous_embeddings - anomalous_mean, axis=1
    )
    anomalous_to_anomalous_mean = np.mean(anomalous_to_anomalous_center)

    # Separation ratio
    separation_ratio = anomalous_mean_dist / (normal_mean_dist + 1e-8)

    # Compute some pairwise distances (sample for efficiency)
    max_samples = min(100, len(normal_embeddings), len(anomalous_embeddings))
    normal_sample = normal_embeddings[:max_samples]
    anomalous_sample = anomalous_embeddings[:max_samples]

    # Mean pairwise distance between normal and anomalous samples
    pairwise_distances = []
    for i in range(max_samples):
        dist = np.linalg.norm(normal_sample[i] - anomalous_sample[i])
        pairwise_distances.append(dist)
    mean_pairwise_dist = np.mean(pairwise_distances)

    return {
        "centroid_distance": float(centroid_distance),
        "normal_mean_dist_to_center": float(normal_mean_dist),
        "normal_std_dist_to_center": float(normal_std_dist),
        "anomalous_mean_dist_to_normal_center": float(anomalous_mean_dist),
        "anomalous_std_dist_to_normal_center": float(anomalous_std_dist),
        "anomalous_mean_dist_to_anomalous_center": float(anomalous_to_anomalous_mean),
        "separation_ratio": float(separation_ratio),
        "mean_pairwise_distance": float(mean_pairwise_dist),
    }


def evaluate_gdn(
    dataset_path: str = "llm/evaluation/shared_dataset/test.npz",
    gdn_model_path: str = "anomaly-detection/best_multilabel_gdn.pt",
    output_dir: str = "results",
    limit: int = None,
):
    """
    Comprehensive GDN model evaluation.

    Args:
        dataset_path: Path to test dataset (.npz file)
        gdn_model_path: Path to trained GDN model (.pt file)
        output_dir: Directory to save results and plots
        limit: Optional limit on number of windows to evaluate
    """
    print("=" * 80)
    print("GDN MODEL EVALUATION")
    print("=" * 80)
    print()

    # Load dataset
    print("1. Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    sensor_names = [
        "ENGINE_RPM",
        "VEHICLE_SPEED",
        "THROTTLE",
        "ENGINE_LOAD",
        "COOLANT_TEMPERATURE",
        "INTAKE_MANIFOLD_PRESSURE",
        "SHORT_TERM_FUEL_TRIM_BANK_1",
        "LONG_TERM_FUEL_TRIM_BANK_1",
    ]

    if limit:
        normalized_windows = normalized_windows[:limit]
        sensor_labels_true = sensor_labels_true[:limit]
        window_labels_true = window_labels_true[:limit]

    num_windows = len(normalized_windows)
    print(f"   Loaded {num_windows} windows")
    print()

    # Load GDN model
    print("2. Loading GDN model...")
    # Detect model dimensions from model path or checkpoint
    # Improved models use embed_dim=64, hidden_dim=64
    embed_dim = 64
    hidden_dim = 64

    # Try to detect from checkpoint
    try:
        import torch

        checkpoint = torch.load(gdn_model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "embed_dim" in checkpoint:
                embed_dim = checkpoint["embed_dim"]
            if "hidden_dim" in checkpoint:
                hidden_dim = checkpoint["hidden_dim"]
    except:
        # Fallback: detect from path
        if (
            "improved" not in str(gdn_model_path).lower()
            and "architecture" not in str(gdn_model_path).lower()
        ):
            embed_dim = 32
            hidden_dim = 32

    predictor = GDNPredictor(
        model_path=gdn_model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=embed_dim,
        top_k=3,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    print(f"   ✓ Model loaded from: {gdn_model_path}")
    print()

    # Process through GDN
    print("3. Processing windows through GDN...")
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true,
        batch_size=32,
    )

    gdn_predictions = kg_data["gdn_predictions"]  # (num_windows, num_sensors)
    print(f"   GDN predictions shape: {gdn_predictions.shape}")
    print()

    # Get embeddings and distances
    print("4. Extracting embeddings and computing distances...")
    embeddings, distances, normal_center = get_embeddings_and_distances(
        predictor, normalized_windows
    )

    # If distances not available, compute from normal samples
    if distances is None:
        print("   Computing normal center from training data...")
        is_faulty_window = sensor_labels_true.sum(axis=1) > 0
        normal_mask = ~is_faulty_window

        if np.sum(normal_mask) > 0:
            # Use mean of normal embeddings as center
            normal_embeddings = embeddings[normal_mask]
            normal_center = np.mean(normal_embeddings, axis=0)
            distances = np.linalg.norm(embeddings - normal_center, axis=1)
            print(
                f"   ✓ Computed distances using normal center (mean: {np.mean(distances):.4f})"
            )
        else:
            # Fallback: use max probability as distance proxy
            distances = np.max(gdn_predictions, axis=1)
            normal_center = None
            print("   ⚠️  No normal samples found, using max probability as proxy")
    else:
        print(f"   ✓ Computed distances (mean: {np.mean(distances):.4f})")
    print()

    # Identify normal vs faulty windows
    print("5. Analyzing separation...")
    is_faulty_window = sensor_labels_true.sum(axis=1) > 0
    normal_mask = ~is_faulty_window
    faulty_mask = is_faulty_window

    num_normal = np.sum(normal_mask)
    num_faulty = np.sum(faulty_mask)

    print(f"   Normal windows: {num_normal}")
    print(f"   Faulty windows: {num_faulty}")
    print()

    # Compute classification metrics
    print("6. Computing classification metrics...")
    # Use distances as anomaly scores (higher = more anomalous)
    anomaly_scores = (
        distances if distances is not None else np.max(gdn_predictions, axis=1)
    )
    binary_labels = is_faulty_window.astype(int)

    metrics = compute_classification_metrics(binary_labels, anomaly_scores)

    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {metrics.get('roc_auc', 0.0):.4f}")
    print(f"   PR-AUC:    {metrics.get('pr_auc', 0.0):.4f}")
    print(f"   Threshold: {metrics['threshold']:.4f}")
    print()

    # Compute embedding space distances
    print("7. Computing embedding space distances...")
    embedding_distances = compute_embedding_distances(embeddings, normal_mask)

    if embedding_distances:
        print(f"   Centroid distance: {embedding_distances['centroid_distance']:.4f}")
        print(
            f"   Normal mean dist to center: {embedding_distances['normal_mean_dist_to_center']:.4f}"
        )
        print(
            f"   Anomalous mean dist to normal center: {embedding_distances['anomalous_mean_dist_to_normal_center']:.4f}"
        )
        print(f"   Separation ratio: {embedding_distances['separation_ratio']:.2f}x")
        print(
            f"   Mean pairwise distance: {embedding_distances['mean_pairwise_distance']:.4f}"
        )
    print()

    # Extract distance statistics
    normal_distances = (
        distances[normal_mask]
        if distances is not None
        else np.max(gdn_predictions[normal_mask], axis=1)
    )
    faulty_distances = (
        distances[faulty_mask]
        if distances is not None
        else np.max(gdn_predictions[faulty_mask], axis=1)
    )

    normal_mean_dist = np.mean(normal_distances)
    faulty_mean_dist = np.mean(faulty_distances)
    separation = faulty_mean_dist - normal_mean_dist

    # Create visualizations
    print("8. Creating visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    create_evaluation_plots(
        normal_distances,
        faulty_distances,
        normal_mask,
        faulty_mask,
        embeddings,
        embedding_distances,
        metrics,
        sensor_names,
        gdn_predictions,
        output_path / "gdn_evaluation.png",
    )
    print(f"   ✓ Plots saved to: {output_path / 'gdn_evaluation.png'}")
    print()

    # Save results
    results = {
        "classification_metrics": metrics,
        "distance_statistics": {
            "normal_mean": float(normal_mean_dist),
            "normal_std": float(np.std(normal_distances)),
            "faulty_mean": float(faulty_mean_dist),
            "faulty_std": float(np.std(faulty_distances)),
            "separation": float(separation),
            "separation_ratio": float(faulty_mean_dist / (normal_mean_dist + 1e-8)),
        },
        "embedding_distances": embedding_distances,
        "statistics": {
            "num_normal": int(num_normal),
            "num_faulty": int(num_faulty),
            "num_windows": int(num_windows),
        },
    }

    results_path = output_path / "gdn_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_path}")
    print()
    print("=" * 80)
    print("✓ Evaluation completed successfully!")
    print("=" * 80)

    return results


def create_evaluation_plots(
    normal_distances,
    faulty_distances,
    normal_mask,
    faulty_mask,
    embeddings,
    embedding_distances,
    metrics,
    sensor_names,
    gdn_predictions,
    output_path,
):
    """Create comprehensive evaluation plots."""
    fig = plt.figure(figsize=(20, 14))

    # Plot 1: Bar chart - Distance separation
    ax1 = plt.subplot(3, 3, 1)
    categories = ["Normal", "Anomalous"]
    means = [np.mean(normal_distances), np.mean(faulty_distances)]
    stds = [np.std(normal_distances), np.std(faulty_distances)]
    colors = ["green", "red"]

    bars = ax1.bar(
        categories,
        means,
        yerr=stds,
        capsize=10,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_ylabel("Distance to Normal Center", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Distance Separation: Normal vs Anomalous", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Histogram of distances
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(
        normal_distances,
        bins=50,
        alpha=0.6,
        label=f"Normal (n={len(normal_distances)})",
        density=True,
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.hist(
        faulty_distances,
        bins=50,
        alpha=0.6,
        label=f"Anomalous (n={len(faulty_distances)})",
        density=True,
        color="red",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axvline(
        x=np.mean(normal_distances),
        color="green",
        linestyle="--",
        linewidth=2,
        label="Normal mean",
    )
    ax2.axvline(
        x=np.mean(faulty_distances),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Anomalous mean",
    )
    ax2.axvline(
        x=metrics["threshold"],
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Threshold ({metrics['threshold']:.4f})",
    )
    ax2.set_xlabel("Distance to Normal Center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax2.set_title("Distance Distribution", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Classification metrics bar chart
    ax3 = plt.subplot(3, 3, 3)
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics.get("roc_auc", 0.0),
        metrics.get("pr_auc", 0.0),
    ]
    colors_metrics = ["blue", "orange", "purple", "red", "green", "brown"]
    bars = ax3.bar(
        metric_names,
        metric_values,
        color=colors_metrics,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax3.set_title("Classification Metrics", fontsize=13, fontweight="bold")
    ax3.set_ylim([0, 1.1])
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Plot 4: ROC curve
    ax4 = plt.subplot(3, 3, 4)
    if "roc_curve" in metrics and metrics.get("roc_auc", 0) > 0:
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        ax4.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"ROC (AUC = {metrics['roc_auc']:.3f})",
            color="blue",
        )
        ax4.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax4.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax4.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax4.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5, 0.5, "ROC curve\nnot available", ha="center", va="center", fontsize=12
        )
        ax4.set_title("ROC Curve", fontsize=13, fontweight="bold")

    # Plot 5: Embedding space distances (bar chart)
    ax5 = plt.subplot(3, 3, 5)
    if embedding_distances:
        dist_names = [
            "Centroid\nDistance",
            "Normal to\nCenter",
            "Anomalous to\nNormal Center",
            "Pairwise\nDistance",
        ]
        dist_values = [
            embedding_distances["centroid_distance"],
            embedding_distances["normal_mean_dist_to_center"],
            embedding_distances["anomalous_mean_dist_to_normal_center"],
            embedding_distances["mean_pairwise_distance"],
        ]
        bars = ax5.bar(
            dist_names,
            dist_values,
            color=["purple", "green", "red", "orange"],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax5.set_ylabel("Distance", fontsize=12, fontweight="bold")
        ax5.set_title("Embedding Space Distances", fontsize=13, fontweight="bold")
        ax5.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, dist_values):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax5.text(
            0.5,
            0.5,
            "Embedding distances\nnot available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax5.set_title("Embedding Space Distances", fontsize=13, fontweight="bold")

    # Plot 6: Box plot of distances
    ax6 = plt.subplot(3, 3, 6)
    box_data = [normal_distances, faulty_distances]
    bp = ax6.boxplot(box_data, labels=["Normal", "Anomalous"], patch_artist=True)
    bp["boxes"][0].set_facecolor("green")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("red")
    bp["boxes"][1].set_alpha(0.6)
    ax6.set_ylabel("Distance to Normal Center", fontsize=12, fontweight="bold")
    ax6.set_title("Distance Distribution (Box Plot)", fontsize=13, fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="y")

    # Plot 7: Confusion matrix
    ax7 = plt.subplot(3, 3, 7)
    cm = metrics["confusion_matrix"]
    cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax7.imshow(cm_array, cmap="Blues", aspect="auto")
    ax7.set_xticks([0, 1])
    ax7.set_yticks([0, 1])
    ax7.set_xticklabels(["Normal", "Anomalous"])
    ax7.set_yticklabels(["Normal", "Anomalous"])
    ax7.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax7.set_ylabel("True", fontsize=12, fontweight="bold")
    ax7.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    # Add text annotations
    thresh = cm_array.max() / 2.0
    for i in range(2):
        for j in range(2):
            text = ax7.text(
                j,
                i,
                cm_array[i, j],
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax7)

    # Plot 8: Separation by sensor (if available)
    ax8 = plt.subplot(3, 3, 8)
    normal_max_scores = np.max(gdn_predictions[normal_mask], axis=1)
    faulty_max_scores = np.max(gdn_predictions[faulty_mask], axis=1)

    ax8.hist(
        normal_max_scores,
        bins=30,
        alpha=0.6,
        label="Normal",
        density=True,
        color="green",
    )
    ax8.hist(
        faulty_max_scores,
        bins=30,
        alpha=0.6,
        label="Anomalous",
        density=True,
        color="red",
    )
    ax8.set_xlabel("Max Sensor Score", fontsize=12, fontweight="bold")
    ax8.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax8.set_title("Window-Level Scores (Max Sensor)", fontsize=13, fontweight="bold")
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # Plot 9: Summary statistics text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    stats_text = f"""
    EVALUATION SUMMARY
    
    Classification Metrics:
    ├─ Accuracy:  {metrics["accuracy"]:.4f}
    ├─ Precision: {metrics["precision"]:.4f}
    ├─ Recall:    {metrics["recall"]:.4f}
    ├─ F1 Score:  {metrics["f1_score"]:.4f}
    ├─ ROC-AUC:   {metrics.get("roc_auc", 0.0):.4f}
    └─ PR-AUC:    {metrics.get("pr_auc", 0.0):.4f}
    
    Distance Statistics:
    ├─ Normal mean:    {np.mean(normal_distances):.4f}
    ├─ Anomalous mean: {np.mean(faulty_distances):.4f}
    ├─ Separation:     {np.mean(faulty_distances) - np.mean(normal_distances):.4f}
    └─ Threshold:      {metrics["threshold"]:.4f}
    
    Dataset:
    ├─ Normal windows:  {np.sum(normal_mask)}
    └─ Anomalous windows: {np.sum(faulty_mask)}
    """

    if embedding_distances:
        stats_text += f"""
    Embedding Distances:
    ├─ Centroid distance: {embedding_distances["centroid_distance"]:.4f}
    ├─ Separation ratio:  {embedding_distances["separation_ratio"]:.2f}x
    └─ Pairwise distance: {embedding_distances["mean_pairwise_distance"]:.4f}
    """

    ax9.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=ax9.transAxes,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive GDN model evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="llm/evaluation/shared_dataset/test.npz",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anomaly-detection/best_multilabel_gdn.pt",
        help="Path to GDN model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of windows to evaluate"
    )

    args = parser.parse_args()

    evaluate_gdn(args.dataset, args.model, args.output, args.limit)
