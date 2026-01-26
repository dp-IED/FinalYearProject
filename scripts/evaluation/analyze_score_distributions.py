#!/usr/bin/env python3
"""
Analyze Anomaly Score Distributions for Class Imbalance

This script analyzes the distribution of anomaly scores for normal vs faulty windows
to determine appropriate thresholds for evidence strength classification.

Outputs recommendations for MODERATE evidence threshold based on:
- P95 of normal windows (to avoid false positives)
- P5 of faulty windows (to ensure recall)
- Overlap analysis
- Class imbalance ratio
"""

import sys
import numpy as np
import json
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add evaluation directory
eval_dir = str(project_root / "llm" / "evaluation")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# Import GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


def analyze_score_distributions(
    dataset_path: str = "llm/evaluation/shared_dataset/test.npz",
    gdn_model_path: str = "checkpoints/best_center_loss_gdn.pt",
    output_path: str = "results/score_distribution_analysis.json",
    limit: int = None,
):
    """
    Analyze anomaly score distributions to determine appropriate thresholds.

    Args:
        dataset_path: Path to test dataset (.npz file)
        gdn_model_path: Path to trained GDN model (.pt file)
        output_path: Path to save analysis results
        limit: Optional limit on number of windows to analyze
    """
    print("=" * 80)
    print("ANOMALY SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Load dataset
    print("1. Loading dataset...")
    dataset_path_full = project_root / dataset_path
    if not dataset_path_full.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path_full}")

    data = np.load(dataset_path_full, allow_pickle=True)
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
    gdn_model_path_full = project_root / gdn_model_path
    if not gdn_model_path_full.exists():
        raise FileNotFoundError(f"Model not found: {gdn_model_path_full}")

    # Detect model dimensions
    embed_dim = 64
    hidden_dim = 64

    try:
        checkpoint = torch.load(gdn_model_path_full, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "embed_dim" in checkpoint:
                embed_dim = checkpoint["embed_dim"]
            if "hidden_dim" in checkpoint:
                hidden_dim = checkpoint["hidden_dim"]
    except:
        if (
            "improved" not in str(gdn_model_path).lower()
            and "architecture" not in str(gdn_model_path).lower()
        ):
            embed_dim = 32
            hidden_dim = 32

    predictor = GDNPredictor(
        model_path=str(gdn_model_path_full),
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=embed_dim,
        top_k=3,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    print(f"   ✓ Model loaded from: {gdn_model_path_full}")
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

    # Get embeddings and compute anomaly scores
    print("4. Computing anomaly scores...")
    embeddings = kg_data.get("embeddings", None)
    distances = kg_data.get("distances", None)

    # Use max probability per window as anomaly score (per-sensor scores)
    # For each window, get the maximum anomaly score across all sensors
    max_scores_per_window = np.max(gdn_predictions, axis=1)  # (num_windows,)

    # Also compute per-sensor scores for detailed analysis
    sensor_scores = gdn_predictions  # (num_windows, num_sensors)

    # If distances available, use them as alternative scoring
    if distances is not None:
        print("   Using embedding distances as anomaly scores")
        anomaly_scores = distances
    else:
        print("   Using max GDN prediction as anomaly scores")
        anomaly_scores = max_scores_per_window

    print(f"   Anomaly scores shape: {anomaly_scores.shape}")
    print()

    # Identify normal vs faulty windows
    print("5. Separating normal vs faulty windows...")
    is_faulty_window = sensor_labels_true.sum(axis=1) > 0
    normal_mask = ~is_faulty_window
    faulty_mask = is_faulty_window

    num_normal = np.sum(normal_mask)
    num_faulty = np.sum(faulty_mask)
    imbalance_ratio = num_normal / num_faulty if num_faulty > 0 else float('inf')

    print(f"   Normal windows: {num_normal}")
    print(f"   Faulty windows: {num_faulty}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    print()

    # Extract scores for each class
    normal_scores = anomaly_scores[normal_mask]
    faulty_scores = anomaly_scores[faulty_mask]

    # Compute statistics
    print("6. Computing distribution statistics...")
    normal_stats = {
        "mean": float(np.mean(normal_scores)),
        "median": float(np.median(normal_scores)),
        "std": float(np.std(normal_scores)),
        "min": float(np.min(normal_scores)),
        "max": float(np.max(normal_scores)),
        "p5": float(np.percentile(normal_scores, 5)),
        "p25": float(np.percentile(normal_scores, 25)),
        "p75": float(np.percentile(normal_scores, 75)),
        "p95": float(np.percentile(normal_scores, 95)),
        "p99": float(np.percentile(normal_scores, 99)),
    }

    faulty_stats = {
        "mean": float(np.mean(faulty_scores)),
        "median": float(np.median(faulty_scores)),
        "std": float(np.std(faulty_scores)),
        "min": float(np.min(faulty_scores)),
        "max": float(np.max(faulty_scores)),
        "p5": float(np.percentile(faulty_scores, 5)),
        "p25": float(np.percentile(faulty_scores, 25)),
        "p75": float(np.percentile(faulty_scores, 75)),
        "p95": float(np.percentile(faulty_scores, 95)),
        "p99": float(np.percentile(faulty_scores, 99)),
    }

    print(f"\n   Normal windows statistics:")
    print(f"      Mean:   {normal_stats['mean']:.4f}")
    print(f"      Median: {normal_stats['median']:.4f}")
    print(f"      Std:    {normal_stats['std']:.4f}")
    print(f"      P95:    {normal_stats['p95']:.4f}")
    print(f"      P99:    {normal_stats['p99']:.4f}")
    print(f"      Max:    {normal_stats['max']:.4f}")

    print(f"\n   Faulty windows statistics:")
    print(f"      Mean:   {faulty_stats['mean']:.4f}")
    print(f"      Median: {faulty_stats['median']:.4f}")
    print(f"      Std:    {faulty_stats['std']:.4f}")
    print(f"      P5:     {faulty_stats['p5']:.4f}")
    print(f"      P25:    {faulty_stats['p25']:.4f}")
    print(f"      Min:    {faulty_stats['min']:.4f}")

    # Analyze overlap
    print("\n7. Analyzing overlap...")
    # Count normal windows above faulty mean
    normal_above_faulty_mean = np.sum(normal_scores > faulty_stats['mean'])
    normal_above_faulty_mean_pct = (normal_above_faulty_mean / len(normal_scores)) * 100

    # Count faulty windows below normal mean
    faulty_below_normal_mean = np.sum(faulty_scores < normal_stats['mean'])
    faulty_below_normal_mean_pct = (faulty_below_normal_mean / len(faulty_scores)) * 100

    # Overlap percentage
    overlap_pct = (normal_above_faulty_mean_pct + faulty_below_normal_mean_pct) / 2

    print(f"   Normal windows above faulty mean: {normal_above_faulty_mean} ({normal_above_faulty_mean_pct:.1f}%)")
    print(f"   Faulty windows below normal mean: {faulty_below_normal_mean} ({faulty_below_normal_mean_pct:.1f}%)")
    print(f"   Overall overlap: {overlap_pct:.1f}%")

    # Separation analysis
    separation = faulty_stats['mean'] - normal_stats['mean']
    separation_ratio = faulty_stats['mean'] / (normal_stats['mean'] + 1e-8)

    print(f"\n   Separation:")
    print(f"      Mean difference: {separation:.4f}")
    print(f"      Separation ratio: {separation_ratio:.2f}x")

    # Determine recommended threshold
    print("\n8. Determining recommended thresholds...")
    
    # Current threshold in code: 0.5 for MODERATE, 0.7 for STRONG
    current_moderate_threshold = 0.5
    current_strong_threshold = 0.7

    # Recommendation: MODERATE threshold should be above P95 of normal to minimize false positives
    # But also consider P5 of faulty to ensure we don't miss too many faults
    recommended_moderate = max(normal_stats['p95'], faulty_stats['p5'] * 0.8)
    # Round to nearest 0.05
    recommended_moderate = round(recommended_moderate / 0.05) * 0.05

    # Ensure recommended is between current and strong threshold
    if recommended_moderate < current_moderate_threshold:
        recommended_moderate = current_moderate_threshold
    if recommended_moderate > current_strong_threshold:
        recommended_moderate = current_strong_threshold * 0.9

    # Check if 0.7 is appropriate for MODERATE
    normal_above_07 = np.sum(normal_scores >= 0.7)
    normal_above_07_pct = (normal_above_07 / len(normal_scores)) * 100

    faulty_below_07 = np.sum(faulty_scores < 0.7)
    faulty_below_07_pct = (faulty_below_07 / len(faulty_scores)) * 100

    print(f"\n   Current thresholds:")
    print(f"      MODERATE: {current_moderate_threshold}")
    print(f"      STRONG:   {current_strong_threshold}")

    print(f"\n   Analysis at threshold 0.7:")
    print(f"      Normal windows >= 0.7: {normal_above_07} ({normal_above_07_pct:.1f}%)")
    print(f"      Faulty windows < 0.7: {faulty_below_07} ({faulty_below_07_pct:.1f}%)")

    # Recommendation
    if normal_stats['p95'] >= 0.7:
        recommendation = "RAISE MODERATE threshold above 0.7 (e.g., 0.75 or 0.8) to reduce false positives"
        recommended_moderate = max(0.75, normal_stats['p95'] + 0.05)
        recommended_moderate = round(recommended_moderate / 0.05) * 0.05
    elif normal_stats['p99'] >= 0.7:
        recommendation = "CONSIDER raising MODERATE threshold to 0.75 to reduce false positives (P99 of normal is >= 0.7)"
        recommended_moderate = 0.75
    else:
        recommendation = "0.7 is appropriate for MODERATE threshold (P95 of normal < 0.7)"
        recommended_moderate = 0.7

    print(f"\n   Recommendation: {recommendation}")
    print(f"   Recommended MODERATE threshold: {recommended_moderate:.2f}")

    # Compile results
    results = {
        "dataset_path": str(dataset_path),
        "gdn_model_path": str(gdn_model_path),
        "num_windows": int(num_windows),
        "class_distribution": {
            "num_normal": int(num_normal),
            "num_faulty": int(num_faulty),
            "imbalance_ratio": float(imbalance_ratio),
        },
        "normal_statistics": normal_stats,
        "faulty_statistics": faulty_stats,
        "overlap_analysis": {
            "normal_above_faulty_mean": int(normal_above_faulty_mean),
            "normal_above_faulty_mean_pct": float(normal_above_faulty_mean_pct),
            "faulty_below_normal_mean": int(faulty_below_normal_mean),
            "faulty_below_normal_mean_pct": float(faulty_below_normal_mean_pct),
            "overlap_percentage": float(overlap_pct),
        },
        "separation_analysis": {
            "mean_difference": float(separation),
            "separation_ratio": float(separation_ratio),
        },
        "threshold_analysis": {
            "current_moderate": float(current_moderate_threshold),
            "current_strong": float(current_strong_threshold),
            "normal_above_07_count": int(normal_above_07),
            "normal_above_07_pct": float(normal_above_07_pct),
            "faulty_below_07_count": int(faulty_below_07),
            "faulty_below_07_pct": float(faulty_below_07_pct),
        },
        "recommendations": {
            "recommended_moderate_threshold": float(recommended_moderate),
            "recommendation_text": recommendation,
            "should_raise_threshold": normal_stats['p95'] >= 0.7,
        },
    }

    # Save results
    print("\n9. Saving results...")
    output_path_full = project_root / output_path
    output_path_full.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_full, "w") as f:
        json.dump(results, f, indent=2)

    print(f"   ✓ Results saved to: {output_path_full}")
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey Findings:")
    print(f"  - P95 of normal windows: {normal_stats['p95']:.4f}")
    print(f"  - P5 of faulty windows: {faulty_stats['p5']:.4f}")
    print(f"  - Recommended MODERATE threshold: {recommended_moderate:.2f}")
    print(f"  - {recommendation}")
    print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze anomaly score distributions for threshold determination"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="llm/evaluation/shared_dataset/test.npz",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_center_loss_gdn.pt",
        help="Path to GDN model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/score_distribution_analysis.json",
        help="Path to save analysis results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to analyze (for testing)",
    )

    args = parser.parse_args()

    analyze_score_distributions(
        dataset_path=args.dataset,
        gdn_model_path=args.model,
        output_path=args.output,
        limit=args.limit,
    )
