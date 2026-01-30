"""
Stage 1 Embedding Diversity Diagnostic

Analyzes embedding quality after Stage 1 self-supervised training to determine:
1. Whether low validation contrastive loss is expected (fewer drives) or concerning (lack of diversity)
2. Embedding separation quality for KAG integration
3. Readiness to proceed to Stage 2 multi-level center loss training

Usage:
    python scripts/evaluation/diagnose_stage1_embeddings.py \
        --checkpoint checkpoints/stage1_best_forecast.pt \
        --data_path data/carOBD/obdiidata \
        --output diagnostics/stage1_embeddings.json
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "anomaly-detection"))

from models.gdn_model import MultiLabelGDN, KAGOptimizedGDN
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingDiversityAnalyzer:
    """Analyzes embedding quality and diversity for KAG readiness."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_embeddings(self, loader):
        """
        Extract embeddings and metadata from dataloader.

        Returns:
            embeddings: (N, hidden_dim) numpy array
            drive_ids: (N,) numpy array
            sensor_embeddings: (N, num_sensors, hidden_dim) numpy array
        """
        embeddings_list = []
        sensor_embeddings_list = []
        drive_ids_list = []

        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)  # (B, W, N)
                drive_ids = batch[2] if len(batch) > 2 else torch.zeros(X_batch.size(0))

                # Window-level embeddings
                window_emb = self.model.get_embeddings(X_batch)  # (B, hidden_dim)
                embeddings_list.append(window_emb.cpu().numpy())

                # Sensor-level embeddings (if available)
                try:
                    sensor_emb = self.model.get_sensor_embeddings(
                        X_batch
                    )  # (B, N, hidden_dim)
                    sensor_embeddings_list.append(sensor_emb.cpu().numpy())
                except:
                    pass  # Model might not have this method

                drive_ids_list.append(drive_ids.cpu().numpy())

        embeddings = np.vstack(embeddings_list)
        drive_ids = np.concatenate(drive_ids_list)

        sensor_embeddings = None
        if sensor_embeddings_list:
            sensor_embeddings = np.vstack(sensor_embeddings_list)

        return embeddings, drive_ids, sensor_embeddings

    def compute_drive_statistics(self, embeddings, drive_ids):
        """
        Compute intra-drive and inter-drive similarity statistics.

        Returns:
            dict with keys:
                - intra_drive_sim: mean similarity within same drive
                - inter_drive_sim: mean similarity across different drives
                - separation: intra - inter
                - intra_std: std of intra-drive similarities
                - inter_std: std of inter-drive similarities
        """
        unique_drives = np.unique(drive_ids)
        intra_sims = []
        inter_sims = []

        for drive in unique_drives:
            drive_mask = drive_ids == drive
            if drive_mask.sum() <= 1:
                continue  # Skip drives with only 1 window

            drive_emb = embeddings[drive_mask]
            other_emb = embeddings[~drive_mask]

            # Intra-drive similarity (same drive)
            intra_sim_matrix = cosine_similarity(drive_emb)
            # Get upper triangle (exclude diagonal)
            triu_indices = np.triu_indices_from(intra_sim_matrix, k=1)
            if len(triu_indices[0]) > 0:
                intra_sims.extend(intra_sim_matrix[triu_indices].tolist())

            # Inter-drive similarity (different drives)
            if len(other_emb) > 0:
                inter_sim_matrix = cosine_similarity(drive_emb, other_emb)
                inter_sims.extend(inter_sim_matrix.flatten().tolist())

        return {
            "intra_drive_sim": np.mean(intra_sims),
            "inter_drive_sim": np.mean(inter_sims),
            "separation": np.mean(intra_sims) - np.mean(inter_sims),
            "intra_std": np.std(intra_sims),
            "inter_std": np.std(inter_sims),
            "num_drives": len(unique_drives),
            "num_windows": len(embeddings),
        }

    def compute_embedding_quality_metrics(self, embeddings):
        """
        Compute general embedding quality metrics.

        Returns:
            dict with quality metrics
        """
        # Normalize embeddings
        embeddings_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Average pairwise similarity (should be moderate, not too high or low)
        sim_matrix = cosine_similarity(embeddings_norm)
        np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
        avg_similarity = sim_matrix.sum() / (
            sim_matrix.shape[0] * (sim_matrix.shape[0] - 1)
        )

        # Embedding variance (should be reasonably high)
        embedding_variance = np.var(embeddings, axis=0).mean()

        # Embedding norm statistics
        norms = np.linalg.norm(embeddings, axis=1)

        return {
            "avg_pairwise_similarity": avg_similarity,
            "embedding_variance": embedding_variance,
            "norm_mean": norms.mean(),
            "norm_std": norms.std(),
            "norm_min": norms.min(),
            "norm_max": norms.max(),
        }

    def diagnose_contrastive_gap(self, train_stats, val_stats):
        """
        Diagnose whether train/val contrastive gap is concerning.

        Returns:
            dict with diagnosis and recommendations
        """
        train_sep = train_stats["separation"]
        val_sep = val_stats["separation"]

        diagnosis = {
            "gap_ratio": val_sep / train_sep if train_sep > 0 else float("inf"),
            "is_concerning": False,
            "reason": "",
            "recommendations": [],
        }

        # Check if validation separation is much higher (concerning)
        if val_sep > train_sep + 0.15:
            diagnosis["is_concerning"] = True
            diagnosis["reason"] = (
                "Validation drives are TOO SIMILAR to each other. "
                "Model easily clusters them, but may not generalize to diverse test scenarios."
            )
            diagnosis["recommendations"] = [
                "Re-split data with stratified sampling to ensure diversity in validation set",
                "Check if validation drives represent full range of driving patterns",
                "Consider using cross-validation to ensure robustness",
            ]

        # Check if validation separation is much lower (also concerning)
        elif val_sep < train_sep - 0.15:
            diagnosis["is_concerning"] = True
            diagnosis["reason"] = (
                "Validation drives are TOO DIVERSE. "
                "Model struggles to cluster them, indicating overfitting to training patterns."
            )
            diagnosis["recommendations"] = [
                "Model may be overfitting to training drive patterns",
                "Consider reducing contrastive loss weight",
                "Add more regularization (dropout, weight decay)",
            ]

        # Check if both separations are too low (concerning for KAG)
        elif train_sep < 0.15 and val_sep < 0.15:
            diagnosis["is_concerning"] = True
            diagnosis["reason"] = (
                "Both train and validation separations are LOW. "
                "Embeddings are not well-clustered by drive, which may hurt KAG performance."
            )
            diagnosis["recommendations"] = [
                "Increase contrastive loss weight (current: 0.3 → try 0.5)",
                "Train for more epochs to improve clustering",
                "Check if drive IDs are correct in dataset",
            ]

        # All good
        else:
            diagnosis["is_concerning"] = False
            diagnosis["reason"] = (
                "Similar separation in train and validation sets. "
                "Low validation contrastive loss is due to fewer drives (expected). "
                "Model should generalize well to test set."
            )
            diagnosis["recommendations"] = [
                "✅ Proceed to Stage 2 multi-level center loss training",
                "Monitor separation ratio in Stage 2 (target: > 5.0)",
            ]

        return diagnosis

    def estimate_kag_quality(self, train_stats, val_stats):
        """
        Estimate expected KAG diagnostic quality based on embedding separation.

        Returns:
            dict with KAG quality estimates
        """
        avg_separation = (train_stats["separation"] + val_stats["separation"]) / 2

        # Rough estimates based on separation
        if avg_separation < 0.15:
            quality_tier = "Poor"
            accuracy = "55-65%"
            confidence = "Low"
            description = "Scores overlap significantly. LLM will hedge and provide vague diagnostics."
        elif avg_separation < 0.25:
            quality_tier = "Moderate"
            accuracy = "65-75%"
            confidence = "Medium"
            description = (
                "Some score gaps. LLM can identify likely issues but with caution."
            )
        elif avg_separation < 0.35:
            quality_tier = "Good"
            accuracy = "75-85%"
            confidence = "High"
            description = (
                "Clear score gaps. LLM provides confident, specific diagnostics."
            )
        else:
            quality_tier = "Excellent"
            accuracy = "85-90%+"
            confidence = "Very High"
            description = (
                "Strong separation. LLM generates expert-level, actionable diagnostics."
            )

        return {
            "quality_tier": quality_tier,
            "estimated_accuracy": accuracy,
            "confidence_level": confidence,
            "description": description,
            "separation": avg_separation,
        }


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_stats_table(train_stats, val_stats):
    """Print statistics in formatted table."""
    print(f"\n{'Metric':<30} {'Train':<20} {'Val':<20} {'Delta':<15}")
    print("-" * 85)

    metrics = [
        ("Number of drives", "num_drives", "{:.0f}"),
        ("Number of windows", "num_windows", "{:.0f}"),
        ("Intra-drive similarity", "intra_drive_sim", "{:.4f}"),
        ("Inter-drive similarity", "inter_drive_sim", "{:.4f}"),
        ("Separation", "separation", "{:.4f}"),
        ("Intra std dev", "intra_std", "{:.4f}"),
        ("Inter std dev", "inter_std", "{:.4f}"),
    ]

    for label, key, fmt in metrics:
        train_val = train_stats[key]
        val_val = val_stats[key]
        delta = (
            val_val - train_val if key != "num_drives" and key != "num_windows" else 0
        )

        train_str = fmt.format(train_val)
        val_str = fmt.format(val_val)
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) and delta != 0 else "-"

        print(f"{label:<30} {train_str:<20} {val_str:<20} {delta_str:<15}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Stage 1 embedding quality for KAG"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to OBD-II data directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save diagnostic report (optional)",
    )

    args = parser.parse_args()

    print_section("STAGE 1 EMBEDDING DIVERSITY DIAGNOSTIC")
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")

    # Load data using same logic as training scripts
    print("\nLoading and preprocessing data...")
    import pandas as pd
    from scripts.training.train_gdn_stage1_forecast import (
        mean_fill_missing_timestamps_and_remove_duplicates,
        remove_zero_variance_columns,
        downsample,
        filter_long_drives,
        add_cross_channel_features,
        build_forecast_windows,
        ID_COL,
        TIME_COL,
        SENSOR_COLS,
        WINDOW_SIZE,
    )
    
    # Load all CSV files
    df_list = []
    for file in os.listdir(args.data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{args.data_path}/{file}", index_col=False)
            df[ID_COL] = file.replace(".csv", "")
            df_list.append(df)
    
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data[ID_COL].nunique()}")
    
    # Preprocessing
    print("Preprocessing data...")
    data = data.drop(columns=["WARM_UPS_SINCE_CODES_CLEARED ()", "TIME_SINCE_TROUBLE_CODES_CLEARED ()"])
    data = mean_fill_missing_timestamps_and_remove_duplicates(data, time_col=TIME_COL, id_cols=[ID_COL])
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=1)
    data = filter_long_drives(data, id_col=ID_COL, min_length=WINDOW_SIZE + 1)
    data = add_cross_channel_features(data)
    
    # Split by drive (70/15/15)
    print("Splitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)
    train_drives = unique_drives[:int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives):int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives):]
    
    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    
    # Build forecast windows (same as Stage 1 training)
    print("Building forecast windows...")
    X_train, y_train_forecast, drive_ids_train, scaler_train = build_forecast_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, horizons=[1, 5, 10], scaler=None
    )
    X_val, y_val_forecast, drive_ids_val, _ = build_forecast_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, horizons=[1, 5, 10], scaler=scaler_train
    )
    
    # Convert drive IDs to integer indices
    all_drive_ids = np.concatenate([drive_ids_train, drive_ids_val])
    unique_drives = np.unique(all_drive_ids)
    drive_to_idx = {drive: idx for idx, drive in enumerate(unique_drives)}
    drive_ids_train_idx = np.array([drive_to_idx[drive] for drive in drive_ids_train])
    drive_ids_val_idx = np.array([drive_to_idx[drive] for drive in drive_ids_val])
    
    # Create datasets with drive IDs
    train_dataset = TensorDataset(
        X_train, torch.zeros_like(X_train[:, 0, 0]), torch.tensor(drive_ids_train_idx, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        X_val, torch.zeros_like(X_val[:, 0, 0]), torch.tensor(drive_ids_val_idx, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print("\nLoading model from checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Initialize model based on checkpoint model_type
    num_sensors = X_train.shape[2]
    window_size = X_train.shape[1]
    model_type = checkpoint.get("model_type", "enhanced")
    
    if model_type == "kag_optimized":
        print(f"  Loading KAG-Optimized model")
        model = KAGOptimizedGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=checkpoint.get("embed_dim", 64),
            top_k=checkpoint.get("top_k", 5),
            hidden_dim=checkpoint.get("hidden_dim", 64),
        ).to(args.device)
    else:
        print(f"  Loading Enhanced MultiLabelGDN model")
        model = MultiLabelGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=checkpoint.get("embed_dim", 32),
            top_k=checkpoint.get("top_k", 5),
            hidden_dim=checkpoint.get("hidden_dim", 64),
        ).to(args.device)
    
    # Load base model state (from GDNWithForecasting wrapper or direct)
    if "base_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["base_model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("  ✓ Model loaded successfully")

    # Run analysis
    analyzer = EmbeddingDiversityAnalyzer(model, args.device)

    print("\nExtracting embeddings...")
    train_emb, train_drives, _ = analyzer.extract_embeddings(train_loader)
    val_emb, val_drives, _ = analyzer.extract_embeddings(val_loader)
    print(
        f"  ✓ Train: {len(train_emb)} windows from {len(np.unique(train_drives))} drives"
    )
    print(f"  ✓ Val: {len(val_emb)} windows from {len(np.unique(val_drives))} drives")

    print("\nComputing statistics...")
    train_stats = analyzer.compute_drive_statistics(train_emb, train_drives)
    val_stats = analyzer.compute_drive_statistics(val_emb, val_drives)

    train_quality = analyzer.compute_embedding_quality_metrics(train_emb)
    val_quality = analyzer.compute_embedding_quality_metrics(val_emb)

    # Print results
    print_section("EMBEDDING STATISTICS BY DRIVE")
    print_stats_table(train_stats, val_stats)

    print_section("EMBEDDING QUALITY METRICS")
    print(f"\n{'Metric':<35} {'Train':<20} {'Val':<20}")
    print("-" * 75)
    quality_metrics = [
        ("Avg pairwise similarity", "avg_pairwise_similarity", "{:.4f}"),
        ("Embedding variance", "embedding_variance", "{:.4f}"),
        ("Norm mean", "norm_mean", "{:.4f}"),
        ("Norm std dev", "norm_std", "{:.4f}"),
    ]

    for label, key, fmt in quality_metrics:
        train_val = train_quality[key]
        val_val = val_quality[key]
        print(f"{label:<35} {fmt.format(train_val):<20} {fmt.format(val_val):<20}")

    # Diagnosis
    print_section("DIAGNOSIS: TRAIN/VAL CONTRASTIVE GAP")
    diagnosis = analyzer.diagnose_contrastive_gap(train_stats, val_stats)

    print(f"\nGap Ratio (Val/Train): {diagnosis['gap_ratio']:.2f}x")
    print(f"Concerning: {'⚠️  YES' if diagnosis['is_concerning'] else '✅ NO'}")
    print(f"\nReason:\n  {diagnosis['reason']}")

    print("\nRecommendations:")
    for i, rec in enumerate(diagnosis["recommendations"], 1):
        print(f"  {i}. {rec}")

    # KAG Quality Estimate
    print_section("ESTIMATED KAG DIAGNOSTIC QUALITY")
    kag_estimate = analyzer.estimate_kag_quality(train_stats, val_stats)

    print(f"\nQuality Tier: {kag_estimate['quality_tier']}")
    print(f"Estimated Accuracy: {kag_estimate['estimated_accuracy']}")
    print(f"Confidence Level: {kag_estimate['confidence_level']}")
    print(f"Average Separation: {kag_estimate['separation']:.4f}")
    print(f"\nDescription:\n  {kag_estimate['description']}")

    # Stage 2 Readiness
    print_section("STAGE 2 READINESS")

    if diagnosis["is_concerning"]:
        print("\n⚠️  WARNING: Address concerns above before proceeding to Stage 2")
        print("   Embedding quality issues may carry over and hurt final performance.")
    elif kag_estimate["separation"] < 0.20:
        print("\n⚠️  CAUTION: Separation is lower than ideal for KAG")
        print("   Consider:")
        print("   - Training for more epochs")
        print("   - Increasing contrastive loss weight")
        print("   - Checking if model architecture is appropriate")
    else:
        print("\n✅ READY: Embeddings show good clustering and separation")
        print("   Proceed to Stage 2 multi-level center loss training")
        print("   Target metrics for Stage 2:")
        print("   - Normal compactness: < 0.30")
        print("   - Anomaly compactness: < 0.40")
        print("   - Separation ratio: > 5.0")
        print("   - Window-sensor ratio: 0.8-1.2")

    # Save report if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        report = {
            "checkpoint": args.checkpoint,
            "train_stats": {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in train_stats.items()
            },
            "val_stats": {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                for k, v in val_stats.items()
            },
            "train_quality": {k: float(v) for k, v in train_quality.items()},
            "val_quality": {k: float(v) for k, v in val_quality.items()},
            "diagnosis": diagnosis,
            "kag_estimate": kag_estimate,
        }

        # Save JSON
        json_path = args.output
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to: {json_path}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
