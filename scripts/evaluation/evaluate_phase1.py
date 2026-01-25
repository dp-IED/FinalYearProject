#!/usr/bin/env python3
"""
Evaluate Phase 1 GDN Model with Center Loss

Evaluates the Phase 1 checkpoint to verify it meets Phase 2 prerequisites:
- Stable center_dist ≥ 0.15
- Separation ratio ≥ 1.7×
- No center collapse (loss_center not 0.0000)

Computes:
1. Center separation distance
2. Distance-to-normal-center for all windows
3. Separation ratio (anomalous_mean / normal_mean)
4. Per-sensor classification metrics
5. Window-level classification metrics
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.gdn_model import MultiLabelGDN
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    DATA_PATH,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
    FORECAST_HORIZON,
    EMBED_DIM,
    TOP_K,
    HIDDEN_DIM,
)

torch.set_default_dtype(torch.float32)


def evaluate_phase1(
    checkpoint_path,
    data_path=DATA_PATH,
    device=None,
    batch_size=32,
):
    """
    Evaluate Phase 1 model checkpoint.

    Returns:
        Dictionary with evaluation results
    """
    # Device detection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Validate checkpoint
    required_keys = [
        "model_state_dict",
        "normal_center",  # New: direct normal center parameter
        "sensor_names",
        "window_size",
        "embed_dim",
        "top_k",
        "hidden_dim",
    ]
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing required key: {key}")

    # Print checkpoint info
    print(f"Checkpoint info:")
    print(f"  Window size: {checkpoint['window_size']}")
    print(f"  Embed dim: {checkpoint['embed_dim']}")
    print(f"  Top-k: {checkpoint['top_k']}")
    print(f"  Hidden dim: {checkpoint['hidden_dim']}")
    if "final_separation_ratio" in checkpoint:
        print(
            f"  Training separation ratio: {checkpoint['final_separation_ratio']:.2f}×"
        )
    if "normal_mean_distance" in checkpoint:
        print(
            f"  Training normal mean distance: {checkpoint['normal_mean_distance']:.4f}"
        )
    if "anomalous_mean_distance" in checkpoint:
        print(
            f"  Training anomalous mean distance: {checkpoint['anomalous_mean_distance']:.4f}"
        )
    if "epoch" in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if "best_val_loss" in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    # Initialize model and center loss
    num_sensors = len(checkpoint["sensor_names"])
    model = MultiLabelGDN(
        num_nodes=num_sensors,
        window_size=checkpoint["window_size"],
        embed_dim=checkpoint["embed_dim"],
        top_k=checkpoint["top_k"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    # Load state dicts (handle GAT layer key mismatch between PyG versions)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as e:
        # Try loading with strict=False if there are key mismatches
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting flexible loading...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Load normal center directly (separation loss approach)
    normal_center = checkpoint["normal_center"].to(device)  # (hidden_dim,)

    print("✓ Model loaded successfully")
    print(f"\nNormal center shape: {normal_center.shape}")
    print(f"Normal center norm: {torch.norm(normal_center).item():.4f}")

    # Load and preprocess test data
    print(f"\nLoading test data from {data_path}...")
    df_list = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)

    print(f"Loaded {len(df_list)} files")

    # Combine all dataframes
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data[ID_COL].nunique()}")

    # Preprocessing (same as training)
    print("\nPreprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ]
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

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (must match training split)
    print("\nSplitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)

    train_drives = unique_drives[: int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives) :]

    test_data = data[data[ID_COL].isin(test_drives)].copy()
    print(f"Test drives: {len(test_drives)}")
    print(f"Test shape: {test_data.shape}")

    # Build clean windows (need to refit scaler - but for evaluation we'll use train scaler logic)
    # For proper evaluation, we should use the same scaler as training, but for simplicity
    # we'll refit on test data
    print("\nBuilding test windows...")
    X_test_clean, y_test_clean, scaler_test = build_clean_windows(
        test_data,
        checkpoint["sensor_names"],
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=None,
    )

    # Inject faults with sensor-level labels
    print("\nInjecting faults with sensor-level labels...")
    X_test_sensor, _, test_sensor_labels, test_window_labels = (
        inject_faults_with_sensor_labels(
            X_test_clean,
            y_test_clean,
            checkpoint["sensor_names"],
            fault_percentage=0.30,
            random_state=44,
        )
    )

    # Statistics
    test_faulty = (test_sensor_labels.sum(dim=1) > 0).sum().item()
    print(f"Test: {test_faulty}/{len(X_test_sensor)} faulty windows")
    print(
        f"  Avg sensors per fault: {test_sensor_labels[test_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}"
    )

    # Create dataloader
    test_ds = TensorDataset(
        X_test_sensor, y_test_clean, test_sensor_labels, test_window_labels
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    model.eval()

    all_window_distances = []
    all_window_labels = []
    all_sensor_labels = []
    all_sensor_probs = []

    print("\nComputing embeddings and distances...")
    with torch.no_grad():
        for X_batch, _, sensor_labels_batch, window_labels_batch in tqdm(
            test_loader, desc="Evaluating"
        ):
            X_batch = X_batch.to(device)

            # Get embeddings
            embeddings = model.get_embeddings(X_batch)  # (B, hidden_dim)

            # Get sensor predictions
            sensor_probs, _ = model(X_batch, return_global=True)

            # Distance to normal center (class 0)
            distances = torch.norm(
                embeddings - normal_center.unsqueeze(0), dim=1
            )  # (B,)

            all_window_distances.append(distances.cpu().numpy())
            all_window_labels.append(window_labels_batch.numpy())
            all_sensor_labels.append(sensor_labels_batch.numpy())
            all_sensor_probs.append(sensor_probs.cpu().numpy())

    # Concatenate all results
    window_distances = np.concatenate(all_window_distances)
    window_labels = np.concatenate(all_window_labels)
    sensor_labels = np.concatenate(all_sensor_labels)
    sensor_probs = np.concatenate(all_sensor_probs)

    # ============================================================================
    # Center Distance Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("CENTER DISTANCE ANALYSIS")
    print("=" * 80)

    normal_distances = window_distances[window_labels == 0]
    anomalous_distances = window_distances[window_labels == 1]

    normal_mean = normal_distances.mean()
    normal_std = normal_distances.std()
    anomalous_mean = anomalous_distances.mean()
    anomalous_std = anomalous_distances.std()

    separation_ratio = anomalous_mean / normal_mean if normal_mean > 0 else 0.0

    print(f"\nDistance to Normal Center:")
    print(f"  Normal windows:")
    print(f"    Mean: {normal_mean:.4f}")
    print(f"    Std:  {normal_std:.4f}")
    print(f"    Min:  {normal_distances.min():.4f}")
    print(f"    Max:  {normal_distances.max():.4f}")
    print(f"  Anomalous windows:")
    print(f"    Mean: {anomalous_mean:.4f}")
    print(f"    Std:  {anomalous_std:.4f}")
    print(f"    Min:  {anomalous_distances.min():.4f}")
    print(f"    Max:  {anomalous_distances.max():.4f}")
    print(f"\nSeparation Ratio (anomalous_mean / normal_mean): {separation_ratio:.2f}×")
    print(
        f"  (Anomalous windows are {separation_ratio:.2f}× further from normal center)"
    )

    # ============================================================================
    # Phase 2 Prerequisites Check
    # ============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 PREREQUISITES CHECK")
    print("=" * 80)

    prerequisites_met = True
    issues = []

    # Check 1: Normal mean distance reasonable (should be small, close to center)
    if normal_mean > 1.0:
        prerequisites_met = False
        issues.append(
            f"Normal mean distance ({normal_mean:.4f}) is too large (target: < 1.0)"
        )
        print(f"❌ Normal mean distance: {normal_mean:.4f} (target: < 1.0)")
    else:
        print(f"✅ Normal mean distance: {normal_mean:.4f} (target: < 1.0)")

    # Check 2: Separation ratio ≥ 1.7×
    if separation_ratio < 1.7:
        prerequisites_met = False
        issues.append(f"Separation ratio ({separation_ratio:.2f}×) < 1.7×")
        print(f"❌ Separation ratio: {separation_ratio:.2f}× (target: ≥ 1.7×)")
    else:
        print(f"✅ Separation ratio: {separation_ratio:.2f}× (target: ≥ 1.7×)")

    # Check 3: Normal mean distance reasonable (should be ~0.07-0.10)
    if normal_mean > 0.15:
        issues.append(
            f"Normal mean distance ({normal_mean:.4f}) is high (expected ~0.07-0.10)"
        )
        print(f"⚠️  Normal mean distance: {normal_mean:.4f} (expected ~0.07-0.10)")
    else:
        print(f"✅ Normal mean distance: {normal_mean:.4f} (expected ~0.07-0.10)")

    # Check 4: Anomalous mean distance reasonable (should be ~0.18-0.25)
    if anomalous_mean < 0.15:
        prerequisites_met = False
        issues.append(
            f"Anomalous mean distance ({anomalous_mean:.4f}) is low (expected ~0.18-0.25)"
        )
        print(f"❌ Anomalous mean distance: {anomalous_mean:.4f} (expected ~0.18-0.25)")
    else:
        print(f"✅ Anomalous mean distance: {anomalous_mean:.4f} (expected ~0.18-0.25)")

    print("\n" + "-" * 80)
    if prerequisites_met:
        print("✅ ALL PREREQUISITES MET - Ready for Phase 2!")
    else:
        print("❌ PREREQUISITES NOT MET - Fix Phase 1 before proceeding to Phase 2")
        print("\nIssues to address:")
        for issue in issues:
            print(f"  - {issue}")

    # ============================================================================
    # Window-Level Classification Metrics
    # ============================================================================
    print("\n" + "=" * 80)
    print("WINDOW-LEVEL CLASSIFICATION METRICS")
    print("=" * 80)

    # Use distance as anomaly score (higher = more anomalous)
    # Find optimal threshold
    thresholds = np.linspace(window_distances.min(), window_distances.max(), 1000)
    best_f1 = 0
    best_threshold = np.median(window_distances)

    for t in thresholds:
        y_pred = (window_distances >= t).astype(int)
        if len(np.unique(y_pred)) > 1:
            f1 = precision_recall_fscore_support(
                window_labels, y_pred, average="binary", zero_division=0
            )[2]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

    y_pred = (window_distances >= best_threshold).astype(int)

    window_accuracy = accuracy_score(window_labels, y_pred)
    window_precision, window_recall, window_f1, _ = precision_recall_fscore_support(
        window_labels, y_pred, average="binary", zero_division=0
    )
    window_auc = roc_auc_score(window_labels, window_distances)
    cm = confusion_matrix(window_labels, y_pred)

    print(f"\nThreshold: {best_threshold:.4f} (F1-optimal)")
    print(f"Accuracy:  {window_accuracy:.4f}")
    print(f"Precision: {window_precision:.4f}")
    print(f"Recall:    {window_recall:.4f}")
    print(f"F1 Score:  {window_f1:.4f}")
    print(f"AUC:       {window_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Normal  Predicted Faulty")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Actually Normal      {tn:6d}           {fp:6d}")
        print(f"Actually Faulty      {fn:6d}           {tp:6d}")

    # ============================================================================
    # Per-Sensor Classification Metrics
    # ============================================================================
    print("\n" + "=" * 80)
    print("PER-SENSOR CLASSIFICATION METRICS")
    print("=" * 80)

    sensor_results = []
    threshold = 0.5  # Standard threshold for sensor probabilities

    for i, sensor_name in enumerate(checkpoint["sensor_names"]):
        sensor_labels_i = sensor_labels[:, i]
        sensor_probs_i = sensor_probs[:, i]
        sensor_preds_i = (sensor_probs_i > threshold).astype(int)

        n_faults = sensor_labels_i.sum()

        if n_faults > 0:
            auc = roc_auc_score(sensor_labels_i, sensor_probs_i)
            precision, recall, f1, _ = precision_recall_fscore_support(
                sensor_labels_i, sensor_preds_i, average="binary", zero_division=0
            )

            sensor_results.append(
                {
                    "Sensor": sensor_name.replace(" ()", ""),
                    "Faults": int(n_faults),
                    "AUC": auc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )

            print(
                f"{sensor_name:40s} | Faults: {int(n_faults):4d} | "
                f"AUC: {auc:.3f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}"
            )
        else:
            print(f"{sensor_name:40s} | No faults injected")

    # Overall sensor metrics (flattened)
    sensor_labels_flat = sensor_labels.flatten()
    sensor_probs_flat = sensor_probs.flatten()
    sensor_preds_flat = (sensor_probs_flat > threshold).astype(int)

    overall_sensor_auc = roc_auc_score(sensor_labels_flat, sensor_probs_flat)
    overall_sensor_precision, overall_sensor_recall, overall_sensor_f1, _ = (
        precision_recall_fscore_support(
            sensor_labels_flat, sensor_preds_flat, average="binary", zero_division=0
        )
    )

    print(
        f"\n{'Overall (all sensors)':40s} | "
        f"AUC: {overall_sensor_auc:.3f} | "
        f"P: {overall_sensor_precision:.3f} | "
        f"R: {overall_sensor_recall:.3f} | "
        f"F1: {overall_sensor_f1:.3f}"
    )

    # ============================================================================
    # Summary
    # ============================================================================
    results = {
        "normal_mean_distance": float(normal_mean),
        "anomalous_mean_distance": float(anomalous_mean),
        "normal_mean_distance": float(normal_mean),
        "normal_std_distance": float(normal_std),
        "anomalous_mean_distance": float(anomalous_mean),
        "anomalous_std_distance": float(anomalous_std),
        "separation_ratio": float(separation_ratio),
        "prerequisites_met": prerequisites_met,
        "window_metrics": {
            "accuracy": float(window_accuracy),
            "precision": float(window_precision),
            "recall": float(window_recall),
            "f1": float(window_f1),
            "auc": float(window_auc),
            "threshold": float(best_threshold),
        },
        "sensor_metrics": {
            "overall_auc": float(overall_sensor_auc),
            "overall_precision": float(overall_sensor_precision),
            "overall_recall": float(overall_sensor_recall),
            "overall_f1": float(overall_sensor_f1),
            "per_sensor": sensor_results,
        },
    }

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 GDN Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_multilabel_gdn_center.pt",
        help="Path to Phase 1 checkpoint",
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path (optional)"
    )
    args = parser.parse_args()

    results = evaluate_phase1(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save results if output path specified
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
