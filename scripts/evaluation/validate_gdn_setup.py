#!/usr/bin/env python3
"""
Validate GDN training data setup before retraining.

Quick checks to ensure:
1. Class balance (imbalance ratio)
2. Data quality (NaNs, Infs, variance)
3. Fault pattern visibility (per-sensor separability)
4. Training readiness

Usage:
    python validate_gdn_setup.py --data-path data/carOBD/obdiidata
"""

import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "anomaly-detection"))

from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
)


def validate_gdn_training_data(data_path, fault_percentage=0.15, random_state=42):
    """
    Quick validation checks before retraining GDN.
    
    Args:
        data_path: Path to directory containing CSV files
        fault_percentage: Percentage of windows to inject faults
        random_state: Random seed for reproducibility
    
    Returns:
        bool: True if data is ready for training, False otherwise
    """
    print("="*70)
    print("GDN TRAINING DATA VALIDATION")
    print("="*70)
    
    # Load data
    print(f"\n1. LOADING DATA from {data_path}...")
    import os
    df_list = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)
    
    if len(df_list) == 0:
        print(f"   ✗ ERROR: No CSV files found in {data_path}")
        return False
    
    print(f"   ✓ Loaded {len(df_list)} files")
    
    # Combine all dataframes
    data = pd.concat(df_list, ignore_index=True)
    print(f"   ✓ Total samples: {len(data):,}")
    print(f"   ✓ Unique drives: {data[ID_COL].nunique()}")
    
    # Preprocessing (same as training script)
    print(f"\n2. PREPROCESSING DATA...")
    data = data.drop(
        columns=["WARM_UPS_SINCE_CODES_CLEARED ()"] if "WARM_UPS_SINCE_CODES_CLEARED ()" in data.columns else []
    )
    
    # Remove zero variance columns
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL, TIME_COL])
    
    # Fill missing timestamps and remove duplicates
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, TIME_COL, id_cols=[ID_COL]
    )
    
    # Downsample
    data = downsample(data, TIME_COL, ID_COL, downsample_factor=2)
    
    # Filter long drives
    data = filter_long_drives(data, ID_COL, min_length=WINDOW_SIZE)
    
    # Add cross-channel features
    data = add_cross_channel_features(data)
    
    print(f"   ✓ Preprocessing complete")
    
    # Check data quality
    print(f"\n3. DATA QUALITY CHECKS...")
    
    # Check for NaNs/Infs in sensor columns
    sensor_data = data[SENSOR_COLS]
    has_nan = sensor_data.isna().any().any()
    has_inf = np.isinf(sensor_data.select_dtypes(include=[np.number])).any().any()
    
    print(f"   NaN values: {has_nan}")
    print(f"   Inf values: {has_inf}")
    
    if has_nan or has_inf:
        print(f"   ✗ CRITICAL: Clean data before training!")
        print(f"      Fill NaNs and remove Infs")
        return False
    
    # Check variance per sensor
    sensor_stds = sensor_data.std()
    low_variance_sensors = sensor_stds[sensor_stds < 0.01].index.tolist()
    
    print(f"   Sensors with low variance (<0.01): {len(low_variance_sensors)}/{len(sensor_stds)}")
    if len(low_variance_sensors) > 0:
        print(f"   ⚠️  Low variance sensors: {low_variance_sensors}")
        print(f"      Consider removing or checking data quality")
    
    # Build windows
    print(f"\n4. BUILDING WINDOWS...")
    X_train, y_train, scaler_train = build_clean_windows(
        data,
        SENSOR_COLS,
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=None,
    )
    
    print(f"   ✓ Built {len(X_train)} windows")
    print(f"   Window shape: {X_train.shape}")
    
    # Inject faults
    print(f"\n5. INJECTING FAULTS (for validation)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = (
        inject_faults_with_sensor_labels(
            X_train,
            y_train,
            SENSOR_COLS,
            fault_percentage=fault_percentage,
            random_state=random_state,
        )
    )
    
    # Check class balance
    print(f"\n6. CLASS BALANCE ANALYSIS...")
    is_faulty = (train_sensor_labels.sum(dim=1) > 0).numpy()
    num_normal = np.sum(~is_faulty)
    num_faulty = np.sum(is_faulty)
    imbalance_ratio = num_normal / num_faulty if num_faulty > 0 else float('inf')
    
    print(f"   Normal windows: {num_normal}")
    print(f"   Faulty windows: {num_faulty}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 5:
        print(f"   ⚠️  SEVERE IMBALANCE (>5:1) - must use weighted loss!")
        print(f"      Recommended pos_weight: {imbalance_ratio:.1f}")
    elif imbalance_ratio > 3:
        print(f"   ⚠️  MODERATE IMBALANCE (>3:1) - should use weighted loss")
        print(f"      Recommended pos_weight: {imbalance_ratio:.1f}")
    else:
        print(f"   ✓ Acceptable balance")
    
    # Check fault pattern visibility
    print(f"\n7. FAULT PATTERN ANALYSIS...")
    
    # Convert to numpy for analysis
    X_train_np = X_train_sensor.numpy()  # (N, W, num_sensors)
    train_window_labels_np = train_window_labels.numpy()
    
    faulty_mask = is_faulty
    normal_mask = ~is_faulty
    
    if np.sum(faulty_mask) == 0:
        print(f"   ✗ ERROR: No faulty windows found!")
        return False
    
    # Per-sensor separability (using mean of window)
    sensor_effects = []
    for i in range(X_train_np.shape[2]):  # For each sensor
        normal_values = X_train_np[normal_mask, :, i].mean(axis=1)  # Mean per window
        faulty_values = X_train_np[faulty_mask, :, i].mean(axis=1)
        
        if normal_values.std() > 1e-6:  # Avoid division by zero
            effect = abs(faulty_values.mean() - normal_values.mean()) / normal_values.std()
            sensor_effects.append((i, effect, SENSOR_COLS[i]))
    
    sensor_effects.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Top 5 discriminative sensors:")
    for idx, effect, name in sensor_effects[:5]:
        print(f"      {name}: effect size = {effect:.3f}")
    
    if len(sensor_effects) == 0 or sensor_effects[0][1] < 0.3:
        print(f"   ⚠️  Even best sensor has weak signal (effect < 0.3)")
        print(f"      Check ground truth labels and fault injection")
        if sensor_effects[0][1] < 0.1:
            print(f"   ✗ CRITICAL: Fault signals too weak - check labels!")
            return False
    else:
        print(f"   ✓ Fault patterns are visible in sensor data")
    
    # Summary and recommendation
    print(f"\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    issues = []
    warnings = []
    
    if has_nan or has_inf:
        issues.append("Data contains NaNs/Infs")
    
    if len(low_variance_sensors) > len(SENSOR_COLS) / 2:
        warnings.append(f"Many low-variance sensors ({len(low_variance_sensors)})")
    
    if imbalance_ratio > 5:
        warnings.append(f"Severe class imbalance ({imbalance_ratio:.1f}:1)")
    
    if sensor_effects[0][1] < 0.3:
        warnings.append(f"Weak fault signals (best effect: {sensor_effects[0][1]:.3f})")
    
    if len(issues) > 0:
        print(f"\n✗ CRITICAL ISSUES (must fix before training):")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    if len(warnings) > 0:
        print(f"\n⚠️  WARNINGS (may affect training quality):")
        for warning in warnings:
            print(f"   - {warning}")
    
    print(f"\n✓ Data quality OK - ready for training")
    print(f"\nRecommended training configuration:")
    print(f"   - pos_weight: {imbalance_ratio:.1f} (for weighted loss)")
    print(f"   - embed_dim: 64 (increased from 32)")
    print(f"   - num_epochs: 150 (with early stopping)")
    print(f"   - learning_rate: 0.001 (with decay)")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate GDN training data setup')
    parser.add_argument('--data-path', type=str, 
                       default='data/carOBD/obdiidata',
                       help='Path to data directory')
    parser.add_argument('--fault-percentage', type=float, default=0.15,
                       help='Percentage of windows to inject faults')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    valid = validate_gdn_training_data(
        args.data_path,
        fault_percentage=args.fault_percentage,
        random_state=args.random_state
    )
    
    if valid:
        print(f"\n✓ Ready to retrain GDN")
        sys.exit(0)
    else:
        print(f"\n✗ Fix data issues first")
        sys.exit(1)
