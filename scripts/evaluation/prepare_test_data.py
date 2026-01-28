#!/usr/bin/env python3
"""
Prepare test data for GDN evaluation.
Extracts test data from the same data loading pipeline used in training.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "anomaly-detection"))

# Import data preprocessing functions from training script
from scripts.training.train_gdn_center_repulsion import (
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

DATA_PATH = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata"


def main():
    print("=" * 80)
    print("Preparing Test Data for GDN Evaluation")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df_list = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{DATA_PATH}/{file}", index_col=False)
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
    data = filter_long_drives(data, id_col=ID_COL, min_length=WINDOW_SIZE + 1)
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15) - same as training
    print("\nSplitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)

    train_drives = unique_drives[: int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives) :]

    print(
        f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}"
    )

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    test_data = data[data[ID_COL].isin(test_drives)].copy()

    # Build clean windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_test_clean, y_test_clean, _ = build_clean_windows(
        test_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Clean test windows: {len(X_test_clean)}")

    # Inject faults with sensor-level labels (same as training: random_state=44, fault_percentage=0.30)
    print("\nInjecting faults with sensor-level labels (30% fault rate)...")
    X_test_sensor, _, test_sensor_labels, test_window_labels = (
        inject_faults_with_sensor_labels(
            X_test_clean,
            y_test_clean,
            SENSOR_COLS,
            fault_percentage=0.30,
            random_state=44,
        )
    )

    # Statistics
    test_faulty = (test_sensor_labels.sum(dim=1) > 0).sum().item()
    print(f"\nTest:  {test_faulty}/{len(X_test_sensor)} faulty windows")
    print(
        f"  Avg sensors per fault: {test_sensor_labels[test_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}"
    )

    # Convert to numpy
    X_test_np = (
        X_test_sensor.numpy()
        if isinstance(X_test_sensor, torch.Tensor)
        else X_test_sensor
    )
    y_test_np = (
        test_window_labels.numpy()
        if isinstance(test_window_labels, torch.Tensor)
        else test_window_labels
    )

    # Save as .npz
    output_path = Path(__file__).parent.parent.parent / "data" / "test_data_gdn.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_test=X_test_np.astype(np.float32),
        y_test=y_test_np.astype(np.int64),
    )

    print(f"\n✓ Test data saved to: {output_path}")
    print(f"  Shape: X_test={X_test_np.shape}, y_test={y_test_np.shape}")
    print(f"  Normal: {(y_test_np == 0).sum()}, Anomalous: {(y_test_np == 1).sum()}")

    return str(output_path)


if __name__ == "__main__":
    main()
