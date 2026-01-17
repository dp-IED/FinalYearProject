"""
Create a test dataset for pipeline testing.

This creates a small synthetic dataset when real data from gdn.ipynb is not available.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

SENSOR_NAMES = [
    'ENGINE_RPM ()',
    'VEHICLE_SPEED ()',
    'THROTTLE ()',
    'ENGINE_LOAD ()',
    'COOLANT_TEMPERATURE ()',
    'INTAKE_MANIFOLD_PRESSURE ()',
    'SHORT_TERM_FUEL_TRIM_BANK_1 ()',
    'LONG_TERM_FUEL_TRIM_BANK_1 ()',
]


def create_test_dataset(num_windows=100, window_size=300, num_sensors=8, output_dir='llm/evaluation/shared_dataset', split='test'):
    """Create a synthetic test dataset."""
    print(f"Creating test dataset: {num_windows} windows, {window_size} timesteps, {num_sensors} sensors")
    
    # Create normalized windows (0-1 range)
    normalized_windows = np.random.rand(num_windows, window_size, num_sensors).astype(np.float32)
    
    # Create unnormalized windows (realistic ranges)
    # Scale to realistic sensor ranges
    ranges = [
        (600, 6000),    # ENGINE_RPM
        (0, 120),       # VEHICLE_SPEED
        (0, 100),       # THROTTLE
        (0, 100),       # ENGINE_LOAD
        (70, 110),      # COOLANT_TEMPERATURE
        (0, 20),        # INTAKE_MANIFOLD_PRESSURE
        (-25, 25),      # SHORT_TERM_FUEL_TRIM
        (-25, 25),      # LONG_TERM_FUEL_TRIM
    ]
    
    unnormalized_windows = np.zeros_like(normalized_windows)
    for i, (min_val, max_val) in enumerate(ranges):
        unnormalized_windows[:, :, i] = normalized_windows[:, :, i] * (max_val - min_val) + min_val
    
    # Create labels (30% faulty windows)
    np.random.seed(42)
    window_labels = np.random.binomial(1, 0.3, num_windows).astype(np.int64)
    sensor_labels = np.zeros((num_windows, num_sensors), dtype=np.float32)
    
    # Mark random sensors as faulty in faulty windows
    for i in range(num_windows):
        if window_labels[i] > 0:
            num_faulty_sensors = np.random.randint(1, 4)  # 1-3 faulty sensors
            faulty_indices = np.random.choice(num_sensors, num_faulty_sensors, replace=False)
            sensor_labels[i, faulty_indices] = 1.0
    
    # Compute statistical features
    statistical_features = np.zeros((num_windows, num_sensors, 9))
    for i in range(num_windows):
        for j in range(num_sensors):
            values = unnormalized_windows[i, :, j]
            statistical_features[i, j, 0] = np.mean(values)  # mean
            statistical_features[i, j, 1] = np.std(values)   # std
            statistical_features[i, j, 2] = np.min(values)    # min
            statistical_features[i, j, 3] = np.max(values)   # max
            statistical_features[i, j, 4] = statistical_features[i, j, 3] - statistical_features[i, j, 2]  # range
            statistical_features[i, j, 5] = np.median(values)  # median
            statistical_features[i, j, 6] = np.mean(values)   # mode (simplified)
            statistical_features[i, j, 7] = 0.0  # skewness (simplified)
            statistical_features[i, j, 8] = 0.0  # kurtosis (simplified)
    
    # Create fault types
    fault_types = np.array(['unknown'] * num_windows)
    for i in range(num_windows):
        if window_labels[i] > 0:
            faulty_sensors = [SENSOR_NAMES[j] for j in range(num_sensors) if sensor_labels[i, j] > 0]
            if 'VEHICLE_SPEED' in str(faulty_sensors):
                fault_types[i] = 'VSS_DROPOUT'
            elif 'COOLANT_TEMPERATURE' in str(faulty_sensors):
                fault_types[i] = 'COOLANT_DROPOUT'
            elif 'THROTTLE' in str(faulty_sensors):
                fault_types[i] = 'TPS_STUCK'
            else:
                fault_types[i] = 'gradual_drift'
    
    # Save dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .npz
    npz_path = output_dir / f'{split}.npz'
    np.savez_compressed(
        npz_path,
        normalized_windows=normalized_windows,
        unnormalized_windows=unnormalized_windows,
        sensor_labels=sensor_labels,
        window_labels=window_labels,
        fault_types=fault_types,
        statistical_features=statistical_features
    )
    print(f"✓ Saved .npz file: {npz_path}")
    
    # Save metadata
    json_path = output_dir / f'{split}_metadata.json'
    metadata = {
        'dataset_info': {
            'name': 'test_evaluation_dataset',
            'split': split,
            'num_windows': int(num_windows),
            'window_size': int(window_size),
            'num_sensors': int(num_sensors),
            'sensor_names': SENSOR_NAMES,
            'created_at': datetime.now().isoformat()
        },
        'sensor_names': SENSOR_NAMES,
        'metadata': {
            'window_ids': list(range(num_windows))
        }
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata JSON: {json_path}")
    
    print(f"\n✓ Test dataset created: {num_windows} windows, {window_labels.sum()} faulty")
    return npz_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-windows', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='llm/evaluation/shared_dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    create_test_dataset(args.num_windows, output_dir=args.output_dir, split=args.split)
