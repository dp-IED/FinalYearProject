"""
Evaluate GDN->KG method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Processes normalized windows through GDN->KG pipeline
3. Extracts fault diagnoses from Knowledge Graph
4. Compares predictions to ground truth
5. Computes evaluation metrics
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys
from tqdm import tqdm

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'anomaly-detection'))
sys.path.insert(0, str(project_root))

from gdn_processor import GDNPredictor
from llm.helpers.KG import KnowledgeGraphBuilder
from metrics import compute_all_metrics, format_metrics_report


def sensor_labels_to_window_label(sensor_labels: np.ndarray) -> int:
    """
    Convert sensor-level labels to window-level sensor-indexed label.
    
    Args:
        sensor_labels: (num_sensors,) binary array - which sensors are faulty
        
    Returns:
        int: 0 if no fault, 1-8 if fault detected (1-indexed sensor index)
             Uses the first faulty sensor index (primary sensor)
    """
    faulty_indices = np.where(sensor_labels > 0)[0]
    if len(faulty_indices) == 0:
        return 0
    # Return first faulty sensor index + 1 (1-indexed: sensor 0 -> label 1, sensor 7 -> label 8)
    return int(faulty_indices[0]) + 1


def extract_predictions_from_kg(
    kg_builder: KnowledgeGraphBuilder,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Extract fault predictions from Knowledge Graph.
    
    Args:
        kg_builder: KnowledgeGraphBuilder instance with built KG
        threshold: Threshold for anomaly scores
        
    Returns:
        Dictionary with:
        - 'window_labels': (N,) binary array
        - 'sensor_labels': (N, num_sensors) binary array
        - 'fault_types': (N,) list of fault type strings
    """
    num_windows = len(kg_builder.window_graphs)
    num_sensors = len(kg_builder.sensor_names)
    
    window_labels = np.zeros(num_windows, dtype=np.int64)
    sensor_labels = np.zeros((num_windows, num_sensors), dtype=np.float32)
    fault_types = []
    
    for window_idx in sorted(kg_builder.window_graphs.keys()):
        stats = kg_builder.window_stats.get(window_idx, {})
        graph = kg_builder.window_graphs[window_idx]
        
        # Extract faulty sensors from window stats
        faulty_sensors_in_window = []
        for sensor_name, stat in stats.items():
            sensor_idx = kg_builder.sensor_to_idx.get(sensor_name, -1)
            if sensor_idx >= 0:
                # Use anomaly_score from stats
                if stat.anomaly_score > threshold:
                    sensor_labels[window_idx, sensor_idx] = 1.0
                    faulty_sensors_in_window.append(sensor_name)
        
        # Convert sensor labels to sensor-indexed window label (0-8)
        window_labels[window_idx] = sensor_labels_to_window_label(sensor_labels[window_idx])
        
        # Determine fault type from KG relationships
        fault_type = "unknown"
        if window_labels[window_idx] > 0:
            # Check for relationship violations
            violations = []
            for u, v, data in graph.edges(data=True):
                if data.get('edge_type') == 'violates_expected_relation':
                    violations.append((u, v))
            
            # Infer fault type from violations and faulty sensors
            faulty_sensor_names = [kg_builder.sensor_names[i] for i in range(num_sensors) 
                                  if sensor_labels[window_idx, i] > 0]
            
            if 'VEHICLE_SPEED ()' in faulty_sensor_names:
                fault_type = "VSS_DROPOUT"
            elif 'COOLANT_TEMPERATURE ()' in faulty_sensor_names:
                fault_type = "COOLANT_DROPOUT"
            elif 'THROTTLE ()' in faulty_sensor_names:
                fault_type = "TPS_STUCK"
            elif 'INTAKE_MANIFOLD_PRESSURE ()' in faulty_sensor_names:
                fault_type = "MAF_SCALE_LOW"
            elif len(faulty_sensor_names) >= 2 and \
                 'ENGINE_RPM ()' in faulty_sensor_names and \
                 'VEHICLE_SPEED ()' in faulty_sensor_names:
                fault_type = "RPM_SPEED_DECOUPLE"
            else:
                fault_type = "gradual_drift"
        
        fault_types.append(fault_type)
    
    return {
        'window_labels': window_labels,
        'sensor_labels': sensor_labels,
        'fault_types': fault_types
    }


def evaluate_gdn_kg(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = 'cpu',
    limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Evaluate GDN->KG method on shared dataset.
    
    Args:
        dataset_path: Path to shared dataset (.npz file)
        model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        limit: Optional limit on number of windows to process (for testing)
        
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("Evaluating GDN->KG Method")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    
    normalized_windows = data['normalized_windows']
    sensor_labels_true = data['sensor_labels']
    window_labels_true = data['window_labels']
    
    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        sensor_names = metadata['dataset_info']['sensor_names']
    else:
        # Fallback: use default sensor names
        sensor_names = [
            'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()',
            'COOLANT_TEMPERATURE ()', 'INTAKE_MANIFOLD_PRESSURE ()',
            'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()'
        ]
    
    num_windows = normalized_windows.shape[0]
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")
    
    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()
    
    # Initialize GDN Predictor
    print("Initializing GDN Predictor...")
    start_time = time.time()
    
    # Try to detect embed_dim from checkpoint
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'sensor_embeddings' in checkpoint:
            detected_embed_dim = checkpoint['sensor_embeddings'].shape[1]
            print(f"  Detected embed_dim from checkpoint: {detected_embed_dim}")
        else:
            detected_embed_dim = 32  # Default
    except:
        detected_embed_dim = 32  # Default
    
    predictor = GDNPredictor(
        model_path=model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=detected_embed_dim,
        top_k=3,
        hidden_dim=32,
        device=device
    )
    
    print(f"  Model loaded in {time.time() - start_time:.2f} seconds")
    print()
    
    # Process data for KG
    print("Processing data through GDN...")
    start_time = time.time()
    
    with tqdm(total=1, desc="GDN Data Processing", unit="step") as pbar:
        kg_data = predictor.process_for_kg(
            X_windows=normalized_windows,
            sensor_labels=sensor_labels_true,  # Use ground truth for KG construction
            window_labels=window_labels_true,
            batch_size=batch_size
        )
        pbar.update(1)
    
    gdn_time = time.time() - start_time
    print(f"  GDN processing completed in {gdn_time:.2f} seconds")
    print()
    
    # Build Knowledge Graph
    print("Building Knowledge Graph...")
    start_time = time.time()
    
    with tqdm(total=1, desc="KG Construction", unit="step") as pbar:
        kg_builder = KnowledgeGraphBuilder(
            sensor_names=kg_data['sensor_names'],
            sensor_embeddings=kg_data['sensor_embeddings'],
            adjacency_matrix=kg_data['adjacency_matrix']
        )
        pbar.update(0.5)
        
        kg = kg_builder.build_from_gdn_windows(
            X_windows=kg_data['X_windows'],
            sensor_labels=kg_data['sensor_labels'],
            window_labels=kg_data['window_labels']
        )
        pbar.update(0.5)
    
    kg_time = time.time() - start_time
    print(f"  Knowledge Graph built in {kg_time:.2f} seconds")
    print(f"  Nodes: {kg.number_of_nodes()}, Edges: {kg.number_of_edges()}")
    print()
    
    # Extract predictions from KG
    print("Extracting predictions from Knowledge Graph...")
    start_time = time.time()
    
    with tqdm(total=1, desc="Prediction Extraction", unit="step") as pbar:
        predictions = extract_predictions_from_kg(kg_builder, threshold=0.5)
        pbar.update(1)
    
    extraction_time = time.time() - start_time
    print(f"  Predictions extracted in {extraction_time:.2f} seconds")
    print()
    
    total_processing_time = gdn_time + kg_time + extraction_time
    
    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = data.get('fault_types', None)
    
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=predictions['window_labels'],
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=predictions['sensor_labels'],
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None
    )
    
    # Add efficiency metrics
    metrics['efficiency'] = {
        'gdn_processing_time_seconds': float(gdn_time),
        'kg_build_time_seconds': float(kg_time),
        'prediction_extraction_time_seconds': float(extraction_time),
        'total_processing_time_seconds': float(total_processing_time),
        'windows_per_second': float(num_windows / total_processing_time),
        'kg_nodes': int(kg.number_of_nodes()),
        'kg_edges': int(kg.number_of_edges())
    }
    
    # Print report
    report = format_metrics_report(metrics)
    print(report)
    
    # Save results
    results = {
        'method': 'gdn_kg',
        'dataset': str(dataset_path),
        'model': str(model_path),
        'num_windows': int(num_windows),
        'metrics': metrics,
        'predictions': {
            'window_labels': predictions['window_labels'].tolist(),
            'sensor_labels': predictions['sensor_labels'].tolist(),
            'fault_types': predictions['fault_types']
        }
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GDN->KG method on shared evaluation dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to shared dataset .npz file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained GDN model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/gdn_kg.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for GDN inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of windows to process (for testing)'
    )
    
    args = parser.parse_args()
    
    evaluate_gdn_kg(
        dataset_path=Path(args.dataset),
        model_path=Path(args.model_path),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
