"""
Evaluate KAG Solver v1 (Heuristic Multi-Step Reasoning)

Evaluates the deterministic KAG solver v1 on shared evaluation dataset.
This solver performs multi-step graph reasoning without LLM planning,
providing a baseline for comparison with LLM-planned KAG (Week 4).

Usage:
    python llm/evaluation/evaluate_kag_v1.py \
        --dataset llm/evaluation/shared_dataset/test.npz \
        --neo4j-uri bolt://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password password \
        --output results/kag_v1_test.json
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import time
import sys
from tqdm import tqdm

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.kag.neo4j_queries import Neo4jKAGQueries
from llm.kag.solver_v1 import KAGSolverV1
from llm.evaluation.metrics import compute_all_metrics, format_metrics_report


def evaluate_kag_v1(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    neo4j_uri: str = 'bolt://127.0.0.1:7687',
    neo4j_user: str = 'neo4j',
    neo4j_password: str = 'password',
    anomaly_threshold: float = 0.5,
    min_anomalous: int = 1,
    confidence_min: float = 0.4,
    limit: Optional[int] = None
) -> Dict[str, any]:
    """
    Evaluate KAG Solver V1 (heuristic multi-step reasoning).
    
    Args:
        dataset_path: Path to shared dataset (.npz file)
        output_path: Optional path to save results JSON
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        anomaly_threshold: Threshold for anomaly detection (default: 0.5)
        min_anomalous: Minimum number of anomalous sensors required (default: 1)
        confidence_min: Minimum confidence threshold to accept a fault (default: 0.4)
        limit: Optional limit on number of windows to process (for testing)
        
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("Evaluating KAG Solver v1 (Heuristic Multi-Step Reasoning)")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Neo4j URI: {neo4j_uri}")
    print(f"Anomaly Threshold: {anomaly_threshold}")
    print(f"Min Anomalous Sensors: {min_anomalous}")
    print(f"Min Confidence: {confidence_min}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    
    sensor_labels_true = data['sensor_labels']
    window_labels_true = data['window_labels']
    
    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        sensor_names = metadata['dataset_info']['sensor_names']
    else:
        # Fallback to default sensor names
        sensor_names = [
            'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()',
            'COOLANT_TEMPERATURE ()', 'INTAKE_MANIFOLD_PRESSURE ()',
            'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()'
        ]
    
    num_windows = len(sensor_labels_true)
    
    # Get fault_types before limiting
    fault_types_true = data.get('fault_types', None)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if fault_types_true is not None:
            fault_types_true = fault_types_true[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")
    
    print(f"  Loaded {num_windows} windows")
    print(f"  Sensors: {len(sensor_names)}")
    print()
    
    # Initialize Neo4j queries
    print("Connecting to Neo4j...")
    try:
        queries = Neo4jKAGQueries(neo4j_uri, neo4j_user, neo4j_password)
        # Test connection with a simple query
        with queries.driver.session() as session:
            session.run("RETURN 1").single()
        print("  ✓ Connected to Neo4j")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Neo4j: {e}. Please ensure Neo4j is running and accessible.")
    
    # Initialize solver
    print("Initializing KAG Solver v1...")
    solver = KAGSolverV1(
        queries, 
        sensor_names, 
        anomaly_threshold=anomaly_threshold,
        min_anomalous=min_anomalous,
        confidence_min=confidence_min
    )
    print(f"  ✓ Solver initialized (anomaly_threshold={anomaly_threshold}, min_anomalous={min_anomalous}, confidence_min={confidence_min})")
    print()
    
    # Process windows
    print("Running KAG Solver v1 on windows...")
    window_labels_pred = []
    sensor_labels_pred = []
    fault_types_pred = []
    reasoning_traces = []
    processing_times = []
    
    with tqdm(total=num_windows, desc="KAG Solver v1", unit="window") as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            
            try:
                result = solver.solve(window_idx)
                
                window_labels_pred.append(result['window_label'])
                sensor_labels_pred.append(result['sensor_labels'])
                fault_types_pred.append(result['fault_type'])
                reasoning_traces.append(result['reasoning_trace'])
            except Exception as e:
                # Fallback to no-fault prediction on error
                print(f"\n  ⚠️  Warning: Error processing window {window_idx}: {e}")
                window_labels_pred.append(0)
                sensor_labels_pred.append(np.zeros(len(sensor_names), dtype=int))
                fault_types_pred.append(None)
                reasoning_traces.append([{'step': 0, 'operation': 'error', 'result': str(e)}])
            
            processing_times.append(time.time() - start_time)
            pbar.update(1)
            
            if (window_idx + 1) % 10 == 0:
                avg_time = np.mean(processing_times[-10:]) if len(processing_times) >= 10 else np.mean(processing_times)
                pbar.set_postfix({"avg_time": f"{avg_time:.3f}s"})
    
    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)
    
    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    
    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total processing time: {total_processing_time:.2f} seconds")
    print()
    
    # Close Neo4j connection
    queries.close()
    
    # Convert window_labels_true to sensor-indexed format
    # window_labels_true from dataset are window indices (0, 1, 2, ...), not sensor-indexed labels
    # We need to convert to sensor-indexed format (0-8) based on sensor_labels_true
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        # Find first faulty sensor
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1  # 1-indexed (sensor 0 -> label 1)
        else:
            window_labels_true_converted[i] = 0  # No fault
    window_labels_true = window_labels_true_converted
    
    # Compute metrics
    print("Computing evaluation metrics...")
    
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None
    )
    
    # Add efficiency metrics
    metrics['efficiency'] = {
        'total_processing_time_seconds': float(total_processing_time),
        'average_processing_time_seconds': float(avg_processing_time),
        'windows_per_second': float(num_windows / total_processing_time) if total_processing_time > 0 else 0.0
    }
    
    # Print report
    report = format_metrics_report(metrics)
    print(report)
    
    # Save results
    results = {
        'method': 'kag_v1',
        'dataset': str(dataset_path),
        'neo4j_uri': neo4j_uri,
        'anomaly_threshold': float(anomaly_threshold),
        'min_anomalous': int(min_anomalous),
        'confidence_min': float(confidence_min),
        'num_windows': int(num_windows),
        'metrics': metrics,
        'predictions': {
            'window_labels': window_labels_pred.tolist(),
            'sensor_labels': sensor_labels_pred.tolist(),
            'fault_types': fault_types_pred,
            'reasoning_traces': reasoning_traces[:10] if len(reasoning_traces) > 10 else reasoning_traces  # Sample traces
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
        description='Evaluate KAG Solver v1 (heuristic multi-step reasoning) on shared evaluation dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to shared dataset .npz file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/kag_v1.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://127.0.0.1:7687',
        help='Neo4j connection URI (default: bolt://127.0.0.1:7687)'
    )
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j username (default: neo4j)'
    )
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default='password',
        help='Neo4j password (default: password)'
    )
    parser.add_argument(
        '--anomaly-threshold',
        type=float,
        default=0.5,
        help='Threshold for anomaly detection (default: 0.5). Sensors with anomaly_score > threshold are considered anomalous.'
    )
    parser.add_argument(
        '--min-anomalous',
        type=int,
        default=1,
        help='Minimum number of anomalous sensors required (default: 1)'
    )
    parser.add_argument(
        '--confidence-min',
        type=float,
        default=0.4,
        help='Minimum confidence threshold to accept a fault (default: 0.4)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of windows to process (for testing)'
    )
    
    args = parser.parse_args()
    
    evaluate_kag_v1(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        anomaly_threshold=args.anomaly_threshold,
        min_anomalous=args.min_anomalous,
        confidence_min=args.confidence_min,
        limit=args.limit
    )


if __name__ == '__main__':
    main()
