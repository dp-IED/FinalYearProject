import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import time
import sys
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.gdn_processor import GDNPredictor
from kg.create_kg import KnowledgeGraph
from evals.metrics import compute_all_metrics, format_metrics_report


def sensor_labels_to_window_label(sensor_labels: np.ndarray) -> int:
    faulty_indices = np.where(sensor_labels > 0)[0]
    if len(faulty_indices) == 0:
        return 0
    return int(faulty_indices[0]) + 1


def extract_predictions_from_kg(kg: KnowledgeGraph, threshold: float = 0.5) -> Dict[str, np.ndarray]:
    num_windows = len(kg.window_graphs)
    num_sensors = len(kg.sensor_names)
    window_labels = np.zeros(num_windows, dtype=np.int64)
    sensor_labels = np.zeros((num_windows, num_sensors), dtype=np.float32)
    fault_types = []

    for window_idx in sorted(kg.window_graphs.keys()):
        stats = kg.window_stats.get(window_idx, {})
        graph = kg.window_graphs[window_idx]

        for sensor_name, stat in stats.items():
            sensor_idx = kg.sensor_to_idx.get(sensor_name, -1)
            if sensor_idx >= 0 and stat.anomaly_score > threshold:
                sensor_labels[window_idx, sensor_idx] = 1.0

        window_labels[window_idx] = sensor_labels_to_window_label(sensor_labels[window_idx])
        fault_type = None
        if window_labels[window_idx] > 0:
            faulty_sensor_names = [kg.sensor_names[i] for i in range(num_sensors) if sensor_labels[window_idx, i] > 0]
            if 'VEHICLE_SPEED ()' in faulty_sensor_names:
                fault_type = "VSS_DROPOUT"
            elif 'COOLANT_TEMPERATURE ()' in faulty_sensor_names:
                fault_type = "COOLANT_DROPOUT"
            elif 'THROTTLE ()' in faulty_sensor_names:
                fault_type = "TPS_STUCK"
            elif 'INTAKE_MANIFOLD_PRESSURE ()' in faulty_sensor_names:
                fault_type = "MAF_SCALE_LOW"
            elif len(faulty_sensor_names) >= 2 and 'ENGINE_RPM ()' in faulty_sensor_names and 'VEHICLE_SPEED ()' in faulty_sensor_names:
                fault_type = "RPM_SPEED_DECOUPLE"
            else:
                fault_type = "gradual_drift"
        fault_types.append(fault_type)

    return {'window_labels': window_labels, 'sensor_labels': sensor_labels, 'fault_types': fault_types}


def evaluate_gdn_kg(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = 'cpu',
    limit: Optional[int] = None
) -> Dict:
    print("="*80)
    print("Evaluating GDN->KG Method")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print()

    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data['normalized_windows']
    sensor_labels_true = data['sensor_labels']
    window_labels_raw = data['window_labels']

    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        sensor_names = metadata['dataset_info']['sensor_names']
    else:
        sensor_names = [
            'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()',
            'COOLANT_TEMPERATURE ()', 'INTAKE_MANIFOLD_PRESSURE ()',
            'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()'
        ]

    num_windows = normalized_windows.shape[0]
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_raw = window_labels_raw[:num_windows]
        print(f"  LIMIT MODE: Processing only {num_windows} windows")

    window_labels_true = np.zeros(len(window_labels_raw), dtype=np.int64)
    for i in range(len(window_labels_raw)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true[i] = faulty_indices[0] + 1
        else:
            window_labels_true[i] = 0

    print(f"  Loaded {num_windows} windows")
    print()

    print("Initializing GDN Predictor...")
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        detected_embed_dim = checkpoint['sensor_embeddings'].shape[1] if 'sensor_embeddings' in checkpoint else 32
    except Exception:
        detected_embed_dim = 32

    predictor = GDNPredictor(
        model_path=model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=detected_embed_dim,
        top_k=3,
        hidden_dim=32,
        device=device
    )

    print("Processing data through GDN...")
    start_time = time.time()
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true,
        batch_size=batch_size
    )
    gdn_time = time.time() - start_time

    print("Building Knowledge Graph...")
    start_time = time.time()
    kg = KnowledgeGraph(
        sensor_names=kg_data['sensor_names'],
        sensor_embeddings=kg_data['sensor_embeddings'],
        adjacency_matrix=kg_data['adjacency_matrix']
    )
    kg.construct(X_windows=kg_data['X_windows'], gdn_predictions=kg_data['gdn_predictions'])
    kg_time = time.time() - start_time
    print(f"  Nodes: {kg.kg.number_of_nodes()}, Edges: {kg.kg.number_of_edges()}")

    print("Extracting predictions from Knowledge Graph...")
    start_time = time.time()
    predictions = extract_predictions_from_kg(kg, threshold=0.5)
    extraction_time = time.time() - start_time

    total_processing_time = gdn_time + kg_time + extraction_time
    fault_types_true = data.get('fault_types', None)

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=predictions['window_labels'],
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=predictions['sensor_labels'],
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None
    )
    metrics['efficiency'] = {
        'gdn_processing_time_seconds': float(gdn_time),
        'kg_build_time_seconds': float(kg_time),
        'total_processing_time_seconds': float(total_processing_time),
        'windows_per_second': float(num_windows / total_processing_time),
        'kg_nodes': int(kg.kg.number_of_nodes()),
        'kg_edges': int(kg.kg.number_of_edges())
    }

    print(format_metrics_report(metrics))

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
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GDN->KG method on shared evaluation dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to shared dataset .npz file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained GDN model checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='results/gdn_kg.json', help='Output path for results JSON')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for GDN inference')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to run on')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of windows to process (for testing)')
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
