"""
Evaluate GDN->KG->LLM method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Processes normalized windows through GDN->KG pipeline
3. Extracts KG context for each window
4. Formats KG-enhanced prompts for LLM
5. Runs LLM inference with KG context
6. Compares predictions to ground truth
7. Computes evaluation metrics

Inspired by KAG (Knowledge-Augmented Generation) research.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import sys
from tqdm import tqdm

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'anomaly-detection'))
sys.path.insert(0, str(project_root))

from gdn_processor import GDNPredictor
from llm.helpers.KG import KnowledgeGraphBuilder, EXPECTED_CORRELATIONS, SENSOR_SUBSYSTEMS
from llm.evaluation.evaluate_llm_baseline import (
    load_llm_model,
    call_llm,
    parse_llm_response,
    format_window_for_llm
)
from metrics import compute_all_metrics, format_metrics_report


def extract_window_kg_context(
    kg_builder: KnowledgeGraphBuilder,
    window_idx: int,
    temporal_context_windows: int = 2
) -> Dict[str, any]:
    """
    Extract KG context for a specific window (KAG-inspired).
    
    Args:
        kg_builder: KnowledgeGraphBuilder instance with built KG
        window_idx: Index of the window to extract context for
        temporal_context_windows: Number of previous windows to include
        
    Returns:
        Dictionary with KG context:
        - 'entities': List of entities with types
        - 'relationships': List of relationship triples
        - 'violations': List of relationship violations
        - 'temporal_context': Temporal information from previous windows
        - 'anomaly_propagation': Relevant anomaly propagation chains
    """
    context = {
        'entities': [],
        'relationships': [],
        'violations': [],
        'temporal_context': [],
        'anomaly_propagation': []
    }
    
    # Get current window graph and stats
    if window_idx not in kg_builder.window_graphs:
        return context
    
    window_graph = kg_builder.window_graphs[window_idx]
    window_stats = kg_builder.window_stats.get(window_idx, {})
    
    # Extract entities with types and subsystems
    sensor_descriptions = kg_builder.get_sensor_descriptions()
    for sensor_name in kg_builder.sensor_names:
        desc = sensor_descriptions.get(sensor_name, {})
        subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, 'Unknown')
        entity_info = {
            'name': sensor_name,
            'type': 'Sensor',
            'subsystem': subsystem,
            'description': desc.get('description', ''),
            'is_faulty': window_stats.get(sensor_name, {}).anomaly_score > 0 if sensor_name in window_stats else False
        }
        context['entities'].append(entity_info)
    
    # Extract relationships from current window
    # Include all violations and relationships involving anomalous sensors
    # Also include significant correlations (threshold 0.3)
    correlation_threshold = 0.3
    anomalous_sensors = {sensor_name for sensor_name, stat in window_stats.items() if stat.anomaly_score > 0}
    
    for u, v, data in window_graph.edges(data=True):
        edge_type = data.get('edge_type', 'correlates_with')
        correlation = data.get('correlation', 0)
        expected_correlation = data.get('expected_correlation', 0)
        deviation = data.get('correlation_deviation', 0)
        
        # Include if:
        # 1. It's a violation (always important)
        # 2. It involves an anomalous sensor (contextual information)
        # 3. It's a significant correlation (above threshold)
        is_violation = edge_type == 'violates_expected_relation'
        involves_anomaly = u in anomalous_sensors or v in anomalous_sensors
        is_significant = abs(correlation) >= correlation_threshold
        
        if not (is_violation or involves_anomaly or is_significant):
            continue
        
        relationship = {
            'source': u,
            'target': v,
            'relation': edge_type,
            'correlation': float(correlation),
            'expected_correlation': float(expected_correlation),
            'deviation': float(deviation)
        }
        
        context['relationships'].append(relationship)
        
        # Track violations separately
        if is_violation:
            context['violations'].append(relationship)
    
    # Extract temporal context from previous windows
    for prev_idx in range(max(0, window_idx - temporal_context_windows), window_idx):
        if prev_idx in kg_builder.window_stats:
            prev_stats = kg_builder.window_stats[prev_idx]
            temporal_info = {
                'window_idx': prev_idx,
                'faulty_sensors': [],
                'anomaly_scores': {}
            }
            
            for sensor_name, stat in prev_stats.items():
                if stat.anomaly_score > 0:
                    temporal_info['faulty_sensors'].append(sensor_name)
                    temporal_info['anomaly_scores'][sensor_name] = float(stat.anomaly_score)
            
            if temporal_info['faulty_sensors']:
                context['temporal_context'].append(temporal_info)
    
    # Extract relevant anomaly propagation chains
    for chain in kg_builder.anomaly_propagation_chains:
        root_window = chain.get('root_window', -1)
        propagation_timeline = chain.get('propagation_timeline', [])
        
        # Check if this window is involved in the chain
        if root_window == window_idx:
            context['anomaly_propagation'].append({
                'type': 'root',
                'root_sensor': chain.get('root_sensor', ''),
                'root_window': root_window,
                'affected_sensors': chain.get('affected_sensors', [])
            })
        else:
            # Check if window appears in propagation timeline
            for timeline_entry in propagation_timeline:
                if timeline_entry.get('window') == window_idx:
                    context['anomaly_propagation'].append({
                        'type': 'propagation',
                        'root_sensor': chain.get('root_sensor', ''),
                        'root_window': root_window,
                        'affected_sensors': timeline_entry.get('affected_sensors', [])
                    })
                    break
    
    return context


def format_kg_context_for_llm(
    kg_context: Dict[str, any],
    window_idx: int,
    kg_builder: KnowledgeGraphBuilder
) -> str:
    """
    Format KG context as structured LLM prompt section following KAG best practices.
    
    Shows structured knowledge graph representation: entities, relationships, violations,
    temporal context, and anomaly propagation. NO raw sensor data.
    
    Args:
        kg_context: Context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg_builder: KnowledgeGraphBuilder instance
        
    Returns:
        Formatted string for LLM prompt with structured KG representation
    """
    lines = []
    lines.append("Knowledge Graph Representation:")
    lines.append("=" * 80)
    
    # All entities with their status and metadata
    lines.append("\nENTITIES:")
    lines.append("-" * 80)
    for entity in kg_context['entities']:
        status = "⚠️ ANOMALOUS" if entity.get('is_faulty') else "✓ Normal"
        lines.append(f"{status}: {entity['name']}")
        lines.append(f"  Subsystem: {entity['subsystem']}")
        if entity.get('description'):
            lines.append(f"  Description: {entity['description']}")
    
    # All relationships (not filtered - show all significant ones)
    lines.append("\nRELATIONSHIPS:")
    lines.append("-" * 80)
    if kg_context['relationships']:
        for rel in kg_context['relationships']:
            rel_type = rel['relation']
            source = rel['source']
            target = rel['target']
            corr = rel.get('correlation', 0)
            exp_corr = rel.get('expected_correlation', 0)
            
            if rel_type == 'violates_expected_relation':
                dev = rel.get('deviation', 0)
                lines.append(f"⚠️ VIOLATION: {source} --[{rel_type}]--> {target}")
                lines.append(f"  Expected correlation: {exp_corr:.3f}")
                lines.append(f"  Actual correlation: {corr:.3f}")
                lines.append(f"  Deviation: {dev:.3f}")
            else:
                lines.append(f"{source} --[{rel_type}]--> {target}")
                lines.append(f"  Correlation: {corr:.3f}")
                if exp_corr != 0:
                    lines.append(f"  Expected correlation: {exp_corr:.3f}")
    else:
        lines.append("No significant relationships detected.")
    
    # Temporal context (all relevant previous windows)
    if kg_context['temporal_context']:
        lines.append("\nTEMPORAL CONTEXT:")
        lines.append("-" * 80)
        for temp in kg_context['temporal_context']:
            lines.append(f"Window {temp['window_idx']}:")
            if temp['faulty_sensors']:
                lines.append(f"  Faulty sensors: {', '.join(temp['faulty_sensors'])}")
                if temp.get('anomaly_scores'):
                    scores_str = ', '.join([f"{sensor}={score:.2f}" for sensor, score in list(temp['anomaly_scores'].items())[:3]])
                    if scores_str:
                        lines.append(f"  Anomaly scores: {scores_str}")
            else:
                lines.append("  No faults detected")
    
    # Anomaly propagation chains
    if kg_context['anomaly_propagation']:
        lines.append("\nANOMALY PROPAGATION:")
        lines.append("-" * 80)
        for prop in kg_context['anomaly_propagation']:
            prop_type = prop.get('type', 'unknown')
            root_sensor = prop.get('root_sensor', 'unknown')
            root_window = prop.get('root_window', -1)
            affected = prop.get('affected_sensors', [])
            
            if prop_type == 'root':
                lines.append(f"Root cause detected:")
                lines.append(f"  Root sensor: {root_sensor} at window {root_window}")
                if affected:
                    lines.append(f"  Affected sensors: {', '.join(affected)}")
            else:
                lines.append(f"Propagation from window {root_window}:")
                lines.append(f"  Root sensor: {root_sensor}")
                if affected:
                    lines.append(f"  Affected sensors in this window: {', '.join(affected)}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def format_window_with_kg_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    kg_context: Dict[str, any],
    window_idx: int,
    kg_builder: KnowledgeGraphBuilder,
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True
) -> str:
    """
    Format prompt with structured KG representation only (KAG-style).
    
    NO raw sensor data - only structured knowledge graph representation.
    This follows KAG best practices where LLM reasons over structured KG,
    not raw time series data.
    
    Args:
        window_data: (window_size, num_sensors) array - unnormalized sensor values (not used, kept for API compatibility)
        sensor_names: List of sensor names
        kg_context: KG context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg_builder: KnowledgeGraphBuilder instance
        statistical_features: Optional statistical features (not used, kept for API compatibility)
        use_statistical_features: Whether to include statistical features (not used, kept for API compatibility)
        
    Returns:
        Complete formatted prompt string with structured KG representation only
    """
    # Format KG context section (structured representation)
    kg_section = format_kg_context_for_llm(kg_context, window_idx, kg_builder)
    
    # Build prompt similar to baseline format for comparability
    lines = []
    lines.append("You are an automotive diagnostic expert analyzing OBD-II sensor data.")
    lines.append("")
    lines.append("Task: Identify which sensors are faulty and describe the fault type.")
    lines.append("")
    lines.append("The following knowledge graph representation was generated from sensor data analysis:")
    lines.append("")
    lines.append(kg_section)
    lines.append("")
    lines.append("Please analyze this knowledge graph representation and provide:")
    lines.append("1. List of faulty sensor names (if any)")
    lines.append("2. Fault type description (e.g., VSS_DROPOUT, COOLANT_DROPOUT, TPS_STUCK, MAF_SCALE_LOW, RPM_SPEED_DECOUPLE, gradual_drift, intermittent_spike, slow_response, bias_offset, electrical_jitter)")
    lines.append("3. Brief reasoning for your diagnosis")
    lines.append("")
    lines.append("Format your response as:")
    lines.append("Faulty Sensors: [sensor1, sensor2, ...] or None")
    lines.append("Fault Type: [fault_type]")
    lines.append("Reasoning: [your analysis]")
    
    return "\n".join(lines)


def evaluate_gdn_kg_llm(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = 'cpu',
    model_repo: Optional[str] = None,
    max_tokens: int = 2048,  # Increased for KG-enhanced prompts
    temperature: float = 0.7,
    use_statistical_features: bool = True
) -> Dict[str, any]:
    """
    Evaluate GDN->KG->LLM method on shared dataset.
    
    Args:
        dataset_path: Path to shared dataset (.npz file)
        model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        model_repo: LLM model repository identifier
        max_tokens: Maximum tokens for LLM generation
        temperature: LLM sampling temperature
        use_statistical_features: Whether to include statistical features in prompts
        
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("Evaluating GDN->KG->LLM Method")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {model_path}")
    print()
    
    # Load LLM model
    if model_repo is None:
        model_repo = "mlx-community/granite-4.0-h-micro-4bit"
    
    try:
        model, tokenizer = load_llm_model(model_repo)
        print()
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM model: {e}. Please ensure mlx-lm is installed.")
    
    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    
    normalized_windows = data['normalized_windows']
    unnormalized_windows = data['unnormalized_windows']
    sensor_labels_true = data['sensor_labels']
    window_labels_true = data['window_labels']
    
    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        sensor_names = metadata['dataset_info']['sensor_names']
        statistical_features = data.get('statistical_features', None)
    else:
        sensor_names = [
            'ENGINE_RPM ()', 'VEHICLE_SPEED ()', 'THROTTLE ()', 'ENGINE_LOAD ()',
            'COOLANT_TEMPERATURE ()', 'INTAKE_MANIFOLD_PRESSURE ()',
            'SHORT_TERM_FUEL_TRIM_BANK_1 ()', 'LONG_TERM_FUEL_TRIM_BANK_1 ()'
        ]
        statistical_features = None
    
    num_windows = normalized_windows.shape[0]
    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()
    
    # Initialize GDN Predictor (reuse from evaluate_gdn_kg.py)
    print("Initializing GDN Predictor...")
    start_time = time.time()
    
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'sensor_embeddings' in checkpoint:
            detected_embed_dim = checkpoint['sensor_embeddings'].shape[1]
            print(f"  Detected embed_dim from checkpoint: {detected_embed_dim}")
        else:
            detected_embed_dim = 32
    except:
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
    
    print(f"  Model loaded in {time.time() - start_time:.2f} seconds")
    print()
    
    # Process data for KG (reuse from evaluate_gdn_kg.py)
    print("Processing data through GDN...")
    start_time = time.time()
    
    with tqdm(total=1, desc="GDN Data Processing", unit="step") as pbar:
        kg_data = predictor.process_for_kg(
            X_windows=normalized_windows,
            sensor_labels=sensor_labels_true,
            window_labels=window_labels_true,
            batch_size=batch_size
        )
        pbar.update(1)
    
    gdn_time = time.time() - start_time
    print(f"  GDN processing completed in {gdn_time:.2f} seconds")
    print()
    
    # Build Knowledge Graph (reuse from evaluate_gdn_kg.py)
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
    
    # Run LLM predictions with KG context
    print("Running LLM predictions with KG context...")
    window_labels_pred = []
    sensor_labels_pred = []
    fault_types_pred = []
    reasoning_list = []
    processing_times = []
    
    with tqdm(total=num_windows, desc="KG-Enhanced LLM Inference", unit="window") as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            
            # Extract KG context for this window
            kg_context = extract_window_kg_context(kg_builder, window_idx, temporal_context_windows=2)
            
            # Get window data
            window_data = unnormalized_windows[window_idx]
            stats = statistical_features[window_idx] if statistical_features is not None and len(statistical_features) > window_idx else None
            
            # Format prompt with KG context
            prompt = format_window_with_kg_for_llm(
                window_data, sensor_names, kg_context, window_idx, kg_builder,
                stats, use_statistical_features
            )
            
            # Call LLM with repetition penalty to prevent degenerate output
            try:
                response = call_llm(
                    prompt, 
                    model, 
                    tokenizer, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    repetition_penalty=1.2,  # Penalize repetition
                    repetition_context_size=20  # Context window for repetition check
                )
                prediction = parse_llm_response(response, sensor_names)
                prediction['reasoning'] = response[:200]  # Store first 200 chars
            except Exception as e:
                # Fallback to no-fault prediction
                prediction = {
                    'window_label': 0,
                    'sensor_labels': np.zeros(len(sensor_names), dtype=np.float32),
                    'fault_type': "unknown",
                    'reasoning': f"Error: {str(e)}"
                }
            
            window_labels_pred.append(prediction['window_label'])
            sensor_labels_pred.append(prediction['sensor_labels'])
            fault_types_pred.append(prediction['fault_type'])
            reasoning_list.append(prediction.get('reasoning', ''))
            processing_times.append(time.time() - start_time)
            
            pbar.update(1)
            if (window_idx + 1) % 10 == 0:
                avg_time = np.mean(processing_times[-10:]) if len(processing_times) >= 10 else np.mean(processing_times)
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s"})
    
    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)
    
    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    llm_time = total_processing_time
    
    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total LLM processing time: {llm_time:.2f} seconds")
    print()
    
    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = data.get('fault_types', None)
    
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None
    )
    
    # Add efficiency metrics
    total_time = gdn_time + kg_time + llm_time
    metrics['efficiency'] = {
        'gdn_processing_time_seconds': float(gdn_time),
        'kg_build_time_seconds': float(kg_time),
        'llm_processing_time_seconds': float(llm_time),
        'total_processing_time_seconds': float(total_time),
        'windows_per_second': float(num_windows / total_time),
        'kg_nodes': int(kg.number_of_nodes()),
        'kg_edges': int(kg.number_of_edges())
    }
    
    # Print report
    report = format_metrics_report(metrics)
    print(report)
    
    # Save results
    results = {
        'method': 'gdn_kg_llm',
        'dataset': str(dataset_path),
        'gdn_model': str(model_path),
        'llm_model': model_repo,
        'num_windows': int(num_windows),
        'metrics': metrics,
        'predictions': {
            'window_labels': window_labels_pred.tolist(),
            'sensor_labels': sensor_labels_pred.tolist(),
            'fault_types': fault_types_pred,
            'reasoning': reasoning_list[:10] if len(reasoning_list) > 10 else reasoning_list  # Sample reasoning
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
        description='Evaluate GDN->KG->LLM method on shared evaluation dataset'
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
        default='results/gdn_kg_llm.json',
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
        '--model-repo',
        type=str,
        default=None,
        help='LLM model repository identifier'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for LLM generation (default: 2048 for KG-enhanced prompts)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM sampling temperature'
    )
    parser.add_argument(
        '--no-statistical-features',
        action='store_true',
        help='Disable statistical features in prompts'
    )
    
    args = parser.parse_args()
    
    evaluate_gdn_kg_llm(
        dataset_path=Path(args.dataset),
        model_path=Path(args.model_path),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
        model_repo=args.model_repo,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_statistical_features=not args.no_statistical_features
    )


if __name__ == '__main__':
    main()
