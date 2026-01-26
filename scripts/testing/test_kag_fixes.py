"""Test KAG v2 fixes on sample windows before full evaluation."""

import numpy as np
import json
import sys
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

from llm.kag.neo4j_queries import Neo4jKAGQueries
from llm.kag.solver_v2 import KAGIterativeSolver
from llm.helpers.KG import KnowledgeGraphBuilder
from llm.evaluation.evaluate_llm_baseline import load_llm_model


def test_on_sample_windows(
    dataset_path: str,
    gdn_model_path: str,
    neo4j_uri: str = "bolt://127.0.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    model_repo: str = None,
):
    """Test fixes on 10 representative windows."""
    
    print("="*80)
    print("TESTING KAG V2 FIXES")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]
    fault_types_true = data.get("fault_types", None)
    
    # Load metadata
    metadata_path = Path(dataset_path).parent / f"{Path(dataset_path).stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]
    
    # Identify faulty and normal windows
    is_faulty = np.sum(sensor_labels_true, axis=1) > 0
    faulty_indices = np.where(is_faulty)[0]
    normal_indices = np.where(~is_faulty)[0]
    
    # Sample: 5 faulty windows + 5 normal windows
    np.random.seed(42)
    test_windows = {
        'faulty': np.random.choice(faulty_indices, min(5, len(faulty_indices)), replace=False).tolist() if len(faulty_indices) > 0 else [],
        'normal': np.random.choice(normal_indices, min(5, len(normal_indices)), replace=False).tolist() if len(normal_indices) > 0 else []
    }
    
    print(f"Selected test windows:")
    print(f"  Faulty: {test_windows['faulty']}")
    print(f"  Normal: {test_windows['normal']}")
    
    # Load LLM model
    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"
    
    print("\nLoading LLM model...")
    try:
        model, tokenizer = load_llm_model(model_repo)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load LLM: {e}")
        return
    
    # Load GDN checkpoint
    print("\nLoading GDN checkpoint...")
    import torch
    checkpoint = torch.load(gdn_model_path, map_location="cpu")
    sensor_embeddings = checkpoint.get("sensor_embeddings")
    
    if sensor_embeddings is None:
        num_sensors = len(sensor_names)
        embed_dim = checkpoint.get("embed_dim", 32)
        sensor_embeddings = np.random.randn(num_sensors, embed_dim).astype(np.float32)
    
    if hasattr(sensor_embeddings, "numpy"):
        sensor_embeddings = sensor_embeddings.numpy()
    elif isinstance(sensor_embeddings, torch.Tensor):
        sensor_embeddings = sensor_embeddings.cpu().numpy()
    
    # Compute adjacency matrix
    norms = np.linalg.norm(sensor_embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    sensor_embeddings_norm = sensor_embeddings / norms
    similarity_matrix = np.dot(sensor_embeddings_norm, sensor_embeddings_norm.T)
    adjacency_matrix = (similarity_matrix + 1.0) / 2.0
    adjacency_matrix = np.clip(adjacency_matrix, 0.1, 1.0)
    np.fill_diagonal(adjacency_matrix, 0.0)
    
    # Create KG builder
    kg_builder = KnowledgeGraphBuilder(
        sensor_names=sensor_names,
        sensor_embeddings=sensor_embeddings,
        adjacency_matrix=adjacency_matrix,
    )
    
    # Initialize Neo4j queries
    print("\nConnecting to Neo4j...")
    try:
        queries = Neo4jKAGQueries(neo4j_uri, neo4j_user, neo4j_password)
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return
    
    # Initialize solver
    print("\nInitializing KAG Solver v2...")
    solver = KAGIterativeSolver(
        kg_builder=kg_builder,
        neo4j_queries=queries,
        sensor_names=sensor_names,
        model=model,
        tokenizer=tokenizer,
        max_iterations=1,
    )
    print("✓ Solver initialized")
    
    # Test windows
    results = {
        'faulty': {'correct': 0, 'total': len(test_windows['faulty'])},
        'normal': {'correct': 0, 'total': len(test_windows['normal'])}
    }
    
    for category, windows in test_windows.items():
        print(f"\n{'='*80}")
        print(f"Testing {category.upper()} windows: {windows}")
        print(f"{'='*80}\n")
        
        for window_idx in windows:
            print(f"\n--- Window {window_idx} ---")
            
            # Get ground truth
            gt_sensor_labels = sensor_labels_true[window_idx]
            gt_fault_type = fault_types_true[window_idx] if fault_types_true is not None else None
            gt_is_faulty = np.sum(gt_sensor_labels) > 0
            
            print(f"Ground Truth: {'FAULT' if gt_is_faulty else 'NORMAL'}")
            if gt_is_faulty:
                faulty_sensors = [sensor_names[i] for i in range(len(sensor_names)) if gt_sensor_labels[i] > 0]
                print(f"  Faulty sensors: {faulty_sensors}")
                print(f"  Fault type: {gt_fault_type}")
            
            # Run KAG v2
            try:
                result = solver.solve(window_idx)
                
                pred_fault_type = result['fault_type']
                pred_is_faulty = (pred_fault_type is not None and pred_fault_type != 'NORMAL')
                pred_sensor_labels = result['sensor_labels']
                pred_faulty_sensors = [sensor_names[i] for i in range(len(sensor_names)) if pred_sensor_labels[i] > 0]
                
                print(f"Prediction: {'FAULT' if pred_is_faulty else 'NORMAL'}")
                print(f"  Faulty sensors: {pred_faulty_sensors}")
                print(f"  Fault type: {pred_fault_type}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                # Check correctness
                if category == 'faulty':
                    correct = pred_is_faulty  # Should predict some fault
                else:
                    correct = not pred_is_faulty  # Should predict NORMAL
                
                if correct:
                    results[category]['correct'] += 1
                    print("✓ CORRECT")
                else:
                    print("✗ INCORRECT")
                    if result.get('reasoning_trace'):
                        print(f"Reasoning trace available: {len(result['reasoning_trace'])} steps")
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    faulty_acc = results['faulty']['correct'] / results['faulty']['total'] if results['faulty']['total'] > 0 else 0.0
    normal_acc = results['normal']['correct'] / results['normal']['total'] if results['normal']['total'] > 0 else 0.0
    total_correct = results['faulty']['correct'] + results['normal']['correct']
    total_windows = results['faulty']['total'] + results['normal']['total']
    overall_acc = total_correct / total_windows if total_windows > 0 else 0.0
    
    print(f"Faulty windows: {results['faulty']['correct']}/{results['faulty']['total']} = {faulty_acc*100:.1f}%")
    print(f"Normal windows: {results['normal']['correct']}/{results['normal']['total']} = {normal_acc*100:.1f}%")
    print(f"Overall: {total_correct}/{total_windows} = {overall_acc*100:.1f}%")
    
    # Requirements check
    print(f"\n{'='*80}")
    print("REQUIREMENTS CHECK")
    print(f"{'='*80}")
    
    if normal_acc >= 0.6:
        print("✓ Not always reporting anomaly (60%+ normal accuracy)")
    else:
        print("✗ Still over-predicting faults on normal windows")
    
    if overall_acc >= 0.5:
        print("✓ Overall accuracy >50%")
    else:
        print("✗ Overall accuracy still below target")
    
    queries.close()
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KAG v2 fixes on sample windows")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .npz file")
    parser.add_argument("--gdn-model", type=str, required=True, help="Path to GDN model checkpoint")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://127.0.0.1:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--model-repo", type=str, default=None, help="LLM model repository")
    
    args = parser.parse_args()
    
    test_on_sample_windows(
        dataset_path=args.dataset,
        gdn_model_path=args.gdn_model,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_repo=args.model_repo,
    )
