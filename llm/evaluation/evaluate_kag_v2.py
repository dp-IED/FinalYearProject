"""
Evaluate KAG (LLM-Planned Iterative Reasoning)

Evaluates the LLM-planned KAG solver on shared evaluation dataset.
This solver uses LLM to generate logical form steps that are executed
over the knowledge graph, enabling flexible, adaptive reasoning.

Usage:
    python llm/evaluation/evaluate_kag_v2.py \
        --dataset llm/evaluation/shared_dataset/test.npz \
        --gdn-model anomaly-detection/models/gdn_model.py \
        --neo4j-uri bolt://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password password \
        --output results/kag_v2_test.json
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
sys.path.insert(0, str(project_root / "anomaly-detection"))

from llm.kag.neo4j_queries import Neo4jKAGQueries
from llm.kag.solver_v2 import KAGIterativeSolver
from llm.helpers.KG import KnowledgeGraphBuilder
from llm.evaluation.evaluate_llm_baseline import load_llm_model
from llm.evaluation.metrics import compute_all_metrics, format_metrics_report
from llm.evaluation.tool_tracker import ToolTracker


def evaluate_kag_v2(
    dataset_path: Path,
    gdn_model_path: Path,
    output_path: Optional[Path] = None,
    neo4j_uri: str = "bolt://127.0.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    model_repo: Optional[str] = None,
    batch_size: int = 32,
    device: str = "cpu",
    max_iterations: int = 1,
    limit: Optional[int] = None,
) -> Dict[str, any]:
    """
    Evaluate KAG (LLM-planned iterative reasoning).

    Args:
        dataset_path: Path to shared dataset (.npz file)
        gdn_model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        model_repo: LLM model repository identifier (default: granite-4.0-h-micro-4bit)
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        max_iterations: Maximum number of refinement iterations (default: 1)
        limit: Optional limit on number of windows to process (for testing)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating KAG (LLM-Planned Iterative Reasoning)")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {gdn_model_path}")
    print(f"Neo4j URI: {neo4j_uri}")
    print(f"Max Iterations: {max_iterations}")
    print()

    # Load LLM model
    if model_repo is None:
        model_repo = "mlx-community/granite-4.0-h-micro-4bit"

    print("Loading LLM model...")
    print(f"Loading LLM model: {model_repo}")
    try:
        model, tokenizer = load_llm_model(model_repo)
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LLM model: {e}. Please ensure mlx-lm is installed."
        )

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)

    normalized_windows = data["normalized_windows"]
    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        # Fallback to default sensor names
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

    num_windows = normalized_windows.shape[0]

    # Get fault_types before limiting
    fault_types_true = data.get("fault_types", None)

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if fault_types_true is not None:
            fault_types_true = fault_types_true[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Load checkpoint to get sensor embeddings and adjacency matrix
    # Note: We don't rebuild the KG here - Neo4j already has it from create_shared_dataset.py
    print("Loading GDN checkpoint for metadata...")
    start_time = time.time()
    import torch

    checkpoint = torch.load(gdn_model_path, map_location="cpu")

    # Get sensor embeddings from checkpoint
    sensor_embeddings = checkpoint.get("sensor_embeddings")

    # If not in checkpoint, create dummy arrays (kg_builder is not used for queries anyway)
    if sensor_embeddings is None:
        num_sensors = len(sensor_names)
        embed_dim = checkpoint.get("embed_dim", 32)
        sensor_embeddings = np.random.randn(num_sensors, embed_dim).astype(np.float32)
        print(f"  ⚠️  Warning: sensor_embeddings not in checkpoint, using dummy array")
    else:
        # Convert torch tensor to numpy if needed
        if hasattr(sensor_embeddings, "numpy"):
            sensor_embeddings = sensor_embeddings.numpy()
        elif isinstance(sensor_embeddings, torch.Tensor):
            sensor_embeddings = sensor_embeddings.cpu().numpy()

    # Compute adjacency matrix from sensor embeddings (cosine similarity)
    # This matches how GDNPredictor.compute_adjacency_matrix() works
    if sensor_embeddings is not None:
        # Normalize embeddings (L2 normalization for cosine similarity)
        norms = np.linalg.norm(sensor_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)  # Avoid division by zero
        sensor_embeddings_norm = sensor_embeddings / norms

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(sensor_embeddings_norm, sensor_embeddings_norm.T)

        # Scale from [-1, 1] to [0.1, 1.0] for compatibility
        adjacency_matrix = (similarity_matrix + 1.0) / 2.0
        adjacency_matrix = np.clip(adjacency_matrix, 0.1, 1.0)

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(adjacency_matrix, 0.0)
    else:
        # Fallback to identity matrix if embeddings are missing
        num_sensors = len(sensor_names)
        adjacency_matrix = np.eye(num_sensors, dtype=np.float32)
        print(
            f"  ⚠️  Warning: Could not compute adjacency_matrix, using identity matrix"
        )

    checkpoint_load_time = time.time() - start_time
    print(f"  ✓ Checkpoint loaded in {checkpoint_load_time:.2f} seconds")
    print()

    # Create minimal kg_builder (not used for queries, but required by solver interface)
    # All queries go through Neo4j, which is already loaded by create_shared_dataset.py
    print("Creating minimal KG builder (queries use Neo4j)...")
    kg_build_start = time.time()

    kg_builder = KnowledgeGraphBuilder(
        sensor_names=sensor_names,
        sensor_embeddings=sensor_embeddings,
        adjacency_matrix=adjacency_matrix,
    )
    kg_time = time.time() - kg_build_start
    print(f"  ✓ KG builder created (using Neo4j for all queries) in {kg_time:.2f} seconds")
    print()
    
    # Set gdn_time to 0 since we're not processing through GDN (using Neo4j instead)
    gdn_time = 0.0

    # Initialize Neo4j queries
    print("Connecting to Neo4j...")
    try:
        queries = Neo4jKAGQueries(neo4j_uri, neo4j_user, neo4j_password)
        # Test connection
        with queries.driver.session() as session:
            session.run("RETURN 1").single()
        print("  ✓ Connected to Neo4j")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Neo4j: {e}. Please ensure Neo4j is running and the KG is loaded."
        )

    # Initialize tool tracker
    tool_tracker = ToolTracker()
    print("  ✓ Tool tracker initialized")

    # Optional: Run diagnostics on sample windows if requested
    # This can be enabled by setting environment variable or adding CLI flag
    if False:  # Set to True to enable diagnostics
        from llm.kag.diagnostics import KAGDiagnostics
        print("\nRunning diagnostics on sample windows...")
        diagnostics = KAGDiagnostics(
            neo4j_queries=queries,
            sensor_names=sensor_names,
            dataset_path=str(dataset_path)
        )
        # Test on first 5 windows
        sample_windows = list(range(min(5, num_windows)))
        summary, reports = diagnostics.batch_diagnose(
            sample_windows,
            output_path=str(output_path.parent / f"{output_path.stem}_diagnostics.json") if output_path else None
        )
        print(f"\nDiagnostic Summary:")
        print(json.dumps(summary, indent=2))

    # Initialize solver
    print("Initializing KAG Solver...")
    solver = KAGIterativeSolver(
        kg_builder=kg_builder,
        neo4j_queries=queries,
        sensor_names=sensor_names,
        model=model,
        tokenizer=tokenizer,
        max_iterations=max_iterations,
        tool_tracker=tool_tracker,
    )
    print(f"  ✓ Solver initialized (max_iterations={max_iterations})")
    print()

    # Process windows
    print("Running KAG Solver on windows...")
    window_labels_pred = []
    sensor_labels_pred = []  # Filtered (root-only) predictions
    sensor_labels_pred_raw = []  # Raw (all sensors) predictions
    fault_types_pred = []
    reasoning_traces = []
    processing_times = []
    faults_detected = 0
    errors_count = 0

    with tqdm(
        total=num_windows,
        desc="KAG Solver",
        unit="window",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5,
        disable=False,
    ) as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()

            try:
                result = solver.solve(window_idx)

                window_labels_pred.append(result["window_label"])
                # Use root-only sensor labels for precision improvement
                sensor_labels_filtered = result.get("sensor_labels", np.zeros(len(sensor_names), dtype=int))  # Root-only
                sensor_labels_raw_val = result.get("sensor_labels_raw", sensor_labels_filtered.copy())  # All sensors
                sensor_labels_pred.append(sensor_labels_filtered)  # Use filtered (root-only) for metrics
                sensor_labels_pred_raw.append(sensor_labels_raw_val)  # Keep raw for analysis
                fault_types_pred.append(result["fault_type"])
                reasoning_traces.append(result["reasoning_trace"])

                # Track faults detected
                if result["window_label"] > 0:
                    faults_detected += 1

                # Determine correctness and mark in tracker
                true_window_label = int(window_labels_true[window_idx])
                pred_window_label = int(result["window_label"])
                is_correct_window = true_window_label == pred_window_label

                # Get sensor-level correctness (use filtered for tracking)
                true_sensor_labels = sensor_labels_true[window_idx]
                pred_sensor_labels = sensor_labels_filtered

                # Convert to sensor names
                true_sensors = [
                    sensor_names[i]
                    for i in range(len(sensor_names))
                    if true_sensor_labels[i] > 0
                ]
                pred_sensors = [
                    sensor_names[i]
                    for i in range(len(sensor_names))
                    if pred_sensor_labels[i] > 0
                ]

                # Mark window result in tracker
                tool_tracker.mark_window_result(
                    window_idx=window_idx,
                    is_correct=is_correct_window,
                    predicted_sensors=pred_sensors,
                    true_sensors=true_sensors,
                    predicted_window_label=pred_window_label,
                    true_window_label=true_window_label,
                )

            except Exception as e:
                # Fallback to no-fault prediction on error
                errors_count += 1
                if errors_count <= 3:  # Only print first 3 errors to avoid spam
                    print(f"\n  ⚠️  Warning: Error processing window {window_idx}: {e}")
                    if errors_count == 1:
                        import traceback

                        traceback.print_exc()
                window_labels_pred.append(0)
                empty_labels = np.zeros(len(sensor_names), dtype=int)
                sensor_labels_pred.append(empty_labels)
                sensor_labels_pred_raw.append(empty_labels.copy())
                fault_types_pred.append(None)
                reasoning_traces.append(
                    [{"step": 0, "operation": "error", "result": str(e)}]
                )

                # Mark error case in tracker (prediction is incorrect)
                true_window_label = int(window_labels_true[window_idx])
                true_sensor_labels = sensor_labels_true[window_idx]
                true_sensors = [
                    sensor_names[i]
                    for i in range(len(sensor_names))
                    if true_sensor_labels[i] > 0
                ]
                tool_tracker.mark_window_result(
                    window_idx=window_idx,
                    is_correct=False,  # Error case is always incorrect
                    predicted_sensors=[],
                    true_sensors=true_sensors,
                    predicted_window_label=0,
                    true_window_label=true_window_label,
                )

            processing_times.append(time.time() - start_time)
            pbar.update(1)

            # Update progress bar with detailed metrics
            if (window_idx + 1) % 5 == 0 or window_idx == num_windows - 1:
                avg_time = (
                    np.mean(processing_times[-10:])
                    if len(processing_times) >= 10
                    else np.mean(processing_times)
                )
                elapsed_time = sum(processing_times)
                remaining_windows = num_windows - (window_idx + 1)
                eta_seconds = (
                    avg_time * remaining_windows if remaining_windows > 0 else 0
                )

                postfix = {
                    "avg": f"{avg_time:.2f}s",
                    "faults": faults_detected,
                    "errors": errors_count if errors_count > 0 else None,
                }
                # Remove None values
                postfix = {k: v for k, v in postfix.items() if v is not None}
                pbar.set_postfix(postfix)

                # Show ETA every 10 windows
                if (window_idx + 1) % 10 == 0:
                    eta_minutes = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    if eta_seconds > 0:
                        pbar.set_description(
                            f"KAG Solver v2 (ETA: {eta_minutes}m{eta_secs}s)"
                        )
                    else:
                        pbar.set_description("KAG Solver")

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)  # Filtered (root-only)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)  # Raw (all sensors)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    windows_per_second = (
        num_windows / total_processing_time if total_processing_time > 0 else 0
    )

    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total processing time: {total_processing_time:.2f} seconds")
    print(f"  Processing rate: {windows_per_second:.2f} windows/second")
    print(
        f"  Faults detected: {faults_detected}/{num_windows} ({100 * faults_detected / num_windows:.1f}%)"
    )
    if errors_count > 0:
        print(f"  Errors encountered: {errors_count}")
    print()

    # Close Neo4j connection
    queries.close()

    # Convert window_labels_true to sensor-indexed format
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1  # 1-indexed
        else:
            window_labels_true_converted[i] = 0
    window_labels_true = window_labels_true_converted

    # Compute metrics (using filtered root-only predictions for precision improvement)
    print("Computing evaluation metrics...")
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,  # Use filtered (root-only) for main metrics
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    
    # Also compute raw metrics for comparison
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred_raw,  # Use raw (all sensors) for comparison
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]

    # Add efficiency metrics
    metrics["efficiency"] = {
        "gdn_processing_time_seconds": float(gdn_time),
        "kg_construction_time_seconds": float(kg_time),
        "total_processing_time_seconds": float(total_processing_time),
        "average_processing_time_seconds": float(avg_processing_time),
        "windows_per_second": float(windows_per_second),
        "faults_detected": int(faults_detected),
        "errors_count": int(errors_count),
    }

    # Generate tool usage report
    print("\nGenerating tool usage analysis report...")
    tool_tracker.print_summary()

    # Save tool usage report
    if output_path:
        tool_report_path = output_path.parent / f"{output_path.stem}_tool_usage.json"
        tool_tracker.save_report(str(tool_report_path))
        print(f"  ✓ Tool usage report saved to: {tool_report_path}")

        # Add tool report path to metrics
        metrics["tool_usage_report_path"] = str(tool_report_path)

    # Print report
    report = format_metrics_report(metrics)
    print(report)

    # Save results
    results = {
        "method": "kag_v2",
        "dataset": str(dataset_path),
        "gdn_model": str(gdn_model_path),
        "neo4j_uri": neo4j_uri,
        "model_repo": model_repo,
        "max_iterations": int(max_iterations),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),  # Filtered (root-only)
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),  # Raw (all sensors)
            "fault_types": fault_types_pred,
            "reasoning_traces": reasoning_traces[:10]
            if len(reasoning_traces) > 10
            else reasoning_traces,  # Sample traces
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KAG (LLM-planned iterative reasoning) on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to shared dataset .npz file"
    )
    parser.add_argument(
        "--gdn-model",
        type=str,
        required=True,
        help="Path to trained GDN model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/kag_v2.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://127.0.0.1:7687",
        help="Neo4j connection URI (default: bolt://127.0.0.1:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="LLM model repository identifier (default: mlx-community/granite-4.0-h-micro-4bit)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GDN inference (default: 32)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Maximum number of refinement iterations (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )

    args = parser.parse_args()

    evaluate_kag_v2(
        dataset_path=Path(args.dataset),
        gdn_model_path=Path(args.gdn_model),
        output_path=Path(args.output),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_repo=args.model_repo,
        batch_size=args.batch_size,
        device=args.device,
        max_iterations=args.max_iterations,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
