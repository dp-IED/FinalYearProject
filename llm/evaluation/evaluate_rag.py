"""
Evaluate RAG (Retrieval-Augmented Generation) method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Loads ChromaDB collection with window descriptions and GDN embeddings
3. For each window, retrieves similar windows from ChromaDB
4. Augments LLM prompts with retrieved similar windows
5. Runs LLM inference with RAG context
6. Compares predictions to ground truth
7. Computes evaluation metrics
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
sys.path.insert(0, str(project_root))

from llm.rag.chromadb_setup import (
    create_chromadb_collection,
    query_similar_windows,
    get_collection_stats,
)
from llm.rag.rule_based_summarizer import generate_all_descriptions
from llm.evaluation.evaluate_llm_baseline import (
    load_llm_model,
    call_llm,
    parse_llm_response,
    format_window_for_llm,
    sensor_labels_to_window_label,
    filter_sensor_labels_to_root_only,
)
from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

# Add anomaly-detection to path for GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


def format_window_with_rag_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    similar_windows: List[Dict[str, any]],
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True,
    top_k: int = 3,
) -> str:
    """
    Format a window of sensor data for LLM prompt with RAG context from similar windows.

    Args:
        window_data: (window_size, num_sensors) array - unnormalized sensor values
        sensor_names: List of sensor names
        similar_windows: List of similar windows retrieved from ChromaDB
        statistical_features: Optional (num_sensors, 9) array with statistical features
        use_statistical_features: Whether to include statistical features in prompt
        top_k: Number of similar windows to include

    Returns:
        Formatted string for LLM prompt with RAG context
    """
    # Start with base prompt
    base_prompt = format_window_for_llm(
        window_data, sensor_names, statistical_features, use_statistical_features
    )

    # Add RAG context from similar windows
    if similar_windows and len(similar_windows) > 0:
        rag_section = []
        rag_section.append("\n" + "=" * 80)
        rag_section.append("RETRIEVED SIMILAR WINDOWS (for reference)")
        rag_section.append("=" * 80)
        rag_section.append(
            f"The following {min(len(similar_windows), top_k)} similar windows were retrieved from the database:"
        )
        rag_section.append("")

        for i, similar in enumerate(similar_windows[:top_k], 1):
            rag_section.append(f"Similar Window {i} (distance: {similar.get('distance', 'N/A'):.4f}):")
            rag_section.append("-" * 80)
            if similar.get("document"):
                rag_section.append(similar["document"])
            if similar.get("metadata"):
                metadata = similar["metadata"]
                if "window_label_true" in metadata:
                    rag_section.append(
                        f"Ground truth: {'Faulty' if metadata['window_label_true'] > 0 else 'Normal'}"
                    )
                if "fault_type" in metadata and metadata["fault_type"] != "None":
                    rag_section.append(f"Fault type: {metadata['fault_type']}")
            rag_section.append("")

        rag_section.append(
            "Use these similar windows as reference, but focus on analyzing the current window above."
        )
        rag_section.append("=" * 80)

        # Insert RAG section before the final instructions
        base_lines = base_prompt.split("\n")
        # Find the line with "Please analyze this sensor data"
        insert_idx = None
        for i, line in enumerate(base_lines):
            if "Please analyze this sensor data" in line:
                insert_idx = i
                break

        if insert_idx is not None:
            base_lines = base_lines[:insert_idx] + rag_section + base_lines[insert_idx:]
        else:
            # If we can't find the insertion point, append at the end
            base_lines.extend(rag_section)

        return "\n".join(base_lines)

    return base_prompt


def evaluate_rag(
    dataset_path: Path,
    chromadb_collection_name: str = "window_data",
    chromadb_persist_dir: str = "chromadb_data",
    output_path: Optional[Path] = None,
    model_repo: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    use_statistical_features: bool = True,
    limit: Optional[int] = None,
    top_k: int = 3,
    use_gdn_embeddings: bool = True,
    gdn_model_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, any]:
    """
    Evaluate RAG method on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        chromadb_collection_name: Name of ChromaDB collection
        chromadb_persist_dir: Directory where ChromaDB is persisted
        output_path: Optional path to save results JSON
        model_repo: Model repository identifier (default: granite-4.0-h-micro-GGUF)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_statistical_features: Whether to include statistical features in prompts
        limit: Optional limit on number of windows to process
        top_k: Number of similar windows to retrieve and include
        use_gdn_embeddings: Whether to use GDN embeddings for retrieval (vs text)
        gdn_model_path: Path to GDN model for extracting embeddings (if use_gdn_embeddings=True)
        batch_size: Batch size for GDN embedding extraction
        device: Device to run GDN model on

    Returns:
        Dictionary with evaluation results and metrics
    """
    print("=" * 80)
    print("Evaluating RAG Method")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"ChromaDB Collection: {chromadb_collection_name}")
    print(f"ChromaDB Persist Dir: {chromadb_persist_dir}")
    print(f"Top-K: {top_k}")
    print(f"Use GDN Embeddings: {use_gdn_embeddings}")
    if limit is not None:
        print(f"⚠️  LIMIT MODE: Processing only {limit} windows")
    print()

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]
    fault_types_true = data.get("fault_types", np.array([None] * len(normalized_windows)))

    # Load metadata for sensor names
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
        statistical_features = data.get("statistical_features", None)
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
        statistical_features = None

    num_windows = normalized_windows.shape[0]
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        fault_types_true = fault_types_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]

    print(f"  Loaded {num_windows} windows.")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Load ChromaDB collection
    print("Loading ChromaDB collection...")
    collection = create_chromadb_collection(
        collection_name=chromadb_collection_name,
        persist_directory=Path(chromadb_persist_dir),
    )
    stats = get_collection_stats(collection)
    collection_count = stats.get('num_windows', stats.get('count', 0))
    print(f"  ✓ ChromaDB collection '{collection.name}' loaded with {collection_count} items.")
    if collection_count == 0:
        print("  ⚠️  ChromaDB collection is empty. RAG will not be effective.")
    print()

    # GDN embeddings not supported - using text-only retrieval
    gdn_predictor = None
    use_gdn_embeddings = False

    # Load LLM model
    print("Loading LLM model...")
    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"
    model, tokenizer = load_llm_model(model_repo)
    print("  ✓ LLM model loaded")
    print()

    # Run evaluation
    print("Running RAG evaluation...")
    print()

    window_predictions = []
    sensor_predictions = []  # Filtered (root-only) predictions
    sensor_predictions_raw = []  # Raw (all sensors) predictions
    fault_type_predictions = []
    processing_times = []
    total_processing_time = 0.0

    for window_idx in tqdm(range(num_windows), desc="RAG Evaluation"):
        start_time = time.time()

        # Get window data
        window_data = unnormalized_windows[window_idx]
        stats = (
            statistical_features[window_idx]
            if statistical_features is not None and window_idx < len(statistical_features)
            else None
        )

        # Retrieve similar windows from ChromaDB using text descriptions
        descriptions_dict = generate_all_descriptions(
            dataset_path=dataset_path,
            output_dir=None,
            save_index=False,
        )
        current_description = descriptions_dict.get(window_idx, "")
        similar_windows = query_similar_windows(
            collection,
            query_text=current_description,
            top_k=top_k + 1,  # +1 because current window might be in results
        )
        # Filter out current window if present
        similar_windows = [
            w
            for w in similar_windows
            if w.get("metadata", {}).get("window_idx") != window_idx
        ][:top_k]

        # Format prompt with RAG context
        prompt = format_window_with_rag_for_llm(
            window_data,
            sensor_names,
            similar_windows,
            stats,
            use_statistical_features,
            top_k,
        )

        # Call LLM
        response = call_llm(
            prompt,
            model,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Parse response
        parsed = parse_llm_response(response, sensor_names)

        # Apply root-only filtering for precision improvement
        sensor_labels_filtered = filter_sensor_labels_to_root_only(parsed, sensor_names)
        sensor_labels_raw = parsed.get("sensor_labels", sensor_labels_filtered.copy())

        # Store predictions
        window_predictions.append(parsed["window_label"])
        sensor_predictions.append(sensor_labels_filtered)  # Use filtered (root-only) for metrics
        sensor_predictions_raw.append(sensor_labels_raw)  # Keep raw for analysis
        fault_type_predictions.append(parsed.get("fault_type"))

        # Track processing time
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        total_processing_time += processing_time

    # Convert to numpy arrays
    window_predictions = np.array(window_predictions)
    sensor_predictions = np.array(sensor_predictions)  # Filtered (root-only)
    sensor_predictions_raw = np.array(sensor_predictions_raw)  # Raw (all sensors)

    # window_predictions already contains sensor-indexed labels (0-8) from parse_llm_response
    window_labels_pred = window_predictions

    # Convert window_labels_true to sensor-indexed format (0-8)
    # The dataset stores window_labels as window indices, not sensor-indexed labels
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1  # 1-indexed (sensor 0 -> label 1)
        else:
            window_labels_true_converted[i] = 0  # No fault
    window_labels_true = window_labels_true_converted

    # Compute metrics
    print("\n" + "=" * 80)
    print("Computing evaluation metrics...")
    print("=" * 80)

    # Compute metrics using filtered (root-only) predictions for precision improvement
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_predictions,  # Use filtered (root-only) for main metrics
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    
    # Also compute raw metrics for comparison
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_predictions_raw,  # Use raw (all sensors) for comparison
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]

    # Add efficiency metrics
    avg_processing_time = total_processing_time / num_windows
    metrics["efficiency"] = {
        "total_processing_time_seconds": float(total_processing_time),
        "average_processing_time_seconds": float(avg_processing_time),
        "windows_per_second": float(num_windows / total_processing_time) if total_processing_time > 0 else 0.0,
    }

    # Format and print report
    report = format_metrics_report(metrics)
    print(report)

    # Prepare results dictionary
    results = {
        "method": "RAG",
        "dataset": str(dataset_path),
        "num_windows": int(num_windows),
        "chromadb_collection": chromadb_collection_name,
        "top_k": top_k,
        "use_gdn_embeddings": use_gdn_embeddings,
        "metrics": metrics,
    }

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG method on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to shared dataset (.npz file)",
    )
    parser.add_argument(
        "--chromadb-collection",
        type=str,
        default="window_data",
        help="Name of ChromaDB collection",
    )
    parser.add_argument(
        "--chromadb-persist-dir",
        type=str,
        default="chromadb_data",
        help="Directory where ChromaDB is persisted",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="granite-4.0-h-micro-GGUF",
        help="Model name in LM Studio (default: granite-4.0-h-micro-GGUF)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar windows to retrieve and include",
    )
    parser.add_argument(
        "--use-gdn-embeddings",
        action="store_true",
        help="Use GDN embeddings for retrieval (vs text descriptions)",
    )
    parser.add_argument(
        "--gdn-model",
        type=str,
        default=None,
        help="Path to GDN model for embedding extraction (required if --use-gdn-embeddings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for GDN embedding extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run GDN model on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )

    args = parser.parse_args()

    gdn_model_path = None
    if args.use_gdn_embeddings:
        if args.gdn_model is None:
            parser.error("--gdn-model is required when --use-gdn-embeddings is set")
        gdn_model_path = Path(args.gdn_model)
        if not gdn_model_path.exists():
            parser.error(f"GDN model not found: {gdn_model_path}")

    results = evaluate_rag(
        dataset_path=Path(args.dataset),
        chromadb_collection_name=args.chromadb_collection,
        chromadb_persist_dir=args.chromadb_persist_dir,
        output_path=Path(args.output),
        model_repo=args.model_repo,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
        top_k=args.top_k,
        use_gdn_embeddings=args.use_gdn_embeddings,
        gdn_model_path=gdn_model_path,
        batch_size=args.batch_size,
        device=args.device,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
