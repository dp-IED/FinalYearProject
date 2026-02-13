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

from llm.kag.neo4j_queries import Neo4jKAGQueries
from llm.kag.solver_v2 import KAGIterativeSolver
from kg.create_kg import KnowledgeGraph
from evals.llm_helpers import load_llm_model
from evals.metrics import compute_all_metrics, format_metrics_report
from evals.tool_tracker import ToolTracker


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
) -> Dict:
    print("=" * 80)
    print("Evaluating KAG (LLM-Planned Iterative Reasoning)")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {gdn_model_path}")
    print()

    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"

    try:
        model, tokenizer = load_llm_model(model_repo)
        print()
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM model: {e}. Please ensure LM Studio is running.")

    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = [
            "ENGINE_RPM ()", "VEHICLE_SPEED ()", "THROTTLE ()", "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()", "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()", "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]

    num_windows = data["normalized_windows"].shape[0]
    fault_types_true = data.get("fault_types", None)

    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if fault_types_true is not None:
            fault_types_true = fault_types_true[:num_windows]
        print(f"  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print()

    print("Loading GDN checkpoint for metadata...")
    import torch
    checkpoint = torch.load(gdn_model_path, map_location="cpu")
    sensor_embeddings = checkpoint.get("sensor_embeddings")
    if sensor_embeddings is None:
        num_sensors = len(sensor_names)
        embed_dim = checkpoint.get("embed_dim", 32)
        sensor_embeddings = np.random.randn(num_sensors, embed_dim).astype(np.float32)
    else:
        sensor_embeddings = sensor_embeddings.cpu().numpy() if hasattr(sensor_embeddings, "cpu") else np.array(sensor_embeddings)

    norms = np.linalg.norm(sensor_embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    sensor_embeddings_norm = sensor_embeddings / norms
    similarity_matrix = np.dot(sensor_embeddings_norm, sensor_embeddings_norm.T)
    adjacency_matrix = (similarity_matrix + 1.0) / 2.0
    adjacency_matrix = np.clip(adjacency_matrix, 0.1, 1.0)
    np.fill_diagonal(adjacency_matrix, 0.0)

    kg = KnowledgeGraph(
        sensor_names=sensor_names,
        sensor_embeddings=sensor_embeddings,
        adjacency_matrix=adjacency_matrix,
    )

    try:
        queries = Neo4jKAGQueries(neo4j_uri, neo4j_user, neo4j_password)
        with queries.driver.session() as session:
            session.run("RETURN 1").single()
        print("  Connected to Neo4j")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Neo4j: {e}. Ensure Neo4j is running and KG is loaded.")

    tool_tracker = ToolTracker()
    solver = KAGIterativeSolver(
        kg_builder=kg,
        neo4j_queries=queries,
        sensor_names=sensor_names,
        model=model,
        tokenizer=tokenizer,
        max_iterations=max_iterations,
        tool_tracker=tool_tracker,
    )

    print("Running KAG Solver on windows...")
    window_labels_pred = []
    sensor_labels_pred = []
    sensor_labels_pred_raw = []
    fault_types_pred = []
    reasoning_traces = []
    processing_times = []

    for window_idx in tqdm(range(num_windows), desc="KAG Solver", unit="window"):
        start_time = time.time()
        try:
            result = solver.solve(window_idx)
            sl = result.get("sensor_labels", np.zeros(len(sensor_names), dtype=np.float32))
            sl_raw = result.get("sensor_labels_raw", sl)
            if not isinstance(sl, np.ndarray):
                sl = np.array(sl, dtype=np.float32)
            if not isinstance(sl_raw, np.ndarray):
                sl_raw = np.array(sl_raw, dtype=np.float32)

            window_labels_pred.append(result["window_label"])
            sensor_labels_pred.append(sl)
            sensor_labels_pred_raw.append(sl_raw)
            fault_types_pred.append(result.get("fault_type"))
            reasoning_traces.append(result.get("reasoning_trace", []))

            faulty = np.where(sensor_labels_true[window_idx] > 0)[0]
            true_wl = int(faulty[0] + 1) if len(faulty) > 0 else 0
            pred_wl = int(result["window_label"])
            true_sensors = [sensor_names[i] for i in range(len(sensor_names)) if sensor_labels_true[window_idx][i] > 0]
            pred_sensors = [sensor_names[i] for i in range(len(sensor_names)) if sl[i] > 0]
            tool_tracker.mark_window_result(
                window_idx=window_idx,
                is_correct=(true_wl == pred_wl),
                predicted_sensors=pred_sensors,
                true_sensors=true_sensors,
                predicted_window_label=pred_wl,
                true_window_label=true_wl,
            )
        except Exception as e:
            window_labels_pred.append(0)
            empty = np.zeros(len(sensor_names), dtype=np.float32)
            sensor_labels_pred.append(empty)
            sensor_labels_pred_raw.append(empty.copy())
            fault_types_pred.append(None)
            reasoning_traces.append([{"step": 0, "operation": "error", "result": str(e)}])
            faulty = np.where(sensor_labels_true[window_idx] > 0)[0]
            true_wl = int(faulty[0] + 1) if len(faulty) > 0 else 0
            tool_tracker.mark_window_result(
                window_idx=window_idx, is_correct=False,
                predicted_sensors=[], true_sensors=[sensor_names[i] for i in range(len(sensor_names)) if sensor_labels_true[window_idx][i] > 0],
                predicted_window_label=0, true_window_label=true_wl,
            )
        processing_times.append(time.time() - start_time)

    queries.close()

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)

    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        window_labels_true_converted[i] = faulty_indices[0] + 1 if len(faulty_indices) > 0 else 0
    window_labels_true = window_labels_true_converted

    total_processing_time = np.sum(processing_times)
    avg_processing_time = np.mean(processing_times)
    windows_per_second = num_windows / total_processing_time if total_processing_time > 0 else 0

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true,
    )
    metrics["efficiency"] = {
        "total_processing_time_seconds": float(total_processing_time),
        "average_processing_time_seconds": float(avg_processing_time),
        "windows_per_second": float(windows_per_second),
    }

    tool_tracker.print_summary()
    if output_path:
        tool_tracker.save_report(str(Path(output_path).parent / f"{Path(output_path).stem}_tool_usage.json"))

    print(format_metrics_report(metrics))

    results = {
        "method": "kag_v2",
        "dataset": str(dataset_path),
        "gdn_model": str(gdn_model_path),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
            "fault_types": fault_types_pred,
            "reasoning_traces": reasoning_traces[:10] if len(reasoning_traces) > 10 else reasoning_traces,
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate KAG on shared evaluation dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to shared dataset .npz file")
    parser.add_argument("--gdn-model", type=str, required=True, help="Path to trained GDN model checkpoint")
    parser.add_argument("--output", type=str, default="results/kag_v2.json", help="Output path for results JSON")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://127.0.0.1:7687", help="Neo4j connection URI")
    parser.add_argument("--neo4j-user", type=str, default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", type=str, default="password", help="Neo4j password")
    parser.add_argument("--model-repo", type=str, default=None, help="LLM model name in LM Studio")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max refinement iterations")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of windows")
    args = parser.parse_args()

    evaluate_kag_v2(
        dataset_path=Path(args.dataset),
        gdn_model_path=Path(args.gdn_model),
        output_path=Path(args.output),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_repo=args.model_repo,
        device=args.device,
        max_iterations=args.max_iterations,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
