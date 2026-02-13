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

from evals.llm_helpers import (
    load_llm_model,
    format_window_for_llm,
    parse_llm_response,
    filter_sensor_labels_to_root_only,
    call_llm,
)
from evals.metrics import compute_all_metrics, format_metrics_report


def evaluate_llm_baseline(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    use_statistical_features: bool = True,
    model_repo: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    limit: Optional[int] = None,
) -> Dict:
    print("=" * 80)
    print("Evaluating LLM Baseline")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print()

    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"

    try:
        model, tokenizer = load_llm_model(model_repo)
        print()
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to LM Studio: {e}. "
            f"Please ensure LM Studio is running with the HTTP server enabled."
        )

    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
        statistical_features = np.array(metadata.get("statistical_features", []))
    else:
        sensor_names = [
            "ENGINE_RPM ()", "VEHICLE_SPEED ()", "THROTTLE ()", "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()", "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()", "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]
        statistical_features = None

    num_windows = unnormalized_windows.shape[0]
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]
        print(f"  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print()

    print("Running LLM predictions...")
    window_labels_pred = []
    sensor_labels_pred = []
    sensor_labels_pred_raw = []
    fault_types_pred = []
    reasoning_list = []
    processing_times = []

    with tqdm(total=num_windows, desc="LLM Inference", unit="window") as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            window_data = unnormalized_windows[window_idx]
            stats = (
                statistical_features[window_idx]
                if statistical_features is not None and len(statistical_features) > window_idx
                else None
            )
            prompt = format_window_for_llm(
                window_data, sensor_names, stats, use_statistical_features
            )
            try:
                response = call_llm(
                    prompt, model, tokenizer,
                    max_tokens=max_tokens, temperature=temperature,
                )
                prediction = parse_llm_response(response, sensor_names)
                prediction["reasoning"] = response[:200]
            except Exception as e:
                empty_labels = np.zeros(len(sensor_names), dtype=np.float32)
                prediction = {
                    "window_label": 0,
                    "sensor_labels": empty_labels,
                    "sensor_labels_root_only": empty_labels.copy(),
                    "fault_type": None,
                    "reasoning": f"Error: {str(e)}",
                }

            sensor_labels_filtered = filter_sensor_labels_to_root_only(prediction, sensor_names)
            sensor_labels_raw = prediction.get("sensor_labels", sensor_labels_filtered.copy())

            window_labels_pred.append(prediction["window_label"])
            sensor_labels_pred.append(sensor_labels_filtered)
            sensor_labels_pred_raw.append(sensor_labels_raw)
            fault_types_pred.append(prediction["fault_type"])
            reasoning_list.append(prediction.get("reasoning", ""))
            processing_times.append(time.time() - start_time)
            pbar.update(1)

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)

    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1
        else:
            window_labels_true_converted[i] = 0
    window_labels_true = window_labels_true_converted

    fault_types_true = data.get("fault_types", None)
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred_raw,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]
    metrics["efficiency"] = {
        "avg_processing_time_seconds": float(avg_processing_time),
        "total_processing_time_seconds": float(total_processing_time),
        "windows_per_second": float(num_windows / total_processing_time),
        "use_statistical_features": use_statistical_features,
    }

    print(format_metrics_report(metrics))

    results = {
        "method": "llm_baseline",
        "dataset": str(dataset_path),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),
            "fault_types": fault_types_pred,
            "reasoning": reasoning_list[:10] if len(reasoning_list) > 10 else reasoning_list,
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
    parser = argparse.ArgumentParser(description="Evaluate LLM baseline on shared evaluation dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to shared dataset .npz file")
    parser.add_argument("--output", type=str, default="results/llm_baseline.json", help="Output path for results JSON")
    parser.add_argument("--use-statistical-features", action="store_true", default=True, help="Include statistical features in LLM prompts")
    parser.add_argument("--model-repo", type=str, default="granite-4.0-h-micro-GGUF", help="Model name in LM Studio")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of windows to process (for testing)")

    args = parser.parse_args()
    evaluate_llm_baseline(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        use_statistical_features=args.use_statistical_features,
        model_repo=args.model_repo,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
