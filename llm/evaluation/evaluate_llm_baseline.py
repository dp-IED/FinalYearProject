"""
Evaluate LLM-only baseline method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Formats unnormalized windows for LLM prompts
3. Runs LLM inference
4. Compares predictions to ground truth
5. Computes evaluation metrics
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time
import re
from tqdm import tqdm

from llm.inference import LMInference, load_llm_model as load_lm_studio_model
from llm.evaluation.metrics import compute_all_metrics, format_metrics_report


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


def format_window_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True,
) -> str:
    """
    Format a window of sensor data for LLM prompt.

    Based on baseline/README.md:
    - Approach 1: Pass entire OBD logs (windows of 300 timesteps)
    - Approach 2: Enrich with statistical features (min, max, range, mean, std, median, mode, skewness, kurtosis)

    Args:
        window_data: (window_size, num_sensors) array - unnormalized sensor values
        sensor_names: List of sensor names
        statistical_features: Optional (num_sensors, 9) array with statistical features
        use_statistical_features: Whether to include statistical features in prompt

    Returns:
        Formatted string for LLM prompt
    """
    lines = []
    lines.append(
        "You are an automotive diagnostic expert analyzing OBD-II sensor data."
    )
    lines.append("")
    lines.append("Task: Identify which sensors are faulty and describe the fault type.")
    lines.append("")
    lines.append("Sensor Data Window (300 timesteps):")
    lines.append("=" * 80)

    # Add statistical features if available and requested
    if statistical_features is not None and use_statistical_features:
        lines.append("\nStatistical Features for each sensor:")
        lines.append("-" * 80)
        feature_names = [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "median",
            "mode",
            "skewness",
            "kurtosis",
        ]
        for i, sensor_name in enumerate(sensor_names):
            if i < len(statistical_features):
                lines.append(f"\n{sensor_name}:")
                for j, feat_name in enumerate(feature_names):
                    if j < len(statistical_features[i]):
                        lines.append(f"  {feat_name}: {statistical_features[i][j]:.4f}")
        lines.append("")

    # Add time series data (sample strategically to reduce context length)
    # Sample more aggressively: every 30-50 timesteps to get ~10-15 key points
    lines.append("\nTime Series Data (key timesteps sampled):")
    lines.append("-" * 80)

    # Sample strategically: beginning, middle sections, and end
    window_size = window_data.shape[0]
    if window_size > 50:
        # Sample ~15 key points: start, middle sections, end
        sample_indices = (
            [0]  # Beginning
            + list(
                range(window_size // 4, window_size // 2, window_size // 8)
            )  # First half
            + list(
                range(window_size // 2, 3 * window_size // 4, window_size // 8)
            )  # Second half
            + [window_size - 1]  # End
        )
        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))
    else:
        # For small windows, use all points
        sample_indices = list(range(window_size))

    # Header
    header = "Time\t" + "\t".join([name.replace(" ()", "") for name in sensor_names])
    lines.append(header)

    # Data rows
    for t in sample_indices:
        values = window_data[t]
        row = f"{t}\t" + "\t".join([f"{val:.2f}" for val in values])
        lines.append(row)

    lines.append(f"\n(Showing {len(sample_indices)} of {window_size} timesteps)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Please analyze this sensor data and provide your diagnosis.")
    lines.append("")
    lines.append("You MUST respond with a valid JSON object in this EXACT format (no other text):")
    lines.append("")
    lines.append("{")
    lines.append('    "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],')
    lines.append('    "fault_type": "VSS_DROPOUT" | "COOLANT_DROPOUT" | "MAF_SCALE" | "TPS_STUCK" | "gradual_drift" or null,')
    lines.append('    "reasoning": "2-4 sentences explaining your diagnosis",')
    lines.append('    "confidence": 0.85')
    lines.append("}")
    lines.append("")
    lines.append("Available sensor names:")
    for name in sensor_names:
        lines.append(f"  - {name.replace(' ()', '')}")
    lines.append("")
    lines.append("Fault types: VSS_DROPOUT, COOLANT_DROPOUT, MAF_SCALE, TPS_STUCK, gradual_drift")
    lines.append("")
    lines.append("CRITICAL: Only output valid JSON. No markdown, no code blocks, no extra text.")
    lines.append("If no faults detected, use: {\"faulty_sensors\": [], \"fault_type\": null, \"reasoning\": \"...\", \"confidence\": 0.9}")

    return "\n".join(lines)


def parse_llm_response(response: str, sensor_names: List[str]) -> Dict[str, any]:
    """
    Parse LLM response to extract fault predictions.
    
    Now supports JSON structured output format:
    {
        "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],
        "fault_type": "VSS_DROPOUT" | "COOLANT_DROPOUT" | "MAF_SCALE" | "TPS_STUCK" | "gradual_drift" or null,
        "reasoning": "2-4 sentences explaining why",
        "confidence": 0.85
    }
    
    Falls back to old text format parsing for backward compatibility.

    Args:
        response: LLM response text (JSON or text format)
        sensor_names: List of sensor names to match against

    Returns:
        Dictionary with predictions:
        - 'window_label': int - is window faulty? (0 or 1)
        - 'sensor_labels': (num_sensors,) binary array - which sensors are faulty
        - 'fault_type': str - predicted fault type
        - 'reasoning': str - LLM reasoning
    """
    sensor_labels = np.zeros(len(sensor_names), dtype=np.float32)
    fault_type = None  # No fault type if no fault detected
    reasoning = ""
    
    # Try to parse as JSON first (structured output)
    try:
        # Clean response - remove markdown code blocks if present
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()
        
        # Extract JSON object using regex (handles cases where there's extra text)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            
            # Extract values from JSON
            faulty_sensors = result.get("faulty_sensors", [])
            fault_type = result.get("fault_type", None)  # None if no fault
            # Convert "unknown" to None for backward compatibility
            if fault_type == "unknown" or fault_type == "":
                fault_type = None
            reasoning = result.get("reasoning", "")
            
            # Validate and map sensor names to indices
            for sensor in faulty_sensors:
                # Try exact match first
                if sensor in sensor_names:
                    idx = sensor_names.index(sensor)
                    sensor_labels[idx] = 1.0
                else:
                    # Try matching without parentheses
                    sensor_clean = sensor.replace(" ()", "").strip()
                    for i, name in enumerate(sensor_names):
                        name_clean = name.replace(" ()", "").strip()
                        if sensor_clean == name_clean:
                            sensor_labels[i] = 1.0
                            break
            
            # Convert sensor labels to sensor-indexed window label (0-8)
            window_label = sensor_labels_to_window_label(sensor_labels)
            
            return {
                "window_label": window_label,
                "sensor_labels": sensor_labels,
                "fault_type": fault_type,
                "reasoning": reasoning if reasoning else "No reasoning provided",
            }
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # JSON parsing failed, fall through to old text parsing
        pass

    # Extract faulty sensors - be more flexible with format
    faulty_sensors_match = re.search(
        r"Faulty Sensors?:\s*(.+?)(?:\n|Fault Type|Reasoning|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    if faulty_sensors_match:
        faulty_sensors_str = faulty_sensors_match.group(1).strip()

        if faulty_sensors_str.lower() not in ["none", "no", "n/a", ""]:
            # Try to extract sensor names from the response
            # Match sensor names (with or without parentheses, with spaces, underscores, etc.)
            for i, sensor_name in enumerate(sensor_names):
                # Try multiple variants including common LLM formatting variations
                sensor_variants = [
                    sensor_name,  # Full name: "VEHICLE_SPEED ()"
                    sensor_name.replace(" ()", ""),  # "VEHICLE_SPEED"
                    sensor_name.replace(" ()", "").replace("_", " "),  # "VEHICLE SPEED"
                    sensor_name.replace(" ()", "").replace(
                        "_", "_ "
                    ),  # "VEHICLE_ SPEED"
                    sensor_name.split()[0]
                    if " " in sensor_name
                    else sensor_name,  # First word
                    sensor_name.replace("ENGINE_", "").replace(" ()", ""),  # Shortened
                ]

                # Also try matching key parts of sensor names
                key_parts = {
                    "VEHICLE_SPEED": ["vehicle", "speed", "vss"],
                    "COOLANT_TEMPERATURE": ["coolant", "temperature"],
                    "THROTTLE": ["throttle", "tps"],
                    "ENGINE_RPM": ["rpm", "engine rpm"],
                    "ENGINE_LOAD": ["engine load", "load"],
                    "INTAKE_MANIFOLD_PRESSURE": [
                        "intake",
                        "manifold",
                        "pressure",
                        "map",
                        "maf",
                    ],
                    "SHORT_TERM_FUEL_TRIM": ["short term", "fuel trim", "stft"],
                    "LONG_TERM_FUEL_TRIM": ["long term", "fuel trim", "ltft"],
                }

                matched = False
                for variant in sensor_variants:
                    # Normalize for comparison (remove spaces, underscores, case-insensitive)
                    variant_clean = variant.replace("_", "").replace(" ", "").lower()
                    response_clean = (
                        faulty_sensors_str.replace("_", "").replace(" ", "").lower()
                    )

                    if variant_clean in response_clean:
                        sensor_labels[i] = 1.0
                        matched = True
                        break

                # If not matched by name, try key parts
                if not matched:
                    sensor_base = sensor_name.replace(" ()", "").replace("_", " ")
                    for key, parts in key_parts.items():
                        if key in sensor_name:
                            if any(
                                part in faulty_sensors_str.lower() for part in parts
                            ):
                                sensor_labels[i] = 1.0
                                break

    # Extract fault type
    fault_type_match = re.search(
        r"Fault Type:\s*(.+?)(?:\n|Reasoning|$)", response, re.IGNORECASE | re.DOTALL
    )
    if fault_type_match:
        fault_type_str = fault_type_match.group(1).strip()
        # Clean up common prefixes/suffixes
        fault_type_str = fault_type_str.split(",")[0].split(".")[0].strip()
        # Only set fault_type if it's not "unknown", "none", or empty
        if fault_type_str.lower() not in ["unknown", "none", "n/a", ""]:
            fault_type = fault_type_str
        else:
            fault_type = None

    # Extract reasoning
    reasoning_match = re.search(
        r"Reasoning:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Fallback 1: Extract sensors from reasoning text if structured format failed
    if np.sum(sensor_labels) == 0 and reasoning:
        # Try to find sensor names in reasoning text
        reasoning_lower = reasoning.lower()
        for i, sensor_name in enumerate(sensor_names):
            sensor_base = sensor_name.replace(" ()", "").replace("_", " ").lower()
            sensor_parts = sensor_base.split()
            # Check if any part of sensor name appears in reasoning
            if any(part in reasoning_lower for part in sensor_parts if len(part) > 3):
                sensor_labels[i] = 1.0
    
    # Fallback 2: Validate parsed sensors against available sensor names
    parsed_sensors = []
    for i, label in enumerate(sensor_labels):
        if label > 0:
            parsed_sensors.append(sensor_names[i])
    
    # If no valid sensors found, check if response explicitly says "no fault"
    if len(parsed_sensors) == 0:
        response_lower = response.lower()
        no_fault_indicators = [
            "no fault",
            "no faults",
            "no faulty sensors",
            "all sensors appear normal",
            "no anomalies",
            "no violations",
            "faulty sensors: []",
            "faulty sensors: none",
        ]
        if any(indicator in response_lower for indicator in no_fault_indicators):
            # Explicitly no fault - this is valid
            pass
        elif "faulty sensors:" in response_lower and "[]" in response_lower:
            # Empty list explicitly provided
            pass
    
    # Convert sensor labels to sensor-indexed window label (0-8)
    window_label = sensor_labels_to_window_label(sensor_labels)

    return {
        "window_label": window_label,
        "sensor_labels": sensor_labels,
        "fault_type": fault_type,
        "reasoning": reasoning if reasoning else "No reasoning provided",
    }


def call_llm(
    prompt: str,
    model,
    tokenizer=None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    repetition_context_size: int = 20,
) -> str:
    """
    Call LLM with prompt and return response.

    Args:
        prompt: Input prompt text
        model: LMInference instance (or MLX model for backward compatibility)
        tokenizer: Tokenizer (ignored if model is LMInference, kept for compatibility)
        max_tokens: Maximum tokens to generate (None = no limit)
        temperature: Sampling temperature
        repetition_penalty: Penalty for repetition (1.0 = no penalty, >1.0 = penalize repetition)
        repetition_context_size: Context window size for repetition penalty (ignored for LM Studio)

    Returns:
        Generated response text
    """
    if model is None:
        raise RuntimeError("Model not loaded. Call load_llm_model() first.")

    # Check if model is LMInference instance
    if isinstance(model, LMInference):
        # Use LM Studio HTTP API
        messages = [{"role": "user", "content": prompt}]
        return model.chat_completions(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    else:
        # Fallback for MLX (if still needed)
        raise RuntimeError(
            "MLX is no longer supported. Please use LM Studio with LMInference."
        )


def load_llm_model(
    model_repo: str = "granite-4.0-h-micro-GGUF",
    base_url: str = "http://localhost:1234/v1",
):
    """
    Load LLM model via LM Studio HTTP server.

    Args:
        model_repo: Model name in LM Studio (default: granite-4.0-h-micro-GGUF)
        base_url: Base URL for LM Studio HTTP server

    Returns:
        Tuple of (model, tokenizer) where both are the same LMInference instance
        for backward compatibility with code expecting (model, tokenizer) tuple
    """
    # Convert old MLX model repo format to LM Studio model name if needed
    if model_repo.startswith("mlx-community/"):
        # Extract model name from MLX format
        model_name = model_repo.replace("mlx-community/", "").replace("-8bit", "").replace("-4bit", "")
        model_name = f"{model_name}-GGUF"
    else:
        model_name = model_repo

    print(f"Connecting to LM Studio for model: {model_name}")
    inference = load_lm_studio_model(model_name=model_name, base_url=base_url)
    
    # Return as tuple for backward compatibility
    # Both model and tokenizer are the same LMInference instance
    return inference, inference


def evaluate_llm_baseline(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    use_statistical_features: bool = True,
    model_repo: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    limit: Optional[int] = None,
) -> Dict[str, any]:
    """
    Evaluate LLM baseline on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        output_path: Optional path to save results JSON
        use_statistical_features: Whether to include statistical features in prompts
        model_repo: Model repository identifier (default: granite-4.0-h-micro-4bit)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Optional limit on number of windows to process (for testing)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating LLM Baseline")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Use statistical features: {use_statistical_features}")
    print()

    # Load LLM model
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

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)

    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
        statistical_features = np.array(metadata.get("statistical_features", []))
    else:
        # Fallback: use default sensor names
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

    num_windows = unnormalized_windows.shape[0]

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {unnormalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Run predictions
    print("Running LLM predictions...")
    window_labels_pred = []
    sensor_labels_pred = []
    fault_types_pred = []
    reasoning_list = []
    processing_times = []
    context_lengths = []

    with tqdm(total=num_windows, desc="LLM Inference", unit="window") as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()

            window_data = unnormalized_windows[window_idx]
            stats = (
                statistical_features[window_idx]
                if statistical_features is not None
                and len(statistical_features) > window_idx
                else None
            )

            # Format prompt
            prompt = format_window_for_llm(
                window_data, sensor_names, stats, use_statistical_features
            )
            context_length = len(prompt.split())  # Approximate token count

            # Call LLM
            try:
                response = call_llm(
                    prompt,
                    model,
                    tokenizer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                prediction = parse_llm_response(response, sensor_names)
                prediction["reasoning"] = response[
                    :200
                ]  # Store first 200 chars of response
            except Exception as e:
                # Fallback to no-fault prediction
                prediction = {
                    "window_label": 0,
                    "sensor_labels": np.zeros(len(sensor_names), dtype=np.float32),
                    "fault_type": None,
                    "reasoning": f"Error: {str(e)}",
                }

            window_labels_pred.append(prediction["window_label"])
            sensor_labels_pred.append(prediction["sensor_labels"])
            fault_types_pred.append(prediction["fault_type"])
            reasoning_list.append(prediction.get("reasoning", ""))
            processing_times.append(time.time() - start_time)
            context_lengths.append(context_length)

            # Update progress bar with current metrics
            pbar.update(1)
            if (window_idx + 1) % 10 == 0:
                avg_time = (
                    np.mean(processing_times[-10:])
                    if len(processing_times) >= 10
                    else np.mean(processing_times)
                )
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s"})

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    avg_context_length = np.mean(context_lengths) if context_lengths else 0

    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total processing time: {total_processing_time:.2f} seconds")
    if avg_context_length > 0:
        print(f"  Average context length: {avg_context_length:.0f} tokens")
    print()

    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = data.get("fault_types", None)

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )

    # Add efficiency metrics
    metrics["efficiency"] = {
        "avg_processing_time_seconds": float(avg_processing_time),
        "total_processing_time_seconds": float(total_processing_time),
        "windows_per_second": float(num_windows / total_processing_time),
        "avg_context_length_tokens": float(avg_context_length),
        "use_statistical_features": use_statistical_features,
    }

    # Print report
    report = format_metrics_report(metrics)
    print(report)

    # Save results
    results = {
        "method": "llm_baseline",
        "dataset": str(dataset_path),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
            "fault_types": fault_types_pred,
            "reasoning": reasoning_list[:10]
            if len(reasoning_list) > 10
            else reasoning_list,  # Sample reasoning
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
        description="Evaluate LLM baseline on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to shared dataset .npz file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_baseline.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--use-statistical-features",
        action="store_true",
        default=True,
        help="Include statistical features in LLM prompts",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="granite-4.0-h-micro-GGUF",
        help="Model name in LM Studio (default: granite-4.0-h-micro-GGUF)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )

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
