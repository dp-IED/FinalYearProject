import numpy as np
import json
import re
from typing import Dict, List, Optional

from llm.inference import LMInference, load_llm_model as load_lm_studio_model


def sensor_labels_to_window_label(sensor_labels: np.ndarray) -> int:
    faulty_indices = np.where(sensor_labels > 0)[0]
    if len(faulty_indices) == 0:
        return 0
    return int(faulty_indices[0]) + 1


def format_window_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True,
) -> str:
    lines = []
    lines.append(
        "You are an automotive diagnostic expert analyzing OBD-II sensor data."
    )
    lines.append("")
    lines.append("Task: Identify which sensors are faulty and describe the fault type.")
    lines.append("")
    lines.append("Sensor Data Window (300 timesteps):")
    lines.append("=" * 80)

    if statistical_features is not None and use_statistical_features:
        lines.append("\nStatistical Features for each sensor:")
        lines.append("-" * 80)
        feature_names = [
            "mean", "std", "min", "max", "range",
            "median", "mode", "skewness", "kurtosis",
        ]
        for i, sensor_name in enumerate(sensor_names):
            if i < len(statistical_features):
                lines.append(f"\n{sensor_name}:")
                for j, feat_name in enumerate(feature_names):
                    if j < len(statistical_features[i]):
                        lines.append(f"  {feat_name}: {statistical_features[i][j]:.4f}")
        lines.append("")

    lines.append("\nTime Series Data (key timesteps sampled):")
    lines.append("-" * 80)
    window_size = window_data.shape[0]
    if window_size > 50:
        sample_indices = (
            [0]
            + list(range(window_size // 4, window_size // 2, window_size // 8))
            + list(range(window_size // 2, 3 * window_size // 4, window_size // 8))
            + [window_size - 1]
        )
        sample_indices = sorted(list(set(sample_indices)))
    else:
        sample_indices = list(range(window_size))

    header = "Time\t" + "\t".join([name.replace(" ()", "") for name in sensor_names])
    lines.append(header)
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
    lines.append('    "root_cause_sensors": ["SENSOR_NAME"] or [],')
    lines.append('    "affected_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],')
    lines.append('    "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],')
    lines.append('    "fault_type": "VSS_DROPOUT" | "COOLANT_DROPOUT" | "MAF_SCALE" | "TPS_STUCK" | "gradual_drift" or null,')
    lines.append('    "reasoning": "2-4 sentences explaining your diagnosis",')
    lines.append('    "confidence": 0.85')
    lines.append("}")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("- root_cause_sensors: The PRIMARY sensor(s) causing the fault (usually 1 sensor)")
    lines.append("- affected_sensors: Secondary sensors that are affected by the root cause")
    lines.append("- faulty_sensors: For backward compatibility, include ALL faulty sensors (root + affected combined)")
    lines.append("")
    lines.append("Available sensor names:")
    for name in sensor_names:
        lines.append(f"  - {name.replace(' ()', '')}")
    lines.append("")
    lines.append("Fault types: VSS_DROPOUT, COOLANT_DROPOUT, MAF_SCALE, TPS_STUCK, gradual_drift")
    lines.append("")
    lines.append("CRITICAL: Only output valid JSON. No markdown, no code blocks, no extra text.")

    return "\n".join(lines)


def parse_llm_response(response: str, sensor_names: List[str]) -> Dict:
    sensor_labels = np.zeros(len(sensor_names), dtype=np.float32)
    sensor_labels_root_only = np.zeros(len(sensor_names), dtype=np.float32)
    root_cause_sensors = []
    affected_sensors = []
    fault_type = None
    reasoning = ""

    try:
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            root_cause_sensors = result.get("root_cause_sensors", [])
            affected_sensors = result.get("affected_sensors", [])
            faulty_sensors = result.get("faulty_sensors", [])
            if not root_cause_sensors and not affected_sensors and faulty_sensors:
                root_cause_sensors = faulty_sensors
                affected_sensors = []

            fault_type = result.get("fault_type", None)
            if fault_type == "unknown" or fault_type == "":
                fault_type = None
            reasoning = result.get("reasoning", "")

            def map_sensor_to_index(sensor: str) -> Optional[int]:
                if sensor in sensor_names:
                    return sensor_names.index(sensor)
                sensor_clean = sensor.replace(" ()", "").strip()
                for i, name in enumerate(sensor_names):
                    name_clean = name.replace(" ()", "").strip()
                    if sensor_clean == name_clean:
                        return i
                return None

            for sensor in root_cause_sensors:
                idx = map_sensor_to_index(sensor)
                if idx is not None:
                    sensor_labels_root_only[idx] = 1.0
                    sensor_labels[idx] = 1.0
            for sensor in affected_sensors:
                idx = map_sensor_to_index(sensor)
                if idx is not None:
                    sensor_labels[idx] = 1.0

            window_label = sensor_labels_to_window_label(sensor_labels_root_only)
            return {
                "window_label": window_label,
                "sensor_labels": sensor_labels,
                "sensor_labels_raw": sensor_labels.copy(),
                "sensor_labels_root_only": sensor_labels_root_only,
                "root_cause_sensors": root_cause_sensors,
                "affected_sensors": affected_sensors,
                "fault_type": fault_type,
                "reasoning": reasoning if reasoning else "No reasoning provided",
            }
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass

    faulty_sensors_match = re.search(
        r"Faulty Sensors?:\s*(.+?)(?:\n|Fault Type|Reasoning|$)",
        response, re.IGNORECASE | re.DOTALL,
    )
    if faulty_sensors_match:
        faulty_sensors_str = faulty_sensors_match.group(1).strip()
        if faulty_sensors_str.lower() not in ["none", "no", "n/a", ""]:
            key_parts = {
                "VEHICLE_SPEED": ["vehicle", "speed", "vss"],
                "COOLANT_TEMPERATURE": ["coolant", "temperature"],
                "THROTTLE": ["throttle", "tps"],
                "ENGINE_RPM": ["rpm", "engine rpm"],
                "ENGINE_LOAD": ["engine load", "load"],
                "INTAKE_MANIFOLD_PRESSURE": ["intake", "manifold", "pressure", "map", "maf"],
                "SHORT_TERM_FUEL_TRIM": ["short term", "fuel trim", "stft"],
                "LONG_TERM_FUEL_TRIM": ["long term", "fuel trim", "ltft"],
            }
            for i, sensor_name in enumerate(sensor_names):
                sensor_variants = [
                    sensor_name, sensor_name.replace(" ()", ""),
                    sensor_name.replace(" ()", "").replace("_", " "),
                ]
                matched = False
                for variant in sensor_variants:
                    variant_clean = variant.replace("_", "").replace(" ", "").lower()
                    response_clean = faulty_sensors_str.replace("_", "").replace(" ", "").lower()
                    if variant_clean in response_clean:
                        sensor_labels[i] = 1.0
                        matched = True
                        break
                if not matched:
                    for key, parts in key_parts.items():
                        if key in sensor_name:
                            if any(part in faulty_sensors_str.lower() for part in parts):
                                sensor_labels[i] = 1.0
                                break

    fault_type_match = re.search(
        r"Fault Type:\s*(.+?)(?:\n|Reasoning|$)", response, re.IGNORECASE | re.DOTALL
    )
    if fault_type_match:
        fault_type_str = fault_type_match.group(1).strip().split(",")[0].split(".")[0].strip()
        if fault_type_str.lower() not in ["unknown", "none", "n/a", ""]:
            fault_type = fault_type_str

    reasoning_match = re.search(
        r"Reasoning:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    if np.sum(sensor_labels) == 0 and reasoning:
        reasoning_lower = reasoning.lower()
        for i, sensor_name in enumerate(sensor_names):
            sensor_base = sensor_name.replace(" ()", "").replace("_", " ").lower()
            sensor_parts = sensor_base.split()
            if any(part in reasoning_lower for part in sensor_parts if len(part) > 3):
                sensor_labels[i] = 1.0

    sensor_labels_root_only = sensor_labels.copy()
    window_label = sensor_labels_to_window_label(sensor_labels_root_only)
    parsed_root_sensors = [sensor_names[i] for i in range(len(sensor_names)) if sensor_labels_root_only[i] > 0]

    return {
        "window_label": window_label,
        "sensor_labels": sensor_labels,
        "sensor_labels_raw": sensor_labels.copy(),
        "sensor_labels_root_only": sensor_labels_root_only,
        "root_cause_sensors": parsed_root_sensors,
        "affected_sensors": [],
        "fault_type": fault_type,
        "reasoning": reasoning if reasoning else "No reasoning provided",
    }


def filter_sensor_labels_to_root_only(parsed_result: Dict, sensor_names: List[str]) -> np.ndarray:
    if "sensor_labels_root_only" in parsed_result:
        return parsed_result["sensor_labels_root_only"].copy()

    sensor_labels = np.zeros(len(sensor_names), dtype=np.float32)
    root_cause_sensors = parsed_result.get("root_cause_sensors", [])

    for sensor in root_cause_sensors:
        if sensor in sensor_names:
            idx = sensor_names.index(sensor)
            sensor_labels[idx] = 1.0
        else:
            sensor_clean = sensor.replace(" ()", "").strip()
            for i, name in enumerate(sensor_names):
                name_clean = name.replace(" ()", "").strip()
                if sensor_clean == name_clean:
                    sensor_labels[i] = 1.0
                    break

    return sensor_labels


def call_llm(
    prompt: str,
    model,
    tokenizer=None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    repetition_context_size: int = 20,
) -> str:
    if model is None:
        raise RuntimeError("Model not loaded. Call load_llm_model() first.")

    if isinstance(model, LMInference):
        messages = [{"role": "user", "content": prompt}]
        return model.chat_completions(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
    raise RuntimeError(
        "MLX is no longer supported. Please use LM Studio with LMInference."
    )


def load_llm_model(
    model_repo: str = "granite-4.0-h-micro-GGUF",
    base_url: str = "http://localhost:1234/v1",
):
    if model_repo.startswith("mlx-community/"):
        model_name = model_repo.replace("mlx-community/", "").replace("-8bit", "").replace("-4bit", "")
        model_name = f"{model_name}-GGUF"
    else:
        model_name = model_repo

    print(f"Connecting to LM Studio for model: {model_name}")
    inference = load_lm_studio_model(model_name=model_name, base_url=base_url)
    return inference, inference
