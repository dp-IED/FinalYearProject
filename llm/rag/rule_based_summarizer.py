"""
Rule-based window summarizer for generating textual descriptions without LLM calls.

This module computes features from sensor windows, maps them to qualitative labels,
and generates template-based descriptions suitable for RAG/vector DB ingestion.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm.rag.sensor_config import get_sensor_config, get_normal_range


def detect_events(sensor_values: np.ndarray, thresholds: Dict) -> Dict[str, bool]:
    """
    Detect events (spikes, dropouts, plateaus) in sensor time series.

    Args:
        sensor_values: (window_size,) array of sensor values
        thresholds: Dictionary with spike_threshold, dropout_threshold,
                   dropout_min_duration, plateau_variance_threshold,
                   plateau_min_duration

    Returns:
        Dictionary with boolean flags: has_spike, has_dropout, has_plateau
    """
    window_size = len(sensor_values)
    events = {
        "has_spike": False,
        "has_dropout": False,
        "has_plateau": False,
    }

    # Detect spikes: any step-to-step change > threshold
    if window_size > 1:
        diffs = np.abs(np.diff(sensor_values))
        spike_threshold = thresholds.get("spike_threshold", 10)
        if np.any(diffs > spike_threshold):
            events["has_spike"] = True

    # Detect dropouts: runs where value ≈ 0 for ≥ N timesteps
    dropout_threshold = thresholds.get("dropout_threshold", 1)
    dropout_min_duration = thresholds.get("dropout_min_duration", 5)

    near_zero = np.abs(sensor_values) < dropout_threshold
    if np.any(near_zero):
        # Find consecutive runs of near-zero values
        runs = []
        in_run = False
        run_start = 0
        for i, is_zero in enumerate(near_zero):
            if is_zero and not in_run:
                run_start = i
                in_run = True
            elif not is_zero and in_run:
                run_length = i - run_start
                if run_length >= dropout_min_duration:
                    runs.append((run_start, i))
                in_run = False
        # Check if run extends to end
        if in_run:
            run_length = window_size - run_start
            if run_length >= dropout_min_duration:
                runs.append((run_start, window_size))

        if len(runs) > 0:
            events["has_dropout"] = True

    # Detect plateaus: long runs with low variance
    plateau_variance_threshold = thresholds.get("plateau_variance_threshold", 5)
    plateau_min_duration = thresholds.get("plateau_min_duration", 20)

    # Use sliding window to detect low-variance segments
    if window_size >= plateau_min_duration:
        for i in range(window_size - plateau_min_duration + 1):
            segment = sensor_values[i : i + plateau_min_duration]
            segment_variance = np.var(segment)
            if segment_variance < plateau_variance_threshold:
                events["has_plateau"] = True
                break

    return events


def compute_window_features(
    unnormalized_windows: np.ndarray, sensor_names: List[str]
) -> Dict[int, Dict[str, Dict]]:
    """
    Compute features for all windows and sensors.

    Args:
        unnormalized_windows: (N, window_size, num_sensors) array
        sensor_names: List of sensor names

    Returns:
        Dictionary: {window_idx: {sensor_name: {feature_name: value}}}
    """
    num_windows, window_size, num_sensors = unnormalized_windows.shape

    features = {}

    for window_idx in range(num_windows):
        window_features = {}

        for sensor_idx, sensor_name in enumerate(sensor_names):
            sensor_values = unnormalized_windows[window_idx, :, sensor_idx]

            # Basic stats
            v_min = float(np.min(sensor_values))
            v_max = float(np.max(sensor_values))
            v_mean = float(np.mean(sensor_values))
            v_std = float(np.std(sensor_values))

            # Start/end values
            v_start = float(sensor_values[0])
            v_end = float(sensor_values[-1])

            # Trend: slope = (v_end - v_start) / window_duration
            window_duration = window_size - 1 if window_size > 1 else 1
            slope = (v_end - v_start) / window_duration

            # Get sensor-specific thresholds
            thresholds = get_sensor_config(sensor_name)

            # Detect events
            events = detect_events(sensor_values, thresholds)

            window_features[sensor_name] = {
                "min": v_min,
                "max": v_max,
                "mean": v_mean,
                "std": v_std,
                "v_start": v_start,
                "v_end": v_end,
                "slope": slope,
                "has_spike": events["has_spike"],
                "has_dropout": events["has_dropout"],
                "has_plateau": events["has_plateau"],
            }

        features[window_idx] = window_features

    return features


def map_features_to_labels(
    features: Dict[int, Dict[str, Dict]], sensor_names: List[str]
) -> Dict[int, Dict[str, Dict]]:
    """
    Map numerical features to qualitative labels.

    Args:
        features: Dictionary from compute_window_features()
        sensor_names: List of sensor names

    Returns:
        Dictionary: {window_idx: {sensor_name: {level, trend, event_description}}}
    """
    labels = {}

    for window_idx, window_features in features.items():
        window_labels = {}

        for sensor_name in sensor_names:
            sensor_features = window_features.get(sensor_name, {})
            thresholds = get_sensor_config(sensor_name)
            normal_range = get_normal_range(sensor_name)

            # Level classification: compare mean to normal range
            mean_val = sensor_features.get("mean", 0)
            normal_min, normal_max = normal_range

            # Use 20% margin for low/high classification
            margin = (normal_max - normal_min) * 0.2
            if mean_val < normal_min - margin:
                level = "low"
            elif mean_val > normal_max + margin:
                level = "high"
            else:
                level = "normal"

            # Trend classification
            slope = sensor_features.get("slope", 0)
            trend_epsilon = thresholds.get("trend_epsilon", 0.5)

            if abs(slope) < trend_epsilon:
                trend = "stable"
            elif slope > trend_epsilon:
                trend = "increasing"
            else:  # slope < -trend_epsilon
                trend = "decreasing"

            # Event classification
            event_descriptions = []
            if sensor_features.get("has_dropout", False):
                event_descriptions.append("drops to near zero for short periods")
            if sensor_features.get("has_spike", False):
                event_descriptions.append("shows occasional sharp spikes")
            if sensor_features.get("has_plateau", False):
                event_descriptions.append("stays flat for a long segment")

            event_clause = ", ".join(event_descriptions) if event_descriptions else ""

            window_labels[sensor_name] = {
                "level": level,
                "trend": trend,
                "event_description": event_clause,
            }

        labels[window_idx] = window_labels

    return labels


def generate_window_description(
    window_idx: int,
    features: Dict[str, Dict],
    labels: Dict[str, Dict],
    sensor_names: List[str],
) -> str:
    """
    Generate textual description for a window using templates.

    Args:
        window_idx: Window index
        features: Dictionary {sensor_name: {feature_name: value}} for this window
        labels: Dictionary {sensor_name: {level, trend, event_description}} for this window
        sensor_names: List of sensor names

    Returns:
        Multi-paragraph description string
    """
    paragraphs = []

    # Helper function to get sensor name without parentheses
    def clean_name(name: str) -> str:
        return name.replace(" ()", "")

    # Helper function to format trend description
    def format_trend(trend: str) -> str:
        if trend == "increasing":
            return "increases gradually"
        elif trend == "decreasing":
            return "decreases gradually"
        else:
            return "stays roughly constant"

    # First paragraph: RPM, SPEED, LOAD, TPS
    first_para_sentences = []

    # RPM
    rpm_name = "ENGINE_RPM ()"
    if rpm_name in features:
        rpm_feat = features[rpm_name]
        rpm_labels = labels[rpm_name]
        rpm_clean = clean_name(rpm_name)
        trend_str = format_trend(rpm_labels["trend"])
        event_clause = (
            f", with {rpm_labels['event_description']}"
            if rpm_labels["event_description"]
            else ""
        )
        first_para_sentences.append(
            f"{rpm_clean} starts around {rpm_feat['v_start']:.0f} and {trend_str} to about {rpm_feat['v_end']:.0f}, "
            f"remaining mostly {rpm_labels['level']} relative to typical operation{event_clause}."
        )

    # SPEED
    speed_name = "VEHICLE_SPEED ()"
    if speed_name in features:
        speed_feat = features[speed_name]
        speed_labels = labels[speed_name]
        speed_clean = clean_name(speed_name)
        trend_str = format_trend(speed_labels["trend"])
        event_clause = (
            f", with {speed_labels['event_description']}"
            if speed_labels["event_description"]
            else ""
        )
        first_para_sentences.append(
            f"{speed_clean} starts around {speed_feat['v_start']:.0f} and {trend_str} to about {speed_feat['v_end']:.0f}, "
            f"remaining mostly {speed_labels['level']} relative to typical operation{event_clause}."
        )

    # Cross-sensor summary: RPM + SPEED
    if rpm_name in features and speed_name in features:
        rpm_feat = features[rpm_name]
        speed_feat = features[speed_name]
        rpm_labels = labels[rpm_name]
        speed_labels = labels[speed_name]

        # Check if both increasing together
        if (
            rpm_labels["trend"] == "increasing"
            and speed_labels["trend"] == "increasing"
        ):
            first_para_sentences.append(
                "ENGINE_RPM and VEHICLE_SPEED evolve consistently, rising together over the window."
            )
        # Check if RPM rises but SPEED stays near zero
        elif rpm_labels["trend"] == "increasing" and speed_feat["mean"] < 5:
            first_para_sentences.append(
                "ENGINE_RPM rises while VEHICLE_SPEED remains near zero."
            )

    # LOAD
    load_name = "ENGINE_LOAD ()"
    if load_name in features:
        load_feat = features[load_name]
        load_labels = labels[load_name]
        load_clean = clean_name(load_name)
        trend_str = format_trend(load_labels["trend"])
        event_clause = (
            f", with {load_labels['event_description']}"
            if load_labels["event_description"]
            else ""
        )
        first_para_sentences.append(
            f"{load_clean} starts around {load_feat['v_start']:.0f} and {trend_str} to about {load_feat['v_end']:.0f}, "
            f"remaining mostly {load_labels['level']} relative to typical operation{event_clause}."
        )

    # TPS (THROTTLE)
    tps_name = "THROTTLE ()"
    if tps_name in features:
        tps_feat = features[tps_name]
        tps_labels = labels[tps_name]
        tps_clean = clean_name(tps_name)
        trend_str = format_trend(tps_labels["trend"])
        event_clause = (
            f", with {tps_labels['event_description']}"
            if tps_labels["event_description"]
            else ""
        )
        first_para_sentences.append(
            f"{tps_clean} starts around {tps_feat['v_start']:.0f} and {trend_str} to about {tps_feat['v_end']:.0f}, "
            f"remaining mostly {tps_labels['level']} relative to typical operation{event_clause}."
        )

    if first_para_sentences:
        paragraphs.append(" ".join(first_para_sentences))

    # Second paragraph: Manifold pressure, coolant, and trims
    second_para_sentences = []

    # INTAKE_MANIFOLD_PRESSURE
    map_name = "INTAKE_MANIFOLD_PRESSURE ()"
    if map_name in features:
        map_feat = features[map_name]
        map_labels = labels[map_name]
        map_clean = clean_name(map_name)
        trend_str = format_trend(map_labels["trend"])
        event_clause = (
            f", with {map_labels['event_description']}"
            if map_labels["event_description"]
            else ""
        )
        second_para_sentences.append(
            f"{map_clean} starts around {map_feat['v_start']:.2f} and {trend_str} to about {map_feat['v_end']:.2f}, "
            f"remaining mostly {map_labels['level']} relative to typical operation{event_clause}."
        )

    # COOLANT_TEMPERATURE
    coolant_name = "COOLANT_TEMPERATURE ()"
    if coolant_name in features:
        coolant_feat = features[coolant_name]
        coolant_labels = labels[coolant_name]
        coolant_clean = clean_name(coolant_name)
        trend_str = format_trend(coolant_labels["trend"])
        event_clause = (
            f", with {coolant_labels['event_description']}"
            if coolant_labels["event_description"]
            else ""
        )
        second_para_sentences.append(
            f"{coolant_clean} starts around {coolant_feat['v_start']:.0f} and {trend_str} to about {coolant_feat['v_end']:.0f}, "
            f"remaining mostly {coolant_labels['level']} relative to typical operation{event_clause}."
        )

    # SHORT_TERM_FUEL_TRIM
    stft_name = "SHORT_TERM_FUEL_TRIM_BANK_1 ()"
    if stft_name in features:
        stft_feat = features[stft_name]
        stft_labels = labels[stft_name]
        stft_clean = clean_name(stft_name)
        trend_str = format_trend(stft_labels["trend"])
        event_clause = (
            f", with {stft_labels['event_description']}"
            if stft_labels["event_description"]
            else ""
        )
        second_para_sentences.append(
            f"{stft_clean} starts around {stft_feat['v_start']:.2f} and {trend_str} to about {stft_feat['v_end']:.2f}, "
            f"remaining mostly {stft_labels['level']} relative to typical operation{event_clause}."
        )

    # LONG_TERM_FUEL_TRIM
    ltft_name = "LONG_TERM_FUEL_TRIM_BANK_1 ()"
    if ltft_name in features:
        ltft_feat = features[ltft_name]
        ltft_labels = labels[ltft_name]
        ltft_clean = clean_name(ltft_name)
        trend_str = format_trend(ltft_labels["trend"])
        event_clause = (
            f", with {ltft_labels['event_description']}"
            if ltft_labels["event_description"]
            else ""
        )
        second_para_sentences.append(
            f"{ltft_clean} starts around {ltft_feat['v_start']:.2f} and {trend_str} to about {ltft_feat['v_end']:.2f}, "
            f"remaining mostly {ltft_labels['level']} relative to typical operation{event_clause}."
        )

    if second_para_sentences:
        paragraphs.append(" ".join(second_para_sentences))

    # Final sentence: Overall consistency check
    # Check if fuel trims deviate significantly
    has_fuel_trim_deviation = False
    stft_name_check = "SHORT_TERM_FUEL_TRIM_BANK_1 ()"
    ltft_name_check = "LONG_TERM_FUEL_TRIM_BANK_1 ()"

    if stft_name_check in features:
        stft_feat_check = features[stft_name_check]
        if abs(stft_feat_check["mean"]) > 10:  # Significant deviation
            has_fuel_trim_deviation = True
    if ltft_name_check in features:
        ltft_feat_check = features[ltft_name_check]
        if abs(ltft_feat_check["mean"]) > 10:  # Significant deviation
            has_fuel_trim_deviation = True

    if has_fuel_trim_deviation:
        paragraphs.append("Fuel trim behaviour deviates from the rest.")
    else:
        paragraphs.append("Overall, signals appear mutually consistent.")

    # Join paragraphs with double newline
    description = "\n\n".join(paragraphs)

    # Add window context at the beginning
    # Detect driving pattern
    driving_pattern = "cruise"
    if speed_name in features:
        speed_feat = features[speed_name]
        speed_labels = labels[speed_name]
        if speed_feat["mean"] < 5:
            driving_pattern = "idle"
        elif speed_labels["trend"] == "increasing":
            driving_pattern = "acceleration"

    full_description = (
        f"Window {window_idx} shows {driving_pattern} operation. {description}"
    )

    return full_description


def generate_all_descriptions(
    dataset_path: Path, output_dir: Optional[Path] = None, save_index: bool = True
) -> Dict[int, str]:
    """
    Generate descriptions for all windows in a dataset.

    Args:
        dataset_path: Path to .npz dataset file
        output_dir: Output directory (default: dataset_path.parent / "descriptions")
        save_index: Whether to save JSON index file with metadata

    Returns:
        Dictionary: {window_idx: description_string}
    """
    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    unnormalized_windows = data["unnormalized_windows"]

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

    num_windows = unnormalized_windows.shape[0]

    # Set output directory
    if output_dir is None:
        output_dir = dataset_path.parent / "descriptions"
    else:
        output_dir = Path(output_dir)

    # Create output directory
    descriptions_dir = output_dir / "descriptions"
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    # Compute features for all windows
    print(f"Computing features for {num_windows} windows...")
    features = compute_window_features(unnormalized_windows, sensor_names)

    # Map features to labels
    print("Mapping features to labels...")
    labels = map_features_to_labels(features, sensor_names)

    # Generate descriptions
    print("Generating descriptions...")
    descriptions = {}
    index_data = {}

    for window_idx in range(num_windows):
        window_features = features[window_idx]
        window_labels = labels[window_idx]

        description = generate_window_description(
            window_idx, window_features, window_labels, sensor_names
        )
        descriptions[window_idx] = description

        # Save individual text file
        filename = f"window_{window_idx:05d}.txt"
        filepath = descriptions_dir / filename
        with open(filepath, "w") as f:
            f.write(description)

        # Prepare index data
        if save_index:
            index_data[f"window_{window_idx}"] = {
                "file": filename,
                "description": description,
                "features": {
                    sensor_name: {
                        k: v
                        for k, v in sensor_feat.items()
                        if k not in ["has_spike", "has_dropout", "has_plateau"]
                    }
                    for sensor_name, sensor_feat in window_features.items()
                },
                "labels": window_labels,
            }

    # Save index file if requested
    if save_index:
        index_path = descriptions_dir / "descriptions_index.json"
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print(f"Saved index file: {index_path}")

    print(f"Generated {num_windows} descriptions in {descriptions_dir}")

    return descriptions
