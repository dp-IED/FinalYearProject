"""
Rule-based window summarizer for generating textual descriptions without LLM calls.

This module computes features from sensor windows, maps them to qualitative labels,
and generates template-based descriptions suitable for RAG/vector DB ingestion.

ENHANCED: Uses GDN predictions instead of hardcoded sensor_config thresholds.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import KnowledgeGraph for KG context (optional)
try:
    from kg.create_kg import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None


def detect_events(
    sensor_values: np.ndarray,
    thresholds: Dict,
    gdn_score: Optional[float] = None,
) -> Dict[str, bool]:
    """
    Detect events (spikes, dropouts, plateaus) in sensor time series.
    
    Uses adaptive thresholds based on GDN scores if provided.

    Args:
        sensor_values: (window_size,) array of sensor values
        thresholds: Dictionary with spike_threshold, dropout_threshold,
                   dropout_min_duration, plateau_variance_threshold,
                   plateau_min_duration
        gdn_score: Optional GDN anomaly score for adaptive thresholds

    Returns:
        Dictionary with boolean flags: has_spike, has_dropout, has_plateau
    """
    window_size = len(sensor_values)
    events = {
        "has_spike": False,
        "has_dropout": False,
        "has_plateau": False,
    }

    # Adaptive thresholds based on GDN score
    if gdn_score is not None:
        # If GDN score is high (>0.7), use more sensitive thresholds
        # If GDN score is low (<0.3), use less sensitive thresholds
        sensitivity_factor = 1.0
        if gdn_score > 0.7:
            sensitivity_factor = 0.7  # More sensitive (lower threshold)
        elif gdn_score < 0.3:
            sensitivity_factor = 1.5  # Less sensitive (higher threshold)
    else:
        sensitivity_factor = 1.0

    # Detect spikes: any step-to-step change > threshold
    if window_size > 1:
        diffs = np.abs(np.diff(sensor_values))
        spike_threshold = thresholds.get("spike_threshold", 10) * sensitivity_factor
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
    plateau_variance_threshold = thresholds.get("plateau_variance_threshold", 5) * sensitivity_factor
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
    unnormalized_windows: np.ndarray,
    sensor_names: List[str],
    gdn_predictions: Optional[np.ndarray] = None,
    distribution_thresholds: Optional[Dict] = None,
) -> Dict[int, Dict[str, Dict]]:
    """
    Compute features for all windows and sensors.
    
    ENHANCED: Accepts GDN predictions and uses adaptive thresholds.

    Args:
        unnormalized_windows: (N, window_size, num_sensors) array
        sensor_names: List of sensor names
        gdn_predictions: Optional (N, num_sensors) array of GDN anomaly scores
        distribution_thresholds: Optional distribution thresholds from KG

    Returns:
        Dictionary: {window_idx: {sensor_name: {feature_name: value}}}
    """
    num_windows, window_size, num_sensors = unnormalized_windows.shape

    features = {}

    # Compute data-driven thresholds for event detection
    # Use percentile-based thresholds computed from all windows
    all_slopes = []
    for window_idx in range(num_windows):
        for sensor_idx in range(num_sensors):
            sensor_values = unnormalized_windows[window_idx, :, sensor_idx]
            if window_size > 1:
                slope = (sensor_values[-1] - sensor_values[0]) / (window_size - 1)
                all_slopes.append(abs(slope))
    
    # Use 25th and 75th percentiles for trend thresholds
    if len(all_slopes) > 0:
        trend_epsilon_low = float(np.percentile(all_slopes, 25))
        trend_epsilon_high = float(np.percentile(all_slopes, 75))
    else:
        trend_epsilon_low = 0.3
        trend_epsilon_high = 0.7

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

            # Get GDN score if available
            gdn_score = None
            if gdn_predictions is not None and window_idx < len(gdn_predictions):
                if sensor_idx < len(gdn_predictions[window_idx]):
                    gdn_score = float(gdn_predictions[window_idx, sensor_idx])

            # Use data-driven thresholds for event detection
            # Default thresholds (can be made adaptive based on GDN score)
            event_thresholds = {
                "spike_threshold": v_std * 3.0,  # 3 standard deviations
                "dropout_threshold": max(v_std * 0.1, 1.0),
                "dropout_min_duration": 5,
                "plateau_variance_threshold": v_std * 0.5,
                "plateau_min_duration": 20,
                "trend_epsilon": (trend_epsilon_low + trend_epsilon_high) / 2,
            }

            # Detect events with adaptive thresholds
            events = detect_events(sensor_values, event_thresholds, gdn_score=gdn_score)

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
                "gdn_anomaly_score": gdn_score if gdn_score is not None else 0.0,
            }

        features[window_idx] = window_features

    return features


def map_features_to_labels(
    features: Dict[int, Dict[str, Dict]],
    sensor_names: List[str],
    gdn_predictions: Optional[np.ndarray] = None,
    distribution_thresholds: Optional[Dict] = None,
) -> Dict[int, Dict[str, Dict]]:
    """
    Map numerical features to qualitative labels using GDN predictions.
    
    ENHANCED: Uses GDN anomaly scores instead of hardcoded normal ranges.

    Args:
        features: Dictionary from compute_window_features()
        sensor_names: List of sensor names
        gdn_predictions: Optional (num_windows, num_sensors) array of GDN anomaly scores
        distribution_thresholds: Optional distribution thresholds from KG

    Returns:
        Dictionary: {window_idx: {sensor_name: {level, trend, event_description}}}
    """
    labels = {}

    # Get default thresholds
    if distribution_thresholds is None:
        anomaly_threshold_global = 0.5
        anomaly_threshold_per_sensor = {}
    else:
        anomaly_threshold_global = distribution_thresholds.get('anomaly_threshold_global', 0.5)
        anomaly_threshold_per_sensor = distribution_thresholds.get('anomaly_threshold_per_sensor', {})

    # Compute data-driven trend thresholds from all slopes
    all_slopes = []
    for window_features in features.values():
        for sensor_name in sensor_names:
            if sensor_name in window_features:
                slope = window_features[sensor_name].get("slope", 0)
                all_slopes.append(abs(slope))
    
    if len(all_slopes) > 0:
        trend_epsilon = float(np.percentile(all_slopes, 50))  # Median as threshold
    else:
        trend_epsilon = 0.5

    for window_idx, window_features in features.items():
        window_labels = {}

        for sensor_idx, sensor_name in enumerate(sensor_names):
            sensor_features = window_features.get(sensor_name, {})
            
            # Get GDN score
            gdn_score = sensor_features.get("gdn_anomaly_score", 0.0)
            if gdn_predictions is not None and window_idx < len(gdn_predictions):
                if sensor_idx < len(gdn_predictions[window_idx]):
                    gdn_score = float(gdn_predictions[window_idx, sensor_idx])

            # Level classification using GDN score
            threshold = anomaly_threshold_per_sensor.get(
                sensor_name, anomaly_threshold_global
            )
            
            if gdn_score > 0.8:
                level = "severe_anomaly"
            elif gdn_score > threshold:
                if gdn_score > 0.5:
                    level = "moderate_anomaly"
                else:
                    level = "mild_anomaly"
            elif gdn_score > 0.3:
                level = "mild_anomaly"
            else:
                level = "normal"

            # Trend classification using data-driven thresholds
            slope = sensor_features.get("slope", 0)
            
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
            
            # Add GDN-based insights
            if gdn_score > threshold:
                confidence = "high" if gdn_score > 0.7 else "moderate"
                event_descriptions.append(
                    f"GDN identifies this sensor as anomalous (score: {gdn_score:.2f}, {confidence} confidence)"
                )

            event_clause = ", ".join(event_descriptions) if event_descriptions else ""

            window_labels[sensor_name] = {
                "level": level,
                "trend": trend,
                "event_description": event_clause,
                "gdn_score": gdn_score,
            }

        labels[window_idx] = window_labels

    return labels


def generate_window_description(
    window_idx: int,
    features: Dict[str, Dict],
    labels: Dict[str, Dict],
    sensor_names: List[str],
    gdn_predictions: Optional[np.ndarray] = None,
    kg_context: Optional[Dict] = None,
    distribution_thresholds: Optional[Dict] = None,
) -> str:
    """
    Generate textual description for a window using templates.
    
    ENHANCED: Includes GDN anomaly insights, correlation violations, and propagation patterns.

    Args:
        window_idx: Window index
        features: Dictionary {sensor_name: {feature_name: value}} for this window
        labels: Dictionary {sensor_name: {level, trend, event_description}} for this window
        sensor_names: List of sensor names
        gdn_predictions: Optional (num_sensors,) array of GDN anomaly scores for this window
        kg_context: Optional KG context from KnowledgeGraph.get_window_kg()
        distribution_thresholds: Optional distribution thresholds from KG

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

    # Get thresholds for anomaly detection
    if distribution_thresholds is None:
        anomaly_threshold_global = 0.5
        anomaly_threshold_per_sensor = {}
    else:
        anomaly_threshold_global = distribution_thresholds.get('anomaly_threshold_global', 0.5)
        anomaly_threshold_per_sensor = distribution_thresholds.get('anomaly_threshold_per_sensor', {})

    # GDN Anomaly Summary Paragraph (NEW)
    if gdn_predictions is not None:
        anomalous_sensors = []
        sensor_scores = []
        
        for sensor_idx, sensor_name in enumerate(sensor_names):
            if sensor_idx < len(gdn_predictions):
                gdn_score = float(gdn_predictions[sensor_idx])
                threshold = anomaly_threshold_per_sensor.get(sensor_name, anomaly_threshold_global)
                
                if gdn_score > threshold:
                    sensor_scores.append((sensor_name, gdn_score))
        
        if sensor_scores:
            # Sort by score (highest first)
            sensor_scores.sort(key=lambda x: x[1], reverse=True)
            
            anomaly_summary = []
            for sensor_name, score in sensor_scores[:3]:  # Top 3
                clean_sensor = clean_name(sensor_name)
                confidence = "high" if score > 0.7 else "moderate"
                anomaly_summary.append(
                    f"{clean_sensor} shows {confidence}-confidence anomaly (score: {score:.2f})"
                )
            
            if anomaly_summary:
                paragraphs.append(
                    "GDN Anomaly Detection: " + "; ".join(anomaly_summary) + "."
                )

    # Correlation Violation Insights (from KG context)
    if kg_context is not None:
        violations = kg_context.get("violations", [])
        if violations:
            violation_descriptions = []
            for violation in violations[:3]:  # Top 3 violations
                source = violation.get("source", "")
                target = violation.get("target", "")
                expected = violation.get("expected_correlation_gdn", 0)
                actual = violation.get("correlation", 0)
                
                if source and target:
                    clean_source = clean_name(source)
                    clean_target = clean_name(target)
                    violation_descriptions.append(
                        f"{clean_source}-{clean_target} correlation breaks down "
                        f"(expected: {expected:.2f}, actual: {actual:.2f})"
                    )
            
            if violation_descriptions:
                paragraphs.append(
                    "Relationship Violations: " + "; ".join(violation_descriptions) + "."
                )

    # Anomaly Propagation Context (from KG context)
    if kg_context is not None:
        propagation = kg_context.get("anomaly_propagation", [])
        if propagation:
            propagation_descriptions = []
            for prop in propagation[:2]:  # Top 2 propagation patterns
                prop_type = prop.get("type", "")
                root_sensor = prop.get("root_sensor", "")
                root_window = prop.get("root_window", -1)
                
                if root_sensor and root_window >= 0:
                    clean_root = clean_name(root_sensor)
                    if prop_type == "root":
                        propagation_descriptions.append(
                            f"Fault originates in {clean_root} (window {root_window})"
                        )
                    elif prop_type == "propagation":
                        propagation_descriptions.append(
                            f"Fault propagates from {clean_root} (window {root_window})"
                        )
            
            if propagation_descriptions:
                paragraphs.append(
                    "Anomaly Propagation: " + "; ".join(propagation_descriptions) + "."
                )

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
        # Use GDN-based level description
        level_desc = rpm_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        first_para_sentences.append(
            f"{rpm_clean} starts around {rpm_feat['v_start']:.0f} and {trend_str} to about {rpm_feat['v_end']:.0f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = speed_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        first_para_sentences.append(
            f"{speed_clean} starts around {speed_feat['v_start']:.0f} and {trend_str} to about {speed_feat['v_end']:.0f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = load_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        first_para_sentences.append(
            f"{load_clean} starts around {load_feat['v_start']:.0f} and {trend_str} to about {load_feat['v_end']:.0f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = tps_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        first_para_sentences.append(
            f"{tps_clean} starts around {tps_feat['v_start']:.0f} and {trend_str} to about {tps_feat['v_end']:.0f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = map_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        second_para_sentences.append(
            f"{map_clean} starts around {map_feat['v_start']:.2f} and {trend_str} to about {map_feat['v_end']:.2f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = coolant_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        second_para_sentences.append(
            f"{coolant_clean} starts around {coolant_feat['v_start']:.0f} and {trend_str} to about {coolant_feat['v_end']:.0f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = stft_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        second_para_sentences.append(
            f"{stft_clean} starts around {stft_feat['v_start']:.2f} and {trend_str} to about {stft_feat['v_end']:.2f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
        level_desc = ltft_labels.get("level", "normal")
        if level_desc == "severe_anomaly":
            level_desc = "severely anomalous"
        elif level_desc == "moderate_anomaly":
            level_desc = "moderately anomalous"
        elif level_desc == "mild_anomaly":
            level_desc = "mildly anomalous"
        else:
            level_desc = "normal"
        
        second_para_sentences.append(
            f"{ltft_clean} starts around {ltft_feat['v_start']:.2f} and {trend_str} to about {ltft_feat['v_end']:.2f}, "
            f"remaining mostly {level_desc} relative to typical operation{event_clause}."
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
    dataset_path: Path,
    gdn_predictions: np.ndarray,
    distribution_thresholds: Optional[Dict] = None,
    kg: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    save_index: bool = True,
) -> Dict[int, str]:
    """
    Generate descriptions for all windows in a dataset.
    
    ENHANCED: Requires GDN predictions and optionally uses KG context.

    Args:
        dataset_path: Path to .npz dataset file
        gdn_predictions: (num_windows, num_sensors) array of GDN anomaly scores (REQUIRED)
        distribution_thresholds: Optional distribution thresholds from KG
        kg: Optional KnowledgeGraph instance for enhanced context
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

    # Validate GDN predictions shape
    if gdn_predictions.shape[0] != num_windows:
        raise ValueError(
            f"GDN predictions shape mismatch: expected {num_windows} windows, "
            f"got {gdn_predictions.shape[0]}"
        )
    if gdn_predictions.shape[1] != len(sensor_names):
        raise ValueError(
            f"GDN predictions shape mismatch: expected {len(sensor_names)} sensors, "
            f"got {gdn_predictions.shape[1]}"
        )

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
    features = compute_window_features(
        unnormalized_windows,
        sensor_names,
        gdn_predictions=gdn_predictions,
        distribution_thresholds=distribution_thresholds,
    )

    # Map features to labels
    print("Mapping features to labels...")
    labels = map_features_to_labels(
        features,
        sensor_names,
        gdn_predictions=gdn_predictions,
        distribution_thresholds=distribution_thresholds,
    )

    # Generate descriptions
    print("Generating descriptions...")
    descriptions = {}
    index_data = {}

    for window_idx in range(num_windows):
        window_features = features[window_idx]
        window_labels = labels[window_idx]
        
        # Get KG context if available
        kg_context = None
        if kg is not None and hasattr(kg, 'get_window_kg'):
            try:
                kg_context = kg.get_window_kg(window_idx)
            except Exception as e:
                print(f"  Warning: Could not get KG context for window {window_idx}: {e}")

        # Get GDN predictions for this window
        window_gdn_predictions = gdn_predictions[window_idx] if gdn_predictions is not None else None

        description = generate_window_description(
            window_idx,
            window_features,
            window_labels,
            sensor_names,
            gdn_predictions=window_gdn_predictions,
            kg_context=kg_context,
            distribution_thresholds=distribution_thresholds,
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
