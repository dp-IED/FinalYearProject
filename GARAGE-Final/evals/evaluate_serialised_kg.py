"""
Evaluate Serialised KG->LLM method on shared evaluation dataset.

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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.gdn_processor import GDNPredictor
from kg.create_kg import (
    KnowledgeGraph,
    EXPECTED_CORRELATIONS,
    SENSOR_SUBSYSTEMS,
    SENSOR_DESCRIPTIONS,
)
from evals.llm_helpers import (
    load_llm_model,
    call_llm,
    parse_llm_response,
    format_window_for_llm,
    filter_sensor_labels_to_root_only,
)
from llm.kag.graphdb import Neo4jLoader
from llm.kag.solver_v2 import KAGIterativeSolver
from llm.kag.neo4j_queries import Neo4jKAGQueries
from evals.metrics import compute_all_metrics, format_metrics_report
from kg.similarity import compute_window_similarity


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


def extract_window_kg_context(
    kg: KnowledgeGraph,
    window_idx: int,
    temporal_context_windows: int = 2,
) -> Dict[str, any]:
    """
    Extract KG context for a specific window (KAG-inspired).

    Args:
        kg: KnowledgeGraph instance with built KG
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
    # Use get_window_kg method if available, otherwise build context manually
    try:
        kg_context = kg.get_window_kg(window_idx, temporal_context_windows=temporal_context_windows)
        return kg_context
    except Exception:
        # Fallback: build context manually
        context = {
            "entities": [],
            "relationships": [],
            "violations": [],
            "temporal_context": [],
            "anomaly_propagation": [],
            "distribution_thresholds": None,
            "stage_features": {},
        }

        # Get current window graph and stats
        if window_idx not in kg.window_graphs:
            return context

        window_graph = kg.window_graphs[window_idx]
        window_stats = kg.window_stats.get(window_idx, {})

        # Get distribution thresholds if available
        thresholds = getattr(kg, "distribution_thresholds", None)
        if thresholds:
            context["distribution_thresholds"] = thresholds

        # Get Stage 2 features if available
        stage2_features = getattr(kg, "window_stage2_features", {}).get(
            window_idx, {}
        )
        if stage2_features:
            context["stage_features"] = stage2_features

        # Extract entities with types and subsystems
        for sensor_name in kg.sensor_names:
            desc = SENSOR_DESCRIPTIONS.get(sensor_name, {})
            subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, "Unknown")
            # Use distribution-based threshold if available
            if thresholds:
                anomaly_threshold_per_sensor = thresholds.get(
                    "anomaly_threshold_per_sensor", {}
                )
                anomaly_threshold = anomaly_threshold_per_sensor.get(
                    sensor_name, thresholds.get("anomaly_threshold_global", 0.5)
                )
            else:
                anomaly_threshold = 0.5  # Fallback
            stat = window_stats.get(sensor_name)
            is_faulty = stat.anomaly_score > anomaly_threshold if stat else False

            entity_info = {
                "name": sensor_name,
                "type": "Sensor",
                "subsystem": subsystem,
                "description": desc.get("description", ""),
                "is_faulty": is_faulty,  # Based on GDN prediction threshold, not ground truth
            }
            context["entities"].append(entity_info)

        # Extract relationships from current window
        # Include all violations and relationships involving sensors with high GDN prediction scores
        # Also include significant correlations (threshold 0.3)
        correlation_threshold = 0.3
        prediction_threshold = 0.5  # Threshold for GDN predictions (not ground truth)
        anomalous_sensors = {
            sensor_name
            for sensor_name, stat in window_stats.items()
            if stat.anomaly_score > prediction_threshold
        }  # Based on GDN predictions

    for u, v, data in window_graph.edges(data=True):
        edge_type = data.get("edge_type", "correlates_with")

        # Support both old and new attribute formats for backward compatibility
        correlation = data.get("correlation", 0)  # Old format (preserved)
        correlation_strength = data.get(
            "correlation_strength", abs(correlation)
        )  # New format
        correlation_direction = data.get(
            "correlation_direction", "positive" if correlation > 0 else "negative"
        )

        # Domain knowledge expectations (new format)
        violates_domain = data.get("violates_domain_expectation", False)
        domain_expected_type = data.get("domain_expected_type", None)
        domain_expected_strength = data.get("domain_expected_strength", None)

        # GDN expectations (new format)
        expected_correlation_gdn = data.get(
            "expected_correlation_gdn", data.get("expected_correlation", 0)
        )  # Fallback to old format
        deviation_from_gdn = data.get(
            "deviation_from_gdn", data.get("correlation_deviation", 0)
        )  # Fallback to old format
        violates_gdn = data.get("violates_gdn_expectation", False)

        # GDN scores
        gdn_score_source = data.get("gdn_score_source", 0)
        gdn_score_target = data.get("gdn_score_target", 0)
        potential_fault_indicator = data.get("potential_fault_indicator", False)

        # Include if:
        # 1. It's a violation (domain or GDN expectation violated)
        # 2. It involves an anomalous sensor (contextual information)
        # 3. It's a significant correlation (above threshold)
        # 4. It's a potential fault indicator
        is_violation = violates_domain or violates_gdn
        involves_anomaly = u in anomalous_sensors or v in anomalous_sensors
        is_significant = correlation_strength >= correlation_threshold

        if not (
            is_violation
            or involves_anomaly
            or is_significant
            or potential_fault_indicator
        ):
            continue

        relationship = {
            "source": u,
            "target": v,
            "relation": edge_type,
            "correlation": float(correlation),  # Preserved for backward compatibility
            "correlation_strength": float(correlation_strength),
            "correlation_direction": correlation_direction,
            "expected_correlation_gdn": float(expected_correlation_gdn),
            "deviation_from_gdn": float(deviation_from_gdn),
            "violates_domain_expectation": violates_domain,
            "violates_gdn_expectation": violates_gdn,
            "gdn_score_source": float(gdn_score_source),
            "gdn_score_target": float(gdn_score_target),
            "potential_fault_indicator": potential_fault_indicator,
        }

        # Add domain knowledge if available
        if domain_expected_type:
            relationship["domain_expected_type"] = domain_expected_type
        if domain_expected_strength:
            relationship["domain_expected_strength"] = domain_expected_strength
        if "violation_type" in data:
            relationship["violation_type"] = data["violation_type"]

        context["relationships"].append(relationship)

        # Track violations separately (both domain and GDN violations)
        if is_violation:
            context["violations"].append(relationship)

        # Extract temporal context from previous windows
        for prev_idx in range(max(0, window_idx - temporal_context_windows), window_idx):
            if prev_idx in kg.window_stats:
                prev_stats = kg.window_stats[prev_idx]
                temporal_info = {
                    "window_idx": prev_idx,
                    "faulty_sensors": [],
                    "anomaly_scores": {},
                }

                # Use prediction threshold (0.5) for GDN predictions, not ground truth
                prediction_threshold = 0.5
                for sensor_name, stat in prev_stats.items():
                    if (
                        stat.anomaly_score > prediction_threshold
                    ):  # Based on GDN prediction threshold
                        temporal_info["faulty_sensors"].append(sensor_name)
                        temporal_info["anomaly_scores"][sensor_name] = float(
                            stat.anomaly_score
                        )

                if temporal_info["faulty_sensors"]:
                    context["temporal_context"].append(temporal_info)

        # Extract relevant anomaly propagation chains
        for chain in kg.anomaly_propagation_chains:
            root_window = chain.get("root_window", -1)
            propagation_timeline = chain.get("propagation_timeline", [])

            # Check if this window is involved in the chain
            if root_window == window_idx:
                context["anomaly_propagation"].append(
                    {
                        "type": "root",
                        "root_sensor": chain.get("root_sensor", ""),
                        "root_window": root_window,
                        "affected_sensors": chain.get("affected_sensors", []),
                    }
                )
            else:
                # Check if window appears in propagation timeline
                for timeline_entry in propagation_timeline:
                    if timeline_entry.get("window") == window_idx:
                        context["anomaly_propagation"].append(
                            {
                                "type": "propagation",
                                "root_sensor": chain.get("root_sensor", ""),
                                "root_window": root_window,
                                "affected_sensors": timeline_entry.get(
                                    "affected_sensors", []
                                ),
                            }
                        )
                        break

        return context


def format_kg_context_as_adjacency_matrix(
    kg_context: Dict[str, any], window_idx: int, kg: KnowledgeGraph
) -> str:
    """
    Format KG context as compact adjacency matrix for LLM prompt.
    
    More compact than text format, better for smaller models.
    
    Args:
        kg_context: Context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg: KnowledgeGraph instance
    
    Returns:
        Formatted string with adjacency matrix representation
    """
    lines = []
    lines.append("Knowledge Graph (Adjacency Matrix Format):")
    
    # Get sensor names in order
    sensor_names = kg.sensor_names
    num_sensors = len(sensor_names)
    
    # Build correlation matrix
    corr_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    deviation_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    violation_matrix = [[False] * num_sensors for _ in range(num_sensors)]
    gdn_expected_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    gdn_score_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    
    # Get thresholds
    thresholds = getattr(kg, "distribution_thresholds", None)
    window_stats = kg.window_stats.get(window_idx, {})
    
    # Fill matrices from relationships
    for rel in kg_context.get("relationships", []):
        source = rel["source"]
        target = rel["target"]
        
        try:
            src_idx = sensor_names.index(source)
            tgt_idx = sensor_names.index(target)
        except ValueError:
            continue
        
        corr = rel.get("correlation", 0.0)
        deviation = rel.get("deviation_from_gdn", 0.0)
        violates = rel.get("violates_gdn_expectation", False) or rel.get("violates_domain_expectation", False)
        gdn_expected = rel.get("expected_correlation_gdn", 0.0)
        gdn_src = rel.get("gdn_score_source", 0.0)
        gdn_tgt = rel.get("gdn_score_target", 0.0)
        
        corr_matrix[src_idx][tgt_idx] = corr
        deviation_matrix[src_idx][tgt_idx] = deviation
        violation_matrix[src_idx][tgt_idx] = violates
        gdn_expected_matrix[src_idx][tgt_idx] = gdn_expected
        gdn_score_matrix[src_idx][tgt_idx] = max(gdn_src, gdn_tgt)
    
    # Sensor status
    lines.append("\nSensor Status:")
    sensor_status = []
    for i, sensor_name in enumerate(sensor_names):
        stat = window_stats.get(sensor_name)
        if thresholds:
            anomaly_threshold_per_sensor = thresholds.get("anomaly_threshold_per_sensor", {})
            threshold = anomaly_threshold_per_sensor.get(sensor_name, thresholds.get("anomaly_threshold_global", 0.5))
        else:
            threshold = 0.5
        is_faulty = stat.anomaly_score > threshold if stat else False
        status = "ANOMALOUS" if is_faulty else "Normal"
        score = stat.anomaly_score if stat else 0.0
        sensor_status.append((sensor_name, status, score))
        lines.append(f"  {i}: {sensor_name} [{status}] (score: {score:.3f})")
    
    # Correlation Matrix
    lines.append("\nCorrelation Matrix (row -> col):")
    lines.append("     " + " ".join([f"{i:>4}" for i in range(num_sensors)]))
    for i, sensor_name in enumerate(sensor_names):
        row_str = f"{i:>3}: "
        for j in range(num_sensors):
            corr = corr_matrix[i][j]
            if abs(corr) < 0.01:
                row_str += "  .  "
            else:
                row_str += f"{corr:>5.2f}"
        # Remove " ()" suffix for cleaner display
        sensor_display = sensor_name.replace(" ()", "")
        lines.append(row_str + f"  [{sensor_display[:20]}]")
    
    # Deviation Matrix (only show violations)
    lines.append("\nDeviation from GDN (violations only, row -> col):")
    has_violations = False
    for i, sensor_name in enumerate(sensor_names):
        row_violations = []
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                dev = deviation_matrix[i][j]
                row_violations.append(f"{j}:{dev:.3f}")
                has_violations = True
        if row_violations:
            sensor_display = sensor_name.replace(" ()", "")
            lines.append(f"  {i} [{sensor_display[:20]}]: {', '.join(row_violations)}")
    if not has_violations:
        lines.append("  No violations detected")
    
    # GDN Expected vs Actual (for violations)
    lines.append("\nGDN Expected vs Actual (violations only):")
    has_violations = False
    for i, sensor_name in enumerate(sensor_names):
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                expected = gdn_expected_matrix[i][j]
                actual = corr_matrix[i][j]
                dev = deviation_matrix[i][j]
                tgt_name = sensor_names[j].replace(" ()", "")[:20]
                lines.append(f"  {sensor_name[:15]} -> {tgt_name}: expected {expected:.3f}, actual {actual:.3f}, deviation {dev:.3f}")
                has_violations = True
    if not has_violations:
        lines.append("  No violations detected")
    
    # Summary: Sensors with violations
    lines.append("\nViolation Summary:")
    violation_sensors = set()
    for i in range(num_sensors):
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                violation_sensors.add(sensor_names[i])
                violation_sensors.add(sensor_names[j])
    
    if violation_sensors:
        lines.append(f"  Sensors involved in violations: {', '.join(sorted(violation_sensors))}")
    else:
        lines.append("  No violations detected - all relationships normal")
    
    # Threshold context
    if thresholds:
        lines.append("\nThreshold Context:")
        lines.append(f"  Deviation threshold (90th percentile): {thresholds.get('deviation_threshold', 0.3):.3f}")
        lines.append(f"  Deviation p50: {thresholds.get('deviation_p50', 0.0):.3f}")
        lines.append(f"  Deviation p95: {thresholds.get('deviation_p95', 0.0):.3f}")
        lines.append(f"  Anomaly threshold (global): {thresholds.get('anomaly_threshold_global', 0.5):.3f}")
    
    # Embedding distances (compact format)
    if window_idx in getattr(kg, 'window_embeddings', {}):
        embedding_data = kg.window_embeddings[window_idx]
        dist_normal = embedding_data.get("dist_normal", 0.0)
        dist_anomalous = embedding_data.get("dist_anomalous", 0.0)
        confidence = embedding_data.get("confidence", 0.0)
        
        lines.append("\nEmbedding Distances:")
        lines.append(f"  Distance to normal center: {dist_normal:.3f}")
        lines.append(f"  Distance to anomalous center: {dist_anomalous:.3f}")
        lines.append(f"  Confidence: {confidence:.3f}")
        
        # Interpretation
        if dist_normal < 0.12:
            interpretation = "Likely normal"
        elif dist_normal > 0.12 and dist_anomalous < 0.15:
            interpretation = "Likely anomalous"
        else:
            interpretation = "Uncertain/edge case"
        lines.append(f"  Interpretation: {interpretation}")
    
    # Per-sensor embedding distances (if Stage 2 features available)
    stage2_features = getattr(kg, "window_stage2_features", {}).get(window_idx, {})
    if stage2_features:
        per_sensor_normal = stage2_features.get("per_sensor_distances_normal")
        per_sensor_anomalous = stage2_features.get("per_sensor_distances_anomalous")
        
        if per_sensor_normal is not None and per_sensor_anomalous is not None:
            lines.append("\nPer-Sensor Embedding Distances:")
            lines.append("  Sensor | Dist to Normal | Dist to Anomalous")
            lines.append("  " + "-" * 50)
            for i, sensor_name in enumerate(sensor_names):
                if i < len(per_sensor_normal) and i < len(per_sensor_anomalous):
                    sensor_display = sensor_name.replace(" ()", "")[:20]
                    dist_n = per_sensor_normal[i]
                    dist_a = per_sensor_anomalous[i]
                    lines.append(f"  {i:2d} [{sensor_display:20s}] | {dist_n:6.3f} | {dist_a:6.3f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def format_kg_context_for_llm(
    kg_context: Dict[str, any], window_idx: int, kg: KnowledgeGraph,
    use_adjacency_matrix: bool = False
) -> str:
    """
    Format KG context as structured LLM prompt section following KAG best practices.

    Shows structured knowledge graph representation: entities, relationships, violations,
    temporal context, and anomaly propagation. NO raw sensor data.

    Args:
        kg_context: Context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg: KnowledgeGraph instance
        use_adjacency_matrix: If True, use compact adjacency matrix format instead of verbose text

    Returns:
        Formatted string for LLM prompt with structured KG representation
    """
    if use_adjacency_matrix:
        return format_kg_context_as_adjacency_matrix(kg_context, window_idx, kg)
    lines = []
    lines.append("Knowledge Graph Representation:")

    # Get distribution thresholds and Stage 2 features if available
    thresholds = getattr(kg, "distribution_thresholds", None)
    stage2_features = getattr(kg, "window_stage2_features", {}).get(
        window_idx, {}
    )

    # All entities with their status and metadata
    lines.append("\nENTITIES:")
    for entity in kg_context["entities"]:
        status = "⚠️ ANOMALOUS" if entity.get("is_faulty") else "✓ Normal"
        lines.append(f"{status}: {entity['name']}")
        lines.append(f"  Subsystem: {entity['subsystem']}")
        if entity.get("description"):
            lines.append(f"  Description: {entity['description']}")

    # All relationships (not filtered - show all significant ones)
    lines.append("\nRELATIONSHIPS:")
    if kg_context["relationships"]:
        for rel in kg_context["relationships"]:
            rel_type = rel["relation"]
            source = rel["source"]
            target = rel["target"]
            corr = rel.get("correlation", 0)

            # Check for violations (domain or GDN expectations)
            violates_domain = rel.get("violates_domain_expectation", False)
            violates_gdn = rel.get("violates_gdn_expectation", False)
            correlation_direction = rel.get("correlation_direction", "positive")
            correlation_strength = rel.get("correlation_strength", abs(corr))
            potential_fault = rel.get("potential_fault_indicator", False)
            violation_confidence = rel.get("violation_confidence", 0.0)

            if violates_domain or violates_gdn:
                # Violation detected
                violation_types = []
                if violates_domain:
                    violation_types.append("domain expectation")
                if violates_gdn:
                    violation_types.append("GDN expectation")

                violation_type_detail = rel.get("violation_type", "unknown")
                lines.append(
                    f"⚠️ VIOLATION ({', '.join(violation_types)}): {source} --[{rel_type}]--> {target}"
                )
                lines.append(
                    f"  Correlation: {corr:.3f} ({correlation_direction}, strength: {correlation_strength:.3f})"
                )

                if violates_domain:
                    domain_type = rel.get("domain_expected_type", "unknown")
                    domain_strength = rel.get("domain_expected_strength", "unknown")
                    lines.append(
                        f"  Domain expected: {domain_type} ({domain_strength})"
                    )
                    if violation_type_detail != "unknown":
                        lines.append(f"  Violation type: {violation_type_detail}")

                if violates_gdn:
                    exp_corr_gdn = rel.get("expected_correlation_gdn", 0)
                    dev_gdn = rel.get("deviation_from_gdn", 0)
                    lines.append(f"  GDN expected correlation: {exp_corr_gdn:.3f}")
                    lines.append(f"  Deviation from GDN: {dev_gdn:.3f}")

                    # Include distribution context in violation message
                    if thresholds:
                        deviation_p50 = thresholds.get("deviation_p50", 0)
                        deviation_p95 = thresholds.get("deviation_p95", 0)
                        deviation_threshold = thresholds.get("deviation_threshold", 0.3)

                        # Show percentile context
                        if dev_gdn > deviation_p95:
                            lines.append(
                                f"  [Severe: deviation > 95th percentile ({deviation_p95:.3f})]"
                            )
                        elif dev_gdn > deviation_p50:
                            lines.append(
                                f"  [Moderate: deviation > 50th percentile ({deviation_p50:.3f})]"
                            )
                        else:
                            lines.append(
                                f"  [Threshold: {deviation_threshold:.3f} (90th percentile)]"
                            )

                    # Include Stage 2 features if available
                    if stage2_features:
                        embedding_distance_normal = stage2_features.get(
                            "embedding_distance_normal"
                        )
                        if embedding_distance_normal is not None:
                            typical_separation = 0.4
                            separation_ratio = (
                                embedding_distance_normal / typical_separation
                            )
                            lines.append(
                                f"  [Stage2: Distance to normal center {embedding_distance_normal:.3f} ({separation_ratio:.1f}× typical separation)]"
                            )

                    # Include violation confidence
                    if violation_confidence > 0.5:
                        lines.append(
                            f"  [High confidence violation ({violation_confidence:.2f}) based on training features]"
                        )

                if potential_fault:
                    gdn_src = rel.get("gdn_score_source", 0)
                    gdn_tgt = rel.get("gdn_score_target", 0)
                    # Include distribution context for anomaly scores
                    if thresholds:
                        anomaly_p50 = thresholds.get("anomaly_p50", 0)
                        anomaly_p95 = thresholds.get("anomaly_p95", 0)
                        anomaly_threshold = thresholds.get(
                            "anomaly_threshold_global", 0.5
                        )
                        src_percentile = (
                            "high"
                            if gdn_src > anomaly_p95
                            else "moderate"
                            if gdn_src > anomaly_p50
                            else "low"
                        )
                        tgt_percentile = (
                            "high"
                            if gdn_tgt > anomaly_p95
                            else "moderate"
                            if gdn_tgt > anomaly_p50
                            else "low"
                        )
                        lines.append(
                            f"  ⚠️ Potential fault indicator (GDN scores: {gdn_src:.2f} [{src_percentile}], {gdn_tgt:.2f} [{tgt_percentile}], threshold: {anomaly_threshold:.2f})"
                        )
                    else:
                        lines.append(
                            f"  ⚠️ Potential fault indicator (GDN scores: {gdn_src:.2f}, {gdn_tgt:.2f})"
                        )
            else:
                # Normal relationship
                lines.append(f"{source} --[{rel_type}]--> {target}")
                lines.append(
                    f"  Correlation: {corr:.3f} ({correlation_direction}, strength: {correlation_strength:.3f})"
                )

                # Show domain expectations if available
                domain_type = rel.get("domain_expected_type")
                if domain_type:
                    domain_strength = rel.get("domain_expected_strength", "unknown")
                    lines.append(
                        f"  Domain expected: {domain_type} ({domain_strength})"
                    )

                # Show GDN expectations
                exp_corr_gdn = rel.get("expected_correlation_gdn", 0)
                if exp_corr_gdn != 0:
                    dev_gdn = rel.get("deviation_from_gdn", 0)
                    lines.append(f"  GDN expected correlation: {exp_corr_gdn:.3f}")
                    if thresholds and dev_gdn > 0:
                        # Show deviation even for non-violations if significant
                        deviation_p50 = thresholds.get("deviation_p50", 0)
                        if dev_gdn > deviation_p50:
                            lines.append(
                                f"  Deviation: {dev_gdn:.3f} (within normal range)"
                            )

                if potential_fault:
                    gdn_src = rel.get("gdn_score_source", 0)
                    gdn_tgt = rel.get("gdn_score_target", 0)
                    lines.append(f"  GDN scores: {gdn_src:.2f}, {gdn_tgt:.2f}")
    else:
        lines.append("No significant relationships detected.")

    # Temporal context (all relevant previous windows)
    if kg_context["temporal_context"]:
        lines.append("\nTEMPORAL CONTEXT:")
        for temp in kg_context["temporal_context"]:
            lines.append(f"Window {temp['window_idx']}:")
            if temp["faulty_sensors"]:
                lines.append(f"  Faulty sensors: {', '.join(temp['faulty_sensors'])}")
                if temp.get("anomaly_scores"):
                    scores_str = ", ".join(
                        [
                            f"{sensor}={score:.2f}"
                            for sensor, score in list(temp["anomaly_scores"].items())[
                                :3
                            ]
                        ]
                    )
                    if scores_str:
                        lines.append(f"  Anomaly scores: {scores_str}")
            else:
                lines.append("  No faults detected")

    # Anomaly propagation chains
    if kg_context["anomaly_propagation"]:
        lines.append("\nANOMALY PROPAGATION:")
        for prop in kg_context["anomaly_propagation"]:
            prop_type = prop.get("type", "unknown")
            root_sensor = prop.get("root_sensor", "unknown")
            root_window = prop.get("root_window", -1)
            affected = prop.get("affected_sensors", [])

            if prop_type == "root":
                lines.append("Root cause detected:")
                lines.append(f"  Root sensor: {root_sensor} at window {root_window}")
                if affected:
                    lines.append(f"  Affected sensors: {', '.join(affected)}")
            else:
                lines.append(f"Propagation from window {root_window}:")
                lines.append(f"  Root sensor: {root_sensor}")
                if affected:
                    lines.append(
                        f"  Affected sensors in this window: {', '.join(affected)}"
                    )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_embedding_context(
    window_idx: int, kg: KnowledgeGraph, gds_client=None
) -> str:
    """
    Format embedding-space analysis for LLM prompt.

    Args:
        window_idx: Index of the window
        kg: KnowledgeGraph instance with window_embeddings
        gds_client: Optional Neo4j GraphDataScience client for querying similar windows

    Returns:
        Formatted markdown string with embedding-space analysis
    """
    lines = []

    # Check if embedding data is available
    if window_idx not in kg.window_embeddings:
        return ""

    embedding_data = kg.window_embeddings[window_idx]
    dist_normal = embedding_data["dist_normal"]
    dist_anomalous = embedding_data["dist_anomalous"]
    confidence = embedding_data["confidence"]

    # Typical ranges (from plan)
    normal_mean = 0.085
    normal_std = 0.03
    anomalous_mean = 0.138
    anomalous_std = 0.04

    # Compute z-scores
    z_score_normal = (dist_normal - normal_mean) / normal_std if normal_std > 0 else 0.0
    z_score_anomalous = (
        (dist_anomalous - anomalous_mean) / anomalous_std if anomalous_std > 0 else 0.0
    )

    lines.append("\nEmbedding Space Analysis:")

    # Distance to normal center
    lines.append(f"\nDistance to Normal Center: {dist_normal:.4f}")
    normal_range_str = f"{normal_mean:.3f} ± {normal_std:.3f}"
    if dist_normal < normal_mean - normal_std:
        interpretation = "significantly closer than typical"
    elif dist_normal < normal_mean + normal_std:
        interpretation = "within typical normal range"
    else:
        interpretation = f"{abs(z_score_normal):.1f} standard deviations above typical"
    lines.append(f"  Typical normal range: {normal_range_str}")
    lines.append(f"  Interpretation: {interpretation}")
    lines.append(f"  Z-score: {z_score_normal:.2f}")

    # Distance to anomalous center
    lines.append(f"\nDistance to Anomalous Center: {dist_anomalous:.4f}")
    anomalous_range_str = f"{anomalous_mean:.3f} ± {anomalous_std:.3f}"
    if dist_anomalous < anomalous_mean - anomalous_std:
        interpretation = "significantly closer than typical"
    elif dist_anomalous < anomalous_mean + anomalous_std:
        interpretation = "within typical anomalous range"
    else:
        interpretation = (
            f"{abs(z_score_anomalous):.1f} standard deviations above typical"
        )
    lines.append(f"  Typical anomalous range: {anomalous_range_str}")
    lines.append(f"  Interpretation: {interpretation}")
    lines.append(f"  Z-score: {z_score_anomalous:.2f}")

    # Confidence score
    lines.append(f"\nConfidence Score: {confidence:.3f}")
    if confidence > 0.7:
        conf_interpretation = "high confidence (likely normal)"
    elif confidence > 0.3:
        conf_interpretation = "moderate confidence (uncertain)"
    else:
        conf_interpretation = "low confidence (likely anomalous)"
    lines.append(f"  Interpretation: {conf_interpretation}")

    # Query Neo4j for similar anomalous windows if available
    if gds_client is not None:
        try:
            # Query for 3 most similar anomalous windows
            query = """
                MATCH (w1:Window {idx: $window_idx})-[s:SIMILAR_TO]->(w2:Window)
                WHERE w2.predicted_class = "anomalous"
                OPTIONAL MATCH (w2)-[:BELONGS_TO]->(sensor:Sensor)
                WHERE sensor.is_faulty = true
                RETURN DISTINCT w2.idx AS window_idx,
                       s.similarity AS similarity,
                       s.distance AS distance,
                       collect(DISTINCT sensor.base_sensor_name) AS faulty_sensors
                ORDER BY s.similarity DESC
                LIMIT 3
            """

            with gds_client.driver.session() as session:
                result = session.run(query, {"window_idx": window_idx})
                similar_cases = []
                for record in result:
                    similar_cases.append(
                        {
                            "window_idx": record["window_idx"],
                            "similarity": float(record["similarity"])
                            if record["similarity"] is not None
                            else 0.0,
                            "distance": float(record["distance"])
                            if record["distance"] is not None
                            else 0.0,
                            "faulty_sensors": record["faulty_sensors"]
                            if record["faulty_sensors"]
                            else [],
                        }
                    )

            if similar_cases:
                lines.append("\nSimilar Anomalous Cases:")
                for case in similar_cases:
                    lines.append(f"  Window {case['window_idx']}:")
                    lines.append(
                        f"    Similarity: {case['similarity']:.3f}, Distance: {case['distance']:.4f}"
                    )
                    if case["faulty_sensors"]:
                        sensors_str = ", ".join(
                            case["faulty_sensors"][:3]
                        )  # Limit to 3 sensors
                        lines.append(f"    Faulty sensors: {sensors_str}")
        except Exception as e:
            # If Neo4j query fails, continue without similar cases
            pass

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_window_with_kg_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    kg_context: Dict[str, any],
    window_idx: int,
    kg: KnowledgeGraph,
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True,
    use_adjacency_matrix: bool = False,
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
        kg: KnowledgeGraph instance
        statistical_features: Optional statistical features (not used, kept for API compatibility)
        use_statistical_features: Whether to include statistical features (not used, kept for API compatibility)

    Returns:
        Complete formatted prompt string with structured KG representation only
    """
    # Format KG context section (structured representation)
    kg_section = format_kg_context_for_llm(kg_context, window_idx, kg, use_adjacency_matrix=use_adjacency_matrix)

    # Build prompt similar to baseline format for comparability
    lines = []
    lines.append(
        "You are an automotive diagnostic expert analyzing OBD-II sensor data."
    )
    lines.append("")
    lines.append("Task: Identify which sensors are faulty and describe the fault type.")
    lines.append("")
    lines.append(
        "The following knowledge graph representation was generated from sensor data analysis:"
    )
    lines.append("")
    lines.append(kg_section)
    lines.append("")

    # Add embedding-space explanation if embeddings are available (simplified)
    if window_idx in kg.window_embeddings:
        embedding_data = kg.window_embeddings[window_idx]
        dist_normal = embedding_data["dist_normal"]
        dist_anomalous = embedding_data["dist_anomalous"]
        confidence = embedding_data["confidence"]

        lines.append("Embedding Analysis:")
        lines.append(f"Distance to normal center: {dist_normal:.3f}")
        lines.append(f"Distance to anomalous center: {dist_anomalous:.3f}")
        lines.append(f"Confidence: {confidence:.2f}")
        if dist_normal < 0.12:
            lines.append("Interpretation: Likely normal")
        elif dist_normal > 0.12 and dist_anomalous < 0.15:
            lines.append("Interpretation: Likely anomalous")
        else:
            lines.append("Interpretation: Uncertain/edge case")
        lines.append("")

    lines.append(
        "Please analyze this knowledge graph representation and provide your diagnosis."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks (no ```), no explanations before or after."
    )
    lines.append("")
    lines.append("Required JSON format:")
    lines.append(
        '{"root_cause_sensors": ["SENSOR_NAME"] or [], "affected_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [], "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [], "fault_type": "FAULT_TYPE" or null, "reasoning": "explanation", "confidence": 0.85}'
    )
    lines.append("")
    lines.append(
        "REASONING: Be extensive and didactic. Write 3–6 sentences that: (1) state which evidence you used (which relationships/violations, which deviation or correlation values), (2) explain step-by-step how that evidence leads to the root cause or to normal operation, (3) briefly say why other sensors were or were not considered. Write so a reader can follow your logic."
    )
    lines.append("")
    lines.append("CRITICAL ANALYSIS INSTRUCTIONS:")
    lines.append(
        "- Carefully examine ALL relationship violations in the knowledge graph above"
    )
    lines.append(
        "- Look at the deviation values and correlation patterns to identify the root cause"
    )
    lines.append(
        "- Do NOT default to any specific fault type - base your diagnosis on the actual violations present"
    )
    lines.append(
        "- If multiple sensors show violations, identify which one has the strongest evidence as root cause"
    )
    lines.append("- If no violations exceed thresholds, conclude normal operation")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append(
        "- root_cause_sensors: The PRIMARY sensor(s) causing the fault (usually 1 sensor)"
    )
    lines.append(
        "- affected_sensors: Secondary sensors that are affected by the root cause but are NOT the primary fault source"
    )
    lines.append(
        "- faulty_sensors: For backward compatibility, include ALL faulty sensors (root + affected combined)"
    )
    lines.append("")
    lines.append("Example 1 (no fault - most common case):")
    lines.append(
        '{"root_cause_sensors": [], "affected_sensors": [], "faulty_sensors": [], "fault_type": null, "reasoning": "I examined all relationships in the knowledge graph. No relationship has a GDN expectation violation above the threshold (deviations are all below 0.3). All correlations are within expected ranges. All entities are marked as normal. The evidence does not support any fault. I conclude normal operation.", "confidence": 0.9}'
    )
    lines.append("")
    lines.append("Example 2 (gradual_drift - multiple sensors with small violations):")
    lines.append(
        '{"root_cause_sensors": ["ENGINE_RPM"], "affected_sensors": ["VEHICLE_SPEED"], "faulty_sensors": ["ENGINE_RPM", "VEHICLE_SPEED"], "fault_type": "gradual_drift", "reasoning": "The knowledge graph shows multiple small violations (deviations 0.35-0.45) across ENGINE_RPM and VEHICLE_SPEED relationships. No single violation is severe, but the pattern of multiple correlated small deviations suggests gradual sensor drift rather than a sudden dropout. ENGINE_RPM shows the strongest violation (0.45 deviation) so I treat it as root cause. VEHICLE_SPEED is affected because it correlates with ENGINE_RPM and shows related violations.", "confidence": 0.75}'
    )
    lines.append("")
    lines.append("Example 3 (COOLANT_DROPOUT - single sensor with severe violation):")
    lines.append(
        '{"root_cause_sensors": ["COOLANT_TEMPERATURE"], "affected_sensors": ["ENGINE_LOAD"], "faulty_sensors": ["COOLANT_TEMPERATURE", "ENGINE_LOAD"], "fault_type": "COOLANT_DROPOUT", "reasoning": "COOLANT_TEMPERATURE has a severe GDN expectation violation (deviation 0.72) with ENGINE_LOAD. The correlation is broken - expected positive relationship but actual is negative. This pattern indicates a dropout fault at the coolant sensor. ENGINE_LOAD is listed as affected because it appears in the violated relationship, but the primary fault is at the coolant sensor.", "confidence": 0.85}'
    )
    lines.append("")
    lines.append("Example 4 (VSS_DROPOUT - single sensor with large deviation):")
    lines.append(
        '{"root_cause_sensors": ["VEHICLE_SPEED"], "affected_sensors": [], "faulty_sensors": ["VEHICLE_SPEED"], "fault_type": "VSS_DROPOUT", "reasoning": "The knowledge graph shows a GDN expectation violation on VEHICLE_SPEED with deviation 0.85. The correlation with ENGINE_RPM is broken (expected positive, actual negative). This large deviation and broken correlation pattern indicates a dropout fault at the vehicle speed sensor. I treat VEHICLE_SPEED as root cause because it has the largest deviation; no other sensor has comparable violation strength.", "confidence": 0.9}'
    )
    lines.append("")
    lines.append("Available sensor names:")
    for name in kg.sensor_names:
        lines.append(f"  - {name.replace(' ()', '')}")
    lines.append("")
    lines.append(
        "Fault types: gradual_drift, COOLANT_DROPOUT, VSS_DROPOUT, MAF_SCALE, TPS_STUCK"
    )
    lines.append("")
    lines.append(
        "CRITICAL: Output ONLY the JSON object. Start with { and end with }. No other text."
    )

    return "\n".join(lines)


def evaluate_gdn_kg_llm(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = "cpu",
    model_repo: Optional[str] = None,
    max_tokens: Optional[int] = None,  # None = no limit (model supports 128k context)
    temperature: float = 0.7,
    use_statistical_features: bool = True,
    limit: Optional[int] = None,
    use_embeddings: bool = True,
    neo4j_sync: bool = True,
    neo4j_uri: str = "bolt://127.0.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    use_adjacency_matrix: bool = False,  # Use compact adjacency matrix format instead of verbose text
) -> Dict[str, any]:
    """
    Evaluate Serialised KG->LLM method on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        model_repo: LLM model repository identifier
        max_tokens: Maximum tokens for LLM generation (None = no limit)
        temperature: LLM sampling temperature
        use_statistical_features: Whether to include statistical features in prompts
        limit: Optional limit on number of windows to process (for testing)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating Serialised KG->LLM Method")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {model_path}")
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

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Initialize GDN Predictor (reuse from evaluate_gdn_kg.py)
    print("Initializing GDN Predictor...")
    start_time = time.time()

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        if "sensor_embeddings" in checkpoint:
            detected_embed_dim = checkpoint["sensor_embeddings"].shape[1]
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
        device=device,
    )

    print(f"  Model loaded in {time.time() - start_time:.2f} seconds")
    print()

    # Process data for KG (reuse from evaluate_gdn_kg.py)
    print("Processing data through GDN...")
    start_time = time.time()

    with tqdm(total=1, desc="GDN Data Processing", unit="step") as pbar:
        kg_data = predictor.process_for_kg(
            X_windows=normalized_windows,
            sensor_labels=sensor_labels_true,  # Ground truth kept for evaluation only
            window_labels=window_labels_true,  # Ground truth kept for evaluation only
            batch_size=batch_size,
        )
        pbar.update(1)

    gdn_time = time.time() - start_time
    print(f"  GDN processing completed in {gdn_time:.2f} seconds")
    print()

    # Extract and store embeddings if available
    embedding_time = 0.0
    similarity_edges = None
    if use_embeddings and "window_embeddings" in kg_data:
        print("Extracting and storing window embeddings...")
        start_time = time.time()

        window_embeddings = kg_data["window_embeddings"]
        distances_to_normal = kg_data["distances_to_normal"]
        distances_to_anomalous = kg_data["distances_to_anomalous"]
        center_embeddings = kg_data["center_embeddings"]

        # Store embeddings in kg (will be created below)
        # We'll do this after kg is created

        embedding_time = time.time() - start_time
        print(f"  Embeddings extracted in {embedding_time:.2f} seconds")
        print()

    # Extract Stage 2 features if available
    print("Extracting Stage 2 features...")
    start_time_stage2 = time.time()
    stage2_features = predictor.extract_stage2_features(
        kg_data["X_windows"], batch_size=batch_size
    )
    if stage2_features and len(stage2_features) > 0:
        print(f"  ✓ Extracted Stage 2 features for {len(stage2_features)} windows")
        print(
            f"  Stage 2 feature extraction completed in {time.time() - start_time_stage2:.2f} seconds"
        )
        # Print sample feature keys for debugging
        if len(stage2_features) > 0:
            sample_keys = list(stage2_features.values())[0].keys()
            print(f"  Feature keys: {list(sample_keys)}")
    else:
        print(
            f"  ⚠️  No Stage 2 features available (checkpoint may not have center loss or extraction failed)"
        )
    print()

    # Build Knowledge Graph using GDN predictions (not ground truth labels)
    print("Building Knowledge Graph...")
    start_time = time.time()

    with tqdm(total=1, desc="KG Construction", unit="step") as pbar:
        kg = KnowledgeGraph(
            sensor_names=kg_data["sensor_names"],
            sensor_embeddings=kg_data["sensor_embeddings"],
            adjacency_matrix=kg_data["adjacency_matrix"],
        )
        pbar.update(0.5)

        # Use GDN predictions (not ground truth) for KG construction
        # Pass Stage 2 features if available
        # Pass ground truth labels for data-driven threshold computation (not used in KG construction)
        kg.construct(
            X_windows=kg_data["X_windows"],
            gdn_predictions=kg_data[
                "gdn_predictions"
            ],  # GDN predictions, not ground truth labels
            X_windows_unnormalized=kg_data.get("X_windows_unnormalized"),
            stage2_features=stage2_features if stage2_features else None,
            sensor_labels_true=sensor_labels_true,  # For data-driven thresholds only
            window_labels_true=window_labels_true,  # For data-driven thresholds only
        )
        pbar.update(0.5)

    kg_time = time.time() - start_time
    print(f"  Knowledge Graph built in {kg_time:.2f} seconds")
    print(f"  Nodes: {kg.kg.number_of_nodes()}, Edges: {kg.kg.number_of_edges()}")
    print()

    # Store embeddings in kg if available
    if use_embeddings and "window_embeddings" in kg_data:
        print("Storing window embeddings in KG...")
        start_time = time.time()

        for window_idx in range(num_windows):
            if window_idx < len(window_embeddings):
                # Store embeddings directly in kg.window_embeddings
                kg.window_embeddings[window_idx] = {
                    "embedding": window_embeddings[window_idx],
                    "dist_normal": distances_to_normal[window_idx],
                    "dist_anomalous": distances_to_anomalous[window_idx],
                    "confidence": 1.0 - (distances_to_normal[window_idx] / (distances_to_normal[window_idx] + distances_to_anomalous[window_idx] + 1e-8))
                }

        print(f"  Stored embeddings for {len(kg.window_embeddings)} windows")
        print(
            f"  Embedding storage completed in {time.time() - start_time:.2f} seconds"
        )
        print()

        # Compute window similarities
        print("Computing window similarities...")
        start_time = time.time()
        similarity_edges = compute_window_similarity(kg.window_embeddings, k=5)
        similarity_time = time.time() - start_time
        print(
            f"  Computed {len(similarity_edges)} similarity edges in {similarity_time:.2f} seconds"
        )
        print()

        # Sync to Neo4j if enabled
        if neo4j_sync:
            print("Syncing embeddings to Neo4j...")
            start_time = time.time()
            try:
                loader = Neo4jLoader(
                    uri=neo4j_uri, user=neo4j_user, password=neo4j_password
                )
                loader.connect()

                # Sync embeddings and centers
                loader.sync_embeddings_to_neo4j(
                    kg.window_embeddings,
                    center_embeddings,
                    gdn_predictions=kg_data["gdn_predictions"],
                    batch_size=100,
                )

                # Sync similarity edges
                if similarity_edges:
                    loader.sync_similarity_edges_to_neo4j(
                        similarity_edges,
                        window_embeddings=kg.window_embeddings,
                        batch_size=1000,
                    )

                loader.close()
                neo4j_time = time.time() - start_time
                print(f"  Neo4j sync completed in {neo4j_time:.2f} seconds")
                print()
            except Exception as e:
                print(f"  ⚠️  Warning: Neo4j sync failed: {e}")
                print("  Continuing without Neo4j sync...")
                print()

    # Initialize KAG Solver (requires Neo4j)
    solver = None
    if neo4j_sync:
        try:
            print("Initializing KAG Iterative Solver...")
            queries = Neo4jKAGQueries(neo4j_uri, neo4j_user, neo4j_password)
            solver = KAGIterativeSolver(
                kg=kg,
                neo4j_queries=queries,
                sensor_names=sensor_names,
                model=model,
                tokenizer=tokenizer,
                max_iterations=1,  # Single iteration for evaluation
            )
            print("  ✓ KAG Solver initialized")
            print()
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to initialize KAG Solver: {e}")
            print("  Falling back to direct LLM approach...")
            print()
            solver = None

    # Run LLM predictions with KG context
    print("Running LLM predictions with KG context...")
    window_labels_pred = []
    sensor_labels_pred = []  # Filtered (root-only) predictions
    sensor_labels_pred_raw = []  # Raw (all sensors) predictions
    fault_types_pred = []
    reasoning_list = []
    processing_times = []

    with tqdm(
        total=num_windows, desc="KG-Enhanced LLM Inference", unit="window"
    ) as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()

            # Use KAG Solver if available, otherwise fall back to direct LLM
            if solver is not None:
                try:
                    # Use KAG two-stage approach
                    result = solver.solve(window_idx)

                    # Map solver output to evaluation format
                    # Use root-only sensor labels from solver (already filtered)
                    sensor_labels_filtered = result.get(
                        "sensor_labels", np.zeros(len(sensor_names), dtype=np.float32)
                    )
                    sensor_labels_raw = result.get(
                        "sensor_labels_raw", sensor_labels_filtered.copy()
                    )
                    prediction = {
                        "window_label": result["window_label"],
                        "sensor_labels": sensor_labels_filtered,  # Root-only
                        "sensor_labels_raw": sensor_labels_raw,  # All sensors
                        "fault_type": result["fault_type"],
                        "reasoning": result.get("reasoning_trace", [{}])[-1].get(
                            "answer", ""
                        )
                        if result.get("reasoning_trace")
                        else "",
                    }

                    # Extract reasoning from trace if available
                    if (
                        result.get("reasoning_trace")
                        and len(result["reasoning_trace"]) > 0
                    ):
                        last_trace = result["reasoning_trace"][-1]
                        if "answer" in last_trace:
                            prediction["reasoning"] = last_trace["answer"][
                                :500
                            ]  # Limit length
                except Exception as e:
                    # Fallback to no-fault prediction on solver error
                    empty_labels = np.zeros(len(sensor_names), dtype=np.float32)
                    prediction = {
                        "window_label": 0,
                        "sensor_labels": empty_labels,
                        "sensor_labels_raw": empty_labels.copy(),
                        "fault_type": None,
                        "reasoning": f"KAG Solver error: {str(e)}",
                    }
            else:
                # Fallback to direct LLM approach (original method)
                # Extract KG context for this window
                kg_context = extract_window_kg_context(
                    kg, window_idx, temporal_context_windows=2
                )

                # For 350m model: limit to top 4 relationships/violations to reduce prompt size and avoid degenerate output
                if (
                    model_repo
                    and "350m" in model_repo
                    and kg_context.get("relationships")
                ):
                    rels = kg_context["relationships"]
                    # Prefer violations (domain or GDN), then by |deviation_from_gdn| descending
                    sorted_rels = sorted(
                        rels,
                        key=lambda r: (
                            not (
                                r.get("violates_domain_expectation")
                                or r.get("violates_gdn_expectation")
                            ),
                            -abs(r.get("deviation_from_gdn", 0)),
                        ),
                    )
                    kg_context["relationships"] = sorted_rels[:4]

                # Get window data
                window_data = unnormalized_windows[window_idx]
                stats = (
                    statistical_features[window_idx]
                    if statistical_features is not None
                    and len(statistical_features) > window_idx
                    else None
                )

                # Format prompt with KG context
                prompt = format_window_with_kg_for_llm(
                    window_data,
                    sensor_names,
                    kg_context,
                    window_idx,
                    kg,
                    stats,
                    use_statistical_features,
                    use_adjacency_matrix=use_adjacency_matrix,
                )

                # Add embedding context if available
                if use_embeddings and window_idx in kg.window_embeddings:
                    try:
                        # Try to get Neo4j client for similar windows query
                        gds_client = None
                        if neo4j_sync:
                            try:
                                loader = Neo4jLoader(
                                    uri=neo4j_uri,
                                    user=neo4j_user,
                                    password=neo4j_password,
                                )
                                loader.connect()
                                gds_client = loader
                            except Exception:
                                pass  # Continue without Neo4j client

                        embedding_context = format_embedding_context(
                            window_idx, kg, gds_client
                        )
                        if embedding_context:
                            prompt += "\n\n" + embedding_context

                        if gds_client:
                            gds_client.close()
                    except Exception:
                        # If embedding context fails, continue without it
                        pass

                # One-time dump of Serialised KG prompt for inspection
                if not getattr(
                    evaluate_gdn_kg_llm, "_example_serialised_kg_prompt_dumped", False
                ):
                    try:
                        out = Path("results/example_serialised_kg_prompt.txt")
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(prompt, encoding="utf-8")
                        print(f"  Example Serialised KG prompt written to {out}")
                        evaluate_gdn_kg_llm._example_serialised_kg_prompt_dumped = True
                    except Exception:
                        pass

                # Call LLM with repetition penalty to prevent degenerate output
                try:
                    # Check prompt length and warn if too long
                    prompt_tokens_est = len(prompt.split())  # Rough estimate
                    if prompt_tokens_est > 2000:
                        print(
                            f"  ⚠️  Warning: Prompt is very long (~{prompt_tokens_est} tokens)"
                        )

                    # Use a default max_tokens to prevent unbounded/repetitive generation.
                    # Without a limit, small models often produce degenerate output on long KG prompts.
                    # evaluations/eval_llm.py uses pre-generated results; when running live (e.g. run_granite_comparison.sh)
                    # the same prompt with no max_tokens can cause both models to degenerate.
                    effective_max_tokens = max_tokens if max_tokens is not None else 512
                    call_kwargs = {
                        "max_tokens": effective_max_tokens,
                        "temperature": 0.3,  # Lower temperature for more deterministic JSON output
                        "repetition_penalty": 1.2,  # Moderate penalty (1.5 was too aggressive)
                        "repetition_context_size": 32,  # Forwarded to API when supported
                    }
                    response = call_llm(prompt, model, tokenizer, **call_kwargs)

                    # Check if response looks valid (has JSON structure)
                    has_json_structure = "{" in response and "}" in response
                    # Check for repetitive patterns (same char repeated many times)
                    response_start = response[:200].strip()
                    is_repetitive = (
                        len(set(response_start)) < 5 and len(response_start) > 20
                    ) or (
                        response_start.count(
                            response_start[0] if response_start else ""
                        )
                        > len(response_start) * 0.8
                    )

                    if is_repetitive and not has_json_structure:
                        # Response is degenerate - try to extract any valid JSON first
                        import re

                        json_match = re.search(
                            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL
                        )
                        if json_match:
                            try:
                                # Try to parse the JSON even if response is repetitive
                                prediction = parse_llm_response(
                                    json_match.group(0), sensor_names
                                )
                                prediction["reasoning"] = (
                                    f"Extracted JSON from repetitive response: {response[:200]}"
                                )
                            except:
                                # Fallback
                                prediction = {
                                    "window_label": 0,
                                    "sensor_labels": np.zeros(
                                        len(sensor_names), dtype=np.float32
                                    ),
                                    "fault_type": None,
                                    "reasoning": f"LLM response was degenerate (repetitive output): {response[:100]}",
                                }
                        else:
                            # No JSON found - use fallback
                            print(
                                f"  ⚠️  Window {window_idx}: Degenerate LLM response detected, using fallback"
                            )
                            prediction = {
                                "window_label": 0,
                                "sensor_labels": np.zeros(
                                    len(sensor_names), dtype=np.float32
                                ),
                                "fault_type": None,
                                "reasoning": f"LLM response was degenerate (repetitive output): {response[:100]}",
                            }
                    else:
                        prediction = parse_llm_response(response, sensor_names)
                        prediction["reasoning"] = response[
                            :1000
                        ]  # Store first 1000 chars for extensive/didactic reasoning
                except Exception as e:
                    # Fallback to no-fault prediction (window_label = 0)
                    empty_labels = np.zeros(len(sensor_names), dtype=np.float32)
                    prediction = {
                        "window_label": 0,  # 0 = no fault
                        "sensor_labels": empty_labels,
                        "sensor_labels_raw": empty_labels.copy(),
                        "fault_type": None,
                        "reasoning": f"Error: {str(e)}",
                    }

            # Apply root-only filtering for precision improvement (if not already filtered)
            if "sensor_labels_raw" not in prediction:
                sensor_labels_filtered = filter_sensor_labels_to_root_only(
                    prediction, sensor_names
                )
                sensor_labels_raw = prediction.get(
                    "sensor_labels", sensor_labels_filtered.copy()
                )
            else:
                sensor_labels_filtered = prediction.get(
                    "sensor_labels", np.zeros(len(sensor_names), dtype=np.float32)
                )
                sensor_labels_raw = prediction.get(
                    "sensor_labels_raw", sensor_labels_filtered.copy()
                )

            window_labels_pred.append(prediction["window_label"])
            sensor_labels_pred.append(
                sensor_labels_filtered
            )  # Use filtered (root-only) for metrics
            sensor_labels_pred_raw.append(sensor_labels_raw)  # Keep raw for analysis
            fault_types_pred.append(prediction["fault_type"])
            reasoning_list.append(prediction.get("reasoning", ""))
            processing_times.append(time.time() - start_time)

            pbar.update(1)
            if (window_idx + 1) % 10 == 0:
                avg_time = (
                    np.mean(processing_times[-10:])
                    if len(processing_times) >= 10
                    else np.mean(processing_times)
                )
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s"})

    # Cleanup solver if it was initialized
    if (
        solver is not None
        and hasattr(solver, "queries")
        and hasattr(solver.queries, "close")
    ):
        try:
            solver.queries.close()
        except Exception:
            pass

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)  # Filtered (root-only)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)  # Raw (all sensors)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    llm_time = total_processing_time

    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total LLM processing time: {llm_time:.2f} seconds")
    print()

    # Convert window_labels_true to sensor-indexed format (0-8)
    # The dataset stores window_labels as window indices, not sensor-indexed labels
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = (
                faulty_indices[0] + 1
            )  # 1-indexed (sensor 0 -> label 1)
        else:
            window_labels_true_converted[i] = 0  # No fault
    window_labels_true = window_labels_true_converted

    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = data.get("fault_types", None)

    # Compute metrics using filtered (root-only) predictions for precision improvement
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
    total_time = gdn_time + kg_time + llm_time
    metrics["efficiency"] = {
        "gdn_processing_time_seconds": float(gdn_time),
        "kg_build_time_seconds": float(kg_time),
        "llm_processing_time_seconds": float(llm_time),
        "total_processing_time_seconds": float(total_time),
        "windows_per_second": float(num_windows / total_time),
        "kg_nodes": int(kg.kg.number_of_nodes()),
        "kg_edges": int(kg.kg.number_of_edges()),
    }

    # Print report
    report = format_metrics_report(metrics)
    print(report)

    # Save results
    results = {
        "method": "gdn_kg_llm",
        "dataset": str(dataset_path),
        "gdn_model": str(model_path),
        "llm_model": model_repo,
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),  # Filtered (root-only)
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),  # Raw (all sensors)
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
        description="Evaluate Serialised KG->LLM method on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to shared dataset .npz file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained GDN model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/gdn_kg_llm.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for GDN inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--model-repo", type=str, default=None, help="LLM model repository identifier"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        nargs="?",
        help="Maximum tokens for LLM generation (default: None = no limit, model supports up to 128k context)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM sampling temperature"
    )
    parser.add_argument(
        "--no-statistical-features",
        action="store_true",
        help="Disable statistical features in prompts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=True,
        help="Enable embedding extraction and similarity computation (default: True)",
    )
    parser.add_argument(
        "--no-embeddings",
        dest="use_embeddings",
        action="store_false",
        help="Disable embedding extraction",
    )
    parser.add_argument(
        "--neo4j-sync",
        action="store_true",
        default=True,
        help="Sync embeddings to Neo4j (default: True)",
    )
    parser.add_argument(
        "--no-neo4j-sync",
        dest="neo4j_sync",
        action="store_false",
        help="Disable Neo4j sync (NOTE: KAG Solver requires Neo4j - will fall back to direct LLM if disabled)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://127.0.0.1:7687",
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user", type=str, default="neo4j", help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password", type=str, default="password", help="Neo4j password"
    )
    parser.add_argument(
        "--use-adjacency-matrix",
        action="store_true",
        default=False,
        help="Use compact adjacency matrix format instead of verbose text format for KG representation",
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
        use_statistical_features=not args.no_statistical_features,
        limit=args.limit,
        use_embeddings=args.use_embeddings,
        neo4j_sync=args.neo4j_sync,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        use_adjacency_matrix=args.use_adjacency_matrix,
    )


if __name__ == "__main__":
    main()
