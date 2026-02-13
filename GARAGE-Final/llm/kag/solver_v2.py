"""
KAG Solver v2 - LLM-Planned Iterative Reasoning

Implements Knowledge-Augmented Generation (KAG) reasoning with LLM planning.
The LLM generates logical form steps that are executed over the knowledge graph,
enabling flexible, adaptive reasoning compared to the deterministic v1 solver.

This solver uses iterative refinement: if confidence is low, it can generate
follow-up questions and refine its reasoning.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from llm.kag.neo4j_queries import Neo4jKAGQueries
from kg.create_kg import KnowledgeGraph
from evals.llm_helpers import call_llm, parse_llm_response


@dataclass
class LogicalFormStep:
    """Represents a single step in a logical form plan."""

    step_id: int
    operator: str  # "Retrieval", "Math", "Deduce", "Sort"
    params: Dict[str, Any]
    description: str


def parse_logical_form(response: str) -> List[LogicalFormStep]:
    """
    Parse LLM response into logical form steps.

    Expected format (supports both single-line and multi-line):
    Step 1: Retrieval(s="ANY", p="HAS_READING", o="ANY", constraints={...})
    Description: Find all sensors with anomaly score > 0.7

    OR multi-line format:
    Step 1: Retrieval(
        operation="has_reading",
        subject={"window_idx": 0},
        ...
    )
    Description: ...

    Args:
        response: LLM response text containing logical form steps

    Returns:
        List of LogicalFormStep objects
    """
    steps = []
    lines = response.split("\n")

    current_step_id = None
    current_operator = None
    current_params = {}
    current_description = ""
    collecting_params = False
    params_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Match step header: "Step <id>: <Operator>(...)" - handle both single-line and multi-line
        step_match = re.match(r"Step\s+(\d+):\s*(\w+)\s*\(", line)
        if step_match:
            # Save previous step if exists
            if current_step_id is not None:
                # Parse accumulated params
                if params_lines:
                    params_str = " ".join(params_lines)
                    current_params = _parse_params_string(params_str)

                steps.append(
                    LogicalFormStep(
                        step_id=current_step_id,
                        operator=current_operator,
                        params=current_params,
                        description=current_description.strip(),
                    )
                )

            current_step_id = int(step_match.group(1))
            current_operator = step_match.group(2)
            current_params = {}
            current_description = ""

            # Check if params are on same line or multi-line
            if ")" in line:
                # Single-line format
                params_str = line[line.find("(") + 1 : line.rfind(")")]
                current_params = _parse_params_string(params_str)
                collecting_params = False
                params_lines = []
            else:
                # Multi-line format - start collecting
                collecting_params = True
                params_lines = []
            i += 1
            continue

        # If collecting params, accumulate lines until we find closing paren
        if collecting_params:
            if ")" in line:
                # Last line of params
                params_lines.append(line[: line.rfind(")")])
                params_str = " ".join(params_lines)
                current_params = _parse_params_string(params_str)
                collecting_params = False
                params_lines = []
            else:
                params_lines.append(line)
            i += 1
            continue

        # Match description line
        if line.startswith("Description:"):
            current_description = line.replace("Description:", "").strip()
        elif current_step_id is not None and not line.startswith("Step"):
            # Continuation of description (if not collecting params)
            if not collecting_params:
                if current_description:
                    current_description += " " + line
                else:
                    current_description = line
        i += 1

    # Save last step
    if current_step_id is not None:
        if params_lines:
            params_str = " ".join(params_lines)
            current_params = _parse_params_string(params_str)
        steps.append(
            LogicalFormStep(
                step_id=current_step_id,
                operator=current_operator,
                params=current_params,
                description=current_description.strip(),
            )
        )

    return steps


def _parse_params_string(params_str: str) -> Dict[str, Any]:
    """Helper to parse parameter string into dict."""
    params = {}
    if not params_str or not params_str.strip():
        return params

    # Try to parse as JSON-like dict first
    try:
        # Normalize the string
        normalized = params_str.strip()
        # Replace single quotes with double quotes
        normalized = normalized.replace("'", '"')
        # Handle boolean values
        normalized = normalized.replace("True", "true").replace("False", "false")
        # Handle None -> null
        normalized = normalized.replace("None", "null")
        # Handle $step references (keep as strings)
        normalized = re.sub(r"\$step(\d+)_results", r'"$step\1_results"', normalized)

        # Try wrapping in braces if not already
        if not normalized.startswith("{"):
            normalized = "{" + normalized + "}"

        params = json.loads(normalized)
    except:
        # Fallback: simple key-value parsing
        # Handle multi-line format with operation/subject/object
        if "operation=" in params_str or "subject=" in params_str:
            # Try to extract key-value pairs
            for part in re.split(r",\s*(?=\w+\s*=)", params_str):
                if "=" in part:
                    key_match = re.match(r"(\w+)\s*=\s*(.+)", part.strip())
                    if key_match:
                        key = key_match.group(1).strip()
                        value_str = key_match.group(2).strip()
                        # Try to parse value
                        try:
                            # Remove trailing commas
                            value_str = value_str.rstrip(",")
                            # Try JSON parsing
                            value_str = value_str.replace("'", '"')
                            value = json.loads(value_str)
                        except:
                            # Keep as string
                            value = value_str.strip("\"'")
                        params[key] = value
        else:
            # Original simple parsing
            for part in params_str.split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip().strip("\"'")
                    value = value.strip().strip("\"'")
                    params[key] = value

    return params


class KGQueryExecutor:
    """Executes logical form steps over the knowledge graph."""

    def __init__(
        self,
        kg_builder: KnowledgeGraph,
        neo4j_queries: Neo4jKAGQueries,
        tool_tracker=None,
    ):
        """
        Initialize KG Query Executor.

        Args:
            kg_builder: KnowledgeGraph instance
            neo4j_queries: Neo4jKAGQueries instance for graph queries
            tool_tracker: Optional ToolTracker instance for tracking tool usage
        """
        self.kg_builder = kg_builder
        self.queries = neo4j_queries
        self.tool_tracker = tool_tracker
        # Initialize embedding operators
        self.embedding_retrieval = EmbeddingRetrievalOperator(neo4j_queries)
        self.anomaly_neighborhood = AnomalyNeighborhoodOperator(neo4j_queries)

    def execute_retrieval(
        self, step: LogicalFormStep, step_results: Dict[int, Any], window_idx: int
    ) -> Any:
        """
        Execute a Retrieval step (KAG Operator 1).

        Implements the Retrieval operator from the KAG paper:
        Retrieval(s=?, p=predicate, o=?, constraints={...})

        Purpose: Query graph triples using SPO (Subject-Predicate-Object) pattern matching.
        Supports both SPO format and convenience format with operation parameter.

        Args:
            step: LogicalFormStep with operator="Retrieval"
            step_results: Dictionary mapping step_id -> results from previous steps
            window_idx: Current window index

        Returns:
            List of retrieved results (sensors, violations, similar windows, etc.)
        """
        # Handle both old format (s, p, o, constraints) and new format (operation, subject, object)
        if "operation" in step.params:
            # New format: operation="has_reading", subject={...}, object={...}
            operation = step.params.get("operation", "").lower()
            subject = step.params.get("subject", {})
            obj = step.params.get("object", {})

            # Extract window_idx from subject if it's a dict
            # Also handle step references in subject
            if isinstance(subject, str) and subject.startswith("$step"):
                # Step reference - resolve it
                step_ref = int(subject.replace("$step", "").replace("_results", ""))
                if step_ref in step_results:
                    # Use results from previous step - but we still need window_idx
                    window_idx_constraint = window_idx
                    # For now, use the window_idx since we can't determine it from step results
                else:
                    window_idx_constraint = window_idx
            elif isinstance(subject, dict):
                window_idx_constraint = subject.get("window_idx", window_idx)
            else:
                window_idx_constraint = window_idx

            # Extract anomaly threshold from object
            anomaly_threshold = None
            if isinstance(obj, dict):
                if "anomaly_score" in obj:
                    anomaly_score_spec = obj["anomaly_score"]
                    if isinstance(anomaly_score_spec, dict):
                        if anomaly_score_spec.get("operation") == "gt":
                            anomaly_threshold = float(
                                anomaly_score_spec.get("value", 0.5)
                            )
                    elif isinstance(anomaly_score_spec, str):
                        if ">" in anomaly_score_spec:
                            anomaly_threshold = float(
                                anomaly_score_spec.replace(">", "")
                            )

            # Determine predicate from operation
            if operation == "has_reading":
                # Auto-threshold detection: query all scores first, compute adaptive threshold
                if anomaly_threshold is None:
                    # Get all scores for auto-threshold detection
                    all_scores = self.queries.get_all_anomaly_scores(
                        window_idx_constraint
                    )

                    if all_scores and len(all_scores) > 0:
                        scores = [s["score"] for s in all_scores]
                        max_score = max(scores)

                        # Only use auto-threshold if max score is meaningful (>0.5)
                        # For normal windows with low scores, use higher threshold (0.5)
                        if max_score > 0.5:
                            # Use 75th percentile OR 0.5, whichever is lower
                            # This ensures we always flag top 25% of sensors
                            percentile_75 = np.percentile(scores, 75)
                            auto_threshold = min(percentile_75, 0.5)
                            threshold = auto_threshold

                            # Filter by threshold
                            result = [s for s in all_scores if s["score"] > threshold]

                            # Guarantee at least top-1 if max score > 0.5 (conservative)
                            if not result and max_score > 0.5:
                                result = [all_scores[0]]  # Return top sensor
                                threshold = all_scores[0]["score"]
                        else:
                            # Low scores - likely normal window, use conservative threshold
                            threshold = 0.5
                            result = [s for s in all_scores if s["score"] > threshold]
                    else:
                        # No sensors found - return empty
                        threshold = 0.5
                        result = []
                else:
                    # Use explicit threshold
                    threshold = anomaly_threshold
                    result = self.queries.get_anomalous_sensors(
                        window_idx_constraint, threshold=threshold
                    )

                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_anomalous_sensors",
                        params={
                            "window_idx": window_idx_constraint,
                            "threshold": threshold,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
            elif operation == "correlates_with":
                is_violation = (
                    obj.get("is_violation", False) if isinstance(obj, dict) else False
                )
                deviation_threshold = (
                    obj.get("deviation_threshold", None)
                    if isinstance(obj, dict)
                    else None
                )

                # Auto-threshold detection for violations
                if deviation_threshold is None:
                    # Get all deviations for auto-threshold detection
                    all_deviations = self.queries.get_all_deviations(
                        window_idx_constraint
                    )

                    if all_deviations and len(all_deviations) > 0:
                        deviations = [d["deviation"] for d in all_deviations]
                        # Use 50th percentile (median deviation)
                        deviation_threshold = np.percentile(deviations, 50)
                        # Filter by threshold
                        result = [
                            d
                            for d in all_deviations
                            if d["deviation"] > deviation_threshold
                        ]
                    else:
                        # No correlations found - return empty
                        deviation_threshold = 0.3
                        result = []
                else:
                    # Use explicit threshold
                    violations = self.queries.get_violations(
                        window_idx_constraint, deviation_threshold
                    )
                    result = violations
                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_violations",
                        params={
                            "window_idx": window_idx_constraint,
                            "deviation_threshold": deviation_threshold,
                            "is_violation": is_violation,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
            elif operation == "sensors_with_violations_and_anomaly":
                # Extract parameters from object
                anomaly_threshold = 0.5  # default
                min_violations = 1  # default
                
                if isinstance(obj, dict):
                    if "anomaly_threshold" in obj:
                        anomaly_threshold = float(obj["anomaly_threshold"])
                    if "min_violations" in obj:
                        min_violations = int(obj["min_violations"])
                
                result = self.queries.get_sensors_with_violations_and_anomaly(
                    window_idx_constraint, 
                    anomaly_threshold=anomaly_threshold,
                    min_violations=min_violations
                )
                
                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_sensors_with_violations_and_anomaly",
                        params={
                            "window_idx": window_idx_constraint,
                            "anomaly_threshold": anomaly_threshold,
                            "min_violations": min_violations,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
            elif operation == "temporal_retrieval":
                # Extract window_range from object
                window_range = None
                if isinstance(obj, dict):
                    window_range = obj.get("window_range", None)
                    if window_range is None:
                        # Try to compute from window_idx and range spec
                        range_spec = obj.get("range", [0, 0])  # e.g., [-2, 0] means [t-2, t]
                        if isinstance(range_spec, list) and len(range_spec) == 2:
                            start_offset = range_spec[0]
                            end_offset = range_spec[1]
                            window_range = list(range(window_idx_constraint + start_offset, window_idx_constraint + end_offset + 1))
                        else:
                            # Default: current window only
                            window_range = [window_idx_constraint]
                
                if window_range is None:
                    window_range = [window_idx_constraint]
                
                # Filter valid window indices (>= 0)
                window_range = [w for w in window_range if w >= 0]
                
                if not window_range:
                    window_range = [window_idx_constraint]
                
                result = self.queries.get_temporal_sensor_history(
                    window_idx_constraint, window_range
                )
                
                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_temporal_sensor_history",
                        params={
                            "window_idx": window_idx_constraint,
                            "window_range": window_range,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
            elif operation == "explore_neighborhood":
                # Extract root sensor and radius from object
                root_sensor = None
                radius = 2  # default
                
                if isinstance(obj, dict):
                    root_sensor = obj.get("root", None)
                    radius = obj.get("radius", 2)
                
                # If root not specified, try to get from step results
                if root_sensor is None or root_sensor == "":
                    # Try to extract from previous step results
                    for step_id, step_result in step_results.items():
                        if isinstance(step_result, list) and len(step_result) > 0:
                            if isinstance(step_result[0], dict):
                                # Check if it's a sensor result
                                if "sensor" in step_result[0]:
                                    # Use first sensor from previous results
                                    root_sensor = step_result[0]["sensor"]
                                    break
                                elif "source" in step_result[0]:
                                    # Use first source from violations
                                    root_sensor = step_result[0]["source"]
                                    break
                
                # If still no root, return empty result
                if root_sensor is None or root_sensor == "":
                    result = {
                        'root_sensor': None,
                        'neighbors': [],
                        'summary': {
                            'total_neighbors': 0,
                            'anomalous_count': 0,
                            'violations_count': 0
                        }
                    }
                else:
                    result = self.queries.explore_neighborhood(
                        root_sensor, window_idx_constraint, radius
                    )
                
                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="explore_neighborhood",
                        params={
                            "window_idx": window_idx_constraint,
                            "root_sensor": root_sensor,
                            "radius": radius,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
        else:
            # Old format: s, p, o, constraints
            constraints = step.params.get("constraints", {})
            subject = step.params.get("s", "ANY")
            predicate = step.params.get("p", "HAS_READING")
            obj = step.params.get("o", "ANY")

            # Resolve step references
            if isinstance(subject, str) and subject.startswith("$step"):
                step_ref = int(subject.replace("$step", "").replace("_results", ""))
                if step_ref in step_results:
                    subject = step_results[step_ref]
                else:
                    subject = "ANY"

            # Parse constraints
            window_idx_constraint = constraints.get("window_idx", window_idx)
            anomaly_threshold = constraints.get("anomaly_score_subject", None)
            if isinstance(anomaly_threshold, str):
                if ">" in anomaly_threshold:
                    anomaly_threshold = float(anomaly_threshold.replace(">", ""))
                else:
                    anomaly_threshold = None

            # Execute appropriate query based on predicate
            if predicate == "HAS_READING" or predicate == "BELONGS_TO":
                # Auto-threshold detection: query all scores first, compute adaptive threshold
                if anomaly_threshold is None:
                    # Get all scores for auto-threshold detection
                    all_scores = self.queries.get_all_anomaly_scores(
                        window_idx_constraint
                    )

                    if all_scores and len(all_scores) > 0:
                        scores = [s["score"] for s in all_scores]
                        max_score = max(scores)

                        # Only use auto-threshold if max score is meaningful (>0.5)
                        # For normal windows with low scores, use higher threshold (0.5)
                        if max_score > 0.5:
                            # Use 75th percentile OR 0.5, whichever is lower
                            # This ensures we always flag top 25% of sensors
                            percentile_75 = np.percentile(scores, 75)
                            auto_threshold = min(percentile_75, 0.5)
                            threshold = auto_threshold

                            # Filter by threshold
                            result = [s for s in all_scores if s["score"] > threshold]

                            # Guarantee at least top-1 if max score > 0.5 (conservative)
                            if not result and max_score > 0.5:
                                result = [all_scores[0]]  # Return top sensor
                                threshold = all_scores[0]["score"]
                        else:
                            # Low scores - likely normal window, use conservative threshold
                            threshold = 0.5
                            result = [s for s in all_scores if s["score"] > threshold]
                    else:
                        # No sensors found - return empty
                        threshold = 0.5
                        result = []
                else:
                    # Use explicit threshold
                    threshold = anomaly_threshold
                    result = self.queries.get_anomalous_sensors(
                        window_idx_constraint, threshold=threshold
                    )

                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_anomalous_sensors",
                        params={
                            "window_idx": window_idx_constraint,
                            "threshold": threshold,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result
            elif predicate == "CORRELATES_WITH":
                deviation_threshold = constraints.get("deviation_threshold", None)
                is_violation = constraints.get("is_violation", False)

                # Auto-threshold detection for violations
                if deviation_threshold is None:
                    # Get all deviations for auto-threshold detection
                    all_deviations = self.queries.get_all_deviations(
                        window_idx_constraint
                    )

                    if all_deviations and len(all_deviations) > 0:
                        deviations = [d["deviation"] for d in all_deviations]
                        # Use 50th percentile (median deviation)
                        deviation_threshold = np.percentile(deviations, 50)
                        # Filter by threshold
                        result = [
                            d
                            for d in all_deviations
                            if d["deviation"] > deviation_threshold
                        ]
                    else:
                        # No correlations found - return empty
                        deviation_threshold = 0.3
                        result = []
                else:
                    # Use explicit threshold
                    violations = self.queries.get_violations(
                        window_idx_constraint, deviation_threshold
                    )
                    result = violations
                # Track tool call
                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="Retrieval",
                        query_method="get_violations",
                        params={
                            "window_idx": window_idx_constraint,
                            "deviation_threshold": deviation_threshold,
                            "is_violation": is_violation,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result

            # Handle embedding-based operations
            elif operation == "find_similar_windows":
                # Find similar windows in embedding space
                k = step.params.get("k", 5)
                class_filter = step.params.get("class_filter", None)

                result = self.embedding_retrieval.find_similar_windows(
                    window_idx_constraint, k=k, class_filter=class_filter
                )

                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="EmbeddingRetrieval",
                        query_method="find_similar_windows",
                        params={
                            "window_idx": window_idx_constraint,
                            "k": k,
                            "class_filter": class_filter,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result

            elif operation == "find_anomalous_neighbors":
                # Find anomalous neighbors in embedding space
                distance_threshold = step.params.get("distance_threshold", 0.2)

                result = self.anomaly_neighborhood.find_anomalous_neighbors(
                    window_idx_constraint, distance_threshold=distance_threshold
                )

                if self.tool_tracker:
                    self.tool_tracker.record_tool_call(
                        tool_name="AnomalyNeighborhood",
                        query_method="find_anomalous_neighbors",
                        params={
                            "window_idx": window_idx_constraint,
                            "distance_threshold": distance_threshold,
                        },
                        result=result,
                        window_idx=window_idx,
                        operator="Retrieval",
                    )
                return result

        return []

    def execute_math(self, step: LogicalFormStep, step_results: Dict[int, Any]) -> Any:
        """
        Execute a Math step (KAG Operator 2).

        Implements the Math operator from the KAG paper:
        Math(operation, operands)

        Purpose: Numerical computation (count, sum, max, mean) over operands.
        Operands can reference previous step results using $step<id>_results syntax.

        Args:
            step: LogicalFormStep with operator="Math"
            step_results: Dictionary mapping step_id -> results from previous steps

        Returns:
            Math operation result (int or float)
        """
        operation = step.params.get("operation", "count")
        operands = step.params.get("operands", [])

        # Resolve operands from previous steps
        resolved_operands = []
        for op in operands:
            if isinstance(op, str) and op.startswith("$step"):
                step_ref = int(op.replace("$step", "").replace("_results", ""))
                if step_ref in step_results:
                    resolved_operands.append(step_results[step_ref])
            else:
                resolved_operands.append(op)

        result = None
        if operation == "count":
            # Count items in first operand
            if resolved_operands:
                data = resolved_operands[0]
                if isinstance(data, list):
                    result = len(data)
                else:
                    result = 1
            else:
                result = 0

        # Track tool call
        if self.tool_tracker:
            # Get window_idx from step if available, otherwise use -1
            window_idx = step.params.get("window_idx", -1)
            self.tool_tracker.record_tool_call(
                tool_name="Math",
                query_method=f"math_{operation}",
                params={"operation": operation, "operands_count": len(operands)},
                result=result,
                window_idx=window_idx,
                operator="Math",
            )

        return result

    def execute_deduce(
        self, step: LogicalFormStep, step_results: Dict[int, Any]
    ) -> Any:
        """
        Execute a Deduce step (KAG Operator 3).

        Implements the Deduce operator from the KAG paper:
        Deduce(left, right, comparison)

        Purpose: Logical comparison (>, <, =, contains) or root cause extraction.
        Supports comparison="extract_root_cause" for identifying root cause sensors
        from violations and anomalies.

        Args:
            step: LogicalFormStep with operator="Deduce"
            step_results: Dictionary mapping step_id -> results from previous steps

        Returns:
            Deduced result (root cause sensor name, boolean comparison result, etc.)
        """
        left_operand_ref = step.params.get("left_operand", "")
        right_operand_ref = step.params.get("right_operand", "")
        comparison = step.params.get("comparison", "extract_root_cause")

        # Resolve operands
        left_data = None
        right_data = None

        if isinstance(left_operand_ref, str) and left_operand_ref.startswith("$step"):
            step_ref = int(
                left_operand_ref.replace("$step", "").replace("_results", "")
            )
            if step_ref in step_results:
                left_data = step_results[step_ref]

        if isinstance(right_operand_ref, str) and right_operand_ref.startswith("$step"):
            step_ref = int(
                right_operand_ref.replace("$step", "").replace("_results", "")
            )
            if step_ref in step_results:
                right_data = step_results[step_ref]

        result = None
        if comparison == "extract_root_cause":
            # Enhanced root cause selection with temporal onset, centrality, and violations
            window_idx = step.params.get("window_idx", -1)

            # Get centrality if window_idx is available
            centrality_map = {}
            if window_idx >= 0:
                try:
                    centrality = self.queries.compute_sensor_centrality(window_idx)
                    centrality_map = {
                        c["sensor"]: c.get("degree", 1) for c in centrality
                    }
                except Exception:
                    pass  # Fall back to simple selection if centrality unavailable

            # Collect candidate sensors from violations and anomalies
            candidate_sensors = {}

            # Process violations
            if isinstance(left_data, list) and len(left_data) > 0:
                if isinstance(left_data[0], dict) and "source" in left_data[0]:
                    for v in left_data:
                        source = v.get("source", "")
                        target = v.get("target", "")
                        deviation = v.get("deviation", 0.0)

                        if source:
                            if source not in candidate_sensors:
                                candidate_sensors[source] = {
                                    "anomaly_score": 0.0,
                                    "violation_count": 0,
                                    "violation_severity": 0.0,
                                    "onset_score": 1.0,
                                    "centrality": centrality_map.get(source, 1),
                                }
                            candidate_sensors[source]["violation_count"] += 1
                            if deviation > 0.5:
                                candidate_sensors[source]["violation_severity"] += (
                                    deviation
                                )

                        if target:
                            if target not in candidate_sensors:
                                candidate_sensors[target] = {
                                    "anomaly_score": 0.0,
                                    "violation_count": 0,
                                    "violation_severity": 0.0,
                                    "onset_score": 1.0,
                                    "centrality": centrality_map.get(target, 1),
                                }
                            candidate_sensors[target]["violation_count"] += 1
                            if deviation > 0.5:
                                candidate_sensors[target]["violation_severity"] += (
                                    deviation
                                )

            # Process anomalous sensors
            if isinstance(right_data, list) and len(right_data) > 0:
                if isinstance(right_data[0], dict) and "sensor" in right_data[0]:
                    for sensor_data in right_data:
                        sensor = sensor_data.get("sensor", "")
                        score = sensor_data.get("score", 0.0)

                        if sensor:
                            if sensor not in candidate_sensors:
                                candidate_sensors[sensor] = {
                                    "anomaly_score": 0.0,
                                    "violation_count": 0,
                                    "violation_severity": 0.0,
                                    "onset_score": 1.0,
                                    "centrality": centrality_map.get(sensor, 1),
                                }
                            candidate_sensors[sensor]["anomaly_score"] = max(
                                candidate_sensors[sensor]["anomaly_score"], score
                            )

            # Enhanced scoring: anomaly_score * onset_score * violation_count * centrality
            # For temporal onset, we need to query previous windows
            # Since we don't have direct access to solver's _find_anomaly_onset here,
            # we'll use a simplified approach: weight by violation_count and anomaly_score
            # Temporal onset can be added later if we pass solver reference

            if candidate_sensors:
                scored_sensors = []
                for sensor, data in candidate_sensors.items():
                    anomaly_score = data["anomaly_score"]
                    violation_count = data["violation_count"]
                    violation_severity = data["violation_severity"]
                    centrality = max(data["centrality"], 1)  # Ensure at least 1

                    # Normalize violation severity (0-1 scale)
                    violation_severity_norm = (
                        min(violation_severity / 2.0, 1.0)
                        if violation_severity > 0
                        else 0.0
                    )

                    # Enhanced score: combine anomaly, violations, and centrality
                    # Formula: anomaly_score * (1 + violation_count) * (1 + violation_severity) * centrality
                    # This rewards sensors with high anomaly scores, many violations, severe violations, and high centrality
                    score = (
                        anomaly_score
                        * (1.0 + violation_count * 0.2)
                        * (1.0 + violation_severity_norm * 0.3)
                        * centrality
                    )
                    scored_sensors.append((sensor, score))

                # Sort by score and return highest
                scored_sensors.sort(key=lambda x: x[1], reverse=True)
                result = scored_sensors[0][0] if scored_sensors else None

            # Fallback to simple selection if enhanced scoring didn't work
            if result is None:
                if isinstance(left_data, list) and len(left_data) > 0:
                    if isinstance(left_data[0], dict) and "source" in left_data[0]:
                        source_counts = {}
                        for v in left_data:
                            source = v.get("source", "")
                            if source:
                                source_counts[source] = source_counts.get(source, 0) + 1
                        if source_counts:
                            result = max(source_counts.items(), key=lambda x: x[1])[0]

                if (
                    result is None
                    and isinstance(right_data, list)
                    and len(right_data) > 0
                ):
                    if isinstance(right_data[0], dict) and "sensor" in right_data[0]:
                        sorted_sensors = sorted(
                            right_data, key=lambda x: x.get("score", 0), reverse=True
                        )
                        result = sorted_sensors[0]["sensor"] if sorted_sensors else None

        # Track tool call
        if self.tool_tracker:
            window_idx = step.params.get("window_idx", -1)
            self.tool_tracker.record_tool_call(
                tool_name="Deduce",
                query_method=f"deduce_{comparison}",
                params={
                    "comparison": comparison,
                    "has_left": left_data is not None,
                    "has_right": right_data is not None,
                },
                result=result,
                window_idx=window_idx,
                operator="Deduce",
            )

        return result

    def execute_sort(self, step: LogicalFormStep, step_results: Dict[int, Any]) -> Any:
        """
        Execute a Sort step (KAG Operator 4).

        Implements the Sort operator from the KAG paper:
        Sort(target, key, direction, limit)

        Purpose: Rank and filter results by a specified key.
        Can sort by anomaly_score, deviation, distance, etc.

        Args:
            step: LogicalFormStep with operator="Sort"
            step_results: Dictionary mapping step_id -> results from previous steps

        Returns:
            Sorted and optionally limited list of results
        """
        target_ref = step.params.get("target_set", "")
        key = step.params.get("key", "deviation")
        direction = step.params.get("direction", "desc")
        limit = step.params.get("limit", None)

        # Resolve target set
        target_data = None
        if isinstance(target_ref, str) and target_ref.startswith("$step"):
            step_ref = int(target_ref.replace("$step", "").replace("_results", ""))
            if step_ref in step_results:
                target_data = step_results[step_ref]

        if not isinstance(target_data, list):
            return []

        # Sort
        reverse = direction.lower() == "desc"
        try:
            sorted_data = sorted(
                target_data, key=lambda x: x.get(key, 0), reverse=reverse
            )
        except:
            sorted_data = target_data

        # Apply limit
        if limit is not None:
            sorted_data = sorted_data[:limit]

        # Track tool call
        if self.tool_tracker:
            window_idx = step.params.get("window_idx", -1)
            self.tool_tracker.record_tool_call(
                tool_name="Sort",
                query_method="sort",
                params={"key": key, "direction": direction, "limit": limit},
                result=sorted_data,
                window_idx=window_idx,
                operator="Sort",
            )

        return sorted_data


class EmbeddingRetrievalOperator:
    """Operator for retrieving similar windows in embedding space."""

    def __init__(self, neo4j_queries: Neo4jKAGQueries):
        """
        Initialize Embedding Retrieval Operator.

        Args:
            neo4j_queries: Neo4jKAGQueries instance for graph queries
        """
        self.queries = neo4j_queries

    def find_similar_windows(
        self, window_idx: int, k: int = 5, class_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find k most similar windows to a given window.

        Args:
            window_idx: Index of the window to find similar windows for
            k: Number of similar windows to return
            class_filter: Optional filter by predicted_class ("normal" or "anomalous")

        Returns:
            List of dicts with keys: window_idx, similarity, distance, dist_normal, dist_anomalous
        """
        with self.queries.driver.session() as session:
            # Build Cypher query
            query = """
                MATCH (w1:Window {idx: $window_idx})-[s:SIMILAR_TO]->(w2:Window)
            """

            if class_filter:
                query += " WHERE w2.predicted_class = $class_filter"

            query += """
                OPTIONAL MATCH (w2)-[d1:DISTANCE_TO_CENTER]->(c1:ClassCenter {class: "normal"})
                OPTIONAL MATCH (w2)-[d2:DISTANCE_TO_CENTER]->(c2:ClassCenter {class: "anomalous"})
                RETURN w2.idx AS window_idx,
                       s.similarity AS similarity,
                       s.distance AS distance,
                       d1.distance AS dist_normal,
                       d2.distance AS dist_anomalous
                ORDER BY s.similarity DESC
                LIMIT $k
            """

            params = {"window_idx": window_idx, "k": k}
            if class_filter:
                params["class_filter"] = class_filter

            result = session.run(query, params)
            similar_windows = []
            for record in result:
                similar_windows.append(
                    {
                        "window_idx": record["window_idx"],
                        "similarity": float(record["similarity"])
                        if record["similarity"] is not None
                        else 0.0,
                        "distance": float(record["distance"])
                        if record["distance"] is not None
                        else 0.0,
                        "dist_normal": float(record["dist_normal"])
                        if record["dist_normal"] is not None
                        else None,
                        "dist_anomalous": float(record["dist_anomalous"])
                        if record["dist_anomalous"] is not None
                        else None,
                    }
                )

            return similar_windows


class AnomalyNeighborhoodOperator:
    """Operator for finding anomalous neighbors in embedding space."""

    def __init__(self, neo4j_queries: Neo4jKAGQueries):
        """
        Initialize Anomaly Neighborhood Operator.

        Args:
            neo4j_queries: Neo4jKAGQueries instance for graph queries
        """
        self.queries = neo4j_queries

    def find_anomalous_neighbors(
        self, window_idx: int, distance_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Find nearby anomalous windows in embedding space.

        Args:
            window_idx: Index of the window to find neighbors for
            distance_threshold: Maximum euclidean distance threshold

        Returns:
            List of dicts with keys: window_idx, distance, similarity, dist_normal, dist_anomalous
            Sorted by distance ascending
        """
        with self.queries.driver.session() as session:
            query = """
                MATCH (w1:Window {idx: $window_idx})-[s:SIMILAR_TO]->(w2:Window)
                WHERE s.distance < $distance_threshold
                  AND w2.predicted_class = "anomalous"
                OPTIONAL MATCH (w2)-[d1:DISTANCE_TO_CENTER]->(c1:ClassCenter {class: "normal"})
                OPTIONAL MATCH (w2)-[d2:DISTANCE_TO_CENTER]->(c2:ClassCenter {class: "anomalous"})
                RETURN w2.idx AS window_idx,
                       s.distance AS distance,
                       s.similarity AS similarity,
                       d1.distance AS dist_normal,
                       d2.distance AS dist_anomalous
                ORDER BY s.distance ASC
            """

            result = session.run(
                query,
                {"window_idx": window_idx, "distance_threshold": distance_threshold},
            )

            neighbors = []
            for record in result:
                neighbors.append(
                    {
                        "window_idx": record["window_idx"],
                        "distance": float(record["distance"])
                        if record["distance"] is not None
                        else 0.0,
                        "similarity": float(record["similarity"])
                        if record["similarity"] is not None
                        else 0.0,
                        "dist_normal": float(record["dist_normal"])
                        if record["dist_normal"] is not None
                        else None,
                        "dist_anomalous": float(record["dist_anomalous"])
                        if record["dist_anomalous"] is not None
                        else None,
                    }
                )

            return neighbors


class KAGIterativeSolver:
    """LLM-planned iterative KAG solver."""

    def __init__(
        self,
        kg_builder: KnowledgeGraph,
        neo4j_queries: Neo4jKAGQueries,
        sensor_names: List[str],
        model,
        tokenizer,
        max_iterations: int = 3,  # Allow 2-3 iterations for reflection and exploration
        tool_tracker=None,
    ):
        """
        Initialize KAG Iterative Solver.

        Args:
            kg_builder: KnowledgeGraph instance
            neo4j_queries: Neo4jKAGQueries instance
            sensor_names: List of sensor names
            model: Loaded LLM model
            tokenizer: Loaded tokenizer
            max_iterations: Maximum number of refinement iterations (default: 1)
            tool_tracker: Optional ToolTracker instance for tracking tool usage
        """
        self.kg_builder = kg_builder
        self.queries = neo4j_queries
        self.sensor_names = sensor_names
        self.sensor_to_idx = {name: idx for idx, name in enumerate(sensor_names)}
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.tool_tracker = tool_tracker
        self.executor = KGQueryExecutor(
            kg_builder, neo4j_queries, tool_tracker=tool_tracker
        )
        self._example_prompt_dumped = False

        # Configuration constants for root cause selection
        self.ROOT_K = 1  # Maximum number of root causes per window
        self.ALPHA = 0.8  # Tie-breaking band: keep sensors within 80% of best score
        self.MIN_ROOT_SCORE = 0.7  # Minimum score to predict fault (filter weak evidence)
        self.CAND_ANOMALY = 0.5  # Candidate threshold: anomaly score or violation count required
        # Scoring weights
        self.W_ANOM = 1.0  # Weight for anomaly score
        self.W_VCOUNT = 0.7  # Weight for violation count
        self.W_VSEV = 0.3  # Weight for violation severity sum
        self.W_NB = 0.3  # Weight for neighbor violations
        # Cache for anomaly scores: (window_idx, sensor_name) -> anomaly_score
        self._anomaly_score_cache = {}
        # Anomaly threshold for temporal analysis (default: 0.7, same as V1)
        self.anomaly_threshold = 0.7

    def _get_anomaly_score(self, sensor: str, window_idx: int) -> float:
        """
        Get anomaly score for a sensor in a specific window.
        Uses cache to avoid repeated Neo4j queries.

        Args:
            sensor: Base sensor name
            window_idx: Window index

        Returns:
            Anomaly score (0.0-1.0)
        """
        cache_key = (window_idx, sensor)
        if cache_key in self._anomaly_score_cache:
            return self._anomaly_score_cache[cache_key]

        # Query Neo4j for anomaly score
        with self.queries.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Sensor {base_sensor_name: $sensor, window: $window_idx})
                RETURN s.anomaly_score AS score
                LIMIT 1
            """,
                sensor=sensor,
                window_idx=window_idx,
            )
            record = result.single()
            score = float(record["score"]) if record else 0.0

        self._anomaly_score_cache[cache_key] = score
        return score

    def _find_anomaly_onset(
        self, sensor: str, window_idx: int, lookback: int = 3
    ) -> int:
        """
        Find the earliest window where sensor became anomalous.

        Searches backwards from window_idx up to lookback windows to find
        when the sensor first exceeded the anomaly threshold.

        Args:
            sensor: Base sensor name
            window_idx: Current window index
            lookback: Maximum number of windows to look back (default: 3)

        Returns:
            Earliest window index where anomaly was detected, or window_idx if not found earlier
        """
        earliest = window_idx
        start_window = max(0, window_idx - lookback)

        for w in range(start_window, window_idx + 1):
            score = self._get_anomaly_score(sensor, w)
            if score > self.anomaly_threshold:
                earliest = w
                break

        return earliest

    def _violation_severity(self, violations: List[Dict]) -> float:
        """
        Compute violation severity score based on deviation magnitude.

        Counts violations with deviation > 0.5 as "severe" and returns
        a normalized score capped at 1.0.

        Args:
            violations: List of violation dicts with 'deviation' key

        Returns:
            Severity score (0.0-1.0)
        """
        if not violations:
            return 0.0

        severe = [v for v in violations if v.get("deviation", 0.0) > 0.5]
        return min(len(severe) / 3.0, 1.0)

    def _assess_evidence_strength(self, exec_results: Dict[int, Any]) -> Dict[str, Any]:
        """
        Assess the strength of evidence from tool execution results.

        Returns:
            dict with keys: max_score, num_anomalous, num_violations,
                           level (NONE/WEAK/MODERATE/STRONG), confidence
        """
        # Extract anomalous sensors
        anomalous_sensors = []
        for step_id, step_results in exec_results.items():
            if isinstance(step_results, list):
                for item in step_results:
                    if isinstance(item, dict) and "score" in item and "sensor" in item:
                        anomalous_sensors.append(item)

        # Extract violations
        violations = []
        for step_id, step_results in exec_results.items():
            if isinstance(step_results, list):
                for item in step_results:
                    if isinstance(item, dict) and "deviation" in item:
                        violations.append(item)

        # Calculate metrics
        max_score = max([s["score"] for s in anomalous_sensors], default=0.0)
        num_anomalous = len([s for s in anomalous_sensors if s["score"] > 0.5])
        num_violations = len([v for v in violations if v.get("deviation", 0) > 0.3])

        # Determine strength level - conservative for normal windows, but allow high scores
        # Note: Based on score distribution analysis, P95 of normal windows is ~0.54,
        # so threshold of 0.7 for MODERATE is appropriate to minimize false positives
        if max_score < 0.5 and num_violations == 0:
            level = "NONE"
            confidence = 0.1
        elif max_score < 0.5:
            # Low scores even with violations - likely normal window with noise
            # OR faulty window that GDN missed (limitation of GDN model)
            level = "WEAK"
            confidence = 0.2
        elif max_score >= 0.7 and num_violations >= 3:
            level = "STRONG"
            confidence = 0.85
        elif max_score >= 0.7:
            # Scores >= 0.7 indicate potential fault - allow MODERATE
            # Based on analysis: P95 of normal is 0.54, so 0.7 is safe threshold
            # This catches faulty windows even if violations are few
            level = "MODERATE"
            confidence = 0.6
        else:
            # Scores between 0.5 and 0.7 - weak evidence, likely normal
            level = "WEAK"
            confidence = 0.3

        return {
            "max_score": max_score,
            "num_anomalous": num_anomalous,
            "num_violations": num_violations,
            "level": level,
            "confidence": confidence,
            "anomalous_sensors": anomalous_sensors,
            "violations": violations,
        }

    def _format_weak_evidence_reasoning(self, evidence: Dict[str, Any]) -> str:
        """Format reasoning string for weak evidence cases."""
        return (
            f"GDN anomaly detection completed. "
            f"Maximum anomaly score: {evidence['max_score']:.3f} (threshold: 0.6). "
            f"Correlation violations detected: {evidence['num_violations']} (threshold: 2). "
            f"Evidence insufficient to diagnose fault. System operating normally."
        )

    def _extract_valid_sensors(self, exec_results: Dict[int, Any]) -> set:
        """Extract set of sensor names that appeared in tool results."""
        valid_sensors = set()

        for step_id, step_results in exec_results.items():
            if isinstance(step_results, list):
                for item in step_results:
                    if isinstance(item, dict):
                        if "sensor" in item:
                            valid_sensors.add(item["sensor"])
                        if "source" in item:
                            valid_sensors.add(item["source"])
                        if "target" in item:
                            valid_sensors.add(item["target"])

        return valid_sensors
    
    def _extract_sensors_from_exec_results(self, exec_results: Dict[int, Any]) -> Dict:
        """
        Extract root cause and affected sensors using scoring-based top-k selection.
        
        Uses weighted features (anomaly score, violation count, violation severity, 
        neighbor violations) to compute scores, then selects top-k roots with tie-breaking.
        
        Args:
            exec_results: Dictionary mapping step_id -> execution results
            
        Returns:
            Dictionary with keys:
            - 'root_cause_sensors': List of root cause sensor names (top-k, max ROOT_K)
            - 'affected_sensors': List of affected sensor names
            - 'all_faulty_sensors': List of all faulty sensors (root + affected)
        """
        from collections import defaultdict
        
        # Aggregate features per sensor
        stats = defaultdict(lambda: {
            "anomaly": 0.0,
            "viol_count": 0,
            "viol_severity": 0.0,
            "neighbor_viol": 0.0,
        })
        
        for step_id, step_results in exec_results.items():
            if isinstance(step_results, list):
                for item in step_results:
                    if isinstance(item, dict):
                        # 1) Direct anomalies
                        if "sensor" in item and "score" in item:
                            sensor = item.get("sensor", "")
                            score = float(item.get("score", 0.0))
                            
                            if sensor and sensor in self.sensor_names:
                                stats[sensor]["anomaly"] = max(stats[sensor]["anomaly"], score)
                        
                        # 2) Violations on edges (weight by deviation magnitude)
                        if "source" in item and "target" in item and "deviation" in item:
                            source = item.get("source", "")
                            target = item.get("target", "")
                            deviation = abs(float(item.get("deviation", 0.0)))
                            
                            for s in [source, target]:
                                if s and s in self.sensor_names:
                                    stats[s]["viol_count"] += 1
                                    stats[s]["viol_severity"] += deviation
                        
                        # 3) Neighborhood exploration results (down-weight by hop distance)
                        if "neighbors" in item:
                            neighbors = item.get("neighbors", [])
                            for neighbor in neighbors:
                                if isinstance(neighbor, dict):
                                    sensor = neighbor.get("sensor", "")
                                    hop = neighbor.get("hop", 1)
                                    vcnt = neighbor.get("violations_count", 0)
                                    
                                    if sensor and sensor in self.sensor_names:
                                        # Down-weight distant neighbors
                                        stats[sensor]["neighbor_viol"] += float(vcnt) / max(hop, 1)
                                        
                                        # Also track anomaly scores from neighbors
                                        nb_score = neighbor.get("score", 0.0)
                                        if nb_score > 0:
                                            stats[sensor]["anomaly"] = max(stats[sensor]["anomaly"], float(nb_score))
        
        # Candidate set: sensors with meaningful evidence
        candidates = {
            s for s, f in stats.items()
            if f["anomaly"] >= self.CAND_ANOMALY or f["viol_count"] > 0
        }
        
        if not candidates:
            return {
                "root_cause_sensors": [],
                "affected_sensors": [],
                "all_faulty_sensors": [],
            }
        
        # Compute weighted scores
        scores = {}
        for s in candidates:
            f = stats[s]
            scores[s] = (
                self.W_ANOM * f["anomaly"]
                + self.W_VCOUNT * min(f["viol_count"], 5) / 5.0  # Normalize count (cap at 5)
                + self.W_VSEV * min(f["viol_severity"], 2.0) / 2.0  # Normalize severity (cap at 2.0)
                + self.W_NB * min(f["neighbor_viol"], 3.0) / 3.0  # Normalize neighbor violations (cap at 3.0)
            )
        
        # Select top-k roots with tie-breaking band
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_score = ranked[0][1]
        
        # If even the best score is weak, treat window as normal
        if best_score < self.MIN_ROOT_SCORE:
            return {
                "root_cause_sensors": [],
                "affected_sensors": [],
                "all_faulty_sensors": [],
            }
        
        # Select roots within ALPHA band of best score, limit to ROOT_K
        root_cause_sensors = [
            s for s, sc in ranked
            if sc >= best_score * self.ALPHA
        ][:self.ROOT_K]
        
        # Affected sensors: candidates that are not root cause
        affected_sensors = [
            s for s in candidates
            if s not in root_cause_sensors
        ]
        
        all_faulty_sensors = root_cause_sensors + affected_sensors
        
        return {
            "root_cause_sensors": root_cause_sensors,
            "affected_sensors": affected_sensors,
            "all_faulty_sensors": all_faulty_sensors,
        }
    
    def _merge_exec_and_llm_predictions(
        self, exec_sensors: Dict, llm_prediction: Dict
    ) -> Dict:
        """
        Merge execution-based sensor labels with LLM fault_type/reasoning.
        
        Execution results take priority for sensor identification (more reliable),
        while LLM provides fault_type and reasoning.
        
        Args:
            exec_sensors: Dictionary from _extract_sensors_from_exec_results()
            llm_prediction: Dictionary from parse_llm_response()
            
        Returns:
            Merged dictionary with:
            - sensor_labels: Binary array (all faulty sensors from execution)
            - sensor_labels_root_only: Binary array (only root cause from execution)
            - fault_type: From LLM (or classified if LLM didn't provide)
            - reasoning: From LLM
            - root_cause_sensors: From execution
            - affected_sensors: From execution
        """
        # Create sensor label arrays from execution results
        sensor_labels = np.zeros(len(self.sensor_names), dtype=np.float32)
        sensor_labels_root_only = np.zeros(len(self.sensor_names), dtype=np.float32)
        
        # Mark all faulty sensors
        for sensor in exec_sensors.get('all_faulty_sensors', []):
            if sensor in self.sensor_names:
                idx = self.sensor_names.index(sensor)
                sensor_labels[idx] = 1.0
        
        # Mark only root cause sensors
        for sensor in exec_sensors.get('root_cause_sensors', []):
            if sensor in self.sensor_names:
                idx = self.sensor_names.index(sensor)
                sensor_labels_root_only[idx] = 1.0
        
        # Use LLM fault_type if available, otherwise classify from execution results
        fault_type = llm_prediction.get("fault_type", None)
        if fault_type is None:
            root_cause = exec_sensors.get('root_cause_sensors', [])
            affected = exec_sensors.get('affected_sensors', [])
            violations = []  # Could extract from exec_results if needed
            fault_type = self._classify_fault_type(
                root_cause[0] if root_cause else None,
                affected,
                violations
            )
        
        # Use LLM reasoning if available
        reasoning = llm_prediction.get("reasoning", "")
        if not reasoning:
            reasoning = f"Root cause identified from execution results: {exec_sensors.get('root_cause_sensors', [])}"
        
        # Determine window label
        window_label = 0
        if exec_sensors.get('root_cause_sensors'):
            # Use first root cause sensor index (1-indexed)
            root_sensor = exec_sensors['root_cause_sensors'][0]
            if root_sensor in self.sensor_names:
                window_label = self.sensor_names.index(root_sensor) + 1
        
        return {
            'sensor_labels': sensor_labels_root_only,  # Root-only for metrics (precision improvement)
            'sensor_labels_root_only': sensor_labels_root_only,  # Explicit root-only
            'sensor_labels_raw': sensor_labels,  # All faulty sensors (root + affected) for analysis
            'fault_type': fault_type,
            'reasoning': reasoning,
            'root_cause_sensors': exec_sensors.get('root_cause_sensors', []),
            'affected_sensors': exec_sensors.get('affected_sensors', []),
            'window_label': window_label
        }

    # Fault type mappings constant
    FAULT_TYPE_MAPPINGS = {
        "VSS_DROPOUT": [
            "VSS_DROPOUT",
            "VSS_DRO",
            "VEHICLE_SPEED_DROPOUT",
            "vss_dropout",
            "vehicle_speed_dropout",
        ],
        "MAF_SCALE_LOW": [
            "MAF_SCALE_LOW",
            "MAF_SCA",
            "MAF_SCALE",
            "maf_scale_low",
            "maf_scale",
        ],
        "COOLANT_DROPOUT": [
            "COOLANT_DROPOUT",
            "COOLANT",
            "coolant_dropout",
            "COOLANT_TEMPERATURE_DROPOUT",
        ],
        "TPS_STUCK": [
            "TPS_STU",
            "TPS_STUCK",
            "THROTTLE_STUCK",
            "tps_stuck",
            "throttle_stuck",
        ],
        "NORMAL": ["NORMAL", "NO_FAULT", "NONE", "normal", "no_fault"],
    }

    def _normalize_fault_type(self, raw_fault: str) -> str:
        """Map any variant to canonical fault type."""
        if not raw_fault:
            return "NORMAL"

        raw_upper = raw_fault.strip().upper()

        for canonical, variants in self.FAULT_TYPE_MAPPINGS.items():
            if raw_upper in [v.upper() for v in variants]:
                return canonical

        # Default to NORMAL if unknown
        return "NORMAL"

    def _hierarchical_diagnosis(self, window_idx: int) -> Dict[int, Any]:
        """
        Multi-stage hierarchical diagnosis pipeline.

        Stage 1: High-threshold anomaly detection (0.7) - clear faults
        Stage 2: If Stage 1 finds sensors, check violations for those sensors
        Stage 3: If Stage 1 finds nothing, use lower threshold (0.5) + violations
        Stage 4: Fuse evidence using weighted combination

        Returns:
            Dictionary of execution results from all stages
        """
        exec_results = {}

        # Stage 1: High-threshold anomaly detection
        stage1_step = LogicalFormStep(
            step_id=1,
            operator="Retrieval",
            params={
                "operation": "has_reading",
                "subject": {"window_idx": window_idx},
                "object": {"anomaly_score": {"operation": "gt", "value": 0.7}},
            },
            description="Stage 1: High-confidence anomaly detection (threshold=0.7)",
        )
        stage1_results = self.execute_logical_form([stage1_step], window_idx)
        exec_results[1] = stage1_results.get(1, [])

        high_conf_anomalies = stage1_results.get(1, [])
        high_conf_sensors = []
        if isinstance(high_conf_anomalies, list) and len(high_conf_anomalies) > 0:
            high_conf_sensors = [
                s.get("sensor", "")
                for s in high_conf_anomalies
                if isinstance(s, dict) and "sensor" in s
            ]

        # Stage 2: If Stage 1 found sensors, check violations for those sensors
        if high_conf_sensors:
            stage2_step = LogicalFormStep(
                step_id=2,
                operator="Retrieval",
                params={
                    "operation": "correlates_with",
                    "subject": {"window_idx": window_idx},
                    "object": {
                        "deviation_threshold": 0.3,
                        "is_violation": True,
                    },
                },
                description="Stage 2: Check violations for high-confidence anomalous sensors",
            )
            stage2_results = self.execute_logical_form([stage2_step], window_idx)
            exec_results[2] = stage2_results.get(2, [])
        else:
            # Stage 3: If Stage 1 found nothing, use lower threshold + violations
            stage3a_step = LogicalFormStep(
                step_id=3,
                operator="Retrieval",
                params={
                    "operation": "has_reading",
                    "subject": {"window_idx": window_idx},
                    "object": {"anomaly_score": {"operation": "gt", "value": 0.5}},
                },
                description="Stage 3a: Lower-threshold anomaly detection (threshold=0.5)",
            )
            stage3b_step = LogicalFormStep(
                step_id=4,
                operator="Retrieval",
                params={
                    "operation": "correlates_with",
                    "subject": {"window_idx": window_idx},
                    "object": {
                        "deviation_threshold": 0.3,
                        "is_violation": True,
                    },
                },
                description="Stage 3b: Check violations",
            )
            stage3_results = self.execute_logical_form(
                [stage3a_step, stage3b_step], window_idx
            )
            exec_results[3] = stage3_results.get(3, [])
            exec_results[4] = stage3_results.get(4, [])

        return exec_results

    def _fuse_evidence(
        self,
        anomalous_sensors: List[Dict],
        violations: List[Dict],
        window_idx: int,
    ) -> Dict[str, float]:
        """
        Fuse multiple evidence sources with reliability weights.

        Weights based on tool success rates:
        - anomaly_score: 0.5 (get_anomalous_sensors: 5.56% success)
        - violations: 0.3 (get_violations: strong signal when present)
        - temporal onset: 0.1 (earlier onset indicates root cause)
        - centrality: 0.1 (if available)

        Args:
            anomalous_sensors: List of dicts with 'sensor' and 'score' keys
            violations: List of dicts with 'source', 'target', 'deviation' keys
            window_idx: Window index

        Returns:
            Dictionary mapping sensor_name -> fused fault score
        """
        fault_scores = {}

        # Get centrality if available
        centrality_map = {}
        try:
            centrality = self.queries.compute_sensor_centrality(window_idx)
            centrality_map = {c["sensor"]: c.get("degree", 1) for c in centrality}
        except Exception:
            pass  # Continue without centrality if unavailable

        # Weight 1: Anomaly scores (0.5 weight, reduced from 0.6 to make room for temporal/centrality)
        for sensor_data in anomalous_sensors:
            if isinstance(sensor_data, dict) and "sensor" in sensor_data:
                sensor_name = sensor_data["sensor"]
                anomaly_score = sensor_data.get("score", 0.0)
                fault_scores[sensor_name] = (
                    fault_scores.get(sensor_name, 0.0) + 0.5 * anomaly_score
                )

        # Weight 2: Violation counts with severity weighting (0.3 weight)
        violation_counts = {}
        violation_deviations = {}
        severe_violation_counts = {}  # Count severe violations (>0.5 deviation)

        for violation in violations:
            if isinstance(violation, dict):
                source = violation.get("source", "")
                target = violation.get("target", "")
                deviation = violation.get("deviation", 0.0)
                is_severe = deviation > 0.5

                if source:
                    violation_counts[source] = violation_counts.get(source, 0) + 1
                    violation_deviations[source] = (
                        violation_deviations.get(source, 0.0) + deviation
                    )
                    if is_severe:
                        severe_violation_counts[source] = (
                            severe_violation_counts.get(source, 0) + 1
                        )
                if target:
                    violation_counts[target] = violation_counts.get(target, 0) + 1
                    violation_deviations[target] = (
                        violation_deviations.get(target, 0.0) + deviation
                    )
                    if is_severe:
                        severe_violation_counts[target] = (
                            severe_violation_counts.get(target, 0) + 1
                        )

        # Normalize violation scores (max deviation = 1.0, count normalized by max count)
        max_deviation = (
            max(violation_deviations.values()) if violation_deviations else 1.0
        )
        max_count = max(violation_counts.values()) if violation_counts else 1
        max_severe = (
            max(severe_violation_counts.values()) if severe_violation_counts else 1
        )

        for sensor_name, count in violation_counts.items():
            deviation_sum = violation_deviations.get(sensor_name, 0.0)
            severe_count = severe_violation_counts.get(sensor_name, 0)

            # Combine count, deviation, and severe violations
            # Weight severe violations more heavily
            violation_score = (
                (count / max_count if max_count > 0 else 0) * 0.3
                + (deviation_sum / max_deviation if max_deviation > 0 else 0) * 0.4
                + (severe_count / max_severe if max_severe > 0 else 0) * 0.3
            )
            fault_scores[sensor_name] = (
                fault_scores.get(sensor_name, 0.0) + 0.3 * violation_score
            )

        # Weight 3: Temporal onset (0.1 weight) - earlier onset indicates root cause
        for sensor_name in fault_scores.keys():
            try:
                onset = self._find_anomaly_onset(sensor_name, window_idx)
                # Earlier onset -> larger score: (window_idx - onset + 1) normalized
                onset_score = (
                    float(window_idx - onset + 1) / (window_idx + 1)
                    if window_idx >= 0
                    else 1.0
                )
                fault_scores[sensor_name] = (
                    fault_scores.get(sensor_name, 0.0) + 0.1 * onset_score
                )
            except Exception:
                pass  # Continue without temporal onset if unavailable

        # Weight 4: Centrality (0.1 weight) - well-connected sensors more likely root cause
        max_centrality = max(centrality_map.values()) if centrality_map else 1
        for sensor_name in fault_scores.keys():
            centrality = centrality_map.get(sensor_name, 1)
            centrality_score = (
                (centrality / max_centrality) if max_centrality > 0 else 0.0
            )
            fault_scores[sensor_name] = (
                fault_scores.get(sensor_name, 0.0) + 0.1 * centrality_score
            )

        return fault_scores

    def _compute_structured_confidence(
        self,
        anomalous_sensors: List[Dict],
        violations: List[Dict],
        window_idx: int,
    ) -> float:
        """
        Compute structured confidence score based on evidence strength.

        Formula: 0.3 (anomalies) + 0.4 (violation_severity) + 0.3 (centrality)
        Uses violation severity weighting (>0.5 deviation = severe).

        Args:
            anomalous_sensors: List of anomalous sensor dicts
            violations: List of violations
            window_idx: Window index

        Returns:
            Confidence score (0.0-1.0)
        """
        score = 0.0

        # Anomalous sensors found (0.3 weight)
        if anomalous_sensors:
            score += 0.3

        # Violation severity (0.4 weight, weighted by deviation magnitude)
        violation_severity_score = self._violation_severity(violations)
        score += 0.4 * violation_severity_score

        # High centrality (0.3 weight) - well-connected sensor indicates root cause
        try:
            centrality = self.queries.compute_sensor_centrality(window_idx)
            if centrality and centrality[0].get("degree", 0) > 3:
                score += 0.3
        except Exception:
            pass  # Continue without centrality if unavailable

        return min(score, 1.0)

    def _build_planning_prompt(
        self, question: str, memory: Dict, trace: List[Dict]
    ) -> str:
        """
        Build prompt for LLM to generate logical form steps following KAG paper format.

        This implements Stage 1 of the KAG two-stage prompting strategy:
        - Planning Prompt: Generates structured logical form with symbolic operators
        - The logical form is then executed to produce structured reasoning results
        """
        schema_block = """
**Knowledge Graph Schema:**

**Entities:**
- **Sensor** with properties:
  - name: Sensor identifier (e.g., "VEHICLE_SPEED")
  - subsystem: Subsystem classification (e.g., "Drivetrain")
  - anomaly_score: GDN model prediction (0.0-1.0), highly reliable indicator
    * Scores > 0.5 indicate likely faults
    * Scores > 0.7 indicate very likely faults

- **Window** with properties:
  - idx: Window index
  - embedding[32]: 32-dimensional embedding vector from GDN model
  - dist_to_normal_center: Euclidean distance to learned normal class center
  - dist_to_anomalous_center: Euclidean distance to learned anomalous class center
  - confidence: Confidence score based on distance comparison
  - label: Ground truth label (if available)
  - fault_type: Fault type classification (if available)

- **ClassCenter** with properties:
  - class: "normal" | "anomalous"
  - embedding[32]: Class center embedding vector
  - mean_radius: Mean distance radius for this class

**Relations:**
- **HAS_READING**(Window -> Sensor) with properties:
  - anomaly_score (float): GDN prediction - reliable fault indicator
  - is_faulty (bool): Derived from anomaly_score threshold

- **CORRELATES_WITH**(Sensor -> Sensor) with properties:
  - correlation (float): Actual correlation value
  - expected_correlation (float): Expected correlation from training
  - deviation (float): |actual - expected| - deviations > 0.3 indicate violations
  - is_violation (bool): True when deviation > threshold (STRONGEST fault signal)

- **SIMILAR_TO**(Window -> Window) with properties:
  - similarity (float): Cosine similarity between window embeddings
  - distance (float): Euclidean distance between embeddings
  - Used to find similar past anomalies for pattern matching

- **DISTANCE_TO_CENTER**(Window -> ClassCenter) with properties:
  - distance (float): Euclidean distance to class center
  - z_score (float): Standardized distance score
"""

        operators_block = """
**Available Logical Operators (KAG Paper Format):**

The following 4 operators are available for traversing the knowledge graph:

1. **Retrieval(s=?, p=predicate, o=?, constraints={{...}})**
   Purpose: Query graph triples using SPO (Subject-Predicate-Object) pattern matching
   
   Parameters:
   - s: Subject (entity type or variable, e.g., Sensor, Window, or $step1_results)
   - p: Predicate (relation name, e.g., HAS_READING, CORRELATES_WITH)
   - o: Object (entity type or variable, or "?" for any)
   - constraints: Dictionary of constraints (e.g., {{"window_idx": 42, "anomaly_score": {{"operation": "gt", "value": 0.6}}}})
   
   Alternative format (for convenience):
   - operation: Operation type ("has_reading", "correlates_with", "find_similar_windows", "find_anomalous_neighbors")
   - subject: Subject constraints dict (e.g., {{"window_idx": 42}})
   - object: Object constraints dict (e.g., {{"anomaly_score": {{"operation": "gt", "value": 0.7}}}})
   
   Examples:
   - Retrieval(operation="has_reading", subject={{"window_idx": 42}}, object={{"anomaly_score": {{"operation": "gt", "value": 0.7}}}})
     → Find all sensors in window 42 with anomaly score > 0.7
   - Retrieval(operation="correlates_with", subject={{"window_idx": 42}}, object={{"deviation_threshold": 0.3, "is_violation": true}})
     → Find correlation violations in window 42 (strongest fault signal - prioritize large deviations)
   - Retrieval(operation="sensors_with_violations_and_anomaly", subject={{"window_idx": 42}}, object={{"anomaly_threshold": 0.5, "min_violations": 2}})
     → Find sensors with high anomaly scores AND multiple violated correlations (combined strong signal)
   - Retrieval(operation="temporal_retrieval", subject={{"sensor": "ENGINE_RPM", "window_idx": 42}}, object={{"window_range": [40, 42]}})
     → Retrieve anomaly history for ENGINE_RPM in windows 40-42 (temporal propagation signal)
   - Retrieval(operation="find_similar_windows", subject={{"window_idx": 42}}, object={{"k": 5, "threshold": 0.2}})
     → Find 5 most similar windows via embedding distance

2. **Math(operation, operands)**
   Purpose: Numerical computation over operands
   
   Parameters:
   - operation: Operation type ("count", "sum", "max", "mean")
   - operands: List of operands (can reference previous steps with $step<id>_results)
   
   Examples:
   - Math(operation="count", operands=["$step1_results"])
     → Count how many items are in step 1 results
   - Math(operation="max", operands=["$step2_results", "anomaly_score"])
     → Find maximum anomaly score from step 2 results

3. **Deduce(left, right, comparison)**
   Purpose: Logical comparison or root cause extraction
   
   Parameters:
   - left: Left operand (can reference previous steps with $step<id>_results)
   - right: Right operand (value or step reference)
   - comparison: Comparison type ("greater", "less", "equal", "contains", "extract_root_cause")
   
   Examples:
   - Deduce(left="$step3_result", right=2, comparison="greater")
     → Check if step 3 result > 2 (indicates multi-sensor fault)
   - Deduce(left="$step1_results", right="$step2_results", comparison="extract_root_cause")
     → Identify root cause sensor from violations (step1) and anomalies (step2)

4. **Sort(target, key, direction, limit)**
   Purpose: Rank and filter results
   
   Parameters:
   - target: Target set to sort (can reference previous steps with $step<id>_results)
   - key: Key to sort by (e.g., "anomaly_score", "deviation", "distance")
   - direction: Sort direction ("asc" or "desc")
   - limit: Optional limit on number of results to return
   
   Examples:
   - Sort(target="$step1_results", key="anomaly_score", direction="desc", limit=3)
     → Sort sensors by anomaly_score descending, return top 3
   - Sort(target="$step2_results", key="deviation", direction="desc", limit=5)
     → Sort violations by deviation descending, return top 5
"""

        strategy_guidance = """
**Diagnostic Strategy Guidance (Aligned with Serialized KG Signals):**

**Priority Signals (use in this order - these are what make serialized KG powerful):**

1. **Violation Edges with Large Deviation** (STRONGEST signal):
   - Use `Retrieval(operation="correlates_with", object={"deviation_threshold": 0.3, "is_violation": true})`
   - Focus on edges with large `deviation_from_gdn` (stored as `deviation` in Neo4j)
   - Violations indicate broken sensor relationships - strongest root cause signal
   - Sensors involved in violations with deviation > 0.5 are very likely root causes
   - Query FIRST when diagnosing faults - this is the most reliable signal

2. **Sensors with High Anomaly + Many Violated Correlations** (Combined signal):
   - Use `Retrieval(operation="correlates_with", object={"is_violation": true})` to get violations
   - Then identify sensors that appear in multiple violations AND have high anomaly scores
   - Sensors with anomaly_score > 0.5 AND 2+ violations are strong root cause candidates
   - This combines GDN predictions with relationship breakdown evidence

3. **Temporal Propagation Edges** (Fault evolution):
   - Use `Retrieval(operation="temporal_retrieval", object={"window_range": [t-2, t-1, t]})`
   - Identify sensors that become anomalous earliest (temporal onset)
   - Sensors that show increasing anomaly scores over time are likely root causes
   - Faults typically propagate from root sensors to affected sensors over time

4. **Anomalous Sensors** (Secondary signal):
   - Use `Retrieval(operation="has_reading", object={"anomaly_score": {"operation": "gt", "value": 0.7}})`
   - Start with threshold=0.7 (high confidence) to reduce false positives
   - ALWAYS combine with violations - anomaly alone is weaker than violations
   - If no high-threshold anomalies, fall back to threshold=0.5

5. **Similar Windows** (Pattern matching):
   - Use `Retrieval(operation="find_similar_windows", object={"k": 5, "threshold": 0.2})`
   - Find windows with similar embedding patterns that were anomalous
   - Helps identify fault types by matching to historical patterns

**Key Insight from Serialized KG:**
The serialized KG method is powerful because it shows ALL violations, temporal patterns, and high-anomaly sensors in one view. Your KAG plan should retrieve these same signals:
- **Violation edges** with large deviation (deviation > 0.3, prioritize deviation > 0.5)
- **Temporal propagation** (sensors that become anomalous early)
- **Sensors with high anomaly + many violations** (combined evidence)

**Recommended Hierarchical Strategy:**
- Step 1: Query violations FIRST (strongest signal) - focus on large deviations (>0.3, prioritize >0.5)
- Step 2: Identify sensors involved in multiple violations AND have high anomaly scores
- Step 3: Query temporal history to see which sensors became anomalous earliest
- Step 4: Query anomalies with threshold=0.7 for sensors with violations (refinement)
- Step 5: Combine evidence - sensors with BOTH high anomaly scores AND violations are very likely faulty

**Exploration Guidance:**
- Prioritize violation edges with large deviation_from_gdn - these are the strongest signals
- Use temporal_retrieval to identify temporal onset (which sensors became anomalous first)
- Focus on sensors that combine multiple signals: high anomaly + many violations + early temporal onset
- If violations are sparse, widen deviation threshold or check temporal propagation patterns
"""

        examples_block = """
**Example Logical Form Generation:**

**Question:** What is the root cause fault in window 42, and which sensors did it affect?

**Logical Form Steps:**

Step 1: Retrieval(operation="correlates_with", subject={{"window_idx": 42}}, 
                  object={{"deviation_threshold": 0.3, "is_violation": true}})
   → Find all correlation violations in window 42 (strongest fault signal)

Step 2: Retrieval(operation="has_reading", subject={{"window_idx": 42}},
                  object={{"anomaly_score": {{"operation": "gt", "value": 0.7}}}})
   → Find sensors with high anomaly scores (>0.7) to validate violations

Step 3: Math(operation="count", operands=["$step1_results"])
   → Count how many violations were found

Step 4: Deduce(left="$step3_result", right=2, comparison="greater")
   → Check if violation count > 2 (indicates multi-sensor fault)

Step 5: Retrieval(operation="find_similar_windows", subject={{"window_idx": 42}},
                  object={{"k": 5, "threshold": 0.2}})
   → Find 5 most similar windows in embedding space to identify fault patterns

Step 6: Deduce(left="$step1_results", right="$step2_results", comparison="extract_root_cause")
   → Identify root cause sensor considering violations, anomaly scores, and embedding similarity
"""

        memory_str = json.dumps(memory, indent=2) if memory else "None"
        trace_brief = json.dumps(trace[-3:], indent=2) if trace else "None"

        return f"""
You are a diagnostic reasoning agent. Decompose this question into logical form steps.

**Question:** {question}

{schema_block}

{operators_block}

{strategy_guidance}

{examples_block}

**Previous Findings (may be empty):**
{memory_str}

**Previous Reasoning Trace (last 3 steps):**
{trace_brief}

**Output Format:**
Generate 3-6 logical form steps. Each step MUST follow this exact format:

Step <id>: <Operator>(<parameters_as_dict>)
   → <short natural language description>

**Constraints:**
- Use only the operators: Retrieval, Math, Deduce, Sort
- Use integer step ids starting from 1
- Use "$step<id>_results" to reference outputs from previous steps
- For Retrieval operations, use operation="has_reading", "correlates_with", "find_similar_windows", "find_anomalous_neighbors", "temporal_retrieval", or "explore_neighborhood"
- Parameters should be valid Python dict syntax (use double braces {{}} for dicts in f-strings)
- Consider using optional exploration operators if initial evidence is insufficient

**Now produce the logical-form steps for this question ONLY. Do not answer the question yet.**
"""

    def generate_logical_form(
        self, question: str, memory: Dict, trace: List[Dict]
    ) -> List[LogicalFormStep]:
        """Generate logical form steps from LLM."""
        prompt = self._build_planning_prompt(question, memory, trace)
        response = call_llm(
            prompt, self.model, self.tokenizer, max_tokens=1024, temperature=0.3
        )
        return parse_logical_form(response)

    def execute_logical_form(
        self, steps: List[LogicalFormStep], window_idx: int
    ) -> Dict[int, Any]:
        """Execute logical form steps and return results."""
        step_results = {}

        for step in steps:
            try:
                # Add window_idx to step params for tracking
                step.params["window_idx"] = window_idx

                if step.operator == "Retrieval":
                    result = self.executor.execute_retrieval(
                        step, step_results, window_idx
                    )
                elif step.operator == "Math":
                    result = self.executor.execute_math(step, step_results)
                elif step.operator == "Deduce":
                    result = self.executor.execute_deduce(step, step_results)
                elif step.operator == "Sort":
                    result = self.executor.execute_sort(step, step_results)
                else:
                    result = None

                step_results[step.step_id] = result
            except Exception:
                # On error, store None for this step
                step_results[step.step_id] = None

        return step_results

    def _format_exec_results(self, exec_results: Dict[int, Any]) -> str:
        """
        Format execution results for LLM answer synthesis following KAG paper format.

        Structures the output to show step-by-step reasoning results with clear
        interpretation guidance, matching the KAG paper's structured reasoning format.
        """
        lines = []
        lines.append("**Structured Reasoning Results:**")
        lines.append("")
        lines.append(
            "The following results were obtained by executing logical form steps over the knowledge graph:"
        )
        lines.append("")

        for step_id, result in sorted(exec_results.items()):
            # Determine operation type from result structure
            operation_name = "Unknown Operation"
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    if "score" in result[0] and "sensor" in result[0]:
                        operation_name = "Anomalous Sensors Found"
                    elif "deviation" in result[0]:
                        operation_name = "Correlation Violations Found"
                    elif "window_idx" in result[0] and "similarity" in result[0]:
                        operation_name = "Similar Windows Found"
                    elif "distance" in result[0]:
                        operation_name = "Anomalous Neighbors Found"
                elif isinstance(result, (int, float)):
                    operation_name = "Math Operation Result"
            elif isinstance(result, (int, float)):
                operation_name = "Math Operation Result"
            elif isinstance(result, str):
                operation_name = "Deduce Operation Result"

            lines.append(f"**Step {step_id} - {operation_name}:**")

            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], dict):
                    # Format list of dicts
                    lines.append(f"Found {len(result)} items:")
                    lines.append("")

                    # Add interpretation for anomalous sensors
                    if "score" in result[0] and "sensor" in result[0]:
                        lines.append(
                            "*Interpretation: Anomaly scores > 0.5 indicate likely faults, scores > 0.7 indicate very likely faults*"
                        )
                        lines.append("")
                        for i, item in enumerate(result[:5]):  # Show first 5
                            score = item.get("score", 0)
                            severity = (
                                "HIGH"
                                if score > 0.7
                                else "MEDIUM"
                                if score > 0.5
                                else "LOW"
                            )
                            sensor_name = item.get("sensor", "UNKNOWN")
                            subsystem = item.get("subsystem", "")
                            lines.append(
                                f"- {sensor_name}"
                                + (f" ({subsystem})" if subsystem else "")
                            )
                            lines.append(
                                f"  - Anomaly Score: {score:.3f} ({severity} severity)"
                            )
                        if len(result) > 5:
                            lines.append(f"- ... and {len(result) - 5} more sensors")

                    # Add interpretation for violations
                    elif "deviation" in result[0]:
                        lines.append(
                            "*Interpretation: Deviations > 0.3 indicate correlation violations - these are STRONG fault signals. Deviations > 0.5 are severe violations.*"
                        )
                        lines.append("")
                        for i, item in enumerate(result[:5]):
                            deviation = item.get("deviation", 0)
                            severity = (
                                "SEVERE"
                                if deviation > 0.5
                                else "MODERATE"
                                if deviation > 0.3
                                else "MILD"
                            )
                            source = item.get("source", "?")
                            target = item.get("target", "?")
                            actual = item.get("actual", 0)
                            expected = item.get("expected", 0)
                            lines.append(f"- {source} ↔ {target}")
                            lines.append(f"  - Deviation: {deviation:.3f} ({severity})")
                            lines.append(
                                f"  - Actual correlation: {actual:.3f}, Expected: {expected:.3f}"
                            )
                        if len(result) > 5:
                            lines.append(f"- ... and {len(result) - 5} more violations")

                    # Add interpretation for similar windows
                    elif "window_idx" in result[0] and "similarity" in result[0]:
                        lines.append(
                            "*Interpretation: Similar windows help identify fault patterns by matching to historical anomalies*"
                        )
                        lines.append("")
                        for i, item in enumerate(result[:5]):
                            window_idx = item.get("window_idx", "?")
                            similarity = item.get("similarity", 0)
                            distance = item.get("distance", 0)
                            lines.append(f"- Window {window_idx}")
                            lines.append(
                                f"  - Similarity: {similarity:.3f}, Distance: {distance:.3f}"
                            )
                            if (
                                "dist_normal" in item
                                and item["dist_normal"] is not None
                            ):
                                lines.append(
                                    f"  - Distance to normal center: {item['dist_normal']:.3f}"
                                )
                            if (
                                "dist_anomalous" in item
                                and item["dist_anomalous"] is not None
                            ):
                                lines.append(
                                    f"  - Distance to anomalous center: {item['dist_anomalous']:.3f}"
                                )
                        if len(result) > 5:
                            lines.append(
                                f"- ... and {len(result) - 5} more similar windows"
                            )

                    # Add interpretation for anomalous neighbors
                    elif "distance" in result[0] and "window_idx" in result[0]:
                        lines.append(
                            "*Interpretation: Anomalous neighbors validate current window's anomaly status via embedding similarity*"
                        )
                        lines.append("")
                        for i, item in enumerate(result[:5]):
                            window_idx = item.get("window_idx", "?")
                            distance = item.get("distance", 0)
                            similarity = item.get("similarity", 0)
                            lines.append(f"- Window {window_idx}")
                            lines.append(
                                f"  - Distance: {distance:.3f}, Similarity: {similarity:.3f}"
                            )
                        if len(result) > 5:
                            lines.append(f"- ... and {len(result) - 5} more neighbors")

                    else:
                        # Generic formatting for other result types
                        for i, item in enumerate(result[:5]):
                            lines.append(
                                f"- Item {i + 1}: {json.dumps(item, indent=2)}"
                            )
                        if len(result) > 5:
                            lines.append(f"- ... and {len(result) - 5} more items")
                elif len(result) == 0:
                    lines.append("No items found.")
                else:
                    lines.append(f"Result: {result}")
            elif isinstance(result, (int, float)):
                lines.append(f"Result: {result}")
                if isinstance(result, int):
                    lines.append(
                        "*Interpretation: Count or numerical result from Math operation*"
                    )
                else:
                    lines.append(
                        "*Interpretation: Numerical result from Math operation*"
                    )
            elif isinstance(result, str):
                lines.append(f"Result: {result}")
                lines.append("*Interpretation: Root cause sensor or deduced result*")
            elif result is None:
                lines.append("(no results)")
            else:
                lines.append(f"Result: {result}")

            lines.append("")

        return "\n".join(lines)

    def _generate_answer(
        self,
        question: str,
        exec_results: Dict,
        memory: Dict,
        window_idx: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Generate final answer from execution results following KAG paper format.

        This implements Stage 2 of the KAG two-stage prompting strategy:
        - Answer Synthesis Prompt: Synthesizes final answer from structured reasoning results
        - Uses only structured evidence from logical form execution
        """
        # STEP 1: Assess evidence strength BEFORE LLM call
        evidence_strength = self._assess_evidence_strength(exec_results)

        # STEP 2: If evidence is weak, immediately return NORMAL prediction
        # Only filter MODERATE if BOTH score is low AND violations are few
        # This preserves faulty windows that have either high scores OR many violations
        if evidence_strength["level"] == "NONE" or evidence_strength["level"] == "WEAK":
            reasoning = self._format_weak_evidence_reasoning(evidence_strength)
            confidence = evidence_strength["confidence"]
            response = f"Faulty Sensors: []\nFault Type: NORMAL\nReasoning: {reasoning}\nConfidence: {confidence:.3f}"
            return response, confidence

        # STEP 3: For MODERATE or STRONG evidence, proceed with LLM
        exec_str = self._format_exec_results(exec_results)

        # Add embedding context if available
        embedding_section = ""
        if (
            window_idx is not None
            and hasattr(self.kg_builder, "window_embeddings")
            and window_idx in self.kg_builder.window_embeddings
        ):
            embedding_data = self.kg_builder.window_embeddings[window_idx]
            dist_normal = embedding_data.get("dist_normal", 0.0)
            dist_anomalous = embedding_data.get("dist_anomalous", 0.0)
            confidence_emb = embedding_data.get("confidence", 0.5)

            embedding_section = f"""
**Embedding Analysis:**
- Distance to normal center: {dist_normal:.4f}
- Distance to anomalous center: {dist_anomalous:.4f}
- Embedding confidence: {confidence_emb:.3f}
- Interpretation: """
            if dist_normal < 0.12:
                embedding_section += "Likely normal (close to normal center)"
            elif dist_normal > 0.12 and dist_anomalous < 0.15:
                embedding_section += "Likely anomalous (closer to anomalous center)"
            else:
                embedding_section += "Uncertain/edge case (intermediate distances)"
            embedding_section += "\n"

        mem_str = json.dumps(memory, indent=2) if memory else "None"

        # Extract valid sensors for validation
        valid_sensors = self._extract_valid_sensors(exec_results)

        evidence_summary = (
            f"- Maximum anomaly score: {evidence_strength['max_score']:.3f}\n"
            f"- Sensors flagged: {evidence_strength['num_anomalous']}\n"
            f"- Correlation violations: {evidence_strength['num_violations']}\n"
            f"- Evidence strength: {evidence_strength['level']}"
        )

        prompt = f"""
You are an automotive diagnostic expert.

**Original Question:** {question}

**Structured Reasoning Results:**

The following results were obtained by executing logical form steps over the knowledge graph. Use ONLY this structured evidence to produce your answer.

{exec_str}
{embedding_section}
**Evidence Summary:**
{evidence_summary}

**Evidence Hierarchy (use in order of reliability):**
1. **Correlation Violations** (STRONGEST SIGNAL): Deviations > 0.3 indicate broken sensor relationships - these are the MOST reliable fault indicators. Sensors with violations are likely root causes.
2. **High Anomaly Scores** (>0.7): Very likely faulty sensors
3. **Medium Anomaly Scores** (0.5-0.7): Possibly faulty, but validate with violations if available
4. **Low Anomaly Scores** (<0.5): Likely normal, but check violations to confirm

**CRITICAL: Missing Violations Indicates Normal Operation**
- **If violations are missing** (Step 1 returns None/empty), this STRONGLY indicates normal operation even if anomaly scores are elevated
- **Only predict faults when BOTH violations exist AND scores are high (>0.7)**
- High scores without violations suggest sensor noise or transient variations, not actual faults

**Evidence Fusion Strategy:**
- **Combine signals**: A sensor with BOTH high anomaly score (>0.7) AND violations is VERY likely faulty
- **Root cause identification**: Sensors with the MOST violations are often root causes
- **Affected sensors**: Sensors connected via violations to root cause are affected sensors
- **No fault prediction**: If max anomaly score < 0.6 AND violations < 2, predict NORMAL (faulty_sensors: [])
- **No violations = normal**: If no violations are found, predict NORMAL even if scores are elevated

**Global Memory (across iterations):**
{mem_str}

**Answer Format:**

Using only the structured evidence above, produce a JSON object in this EXACT format (no other text, no markdown, no code blocks):

{{
    "root_cause_sensors": ["SENSOR_NAME"] or [],
    "affected_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],
    "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [],  # backward compatibility (root + affected combined)
    "fault_type": "VSS_DROPOUT" | "COOLANT_DROPOUT" | "MAF_SCALE_LOW" | "TPS_STUCK" | "gradual_drift" or null,
    "reasoning": "3-6 sentences, extensive and didactic: (1) state which evidence you used (which Step results, which violations/scores and values), (2) explain step-by-step how that evidence leads to the root cause or to normal operation, (3) briefly say why other sensors were or were not considered. Write so a reader can follow your logic.",
    "confidence": 0.85
}}

**IMPORTANT:**
- **root_cause_sensors**: The PRIMARY sensor(s) causing the fault (usually 1 sensor, the one with most violations or highest anomaly score)
- **affected_sensors**: Secondary sensors that are affected by the root cause but are NOT the primary fault source
- **faulty_sensors**: For backward compatibility, include ALL faulty sensors (root + affected combined)

**Available Sensor Names:** {", ".join([s.replace(" ()", "") for s in self.sensor_names])}

**Example Valid JSON Responses:**

Example 1 (with fault):
{{"root_cause_sensors": ["VEHICLE_SPEED"], "affected_sensors": [], "faulty_sensors": ["VEHICLE_SPEED"], "fault_type": "VSS_DROPOUT", "reasoning": "VEHICLE_SPEED has multiple severe correlation violations (deviations >0.5) with ENGINE_RPM and INTAKE_MANIFOLD_PRESSURE, indicating broken relationships. Combined with high anomaly score (0.85) from Step 2 results, this strongly indicates VSS dropout fault as root cause.", "confidence": 0.90}}

Example 2 (with fault and affected sensors):
{{"root_cause_sensors": ["COOLANT_TEMPERATURE"], "affected_sensors": ["ENGINE_LOAD"], "faulty_sensors": ["COOLANT_TEMPERATURE", "ENGINE_LOAD"], "fault_type": "COOLANT_DROPOUT", "reasoning": "COOLANT_TEMPERATURE shows correlation violations with ENGINE_LOAD (deviation 0.42) from Step 1 results and anomaly score of 0.72 from Step 2 results. ENGINE_LOAD is affected by the coolant sensor dropout but is not the root cause.", "confidence": 0.80}}

Example 3 (no fault):
{{"root_cause_sensors": [], "affected_sensors": [], "faulty_sensors": [], "fault_type": null, "reasoning": "All sensors show low anomaly scores (<0.5) from Step 2 results and NO correlation violations detected in Step 1 results. This indicates normal operation with no faults present.", "confidence": 0.90}}

Example 4 (no fault, despite high scores):
{{"root_cause_sensors": [], "affected_sensors": [], "faulty_sensors": [], "fault_type": null, "reasoning": "Step 2 results show some sensors with elevated anomaly scores (0.65-0.75), but Step 1 found NO correlation violations. Since correlation violations are the STRONGEST signal and none were detected, this indicates normal operation despite elevated scores. High scores without violations suggest sensor noise or transient variations, not actual faults.", "confidence": 0.85}}

**Critical Rules:**
1. **You can ONLY name sensors that appear in the structured results above** - Do NOT invent sensors
2. **Prioritize violations**: If violations exist in Step results, use them to identify root cause (sensor with most violations)
3. **Combine evidence**: Use BOTH anomaly scores AND violations when both are available
4. **NORMAL prediction rule**: If max anomaly score < 0.6 AND violations < 2, predict NORMAL (root_cause_sensors: [], affected_sensors: [], faulty_sensors: [], fault_type: null)
5. **Reasoning must be extensive and didactic**: Write 3-6 sentences. State which evidence you used (Step 1 violations, Step 2 scores, exact values), explain step-by-step how it leads to your conclusion, and why other sensors were or were not chosen. Refer to specific Step results (e.g., "from Step 1 results", "Step 2 found...").
6. **Violation severity**: Severe violations (deviation >0.5) are stronger signals than mild violations
7. **Root cause = sensor with most violations**: When multiple sensors have violations, prioritize the one with most violations as root_cause_sensors
8. **Affected sensors**: Sensors that show anomalies but are NOT the primary fault source should go in affected_sensors
9. **JSON ONLY**: Output ONLY valid JSON. No markdown code blocks, no explanatory text, just the JSON object.

**Available sensor names from tool results:** {", ".join(sorted(valid_sensors)) if valid_sensors else "None"}

Now produce the JSON response using ONLY the structured reasoning results above. Do NOT invent extra sensors or signals.
"""

        # Log prompt length for debugging (chars, words as rough token proxy)
        n_chars = len(prompt)
        n_words = len(prompt.split())
        if window_idx is not None:
            print(f"  [Window {window_idx}] Answer prompt: {n_chars} chars, ~{n_words} words (~{n_words * 4 // 3} tokens est.)")

        # One-time dump of example prompt to file for inspection
        if not getattr(self, "_example_prompt_dumped", True):
            try:
                out = Path("results/example_kg_answer_prompt.txt")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(prompt, encoding="utf-8")
                print(f"  Example prompt written to {out}")
                self._example_prompt_dumped = True
            except Exception:
                pass

        response = call_llm(
            prompt,
            self.model,
            self.tokenizer,
            max_tokens=512,
            temperature=0.3,  # Lower temperature for more consistent JSON
        )

        # STEP 4: Parse JSON response and validate
        confidence = 0.5
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
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_clean, re.DOTALL
            )
            if json_match:
                result = json.loads(json_match.group(0))
                confidence = float(result.get("confidence", 0.5))

                # STEP 5: Post-LLM validation - filter hallucinations
                claimed_sensors = result.get("faulty_sensors", [])
                validated_sensors = [s for s in claimed_sensors if s in valid_sensors]

                # Remove hallucinations
                if len(validated_sensors) < len(claimed_sensors):
                    num_removed = len(claimed_sensors) - len(validated_sensors)
                    # Adjust confidence downward if hallucinations detected
                    confidence = max(0.0, confidence - 0.1 * num_removed)

                # Normalize fault type
                raw_fault_type = result.get("fault_type", None)
                fault_type = (
                    self._normalize_fault_type(raw_fault_type)
                    if raw_fault_type
                    else None
                )

                # If no valid sensors, ensure fault_type is None
                if not validated_sensors:
                    fault_type = None

                reasoning = result.get("reasoning", "")
                response = f"Faulty Sensors: {validated_sensors}\nFault Type: {fault_type}\nReasoning: {reasoning}\nConfidence: {confidence:.3f}"
            else:
                # Fallback: try to extract confidence from text format
                if "Confidence:" in response:
                    try:
                        conf_str = (
                            response.split("Confidence:")[1].split("\n")[0].strip()
                        )
                        confidence = float(conf_str)
                    except Exception:
                        pass
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            # JSON parsing failed - return NORMAL with low confidence
            reasoning = f"LLM output parsing failed: {str(e)}. Evidence insufficient."
            confidence = 0.1
            response = f"Faulty Sensors: []\nFault Type: NORMAL\nReasoning: {reasoning}\nConfidence: {confidence:.3f}"

        return response, confidence

    def _reflect_and_refine(
        self,
        original_question: str,
        current_answer: str,
        exec_results: Dict,
        memory: Dict,
    ) -> str:
        """Reflect on low-confidence answer and generate exploratory follow-up question."""
        
        # Summarize what's been examined
        sensors_inspected = set()
        violations_checked = []
        candidates_ruled_out = []
        
        for step_id, step_results in exec_results.items():
            if isinstance(step_results, list):
                for item in step_results:
                    if isinstance(item, dict):
                        if "sensor" in item:
                            sensors_inspected.add(item["sensor"])
                        if "source" in item and "deviation" in item:
                            violations_checked.append({
                                "source": item.get("source", ""),
                                "target": item.get("target", ""),
                                "deviation": item.get("deviation", 0.0)
                            })
                        # Check for candidates that were examined but ruled out
                        if "score" in item and item.get("score", 0.0) < 0.5:
                            if "sensor" in item:
                                candidates_ruled_out.append(item["sensor"])
        
        # Format summary
        sensors_summary = ", ".join(sorted(list(sensors_inspected))[:10])  # Limit to 10 for brevity
        if len(sensors_inspected) > 10:
            sensors_summary += f" (and {len(sensors_inspected) - 10} more)"
        
        violations_summary = f"{len(violations_checked)} violations checked"
        if violations_checked:
            top_violation = max(violations_checked, key=lambda x: x.get("deviation", 0.0))
            violations_summary += f" (max deviation: {top_violation.get('deviation', 0.0):.3f})"
        
        candidates_summary = ", ".join(candidates_ruled_out[:5]) if candidates_ruled_out else "None"
        
        prompt = f"""
You attempted to answer this diagnostic question but the confidence was low.

Original question:
{original_question}

Current answer:
{current_answer}

**Summary of What Has Been Examined:**
- Sensors inspected: {sensors_summary if sensors_inspected else "None"}
- Violations checked: {violations_summary}
- Candidates ruled out: {candidates_summary}

Structured reasoning results (truncated):
{self._format_exec_results(exec_results)[:2000]}

Global memory:
{json.dumps(memory, indent=2)}

**Exploratory Reflection:**
1. Briefly state what is still uncertain (e.g., multiple candidate root sensors, missing temporal evidence, unexplored subsystems).
2. Identify which sensors or relationships have **not** been examined yet but might change the diagnosis.
3. Consider exploring:
   - Other subsystems that haven't been checked
   - Non-obvious correlations that might reveal root causes
   - Temporal patterns (how faults developed over time)
   - Neighborhood exploration around candidate sensors
4. Generate a follow-up query using the available operators to explore new evidence.

**Available Operators for Exploration:**
- Retrieval(operation="correlates_with", is_violation=true) - Find violations
- Retrieval(operation="correlates_with", constraints={{correlation < 0.2}}) - Find broken correlations
- Retrieval(operation="temporal_retrieval", window_range=[t-2, t]) - Get sensor history
- Retrieval(operation="explore_neighborhood", root=?, radius=2) - Expand k-hop neighborhood
- Retrieval(operation="has_reading", object={{"anomaly_score": {{"operation": "gt", "value": 0.5}}}}) - Find anomalous sensors

Format:
Uncertainty: <text>
Unexplored Areas: <sensors, relationships, or subsystems not yet examined>
Follow-up Question: <single new question to the knowledge graph that explores new evidence>
"""
        response = call_llm(
            prompt, self.model, self.tokenizer, max_tokens=512, temperature=0.7
        )

        # Extract follow-up question
        for line in response.splitlines():
            if line.startswith("Follow-up Question:"):
                return line.split(":", 1)[1].strip()

        return original_question  # fallback

    def _classify_fault_type(
        self, root_cause: str, affected: List[str], violations: List[Dict]
    ) -> Optional[str]:
        """
        Map root cause + affected sensors to fault type.

        Uses exact fault types from codebase:
        - VSS_DROPOUT for VEHICLE_SPEED faults
        - COOLANT_DROPOUT for COOLANT_TEMPERATURE faults
        - TPS_STUCK for THROTTLE faults
        - MAF_SCALE_LOW for INTAKE_MANIFOLD_PRESSURE faults
        - RPM_SPEED_DECOUPLE for RPM+SPEED decoupling
        - gradual_drift as default

        Args:
            root_cause: Base sensor name of root cause
            affected: List of affected base sensor names
            violations: List of violation dicts (unused but kept for API consistency)

        Returns:
            Fault type string or None if no root cause
        """
        if not root_cause:
            return None

        # Check for RPM_SPEED_DECOUPLE (both sensors affected)
        all_faulty = [root_cause] + affected
        has_rpm = any("ENGINE_RPM" in s for s in all_faulty)
        has_speed = any("VEHICLE_SPEED" in s for s in all_faulty)

        if has_rpm and has_speed:
            return "RPM_SPEED_DECOUPLE"

        # Check root cause sensor
        if "VEHICLE_SPEED" in root_cause:
            return "VSS_DROPOUT"
        elif "COOLANT_TEMPERATURE" in root_cause:
            return "COOLANT_DROPOUT"
        elif "THROTTLE" in root_cause:
            return "TPS_STUCK"
        elif "INTAKE_MANIFOLD_PRESSURE" in root_cause:
            return "MAF_SCALE_LOW"
        else:
            return "gradual_drift"

    def solve(self, window_idx: int) -> Dict:
        """
        Solve diagnostic question for a window using iterative LLM planning.

        Args:
            window_idx: Window index to analyze

        Returns:
            Dictionary with keys:
            - 'root_cause_sensor': str or None
            - 'affected_sensors': List[str]
            - 'fault_type': str
            - 'reasoning_trace': List[Dict]
            - 'confidence': float
            - 'sensor_labels': np.ndarray
            - 'window_label': int
        """
        memory = {}
        trace = []
        question = f"Analyze window {window_idx}: Are there any faulty sensors? If yes, identify which sensors are faulty and determine the root cause."

        for iteration in range(self.max_iterations):
            # Generate logical form
            steps = self.generate_logical_form(question, memory, trace)
            trace.append(
                {
                    "iteration": iteration + 1,
                    "question": question,
                    "steps": [
                        {
                            "id": s.step_id,
                            "operator": s.operator,
                            "description": s.description,
                        }
                        for s in steps
                    ],
                }
            )

            # Execute logical form
            exec_results = self.execute_logical_form(steps, window_idx)
            trace[-1]["execution_results"] = {
                k: str(v)[:100] for k, v in exec_results.items()
            }

            # Generate answer
            answer, llm_confidence = self._generate_answer(
                question, exec_results, memory, window_idx=window_idx
            )

            # Extract anomalous sensors and violations for structured confidence
            anomalous_sensors = []
            violations = []
            for step_id, result in exec_results.items():
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        if "sensor" in result[0] and "score" in result[0]:
                            # Anomalous sensors
                            anomalous_sensors.extend(result)
                        elif "source" in result[0] and "deviation" in result[0]:
                            # Violations
                            violations.extend(result)

            # Compute structured confidence
            structured_confidence = self._compute_structured_confidence(
                anomalous_sensors, violations, window_idx
            )

            # Hybrid confidence: 0.6 * LLM + 0.4 * structured
            confidence = 0.6 * llm_confidence + 0.4 * structured_confidence

            trace[-1]["answer"] = answer[:200]
            trace[-1]["confidence"] = confidence
            trace[-1]["llm_confidence"] = llm_confidence
            trace[-1]["structured_confidence"] = structured_confidence

            # Update memory
            memory[f"iteration_{iteration + 1}"] = {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "llm_confidence": llm_confidence,
                "structured_confidence": structured_confidence,
                "exec_results_summary": {
                    k: str(v)[:50] for k, v in exec_results.items()
                },
            }

            # Check if confidence is high enough
            if confidence >= 0.6:
                break

            # Violation-first fallback for low confidence
            if confidence < 0.5 and iteration == 0:
                # Try violation-first strategy: query violations first, then anomalies
                violation_steps = [
                    LogicalFormStep(
                        step_id=100,
                        operator="Retrieval",
                        params={
                            "operation": "correlates_with",
                            "subject": {"window_idx": window_idx},
                            "object": {
                                "deviation_threshold": 0.3,
                                "is_violation": True,
                            },
                        },
                        description="Query correlation violations (strong fault signal)",
                    )
                ]
                violation_results = self.execute_logical_form(
                    violation_steps, window_idx
                )

                # If violations found, query anomalies for those sensors
                if violation_results.get(100) and len(violation_results[100]) > 0:
                    violations = violation_results[100]
                    # Extract unique sensors from violations
                    violation_sensors = set()
                    for v in violations:
                        if isinstance(v, dict):
                            violation_sensors.add(v.get("source", ""))
                            violation_sensors.add(v.get("target", ""))
                    violation_sensors = [s for s in violation_sensors if s]

                    # Query anomalies with lower threshold for violation sensors
                    if violation_sensors:
                        anomaly_steps = [
                            LogicalFormStep(
                                step_id=101,
                                operator="Retrieval",
                                params={
                                    "operation": "has_reading",
                                    "subject": {"window_idx": window_idx},
                                    "object": {
                                        "anomaly_score": {
                                            "operation": "gt",
                                            "value": 0.5,
                                        }
                                    },
                                },
                                description="Query anomalous sensors to validate violations",
                            )
                        ]
                        anomaly_results = self.execute_logical_form(
                            anomaly_steps, window_idx
                        )

                        # Merge results
                        exec_results[100] = violation_results[100]
                        exec_results[101] = anomaly_results.get(101, [])

                        # Regenerate answer with violation-first evidence
                        answer, llm_confidence = self._generate_answer(
                            question, exec_results, memory, window_idx=window_idx
                        )

                        # Extract anomalous sensors and violations for structured confidence
                        anomalous_sensors_vf = []
                        violations_vf = []
                        for step_id, result in exec_results.items():
                            if isinstance(result, list) and len(result) > 0:
                                if isinstance(result[0], dict):
                                    if "sensor" in result[0] and "score" in result[0]:
                                        anomalous_sensors_vf.extend(result)
                                    elif (
                                        "source" in result[0]
                                        and "deviation" in result[0]
                                    ):
                                        violations_vf.extend(result)

                        # Compute structured confidence
                        structured_confidence_vf = self._compute_structured_confidence(
                            anomalous_sensors_vf, violations_vf, window_idx
                        )

                        # Hybrid confidence: 0.6 * LLM + 0.4 * structured
                        confidence = (
                            0.6 * llm_confidence + 0.4 * structured_confidence_vf
                        )

                        trace[-1]["answer"] = answer[:200]
                        trace[-1]["confidence"] = confidence
                        trace[-1]["llm_confidence"] = llm_confidence
                        trace[-1]["structured_confidence"] = structured_confidence_vf
                        trace[-1]["violation_first"] = True

            # Confidence-based exploration: trigger broader exploration if confidence is low
            confidence_threshold = 0.8
            
            # If confidence is high, stop early (unless we're in first iteration and want to explore)
            if confidence >= confidence_threshold and iteration > 0:
                # High confidence - no need for further exploration
                break
            
            # If confidence is medium/low, trigger reflection for broader exploration
            if confidence < confidence_threshold and iteration < self.max_iterations - 1:
                # Enhance reflection prompt with exploration suggestions
                enhanced_question = self._reflect_and_refine(
                    question, answer, exec_results, memory
                )
                
                # If reflection suggests exploration, update question
                if enhanced_question != question:
                    question = enhanced_question
                else:
                    # If reflection didn't change question, suggest exploration strategies
                    # Check if we should widen cutoffs or switch strategies
                    max_score = max([item.get("score", 0.0) for step_result in exec_results.values() 
                                    if isinstance(step_result, list) 
                                    for item in step_result 
                                    if isinstance(item, dict) and "score" in item], default=0.0)
                    num_violations = sum([1 for step_result in exec_results.values() 
                                         if isinstance(step_result, list) 
                                         for item in step_result 
                                         if isinstance(item, dict) and "deviation" in item])
                    
                    # Suggest widening anomaly cutoffs if scores are moderate
                    if 0.5 <= max_score < 0.7 and num_violations < 2:
                        question = f"Analyze window {window_idx} with broader criteria: include moderately anomalous sensors (score > 0.5) and check sensors with many correlations even if anomaly is moderate. What is the root cause?"
                    # Suggest checking correlations if violations are few
                    elif max_score >= 0.6 and num_violations < 2:
                        question = f"Analyze window {window_idx}: Check sensors with many correlations even if anomaly is moderate. Explore neighborhood around candidate sensors. What is the root cause?"
                    # Otherwise use enhanced reflection question
                    else:
                        question = enhanced_question

        # Extract sensors directly from execution results (more reliable than NL parsing)
        exec_sensors = self._extract_sensors_from_exec_results(exec_results)
        
        # Parse LLM answer for fault_type and reasoning
        prediction = parse_llm_response(answer, self.sensor_names)
        
        # Merge execution results (sensor labels) with LLM prediction (fault_type, reasoning)
        merged_prediction = self._merge_exec_and_llm_predictions(exec_sensors, prediction)
        
        # Use merged prediction
        # Note: merged_prediction["sensor_labels"] is root-only (for metrics)
        #       merged_prediction["sensor_labels_raw"] is all faulty sensors (for analysis)
        sensor_labels = merged_prediction["sensor_labels"]  # Root-only (for metrics)
        sensor_labels_raw = merged_prediction["sensor_labels_raw"]  # All sensors (root + affected)

        # Check if LLM explicitly said "no fault" (empty list or explicit text)
        llm_said_no_fault = (
            np.sum(sensor_labels) == 0
            and answer
            and (
                "Faulty Sensors: []" in answer
                or '"faulty_sensors": []' in answer
                or '"root_cause_sensors": []' in answer
                or "no fault" in answer.lower()
                or "no faulty sensors" in answer.lower()
            )
        )

        # Check for strong contradictory evidence (high scores >= 0.8 AND violations exist)
        has_strong_evidence = False
        max_score_in_results = 0.0
        has_violations = False

        for step_id, result in exec_results.items():
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    if "score" in result[0] and "sensor" in result[0]:
                        # Anomalous sensors
                        max_score_in_results = max(
                            max_score_in_results,
                            max(
                                [
                                    item.get("score", 0.0)
                                    for item in result
                                    if isinstance(item, dict)
                                ]
                            ),
                        )
                    elif "source" in result[0] and "deviation" in result[0]:
                        # Violations
                        has_violations = True

        # Only consider strong evidence if both high scores AND violations exist
        has_strong_evidence = max_score_in_results >= 0.8 and has_violations

        # Only use fallback if:
        # 1. LLM didn't explicitly say "no fault" AND
        # 2. Either parsing failed (no answer) OR there's strong contradictory evidence
        if np.sum(sensor_labels) == 0 and not llm_said_no_fault:
            # Only apply fallback if there's strong contradictory evidence
            if has_strong_evidence:
                # Try to extract from execution results (only with strong evidence)
                for step_id, result in exec_results.items():
                    if isinstance(result, list) and len(result) > 0:
                        # Check if this is anomalous sensors result
                        if isinstance(result[0], dict) and "sensor" in result[0]:
                            # Extract sensors with very high scores (>= 0.8) only
                            for item in result:
                                if isinstance(item, dict):
                                    sensor_name = item.get("sensor", "")
                                    score = item.get("score", 0.0)
                                    if (
                                        score >= 0.8
                                        and sensor_name in self.sensor_names
                                    ):
                                        idx = self.sensor_names.index(sensor_name)
                                        sensor_labels[idx] = 1.0

                        # Check if this is violations result (only use if we also have high scores)
                        elif (
                            isinstance(result[0], dict)
                            and "source" in result[0]
                            and max_score_in_results >= 0.8
                        ):
                            # Count violations per sensor
                            violation_counts = {}
                            for item in result:
                                if isinstance(item, dict):
                                    source = item.get("source", "")
                                    target = item.get("target", "")
                                    if source and source in self.sensor_names:
                                        violation_counts[source] = (
                                            violation_counts.get(source, 0) + 1
                                        )
                                    if target and target in self.sensor_names:
                                        violation_counts[target] = (
                                            violation_counts.get(target, 0) + 1
                                        )

                            # Mark sensors with most violations (only if we have strong evidence)
                            if violation_counts:
                                max_violations = max(violation_counts.values())
                                for sensor_name, count in violation_counts.items():
                                    if (
                                        count == max_violations
                                        and sensor_name in self.sensor_names
                                    ):
                                        idx = self.sensor_names.index(sensor_name)
                                        sensor_labels[idx] = 1.0

        # Extract root cause and affected sensors from merged prediction
        root_cause_sensors_list = merged_prediction.get("root_cause_sensors", [])
        affected_sensors_list = merged_prediction.get("affected_sensors", [])
        
        root_cause_sensor = root_cause_sensors_list[0] if root_cause_sensors_list else None
        fault_type = merged_prediction.get("fault_type", None)

        return {
            "root_cause_sensor": root_cause_sensor,
            "affected_sensors": affected_sensors_list,
            "fault_type": fault_type,
            "reasoning_trace": trace,
            "confidence": confidence,
            "sensor_labels": sensor_labels,  # Root-only (for precision improvement)
            "sensor_labels_raw": sensor_labels_raw,  # All sensors (root + affected)
            "window_label": merged_prediction.get("window_label", 0),
        }
