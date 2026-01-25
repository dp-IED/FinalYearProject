"""
Tool Usage Tracker for Solver V2 Evaluation

Tracks which tools/functions are called during reasoning and analyzes
their usefulness by correlating tool usage with prediction correctness.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import numpy as np


class ToolTracker:
    """Tracks tool calls and their outcomes to analyze usefulness."""

    def __init__(self):
        """Initialize tool tracker."""
        # Store raw tool calls: window_idx -> list of tool call records
        self.tool_calls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        # Store window results: window_idx -> correctness info
        self.window_results: Dict[int, Dict[str, Any]] = {}

        # Aggregated statistics (computed on demand)
        self._tool_stats: Optional[Dict[str, Any]] = None
        self._operator_stats: Optional[Dict[str, Any]] = None

    def record_tool_call(
        self,
        tool_name: str,
        query_method: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        result: Any = None,
        window_idx: int = -1,
        operator: Optional[str] = None,
    ):
        """
        Record a tool call.

        Args:
            tool_name: Name of the tool/operator (e.g., "Retrieval", "Math")
            query_method: Specific query method called (e.g., "get_anomalous_sensors")
            params: Dictionary of parameters used
            result: Result returned by the tool
            window_idx: Window index where tool was called
            operator: Operator type (same as tool_name, for consistency)
        """
        if operator is None:
            operator = tool_name

        # Determine result size and type
        result_size = 0
        result_type = type(result).__name__
        if isinstance(result, list):
            result_size = len(result)
        elif isinstance(result, (int, float)):
            result_size = 1
            result_type = "numeric"
        elif result is not None:
            result_size = 1

        call_record = {
            "tool_name": tool_name,
            "operator": operator,
            "query_method": query_method,
            "params": params or {},
            "result_size": result_size,
            "result_type": result_type,
            "window_idx": window_idx,
        }

        self.tool_calls[window_idx].append(call_record)

        # Invalidate cached stats
        self._tool_stats = None
        self._operator_stats = None

    def mark_window_result(
        self,
        window_idx: int,
        is_correct: bool,
        predicted_sensors: Optional[List[str]] = None,
        true_sensors: Optional[List[str]] = None,
        predicted_window_label: Optional[int] = None,
        true_window_label: Optional[int] = None,
    ):
        """
        Mark the final result for a window.

        Args:
            window_idx: Window index
            is_correct: Whether the prediction was correct
            predicted_sensors: List of predicted faulty sensor names
            true_sensors: List of true faulty sensor names
            predicted_window_label: Predicted window label
            true_window_label: True window label
        """
        self.window_results[window_idx] = {
            "is_correct": is_correct,
            "predicted_sensors": predicted_sensors or [],
            "true_sensors": true_sensors or [],
            "predicted_window_label": predicted_window_label,
            "true_window_label": true_window_label,
        }

        # Invalidate cached stats
        self._tool_stats = None
        self._operator_stats = None

    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics per tool.

        Returns:
            Dictionary with tool statistics
        """
        if self._tool_stats is not None:
            return self._tool_stats

        # Initialize counters
        tool_stats = defaultdict(
            lambda: {
                "usage_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_result_size": 0,
                "windows_used": set(),
                "param_distributions": defaultdict(int),
            }
        )

        # Process each window
        for window_idx, calls in self.tool_calls.items():
            # Get correctness for this window
            is_correct = self.window_results.get(window_idx, {}).get("is_correct", None)

            for call in calls:
                tool_key = call["query_method"] or call["tool_name"]

                tool_stats[tool_key]["usage_count"] += 1
                tool_stats[tool_key]["windows_used"].add(window_idx)
                tool_stats[tool_key]["total_result_size"] += call["result_size"]

                # Track parameter distributions
                if call["params"]:
                    for param_name, param_value in call["params"].items():
                        param_key = f"{param_name}={param_value}"
                        tool_stats[tool_key]["param_distributions"][param_key] += 1

                # Track success/failure
                if is_correct is not None:
                    if is_correct:
                        tool_stats[tool_key]["success_count"] += 1
                    else:
                        tool_stats[tool_key]["failure_count"] += 1

        # Compute derived metrics
        result = {}
        for tool_key, stats in tool_stats.items():
            usage_count = stats["usage_count"]
            success_count = stats["success_count"]
            failure_count = stats["failure_count"]
            total_attempts = success_count + failure_count

            result[tool_key] = {
                "usage_count": usage_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / total_attempts
                if total_attempts > 0
                else 0.0,
                "failure_rate": failure_count / total_attempts
                if total_attempts > 0
                else 0.0,
                "avg_result_size": stats["total_result_size"] / usage_count
                if usage_count > 0
                else 0.0,
                "num_windows_used": len(stats["windows_used"]),
                "windows_used": sorted(list(stats["windows_used"])),
                "common_params": dict(
                    sorted(
                        stats["param_distributions"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                ),  # Top 5 most common parameter combinations
            }

        self._tool_stats = result
        return result

    def get_operator_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics per operator type.

        Returns:
            Dictionary with operator statistics
        """
        if self._operator_stats is not None:
            return self._operator_stats

        # Initialize counters
        operator_stats = defaultdict(
            lambda: {
                "usage_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "windows_used": set(),
            }
        )

        # Process each window
        for window_idx, calls in self.tool_calls.items():
            is_correct = self.window_results.get(window_idx, {}).get("is_correct", None)

            for call in calls:
                operator = call["operator"]

                operator_stats[operator]["usage_count"] += 1
                operator_stats[operator]["windows_used"].add(window_idx)

                if is_correct is not None:
                    if is_correct:
                        operator_stats[operator]["success_count"] += 1
                    else:
                        operator_stats[operator]["failure_count"] += 1

        # Compute derived metrics
        result = {}
        for operator, stats in operator_stats.items():
            success_count = stats["success_count"]
            failure_count = stats["failure_count"]
            total_attempts = success_count + failure_count

            result[operator] = {
                "usage_count": stats["usage_count"],
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": success_count / total_attempts
                if total_attempts > 0
                else 0.0,
                "failure_rate": failure_count / total_attempts
                if total_attempts > 0
                else 0.0,
                "num_windows_used": len(stats["windows_used"]),
            }

        self._operator_stats = result
        return result

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive tool usage report.

        Returns:
            Dictionary with complete report including rankings and analysis
        """
        tool_stats = self.get_tool_statistics()
        operator_stats = self.get_operator_statistics()

        # Rank tools by success rate (minimum usage threshold)
        min_usage = 5  # Require at least 5 uses to be ranked
        ranked_tools = [
            {
                "tool": tool,
                "success_rate": stats["success_rate"],
                "usage": stats["usage_count"],
                "success_count": stats["success_count"],
                "failure_count": stats["failure_count"],
            }
            for tool, stats in tool_stats.items()
            if stats["usage_count"] >= min_usage
        ]
        ranked_tools.sort(key=lambda x: x["success_rate"], reverse=True)

        # Most useful tools (top 10)
        most_useful = ranked_tools[:10]

        # Least useful tools (bottom 10)
        least_useful = ranked_tools[-10:] if len(ranked_tools) >= 10 else ranked_tools
        least_useful.reverse()  # Show worst first

        # Most frequently used tools
        most_frequent = sorted(
            [
                {
                    "tool": tool,
                    "usage": stats["usage_count"],
                    "success_rate": stats["success_rate"],
                }
                for tool, stats in tool_stats.items()
            ],
            key=lambda x: x["usage"],
            reverse=True,
        )[:10]

        # Compute correlation between tool usage and correctness
        # For each tool, compute correlation: windows where tool was used vs correctness
        tool_correlations = {}
        for tool_key, stats in tool_stats.items():
            windows_used = set(stats["windows_used"])
            if len(windows_used) < 2:
                continue

            # Get correctness for windows where tool was used
            correctness_values = [
                self.window_results.get(w, {}).get("is_correct", False)
                for w in windows_used
            ]

            # Simple correlation: success rate
            tool_correlations[tool_key] = {
                "correlation": stats["success_rate"],
                "num_windows": len(windows_used),
            }

        # Overall correlation analysis
        total_windows = len(self.window_results)
        correct_windows = sum(
            1 for r in self.window_results.values() if r.get("is_correct", False)
        )
        overall_accuracy = correct_windows / total_windows if total_windows > 0 else 0.0

        report = {
            "summary": {
                "total_windows": total_windows,
                "correct_windows": correct_windows,
                "incorrect_windows": total_windows - correct_windows,
                "overall_accuracy": overall_accuracy,
                "total_tool_calls": sum(
                    len(calls) for calls in self.tool_calls.values()
                ),
                "unique_tools": len(tool_stats),
                "unique_operators": len(operator_stats),
            },
            "tool_statistics": tool_stats,
            "operator_statistics": operator_stats,
            "most_useful_tools": most_useful,
            "least_useful_tools": least_useful,
            "most_frequent_tools": most_frequent,
            "correlation_analysis": tool_correlations,
        }

        return report

    def save_report(self, filepath: str):
        """
        Save report to JSON file.

        Args:
            filepath: Path to save JSON report
        """
        report = self.generate_report()

        # Convert sets to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, set):
                return sorted(list(obj))
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        report_json = convert_for_json(report)

        with open(filepath, "w") as f:
            json.dump(report_json, f, indent=2)

    def print_summary(self):
        """Print summary report to console."""
        report = self.generate_report()

        print("\n" + "=" * 80)
        print("Tool Usage Analysis Summary")
        print("=" * 80)
        print(f"Total windows evaluated: {report['summary']['total_windows']}")
        print(f"Overall accuracy: {report['summary']['overall_accuracy']:.2%}")
        print(f"Total tool calls: {report['summary']['total_tool_calls']}")
        print(f"Unique tools: {report['summary']['unique_tools']}")
        print()

        # Most useful tools
        if report["most_useful_tools"]:
            print("Top 5 Most Useful Tools (by success rate):")
            print("-" * 80)
            for i, tool_info in enumerate(report["most_useful_tools"][:5], 1):
                print(
                    f"  {i}. {tool_info['tool']}: "
                    f"{tool_info['success_rate']:.2%} success rate "
                    f"({tool_info['success_count']}/{tool_info['success_count'] + tool_info['failure_count']} correct, "
                    f"used {tool_info['usage']} times)"
                )
            print()

        # Least useful tools
        if report["least_useful_tools"]:
            print("Bottom 5 Least Useful Tools (by success rate):")
            print("-" * 80)
            for i, tool_info in enumerate(report["least_useful_tools"][:5], 1):
                print(
                    f"  {i}. {tool_info['tool']}: "
                    f"{tool_info['success_rate']:.2%} success rate "
                    f"({tool_info['success_count']}/{tool_info['success_count'] + tool_info['failure_count']} correct, "
                    f"used {tool_info['usage']} times)"
                )
            print()

        # Most frequent tools
        if report["most_frequent_tools"]:
            print("Top 5 Most Frequently Used Tools:")
            print("-" * 80)
            for i, tool_info in enumerate(report["most_frequent_tools"][:5], 1):
                print(
                    f"  {i}. {tool_info['tool']}: "
                    f"used {tool_info['usage']} times "
                    f"({tool_info['success_rate']:.2%} success rate)"
                )
            print()

        # Operator statistics
        if report["operator_statistics"]:
            print("Operator Statistics:")
            print("-" * 80)
            for operator, stats in sorted(
                report["operator_statistics"].items(),
                key=lambda x: x[1]["usage_count"],
                reverse=True,
            ):
                print(
                    f"  {operator}: "
                    f"{stats['usage_count']} uses, "
                    f"{stats['success_rate']:.2%} success rate "
                    f"({stats['success_count']}/{stats['success_count'] + stats['failure_count']} correct)"
                )
            print()

        print("=" * 80)
