"""
Compare KAG Solver v1 (Heuristic) vs KAG Solver v2 (LLM-Planned).

Generates an HTML comparison report for the two KAG solvers.
"""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compare_kag_v1_v2(kag_v1_results: Dict, kag_v2_results: Dict) -> Dict[str, any]:
    """
    Compare metrics between KAG v1 and v2.

    Args:
        kag_v1_results: Results from KAG Solver v1 evaluation
        kag_v2_results: Results from KAG Solver v2 evaluation

    Returns:
        Dictionary with comparison metrics
    """
    v1_metrics = kag_v1_results["metrics"]
    v2_metrics = kag_v2_results["metrics"]

    comparison = {"window_level": {}, "sensor_level": {}, "efficiency": {}}

    # Window-level comparison
    wl_v1 = v1_metrics["window_level"]
    wl_v2 = v2_metrics["window_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"window_{metric}"
        v1_val = wl_v1.get(key, 0)
        v2_val = wl_v2.get(key, 0)
        diff = v2_val - v1_val

        comparison["window_level"][metric] = {
            "kag_v1": float(v1_val),
            "kag_v2": float(v2_val),
            "difference": float(diff),
            "improvement_pct": float(diff / v1_val * 100) if v1_val > 0 else 0.0,
        }

    # Sensor-level comparison
    sl_v1 = v1_metrics["sensor_level"]
    sl_v2 = v2_metrics["sensor_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"sensor_{metric}"
        v1_val = sl_v1.get(key, 0)
        v2_val = sl_v2.get(key, 0)
        diff = v2_val - v1_val

        comparison["sensor_level"][metric] = {
            "kag_v1": float(v1_val),
            "kag_v2": float(v2_val),
            "difference": float(diff),
            "improvement_pct": float(diff / v1_val * 100) if v1_val > 0 else 0.0,
        }

    # Efficiency comparison
    eff_v1 = v1_metrics.get("efficiency", {})
    eff_v2 = v2_metrics.get("efficiency", {})

    comparison["efficiency"] = {
        "kag_v1": {
            "total_processing_time": eff_v1.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_v1.get("windows_per_second", 0),
        },
        "kag_v2": {
            "total_processing_time": eff_v2.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_v2.get("windows_per_second", 0),
        },
    }

    # Per-fault-type comparison
    if "per_fault_type" in v1_metrics and "per_fault_type" in v2_metrics:
        comparison["per_fault_type"] = {}

        v1_ft = v1_metrics["per_fault_type"]
        v2_ft = v2_metrics["per_fault_type"]

        all_fault_types = set(v1_ft.keys()) | set(v2_ft.keys())

        for fault_type in all_fault_types:
            v1_ft_metrics = v1_ft.get(fault_type, {})
            v2_ft_metrics = v2_ft.get(fault_type, {})

            comparison["per_fault_type"][fault_type] = {
                "kag_v1": {
                    "window_f1": v1_ft_metrics.get("window_f1", 0),
                    "sensor_f1": v1_ft_metrics.get("sensor_f1", 0),
                },
                "kag_v2": {
                    "window_f1": v2_ft_metrics.get("window_f1", 0),
                    "sensor_f1": v2_ft_metrics.get("sensor_f1", 0),
                },
                "improvement": {
                    "window_f1": v2_ft_metrics.get("window_f1", 0)
                    - v1_ft_metrics.get("window_f1", 0),
                    "sensor_f1": v2_ft_metrics.get("sensor_f1", 0)
                    - v1_ft_metrics.get("sensor_f1", 0),
                },
            }

    return comparison


def generate_html_report(comparison: Dict, output_path: Path, num_windows: int):
    """Generate HTML comparison report."""
    title = f"KAG Solver Comparison: v1 (Heuristic) vs v2 (LLM-Planned) - {num_windows} Windows"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>KAG Solver Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .improvement {{ color: #2e7d32; font-weight: bold; }}
        .degradation {{ color: #c62828; font-weight: bold; }}
        .neutral {{ color: #666; }}
        .summary {{ background-color: #e8f5e9; padding: 15px; margin: 20px 0; border-left: 4px solid #4CAF50; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
    <h1>{title}</h1>
    
    <div class="summary">
        <h3>Summary</h3>
        <p>This report compares the performance of KAG Solver v1 (heuristic-based) and KAG Solver v2 (LLM-planned iterative reasoning) on {num_windows} test windows.</p>
    </div>
    
    <h2>Window-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (LLM-Planned)</th>
            <th>Difference</th>
            <th>Improvement %</th>
        </tr>
"""

    for metric, values in comparison["window_level"].items():
        diff = values["difference"]
        diff_class = (
            "improvement" if diff > 0 else "degradation" if diff < 0 else "neutral"
        )
        html += f"""
        <tr>
            <td><strong>{metric.capitalize()}</strong></td>
            <td>{values["kag_v1"]:.4f}</td>
            <td>{values["kag_v2"]:.4f}</td>
            <td class="{diff_class}">{diff:+.4f}</td>
            <td class="{diff_class}">{values["improvement_pct"]:+.2f}%</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Sensor-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (LLM-Planned)</th>
            <th>Difference</th>
            <th>Improvement %</th>
        </tr>
"""

    for metric, values in comparison["sensor_level"].items():
        diff = values["difference"]
        diff_class = (
            "improvement" if diff > 0 else "degradation" if diff < 0 else "neutral"
        )
        html += f"""
        <tr>
            <td><strong>{metric.capitalize()}</strong></td>
            <td>{values["kag_v1"]:.4f}</td>
            <td>{values["kag_v2"]:.4f}</td>
            <td class="{diff_class}">{diff:+.4f}</td>
            <td class="{diff_class}">{values["improvement_pct"]:+.2f}%</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Efficiency Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (LLM-Planned)</th>
        </tr>
"""

    eff_v1 = comparison["efficiency"]["kag_v1"]
    eff_v2 = comparison["efficiency"]["kag_v2"]

    html += f"""
        <tr>
            <td><strong>Total Processing Time (seconds)</strong></td>
            <td>{eff_v1["total_processing_time"]:.2f}</td>
            <td>{eff_v2["total_processing_time"]:.2f}</td>
        </tr>
        <tr>
            <td><strong>Windows per Second</strong></td>
            <td>{eff_v1["windows_per_second"]:.4f}</td>
            <td>{eff_v2["windows_per_second"]:.4f}</td>
        </tr>
"""

    # Add per-fault-type comparison if available
    if "per_fault_type" in comparison:
        html += """
    </table>
    
    <h2>Per-Fault-Type Comparison</h2>
    <table>
        <tr>
            <th>Fault Type</th>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (LLM-Planned)</th>
            <th>Improvement</th>
        </tr>
"""
        for fault_type, ft_comp in comparison["per_fault_type"].items():
            wf1_diff = ft_comp["improvement"]["window_f1"]
            sf1_diff = ft_comp["improvement"]["sensor_f1"]
            wf1_class = (
                "improvement"
                if wf1_diff > 0
                else "degradation"
                if wf1_diff < 0
                else "neutral"
            )
            sf1_class = (
                "improvement"
                if sf1_diff > 0
                else "degradation"
                if sf1_diff < 0
                else "neutral"
            )

            html += f"""
        <tr>
            <td rowspan="2"><strong>{fault_type}</strong></td>
            <td>Window F1</td>
            <td>{ft_comp["kag_v1"]["window_f1"]:.4f}</td>
            <td>{ft_comp["kag_v2"]["window_f1"]:.4f}</td>
            <td class="{wf1_class}">{wf1_diff:+.4f}</td>
        </tr>
        <tr>
            <td>Sensor F1</td>
            <td>{ft_comp["kag_v1"]["sensor_f1"]:.4f}</td>
            <td>{ft_comp["kag_v2"]["sensor_f1"]:.4f}</td>
            <td class="{sf1_class}">{sf1_diff:+.4f}</td>
        </tr>
"""
        html += """
    </table>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Compare KAG Solver v1 vs v2 evaluation results"
    )
    parser.add_argument(
        "--kag-v1-results",
        type=str,
        required=True,
        help="Path to KAG v1 results JSON",
    )
    parser.add_argument(
        "--kag-v2-results",
        type=str,
        required=True,
        help="Path to KAG v2 results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/kag_comparison.html",
        help="Output path for comparison report (HTML)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Comparing KAG Solver v1 vs v2")
    print("=" * 80)
    print()

    print("Loading results...")
    kag_v1_results = load_results(Path(args.kag_v1_results))
    kag_v2_results = load_results(Path(args.kag_v2_results))
    print(f"  KAG v1 results: {args.kag_v1_results}")
    print(f"  KAG v2 results: {args.kag_v2_results}")
    print()

    print("Computing comparison...")
    comparison = compare_kag_v1_v2(kag_v1_results, kag_v2_results)
    print()

    num_windows = kag_v1_results.get(
        "num_windows", kag_v2_results.get("num_windows", 0)
    )
    print(f"Generating HTML report for {num_windows} windows...")
    generate_html_report(comparison, Path(args.output), num_windows)
    print(f"✓ HTML report saved to: {args.output}")


if __name__ == "__main__":
    main()
