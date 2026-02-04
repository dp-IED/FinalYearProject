"""
Compare KAG Solver v1 (Heuristic) vs KAG Solver v2 with Granite vs KAG Solver v2 with GLM.

Generates an HTML comparison report for the three KAG solver configurations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compare_three_way(
    v1_results: Dict, v2_granite_results: Dict, v2_glm_results: Dict
) -> Dict[str, any]:
    """
    Compare metrics between KAG v1, v2 (Granite), and v2 (GLM).

    Args:
        v1_results: Results from KAG Solver v1 evaluation
        v2_granite_results: Results from KAG Solver v2 with Granite model
        v2_glm_results: Results from KAG Solver v2 with GLM model

    Returns:
        Dictionary with comparison metrics
    """
    v1_metrics = v1_results["metrics"]
    v2_granite_metrics = v2_granite_results["metrics"]
    v2_glm_metrics = v2_glm_results["metrics"]

    comparison = {"window_level": {}, "sensor_level": {}, "efficiency": {}}

    # Window-level comparison
    wl_v1 = v1_metrics["window_level"]
    wl_v2_g = v2_granite_metrics["window_level"]
    wl_v2_glm = v2_glm_metrics["window_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"window_{metric}"
        v1_val = wl_v1.get(key, 0)
        v2_g_val = wl_v2_g.get(key, 0)
        v2_glm_val = wl_v2_glm.get(key, 0)

        comparison["window_level"][metric] = {
            "kag_v1": float(v1_val),
            "kag_v2_granite": float(v2_g_val),
            "kag_v2_glm": float(v2_glm_val),
            "v2_granite_vs_v1": float(v2_g_val - v1_val),
            "v2_glm_vs_v1": float(v2_glm_val - v1_val),
            "v2_glm_vs_granite": float(v2_glm_val - v2_g_val),
        }

    # Sensor-level comparison
    sl_v1 = v1_metrics["sensor_level"]
    sl_v2_g = v2_granite_metrics["sensor_level"]
    sl_v2_glm = v2_glm_metrics["sensor_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"sensor_{metric}"
        v1_val = sl_v1.get(key, 0)
        v2_g_val = sl_v2_g.get(key, 0)
        v2_glm_val = sl_v2_glm.get(key, 0)

        comparison["sensor_level"][metric] = {
            "kag_v1": float(v1_val),
            "kag_v2_granite": float(v2_g_val),
            "kag_v2_glm": float(v2_glm_val),
            "v2_granite_vs_v1": float(v2_g_val - v1_val),
            "v2_glm_vs_v1": float(v2_glm_val - v1_val),
            "v2_glm_vs_granite": float(v2_glm_val - v2_g_val),
        }

    # Efficiency comparison
    eff_v1 = v1_metrics.get("efficiency", {})
    eff_v2_g = v2_granite_metrics.get("efficiency", {})
    eff_v2_glm = v2_glm_metrics.get("efficiency", {})

    comparison["efficiency"] = {
        "kag_v1": {
            "total_processing_time": eff_v1.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_v1.get("windows_per_second", 0),
        },
        "kag_v2_granite": {
            "total_processing_time": eff_v2_g.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_v2_g.get("windows_per_second", 0),
        },
        "kag_v2_glm": {
            "total_processing_time": eff_v2_glm.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_v2_glm.get("windows_per_second", 0),
        },
    }

    return comparison


def generate_html_report(comparison: Dict, output_path: Path, num_windows: int):
    """Generate HTML comparison report."""
    title = f"KAG Solver Comparison: v1 (Heuristic) vs v2 (Granite) vs v2 (GLM) - {num_windows} Windows"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>KAG Solver Three-Way Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
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
        .best {{ background-color: #c8e6c9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
    <h1>{title}</h1>
    
    <div class="summary">
        <h3>Summary</h3>
        <p>This report compares the performance of three KAG solver configurations on {num_windows} test windows:</p>
        <ul>
            <li><strong>KAG v1 (Heuristic)</strong>: Deterministic heuristic-based reasoning</li>
            <li><strong>KAG v2 (Granite)</strong>: LLM-planned reasoning using mlx-community/granite-4.0-h-micro-4bit</li>
            <li><strong>KAG v2 (GLM)</strong>: LLM-planned reasoning using zai-org/glm-4.6v-flash</li>
        </ul>
    </div>
    
    <h2>Window-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (Granite)</th>
            <th>KAG v2 (GLM)</th>
            <th>Granite vs v1</th>
            <th>GLM vs v1</th>
            <th>GLM vs Granite</th>
        </tr>
"""

    for metric, values in comparison["window_level"].items():
        # Find best value
        best_val = max(values["kag_v1"], values["kag_v2_granite"], values["kag_v2_glm"])
        v1_class = "best" if values["kag_v1"] == best_val else ""
        v2_g_class = "best" if values["kag_v2_granite"] == best_val else ""
        v2_glm_class = "best" if values["kag_v2_glm"] == best_val else ""

        g_vs_v1_diff = values["v2_granite_vs_v1"]
        glm_vs_v1_diff = values["v2_glm_vs_v1"]
        glm_vs_g_diff = values["v2_glm_vs_granite"]

        g_vs_v1_class = (
            "improvement"
            if g_vs_v1_diff > 0
            else "degradation"
            if g_vs_v1_diff < 0
            else "neutral"
        )
        glm_vs_v1_class = (
            "improvement"
            if glm_vs_v1_diff > 0
            else "degradation"
            if glm_vs_v1_diff < 0
            else "neutral"
        )
        glm_vs_g_class = (
            "improvement"
            if glm_vs_g_diff > 0
            else "degradation"
            if glm_vs_g_diff < 0
            else "neutral"
        )

        html += f"""
        <tr>
            <td><strong>{metric.capitalize()}</strong></td>
            <td class="{v1_class}">{values["kag_v1"]:.4f}</td>
            <td class="{v2_g_class}">{values["kag_v2_granite"]:.4f}</td>
            <td class="{v2_glm_class}">{values["kag_v2_glm"]:.4f}</td>
            <td class="{g_vs_v1_class}">{g_vs_v1_diff:+.4f}</td>
            <td class="{glm_vs_v1_class}">{glm_vs_v1_diff:+.4f}</td>
            <td class="{glm_vs_g_class}">{glm_vs_g_diff:+.4f}</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Sensor-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (Granite)</th>
            <th>KAG v2 (GLM)</th>
            <th>Granite vs v1</th>
            <th>GLM vs v1</th>
            <th>GLM vs Granite</th>
        </tr>
"""

    for metric, values in comparison["sensor_level"].items():
        # Find best value
        best_val = max(values["kag_v1"], values["kag_v2_granite"], values["kag_v2_glm"])
        v1_class = "best" if values["kag_v1"] == best_val else ""
        v2_g_class = "best" if values["kag_v2_granite"] == best_val else ""
        v2_glm_class = "best" if values["kag_v2_glm"] == best_val else ""

        g_vs_v1_diff = values["v2_granite_vs_v1"]
        glm_vs_v1_diff = values["v2_glm_vs_v1"]
        glm_vs_g_diff = values["v2_glm_vs_granite"]

        g_vs_v1_class = (
            "improvement"
            if g_vs_v1_diff > 0
            else "degradation"
            if g_vs_v1_diff < 0
            else "neutral"
        )
        glm_vs_v1_class = (
            "improvement"
            if glm_vs_v1_diff > 0
            else "degradation"
            if glm_vs_v1_diff < 0
            else "neutral"
        )
        glm_vs_g_class = (
            "improvement"
            if glm_vs_g_diff > 0
            else "degradation"
            if glm_vs_g_diff < 0
            else "neutral"
        )

        html += f"""
        <tr>
            <td><strong>{metric.capitalize()}</strong></td>
            <td class="{v1_class}">{values["kag_v1"]:.4f}</td>
            <td class="{v2_g_class}">{values["kag_v2_granite"]:.4f}</td>
            <td class="{v2_glm_class}">{values["kag_v2_glm"]:.4f}</td>
            <td class="{g_vs_v1_class}">{g_vs_v1_diff:+.4f}</td>
            <td class="{glm_vs_v1_class}">{glm_vs_v1_diff:+.4f}</td>
            <td class="{glm_vs_g_class}">{glm_vs_g_diff:+.4f}</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Efficiency Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>KAG v1 (Heuristic)</th>
            <th>KAG v2 (Granite)</th>
            <th>KAG v2 (GLM)</th>
        </tr>
"""

    eff_v1 = comparison["efficiency"]["kag_v1"]
    eff_v2_g = comparison["efficiency"]["kag_v2_granite"]
    eff_v2_glm = comparison["efficiency"]["kag_v2_glm"]

    html += f"""
        <tr>
            <td><strong>Total Processing Time (seconds)</strong></td>
            <td>{eff_v1["total_processing_time"]:.2f}</td>
            <td>{eff_v2_g["total_processing_time"]:.2f}</td>
            <td>{eff_v2_glm["total_processing_time"]:.2f}</td>
        </tr>
        <tr>
            <td><strong>Windows per Second</strong></td>
            <td>{eff_v1["windows_per_second"]:.4f}</td>
            <td>{eff_v2_g["windows_per_second"]:.4f}</td>
            <td>{eff_v2_glm["windows_per_second"]:.4f}</td>
        </tr>
"""

    html += """
    </table>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Compare KAG Solver v1 vs v2 (Granite) vs v2 (GLM) evaluation results"
    )
    parser.add_argument(
        "--v1-results",
        type=str,
        required=True,
        help="Path to KAG v1 results JSON",
    )
    parser.add_argument(
        "--v2-granite-results",
        type=str,
        required=True,
        help="Path to KAG v2 (Granite) results JSON",
    )
    parser.add_argument(
        "--v2-glm-results",
        type=str,
        required=True,
        help="Path to KAG v2 (GLM) results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/kag_three_way_comparison.html",
        help="Output path for comparison report (HTML)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Comparing KAG Solver v1 vs v2 (Granite) vs v2 (GLM)")
    print("=" * 80)
    print()

    print("Loading results...")
    v1_results = load_results(Path(args.v1_results))
    v2_granite_results = load_results(Path(args.v2_granite_results))
    v2_glm_results = load_results(Path(args.v2_glm_results))
    print(f"  KAG v1 results: {args.v1_results}")
    print(f"  KAG v2 (Granite) results: {args.v2_granite_results}")
    print(f"  KAG v2 (GLM) results: {args.v2_glm_results}")
    print()

    print("Computing comparison...")
    comparison = compare_three_way(v1_results, v2_granite_results, v2_glm_results)
    print()

    num_windows = v1_results.get("num_windows", 0)
    print(f"Generating HTML report for {num_windows} windows...")
    generate_html_report(comparison, Path(args.output), num_windows)
    print(f"✓ HTML report saved to: {args.output}")


if __name__ == "__main__":
    main()
