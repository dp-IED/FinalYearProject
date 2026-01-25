"""
Compare LLM-only vs GDN->KG vs GDN->KG->LLM methods.

This script:
1. Loads results from evaluation methods (2-way or 3-way)
2. Generates comparison report
3. Visualizes differences in performance
4. Shows KG enhancement impact on LLM performance
"""

import json
import argparse
from pathlib import Path
from typing import Dict


def load_results(results_path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def compare_metrics(llm_results: Dict, gdn_kg_results: Dict) -> Dict[str, any]:
    """
    Compare metrics between two methods.

    Args:
        llm_results: Results from LLM baseline evaluation
        gdn_kg_results: Results from GDN->KG evaluation

    Returns:
        Dictionary with comparison metrics
    """
    llm_metrics = llm_results["metrics"]
    gdn_kg_metrics = gdn_kg_results["metrics"]

    comparison = {"window_level": {}, "sensor_level": {}, "efficiency": {}}

    # Window-level comparison
    wl_llm = llm_metrics["window_level"]
    wl_gdn = gdn_kg_metrics["window_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"window_{metric}"
        llm_val = wl_llm.get(key, 0)
        gdn_val = wl_gdn.get(key, 0)
        diff = gdn_val - llm_val

        comparison["window_level"][metric] = {
            "llm": float(llm_val),
            "gdn_kg": float(gdn_val),
            "difference": float(diff),
            "improvement": float(diff / llm_val * 100) if llm_val > 0 else 0.0,
        }

    # Sensor-level comparison
    sl_llm = llm_metrics["sensor_level"]
    sl_gdn = gdn_kg_metrics["sensor_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"sensor_{metric}"
        llm_val = sl_llm.get(key, 0)
        gdn_val = sl_gdn.get(key, 0)
        diff = gdn_val - llm_val

        comparison["sensor_level"][metric] = {
            "llm": float(llm_val),
            "gdn_kg": float(gdn_val),
            "difference": float(diff),
            "improvement": float(diff / llm_val * 100) if llm_val > 0 else 0.0,
        }

    # Computational speed comparison
    eff_llm = llm_metrics.get("efficiency", {})
    eff_gdn = gdn_kg_metrics.get("efficiency", {})

    comparison["efficiency"] = {
        "llm": {
            "avg_processing_time": eff_llm.get("avg_processing_time_seconds", 0),
            "total_processing_time": eff_llm.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_llm.get("windows_per_second", 0),
        },
        "gdn_kg": {
            "avg_processing_time": eff_gdn.get("total_processing_time_seconds", 0)
            / gdn_kg_results["num_windows"],
            "total_processing_time": eff_gdn.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_gdn.get("windows_per_second", 0),
            "gdn_time": eff_gdn.get("gdn_processing_time_seconds", 0),
            "kg_time": eff_gdn.get("kg_build_time_seconds", 0),
        },
    }

    # Per-fault-type comparison
    if "per_fault_type" in llm_metrics and "per_fault_type" in gdn_kg_metrics:
        comparison["per_fault_type"] = {}

        llm_ft = llm_metrics["per_fault_type"]
        gdn_ft = gdn_kg_metrics["per_fault_type"]

        all_fault_types = set(llm_ft.keys()) | set(gdn_ft.keys())

        for fault_type in all_fault_types:
            llm_ft_metrics = llm_ft.get(fault_type, {})
            gdn_ft_metrics = gdn_ft.get(fault_type, {})

            comparison["per_fault_type"][fault_type] = {
                "llm": {
                    "window_f1": llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1": llm_ft_metrics.get("sensor_f1", 0),
                },
                "gdn_kg": {
                    "window_f1": gdn_ft_metrics.get("window_f1", 0),
                    "sensor_f1": gdn_ft_metrics.get("sensor_f1", 0),
                },
                "improvement": {
                    "window_f1": gdn_ft_metrics.get("window_f1", 0)
                    - llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1": gdn_ft_metrics.get("sensor_f1", 0)
                    - llm_ft_metrics.get("sensor_f1", 0),
                },
            }

    return comparison


def compare_five_methods(
    llm_results: Dict,
    gdn_kg_results: Dict,
    gdn_kg_llm_results: Dict,
    kag_v1_results: Dict,
    kag_v2_results: Dict,
) -> Dict[str, any]:
    """
    Compare metrics between five methods.

    Args:
        llm_results: Results from LLM baseline evaluation
        gdn_kg_results: Results from GDN->KG evaluation
        gdn_kg_llm_results: Results from Serialised KG->LLM evaluation
        kag_v1_results: Results from KAG Heuristics (heuristic) evaluation
        kag_v2_results: Results from Our Method (KAG Solver v2 - LLM-planned) evaluation

    Returns:
        Dictionary with comparison metrics
    """
    llm_metrics = llm_results["metrics"]
    gdn_kg_metrics = gdn_kg_results["metrics"]
    gdn_kg_llm_metrics = gdn_kg_llm_results["metrics"]
    kag_v1_metrics = kag_v1_results["metrics"]
    kag_v2_metrics = kag_v2_results["metrics"]

    comparison = {"window_level": {}, "sensor_level": {}, "efficiency": {}}

    # Window-level comparison
    wl_llm = llm_metrics["window_level"]
    wl_gdn = gdn_kg_metrics["window_level"]
    wl_gdn_llm = gdn_kg_llm_metrics["window_level"]
    wl_kag_v1 = kag_v1_metrics["window_level"]
    wl_kag_v2 = kag_v2_metrics["window_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"window_{metric}"
        llm_val = wl_llm.get(key, 0)
        gdn_val = wl_gdn.get(key, 0)
        gdn_llm_val = wl_gdn_llm.get(key, 0)
        kag_v1_val = wl_kag_v1.get(key, 0)
        kag_v2_val = wl_kag_v2.get(key, 0)

        comparison["window_level"][metric] = {
            "llm_baseline": float(llm_val),
            "gdn_kg": float(gdn_val),
            "gdn_kg_llm": float(gdn_llm_val),
            "kag_v1": float(kag_v1_val),
            "kag_v2": float(kag_v2_val),
            "kg_llm_improvement_over_baseline": float(gdn_llm_val - llm_val),
            "kg_llm_improvement_over_gdn_kg": float(gdn_llm_val - gdn_val),
            "kag_v1_improvement_over_baseline": float(kag_v1_val - llm_val),
            "kag_v1_improvement_over_gdn_kg": float(kag_v1_val - gdn_val),
            "kag_v2_improvement_over_baseline": float(kag_v2_val - llm_val),
            "kag_v2_improvement_over_gdn_kg": float(kag_v2_val - gdn_val),
            "kag_v2_improvement_over_kg_llm": float(kag_v2_val - gdn_llm_val),
            "kag_v2_improvement_over_kag_v1": float(kag_v2_val - kag_v1_val),
            "kg_llm_improvement_pct_over_baseline": float(
                (gdn_llm_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kg_llm_improvement_pct_over_gdn_kg": float(
                (gdn_llm_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v1_improvement_pct_over_baseline": float(
                (kag_v1_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kag_v1_improvement_pct_over_gdn_kg": float(
                (kag_v1_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_baseline": float(
                (kag_v2_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_gdn_kg": float(
                (kag_v2_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_kg_llm": float(
                (kag_v2_val - gdn_llm_val) / gdn_llm_val * 100
            )
            if gdn_llm_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_kag_v1": float(
                (kag_v2_val - kag_v1_val) / kag_v1_val * 100
            )
            if kag_v1_val > 0
            else 0.0,
        }

    # Sensor-level comparison
    sl_llm = llm_metrics["sensor_level"]
    sl_gdn = gdn_kg_metrics["sensor_level"]
    sl_gdn_llm = gdn_kg_llm_metrics["sensor_level"]
    sl_kag_v1 = kag_v1_metrics["sensor_level"]
    sl_kag_v2 = kag_v2_metrics["sensor_level"]

    for metric in ["accuracy", "precision", "recall", "f1"]:
        key = f"sensor_{metric}"
        llm_val = sl_llm.get(key, 0)
        gdn_val = sl_gdn.get(key, 0)
        gdn_llm_val = sl_gdn_llm.get(key, 0)
        kag_v1_val = sl_kag_v1.get(key, 0)
        kag_v2_val = sl_kag_v2.get(key, 0)

        comparison["sensor_level"][metric] = {
            "llm_baseline": float(llm_val),
            "gdn_kg": float(gdn_val),
            "gdn_kg_llm": float(gdn_llm_val),
            "kag_v1": float(kag_v1_val),
            "kag_v2": float(kag_v2_val),
            "kg_llm_improvement_over_baseline": float(gdn_llm_val - llm_val),
            "kg_llm_improvement_over_gdn_kg": float(gdn_llm_val - gdn_val),
            "kag_v1_improvement_over_baseline": float(kag_v1_val - llm_val),
            "kag_v1_improvement_over_gdn_kg": float(kag_v1_val - gdn_val),
            "kag_v2_improvement_over_baseline": float(kag_v2_val - llm_val),
            "kag_v2_improvement_over_gdn_kg": float(kag_v2_val - gdn_val),
            "kag_v2_improvement_over_kg_llm": float(kag_v2_val - gdn_llm_val),
            "kag_v2_improvement_over_kag_v1": float(kag_v2_val - kag_v1_val),
            "kg_llm_improvement_pct_over_baseline": float(
                (gdn_llm_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kg_llm_improvement_pct_over_gdn_kg": float(
                (gdn_llm_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v1_improvement_pct_over_baseline": float(
                (kag_v1_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kag_v1_improvement_pct_over_gdn_kg": float(
                (kag_v1_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_baseline": float(
                (kag_v2_val - llm_val) / llm_val * 100
            )
            if llm_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_gdn_kg": float(
                (kag_v2_val - gdn_val) / gdn_val * 100
            )
            if gdn_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_kg_llm": float(
                (kag_v2_val - gdn_llm_val) / gdn_llm_val * 100
            )
            if gdn_llm_val > 0
            else 0.0,
            "kag_v2_improvement_pct_over_kag_v1": float(
                (kag_v2_val - kag_v1_val) / kag_v1_val * 100
            )
            if kag_v1_val > 0
            else 0.0,
        }

    # Efficiency comparison
    eff_llm = llm_metrics.get("efficiency", {})
    eff_gdn = gdn_kg_metrics.get("efficiency", {})
    eff_gdn_llm = gdn_kg_llm_metrics.get("efficiency", {})
    eff_kag_v1 = kag_v1_metrics.get("efficiency", {})
    eff_kag_v2 = kag_v2_metrics.get("efficiency", {})

    comparison["efficiency"] = {
        "llm_baseline": {
            "avg_processing_time": eff_llm.get("avg_processing_time_seconds", 0),
            "total_processing_time": eff_llm.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_llm.get("windows_per_second", 0),
        },
        "gdn_kg": {
            "avg_processing_time": eff_gdn.get("total_processing_time_seconds", 0)
            / gdn_kg_results["num_windows"],
            "total_processing_time": eff_gdn.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_gdn.get("windows_per_second", 0),
            "gdn_time": eff_gdn.get("gdn_processing_time_seconds", 0),
            "kg_time": eff_gdn.get("kg_build_time_seconds", 0),
        },
        "gdn_kg_llm": {
            "avg_processing_time": eff_gdn_llm.get("total_processing_time_seconds", 0)
            / gdn_kg_llm_results["num_windows"],
            "total_processing_time": eff_gdn_llm.get(
                "total_processing_time_seconds", 0
            ),
            "windows_per_second": eff_gdn_llm.get("windows_per_second", 0),
            "gdn_time": eff_gdn_llm.get("gdn_processing_time_seconds", 0),
            "kg_time": eff_gdn_llm.get("kg_build_time_seconds", 0),
            "llm_time": eff_gdn_llm.get("llm_processing_time_seconds", 0),
        },
        "kag_v1": {
            "avg_processing_time": eff_kag_v1.get("average_processing_time_seconds", 0),
            "total_processing_time": eff_kag_v1.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_kag_v1.get("windows_per_second", 0),
        },
        "kag_v2": {
            "avg_processing_time": eff_kag_v2.get("average_processing_time_seconds", 0),
            "total_processing_time": eff_kag_v2.get("total_processing_time_seconds", 0),
            "windows_per_second": eff_kag_v2.get("windows_per_second", 0),
            "gdn_time": eff_kag_v2.get("gdn_processing_time_seconds", 0),
            "kg_time": eff_kag_v2.get("kg_construction_time_seconds", 0),
        },
    }

    # Per-fault-type comparison
    if (
        "per_fault_type" in llm_metrics
        and "per_fault_type" in gdn_kg_metrics
        and "per_fault_type" in gdn_kg_llm_metrics
        and "per_fault_type" in kag_v1_metrics
        and "per_fault_type" in kag_v2_metrics
    ):
        comparison["per_fault_type"] = {}

        llm_ft = llm_metrics["per_fault_type"]
        gdn_ft = gdn_kg_metrics["per_fault_type"]
        gdn_llm_ft = gdn_kg_llm_metrics["per_fault_type"]
        kag_v1_ft = kag_v1_metrics["per_fault_type"]
        kag_v2_ft = kag_v2_metrics["per_fault_type"]

        all_fault_types = (
            set(llm_ft.keys())
            | set(gdn_ft.keys())
            | set(gdn_llm_ft.keys())
            | set(kag_v1_ft.keys())
            | set(kag_v2_ft.keys())
        )

        for fault_type in all_fault_types:
            llm_ft_metrics = llm_ft.get(fault_type, {})
            gdn_ft_metrics = gdn_ft.get(fault_type, {})
            gdn_llm_ft_metrics = gdn_llm_ft.get(fault_type, {})
            kag_v1_ft_metrics = kag_v1_ft.get(fault_type, {})
            kag_v2_ft_metrics = kag_v2_ft.get(fault_type, {})

            comparison["per_fault_type"][fault_type] = {
                "llm_baseline": {
                    "window_f1": llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1": llm_ft_metrics.get("sensor_f1", 0),
                },
                "gdn_kg": {
                    "window_f1": gdn_ft_metrics.get("window_f1", 0),
                    "sensor_f1": gdn_ft_metrics.get("sensor_f1", 0),
                },
                "gdn_kg_llm": {
                    "window_f1": gdn_llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1": gdn_llm_ft_metrics.get("sensor_f1", 0),
                },
                "kag_v1": {
                    "window_f1": kag_v1_ft_metrics.get("window_f1", 0),
                    "sensor_f1": kag_v1_ft_metrics.get("sensor_f1", 0),
                },
                "kag_v2": {
                    "window_f1": kag_v2_ft_metrics.get("window_f1", 0),
                    "sensor_f1": kag_v2_ft_metrics.get("sensor_f1", 0),
                },
                "kg_llm_improvement": {
                    "window_f1_over_baseline": gdn_llm_ft_metrics.get("window_f1", 0)
                    - llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_baseline": gdn_llm_ft_metrics.get("sensor_f1", 0)
                    - llm_ft_metrics.get("sensor_f1", 0),
                    "window_f1_over_gdn_kg": gdn_llm_ft_metrics.get("window_f1", 0)
                    - gdn_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_gdn_kg": gdn_llm_ft_metrics.get("sensor_f1", 0)
                    - gdn_ft_metrics.get("sensor_f1", 0),
                },
                "kag_v1_improvement": {
                    "window_f1_over_baseline": kag_v1_ft_metrics.get("window_f1", 0)
                    - llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_baseline": kag_v1_ft_metrics.get("sensor_f1", 0)
                    - llm_ft_metrics.get("sensor_f1", 0),
                    "window_f1_over_gdn_kg": kag_v1_ft_metrics.get("window_f1", 0)
                    - gdn_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_gdn_kg": kag_v1_ft_metrics.get("sensor_f1", 0)
                    - gdn_ft_metrics.get("sensor_f1", 0),
                },
                "kag_v2_improvement": {
                    "window_f1_over_baseline": kag_v2_ft_metrics.get("window_f1", 0)
                    - llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_baseline": kag_v2_ft_metrics.get("sensor_f1", 0)
                    - llm_ft_metrics.get("sensor_f1", 0),
                    "window_f1_over_gdn_kg": kag_v2_ft_metrics.get("window_f1", 0)
                    - gdn_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_gdn_kg": kag_v2_ft_metrics.get("sensor_f1", 0)
                    - gdn_ft_metrics.get("sensor_f1", 0),
                    "window_f1_over_kg_llm": kag_v2_ft_metrics.get("window_f1", 0)
                    - gdn_llm_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_kg_llm": kag_v2_ft_metrics.get("sensor_f1", 0)
                    - gdn_llm_ft_metrics.get("sensor_f1", 0),
                    "window_f1_over_kag_v1": kag_v2_ft_metrics.get("window_f1", 0)
                    - kag_v1_ft_metrics.get("window_f1", 0),
                    "sensor_f1_over_kag_v1": kag_v2_ft_metrics.get("sensor_f1", 0)
                    - kag_v1_ft_metrics.get("sensor_f1", 0),
                },
            }

    return comparison


def format_comparison_report(comparison: Dict) -> str:
    """Format comparison results as a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append(
        "METHOD COMPARISON REPORT: LLM-only vs GDN->KG vs Serialised KG->LLM vs KAG Heuristics vs Our Method"
    )
    lines.append("=" * 80)
    lines.append("")

    # Add direct comparison: Serialised KG->LLM vs KAG Heuristics
    lines.append("=" * 80)
    lines.append("DIRECT COMPARISON: Serialised KG->LLM vs KAG Heuristics")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Window-Level Metrics:")
    lines.append("-" * 80)
    lines.append(
        f"{'Metric':<15} {'Serialised KG->LLM':<20} {'KAG Heuristics':<18} {'Difference':<15} {'KAG Heuristics vs Serialised':<30}"
    )
    lines.append("-" * 80)

    for metric, values in comparison["window_level"].items():
        our_val = values["gdn_kg_llm"]
        kag_val = values["kag_v1"]
        diff = kag_val - our_val
        pct_diff = (diff / our_val * 100) if our_val > 0 else 0.0
        lines.append(
            f"{metric.capitalize():<15} "
            f"{our_val:<15.4f} "
            f"{kag_val:<15.4f} "
            f"{diff:+.4f} "
            f"({pct_diff:+.2f}%)"
        )
    lines.append("")

    lines.append("Sensor-Level Metrics:")
    lines.append("-" * 80)
    lines.append(
        f"{'Metric':<15} {'Serialised KG->LLM':<20} {'KAG Heuristics':<18} {'Difference':<15} {'KAG Heuristics vs Serialised':<30}"
    )
    lines.append("-" * 80)

    for metric, values in comparison["sensor_level"].items():
        our_val = values["gdn_kg_llm"]
        kag_val = values["kag_v1"]
        diff = kag_val - our_val
        pct_diff = (diff / our_val * 100) if our_val > 0 else 0.0
        lines.append(
            f"{metric.capitalize():<15} "
            f"{our_val:<15.4f} "
            f"{kag_val:<15.4f} "
            f"{diff:+.4f} "
            f"({pct_diff:+.2f}%)"
        )
    lines.append("")

    lines.append("=" * 80)
    lines.append("FULL COMPARISON: All Methods")
    lines.append("=" * 80)
    lines.append("")

    # Five-way comparison: Window-level
    lines.append("Window-Level Metrics:")
    lines.append("-" * 80)
    lines.append(
        f"{'Metric':<15} {'LLM':<12} {'GDN->KG':<12} {'Serialised':<15} {'KAG Heuristics':<18} {'Our Method':<15} {'Our vs Serialised':<20} {'Our vs KAG':<18}"
    )
    lines.append("-" * 80)

    for metric, values in comparison["window_level"].items():
        our_vs_serialised = values["kag_v2"] - values["gdn_kg_llm"]
        our_vs_kag = values["kag_v2"] - values["kag_v1"]
        lines.append(
            f"{metric.capitalize():<15} "
            f"{values['llm_baseline']:<12.4f} "
            f"{values['gdn_kg']:<12.4f} "
            f"{values['gdn_kg_llm']:<15.4f} "
            f"{values['kag_v1']:<18.4f} "
            f"{values['kag_v2']:<15.4f} "
            f"{our_vs_serialised:+.4f} "
            f"{our_vs_kag:+.4f}"
        )
    lines.append("")

    # Five-way comparison: Sensor-level
    lines.append("Sensor-Level Metrics:")
    lines.append("-" * 80)
    lines.append(
        f"{'Metric':<15} {'LLM':<12} {'GDN->KG':<12} {'Serialised':<15} {'KAG Heuristics':<18} {'Our Method':<15} {'Our vs Serialised':<20} {'Our vs KAG':<18}"
    )
    lines.append("-" * 80)

    for metric, values in comparison["sensor_level"].items():
        our_vs_serialised = values["kag_v2"] - values["gdn_kg_llm"]
        our_vs_kag = values["kag_v2"] - values["kag_v1"]
        lines.append(
            f"{metric.capitalize():<15} "
            f"{values['llm_baseline']:<12.4f} "
            f"{values['gdn_kg']:<12.4f} "
            f"{values['gdn_kg_llm']:<15.4f} "
            f"{values['kag_v1']:<18.4f} "
            f"{values['kag_v2']:<15.4f} "
            f"{our_vs_serialised:+.4f} "
            f"{our_vs_kag:+.4f}"
        )
    lines.append("")

    # Efficiency comparison
    lines.append("Efficiency Metrics:")
    lines.append("-" * 80)

    eff_llm = comparison["efficiency"]["llm_baseline"]
    eff_gdn = comparison["efficiency"]["gdn_kg"]
    eff_gdn_llm = comparison["efficiency"]["gdn_kg_llm"]
    eff_kag_v1 = comparison["efficiency"]["kag_v1"]

    lines.append("Processing Time:")
    lines.append(
        f"  LLM Baseline:     {eff_llm['total_processing_time']:.2f} seconds "
        f"({eff_llm['windows_per_second']:.2f} windows/sec)"
    )
    lines.append(
        f"  GDN->KG:          {eff_gdn['total_processing_time']:.2f} seconds "
        f"({eff_gdn['windows_per_second']:.2f} windows/sec)"
    )
    lines.append(f"    - GDN processing: {eff_gdn.get('gdn_time', 0):.2f} seconds")
    lines.append(f"    - KG building: {eff_gdn.get('kg_time', 0):.2f} seconds")
    lines.append(
        f"  Serialised KG->LLM:        {eff_gdn_llm['total_processing_time']:.2f} seconds "
        f"({eff_gdn_llm['windows_per_second']:.2f} windows/sec)"
    )
    lines.append(f"    - GDN processing: {eff_gdn_llm.get('gdn_time', 0):.2f} seconds")
    lines.append(f"    - KG building: {eff_gdn_llm.get('kg_time', 0):.2f} seconds")
    lines.append(f"    - LLM processing: {eff_gdn_llm.get('llm_time', 0):.2f} seconds")
    lines.append(
        f"  KAG Heuristics:    {eff_kag_v1['total_processing_time']:.2f} seconds "
        f"({eff_kag_v1['windows_per_second']:.2f} windows/sec)"
    )
    eff_kag_v2 = comparison["efficiency"]["kag_v2"]
    lines.append(
        f"  Our Method:        {eff_kag_v2['total_processing_time']:.2f} seconds "
        f"({eff_kag_v2['windows_per_second']:.2f} windows/sec)"
    )
    if eff_kag_v2.get("gdn_time", 0) > 0:
        lines.append(
            f"    - GDN processing: {eff_kag_v2.get('gdn_time', 0):.2f} seconds"
        )
        lines.append(
            f"    - KG construction: {eff_kag_v2.get('kg_time', 0):.2f} seconds"
        )
    lines.append("")

    # Per-fault-type comparison
    if "per_fault_type" in comparison:
        lines.append("Per-Fault-Type Comparison:")
        lines.append("-" * 80)

        for fault_type, ft_comp in comparison["per_fault_type"].items():
            lines.append(f"\n{fault_type}:")
            lines.append("  Window F1:")
            lines.append(
                f"    LLM Baseline:     {ft_comp['llm_baseline']['window_f1']:.4f}"
            )
            lines.append(f"    GDN->KG:          {ft_comp['gdn_kg']['window_f1']:.4f}")
            lines.append(
                f"    Serialised KG->LLM:       {ft_comp['gdn_kg_llm']['window_f1']:.4f}"
            )
            lines.append(f"    KAG Heuristics:    {ft_comp['kag_v1']['window_f1']:.4f}")
            lines.append(
                f"    Serialised KG->LLM vs Baseline: {ft_comp['kg_llm_improvement']['window_f1_over_baseline']:+.4f}"
            )
            lines.append(
                f"    KAG Heuristics vs Baseline: {ft_comp['kag_v1_improvement']['window_f1_over_baseline']:+.4f}"
            )
            our_vs_kag_wf1 = (
                ft_comp["kag_v1"]["window_f1"] - ft_comp["gdn_kg_llm"]["window_f1"]
            )
            lines.append(f"    KAG Heuristics vs Serialised: {our_vs_kag_wf1:+.4f}")
            lines.append("  Sensor F1:")
            lines.append(
                f"    LLM Baseline:     {ft_comp['llm_baseline']['sensor_f1']:.4f}"
            )
            lines.append(f"    GDN->KG:          {ft_comp['gdn_kg']['sensor_f1']:.4f}")
            lines.append(
                f"    Serialised KG->LLM:       {ft_comp['gdn_kg_llm']['sensor_f1']:.4f}"
            )
            lines.append(f"    KAG Heuristics:    {ft_comp['kag_v1']['sensor_f1']:.4f}")
            lines.append(
                f"    Serialised KG->LLM vs Baseline: {ft_comp['kg_llm_improvement']['sensor_f1_over_baseline']:+.4f}"
            )
            lines.append(
                f"    KAG Heuristics vs Baseline: {ft_comp['kag_v1_improvement']['sensor_f1_over_baseline']:+.4f}"
            )
            our_vs_kag_sf1 = (
                ft_comp["kag_v1"]["sensor_f1"] - ft_comp["gdn_kg_llm"]["sensor_f1"]
            )
            lines.append(f"    KAG Heuristics vs Serialised: {our_vs_kag_sf1:+.4f}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def generate_html_report(comparison: Dict, output_path: Path):
    """Generate HTML comparison report."""
    title = "Method Comparison Report: LLM-only vs GDN->KG vs Serialised KG->LLM vs KAG Heuristics vs Our Method"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Method Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .improvement {{ color: green; font-weight: bold; }}
        .degradation {{ color: red; font-weight: bold; }}
        .kg-impact {{ background-color: #e8f5e9; padding: 15px; margin: 20px 0; border-left: 4px solid #4CAF50; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <h2>Window-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>Serialised KG->LLM</th>
            <th>KAG Heuristics</th>
            <th>Our Method</th>
            <th>Serialised vs Baseline</th>
            <th>KAG Heuristics vs Baseline</th>
            <th>Our Method vs Baseline</th>
            <th>Our Method vs Serialised</th>
            <th>Our Method vs KAG Heuristics</th>
        </tr>
"""

    for metric, values in comparison["window_level"].items():
        baseline_imp_class = (
            "improvement"
            if values["kg_llm_improvement_over_baseline"] > 0
            else "degradation"
        )
        kag_imp_class = (
            "improvement"
            if values["kag_v1_improvement_over_baseline"] > 0
            else "degradation"
        )
        our_imp_class = (
            "improvement"
            if values["kag_v2_improvement_over_baseline"] > 0
            else "degradation"
        )
        our_vs_serialised_diff = values["kag_v2"] - values["gdn_kg_llm"]
        our_vs_serialised_pct = (
            (our_vs_serialised_diff / values["gdn_kg_llm"] * 100)
            if values["gdn_kg_llm"] > 0
            else 0.0
        )
        our_vs_serialised_class = (
            "improvement" if our_vs_serialised_diff > 0 else "degradation"
        )
        our_vs_kag_diff = values["kag_v2"] - values["kag_v1"]
        our_vs_kag_pct = (
            (our_vs_kag_diff / values["kag_v1"] * 100) if values["kag_v1"] > 0 else 0.0
        )
        our_vs_kag_class = "improvement" if our_vs_kag_diff > 0 else "degradation"
        html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values["llm_baseline"]:.4f}</td>
            <td>{values["gdn_kg"]:.4f}</td>
            <td>{values["gdn_kg_llm"]:.4f}</td>
            <td>{values["kag_v1"]:.4f}</td>
            <td>{values["kag_v2"]:.4f}</td>
            <td class="{baseline_imp_class}">{values["kg_llm_improvement_over_baseline"]:+.4f} ({values["kg_llm_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{kag_imp_class}">{values["kag_v1_improvement_over_baseline"]:+.4f} ({values["kag_v1_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{our_imp_class}">{values["kag_v2_improvement_over_baseline"]:+.4f} ({values["kag_v2_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{our_vs_serialised_class}">{our_vs_serialised_diff:+.4f} ({our_vs_serialised_pct:+.2f}%)</td>
            <td class="{our_vs_kag_class}">{our_vs_kag_diff:+.4f} ({our_vs_kag_pct:+.2f}%)</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Sensor-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>Serialised KG->LLM</th>
            <th>KAG Heuristics</th>
            <th>Our Method</th>
            <th>Serialised vs Baseline</th>
            <th>KAG Heuristics vs Baseline</th>
            <th>Our Method vs Baseline</th>
            <th>Our Method vs Serialised</th>
            <th>Our Method vs KAG Heuristics</th>
        </tr>
"""

    for metric, values in comparison["sensor_level"].items():
        baseline_imp_class = (
            "improvement"
            if values["kg_llm_improvement_over_baseline"] > 0
            else "degradation"
        )
        kag_imp_class = (
            "improvement"
            if values["kag_v1_improvement_over_baseline"] > 0
            else "degradation"
        )
        our_imp_class = (
            "improvement"
            if values["kag_v2_improvement_over_baseline"] > 0
            else "degradation"
        )
        our_vs_serialised_diff = values["kag_v2"] - values["gdn_kg_llm"]
        our_vs_serialised_pct = (
            (our_vs_serialised_diff / values["gdn_kg_llm"] * 100)
            if values["gdn_kg_llm"] > 0
            else 0.0
        )
        our_vs_serialised_class = (
            "improvement" if our_vs_serialised_diff > 0 else "degradation"
        )
        our_vs_kag_diff = values["kag_v2"] - values["kag_v1"]
        our_vs_kag_pct = (
            (our_vs_kag_diff / values["kag_v1"] * 100) if values["kag_v1"] > 0 else 0.0
        )
        our_vs_kag_class = "improvement" if our_vs_kag_diff > 0 else "degradation"
        html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values["llm_baseline"]:.4f}</td>
            <td>{values["gdn_kg"]:.4f}</td>
            <td>{values["gdn_kg_llm"]:.4f}</td>
            <td>{values["kag_v1"]:.4f}</td>
            <td>{values["kag_v2"]:.4f}</td>
            <td class="{baseline_imp_class}">{values["kg_llm_improvement_over_baseline"]:+.4f} ({values["kg_llm_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{kag_imp_class}">{values["kag_v1_improvement_over_baseline"]:+.4f} ({values["kag_v1_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{our_imp_class}">{values["kag_v2_improvement_over_baseline"]:+.4f} ({values["kag_v2_improvement_pct_over_baseline"]:+.2f}%)</td>
            <td class="{our_vs_serialised_class}">{our_vs_serialised_diff:+.4f} ({our_vs_serialised_pct:+.2f}%)</td>
            <td class="{our_vs_kag_class}">{our_vs_kag_diff:+.4f} ({our_vs_kag_pct:+.2f}%)</td>
        </tr>
"""

    html += """
    </table>
    
    <h2>Efficiency Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>Serialised KG->LLM</th>
            <th>KAG Heuristics</th>
            <th>Our Method</th>
        </tr>
"""

    eff_llm = comparison["efficiency"]["llm_baseline"]
    eff_gdn = comparison["efficiency"]["gdn_kg"]
    eff_gdn_llm = comparison["efficiency"]["gdn_kg_llm"]
    eff_kag_v1 = comparison["efficiency"]["kag_v1"]
    eff_kag_v2 = comparison["efficiency"]["kag_v2"]

    html += f"""
        <tr>
            <td>Total Processing Time (seconds)</td>
            <td>{eff_llm["total_processing_time"]:.2f}</td>
            <td>{eff_gdn["total_processing_time"]:.2f}</td>
            <td>{eff_gdn_llm["total_processing_time"]:.2f}</td>
            <td>{eff_kag_v1["total_processing_time"]:.2f}</td>
            <td>{eff_kag_v2["total_processing_time"]:.2f}</td>
        </tr>
        <tr>
            <td>Windows per Second</td>
            <td>{eff_llm["windows_per_second"]:.2f}</td>
            <td>{eff_gdn["windows_per_second"]:.2f}</td>
            <td>{eff_gdn_llm["windows_per_second"]:.2f}</td>
            <td>{eff_kag_v1["windows_per_second"]:.2f}</td>
            <td>{eff_kag_v2["windows_per_second"]:.2f}</td>
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
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>Serialised KG->LLM</th>
            <th>KAG Heuristics</th>
            <th>Our Method</th>
            <th>Serialised vs Baseline</th>
            <th>KAG Heuristics vs Baseline</th>
            <th>Our Method vs Baseline</th>
            <th>Our Method vs Serialised</th>
            <th>Our Method vs KAG Heuristics</th>
        </tr>
"""
        for fault_type, ft_comp in comparison["per_fault_type"].items():
            kg_llm_wf1_imp = ft_comp["kg_llm_improvement"]["window_f1_over_baseline"]
            kag_v1_wf1_imp = ft_comp["kag_v1_improvement"]["window_f1_over_baseline"]
            kag_v2_wf1_imp = ft_comp["kag_v2_improvement"]["window_f1_over_baseline"]
            kg_llm_sf1_imp = ft_comp["kg_llm_improvement"]["sensor_f1_over_baseline"]
            kag_v1_sf1_imp = ft_comp["kag_v1_improvement"]["sensor_f1_over_baseline"]
            kag_v2_sf1_imp = ft_comp["kag_v2_improvement"]["sensor_f1_over_baseline"]

            serialised_wf1 = ft_comp["gdn_kg_llm"]["window_f1"]
            kag_wf1 = ft_comp["kag_v1"]["window_f1"]
            our_wf1 = ft_comp["kag_v2"]["window_f1"]
            serialised_sf1 = ft_comp["gdn_kg_llm"]["sensor_f1"]
            kag_sf1 = ft_comp["kag_v1"]["sensor_f1"]
            our_sf1 = ft_comp["kag_v2"]["sensor_f1"]

            wf1_kg_class = "improvement" if kg_llm_wf1_imp > 0 else "degradation"
            wf1_kag_class = "improvement" if kag_v1_wf1_imp > 0 else "degradation"
            wf1_our_class = "improvement" if kag_v2_wf1_imp > 0 else "degradation"
            wf1_our_vs_serialised = our_wf1 - serialised_wf1
            wf1_our_vs_serialised_class = (
                "improvement" if wf1_our_vs_serialised > 0 else "degradation"
            )
            wf1_our_vs_kag = our_wf1 - kag_wf1
            wf1_our_vs_kag_class = (
                "improvement" if wf1_our_vs_kag > 0 else "degradation"
            )

            sf1_kg_class = "improvement" if kg_llm_sf1_imp > 0 else "degradation"
            sf1_kag_class = "improvement" if kag_v1_sf1_imp > 0 else "degradation"
            sf1_our_class = "improvement" if kag_v2_sf1_imp > 0 else "degradation"
            sf1_our_vs_serialised = our_sf1 - serialised_sf1
            sf1_our_vs_serialised_class = (
                "improvement" if sf1_our_vs_serialised > 0 else "degradation"
            )
            sf1_our_vs_kag = our_sf1 - kag_sf1
            sf1_our_vs_kag_class = (
                "improvement" if sf1_our_vs_kag > 0 else "degradation"
            )

            html += f"""
        <tr>
            <td rowspan="2">{fault_type}</td>
            <td>Window F1</td>
            <td>{ft_comp["llm_baseline"]["window_f1"]:.4f}</td>
            <td>{ft_comp["gdn_kg"]["window_f1"]:.4f}</td>
            <td>{serialised_wf1:.4f}</td>
            <td>{kag_wf1:.4f}</td>
            <td>{our_wf1:.4f}</td>
            <td class="{wf1_kg_class}">{kg_llm_wf1_imp:+.4f}</td>
            <td class="{wf1_kag_class}">{kag_v1_wf1_imp:+.4f}</td>
            <td class="{wf1_our_class}">{kag_v2_wf1_imp:+.4f}</td>
            <td class="{wf1_our_vs_serialised_class}">{wf1_our_vs_serialised:+.4f}</td>
            <td class="{wf1_our_vs_kag_class}">{wf1_our_vs_kag:+.4f}</td>
        </tr>
        <tr>
            <td>Sensor F1</td>
            <td>{ft_comp["llm_baseline"]["sensor_f1"]:.4f}</td>
            <td>{ft_comp["gdn_kg"]["sensor_f1"]:.4f}</td>
            <td>{serialised_sf1:.4f}</td>
            <td>{kag_sf1:.4f}</td>
            <td>{our_sf1:.4f}</td>
            <td class="{sf1_kg_class}">{kg_llm_sf1_imp:+.4f}</td>
            <td class="{sf1_kag_class}">{kag_v1_sf1_imp:+.4f}</td>
            <td class="{sf1_our_class}">{kag_v2_sf1_imp:+.4f}</td>
            <td class="{sf1_our_vs_serialised_class}">{sf1_our_vs_serialised:+.4f}</td>
            <td class="{sf1_our_vs_kag_class}">{sf1_our_vs_kag:+.4f}</td>
        </tr>
"""
        html += """
    </table>
"""

    # Add KG Enhancement Impact section
    # Add direct comparison: Our Method vs others
    html += """
    
    <h2>Direct Comparison: Our Method vs Others</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Serialised KG->LLM</th>
            <th>KAG Heuristics</th>
            <th>Our Method</th>
            <th>Our Method vs Serialised</th>
            <th>Our Method vs KAG Heuristics</th>
        </tr>
"""

    for metric, values in comparison["window_level"].items():
        serialised_val = values["gdn_kg_llm"]
        kag_val = values["kag_v1"]
        our_val = values["kag_v2"]
        diff_vs_serialised = our_val - serialised_val
        diff_vs_kag = our_val - kag_val
        pct_diff_serialised = (
            (diff_vs_serialised / serialised_val * 100) if serialised_val > 0 else 0.0
        )
        pct_diff_kag = (diff_vs_kag / kag_val * 100) if kag_val > 0 else 0.0
        diff_class_serialised = (
            "improvement" if diff_vs_serialised > 0 else "degradation"
        )
        diff_class_kag = "improvement" if diff_vs_kag > 0 else "degradation"
        html += f"""
        <tr>
            <td>Window {metric.capitalize()}</td>
            <td>{serialised_val:.4f}</td>
            <td>{kag_val:.4f}</td>
            <td>{our_val:.4f}</td>
            <td class="{diff_class_serialised}">{diff_vs_serialised:+.4f} ({pct_diff_serialised:+.2f}%)</td>
            <td class="{diff_class_kag}">{diff_vs_kag:+.4f} ({pct_diff_kag:+.2f}%)</td>
        </tr>
"""

    for metric, values in comparison["sensor_level"].items():
        serialised_val = values["gdn_kg_llm"]
        kag_val = values["kag_v1"]
        our_val = values["kag_v2"]
        diff_vs_serialised = our_val - serialised_val
        diff_vs_kag = our_val - kag_val
        pct_diff_serialised = (
            (diff_vs_serialised / serialised_val * 100) if serialised_val > 0 else 0.0
        )
        pct_diff_kag = (diff_vs_kag / kag_val * 100) if kag_val > 0 else 0.0
        diff_class_serialised = (
            "improvement" if diff_vs_serialised > 0 else "degradation"
        )
        diff_class_kag = "improvement" if diff_vs_kag > 0 else "degradation"
        html += f"""
        <tr>
            <td>Sensor {metric.capitalize()}</td>
            <td>{serialised_val:.4f}</td>
            <td>{kag_val:.4f}</td>
            <td>{our_val:.4f}</td>
            <td class="{diff_class_serialised}">{diff_vs_serialised:+.4f} ({pct_diff_serialised:+.2f}%)</td>
            <td class="{diff_class_kag}">{diff_vs_kag:+.4f} ({pct_diff_kag:+.2f}%)</td>
        </tr>
"""

    html += """
    </table>
    
    <div class="kg-impact">
        <h2>KG Enhancement Impact</h2>
        <p>This section highlights the improvements achieved by adding Knowledge Graph context to the LLM:</p>
        <ul>
"""

    # Find best improvements
    best_improvements = []
    for metric, values in comparison["window_level"].items():
        if values["kg_llm_improvement_over_baseline"] > 0:
            best_improvements.append(
                (
                    f"Window {metric}",
                    values["kg_llm_improvement_over_baseline"],
                    values["kg_llm_improvement_pct_over_baseline"],
                )
            )

    if best_improvements:
        best_improvements.sort(key=lambda x: x[1], reverse=True)
        html += "<li><strong>Top improvements from KG enhancement:</strong><ul>"
        for metric_name, abs_imp, pct_imp in best_improvements[:5]:
            html += f"<li>{metric_name}: +{abs_imp:.4f} ({pct_imp:+.2f}%)</li>"
        html += "</ul></li>"

    html += """
        </ul>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results: LLM vs GDN->KG vs Serialised KG->LLM vs KAG Heuristics vs Our Method"
    )
    parser.add_argument(
        "--llm-results",
        type=str,
        required=True,
        help="Path to LLM baseline results JSON",
    )
    parser.add_argument(
        "--gdn-kg-results", type=str, required=True, help="Path to GDN->KG results JSON"
    )
    parser.add_argument(
        "--gdn-kg-llm-results",
        type=str,
        required=True,
        help="Path to GDN->KG->LLM results JSON",
    )
    parser.add_argument(
        "--kag-v1-results",
        type=str,
        required=False,
        default=None,
        help="Path to KAG Heuristics results JSON (optional)",
    )
    parser.add_argument(
        "--kag-v2-results",
        type=str,
        required=False,
        default=None,
        help="Path to Our Method (KAG Solver v2) results JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.html",
        help="Output path for comparison report (HTML)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional: Also save comparison as JSON",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "Comparing Methods: LLM-only vs GDN->KG vs Serialised KG->LLM vs KAG Heuristics vs Our Method"
    )
    print("=" * 80)
    print()

    print("Loading results...")
    llm_results = load_results(Path(args.llm_results))
    gdn_kg_results = load_results(Path(args.gdn_kg_results))
    gdn_kg_llm_results = load_results(Path(args.gdn_kg_llm_results))
    
    has_kag_v1 = args.kag_v1_results is not None and Path(args.kag_v1_results).exists()
    has_kag_v2 = args.kag_v2_results is not None and Path(args.kag_v2_results).exists()
    
    kag_v1_results = None
    kag_v2_results = None
    
    if has_kag_v1:
        kag_v1_results = load_results(Path(args.kag_v1_results))
    if has_kag_v2:
        kag_v2_results = load_results(Path(args.kag_v2_results))
    
    print(f"  LLM results: {args.llm_results}")
    print(f"  GDN->KG results: {args.gdn_kg_results}")
    print(f"  Serialised KG->LLM results: {args.gdn_kg_llm_results}")
    if has_kag_v1:
        print(f"  KAG Heuristics results: {args.kag_v1_results}")
    if has_kag_v2:
        print(f"  Our Method (KAG v2) results: {args.kag_v2_results}")
    print()

    print("Computing comparison...")
    if has_kag_v1 and has_kag_v2:
        # Five-way comparison (all methods available)
        comparison = compare_five_methods(
            llm_results, gdn_kg_results, gdn_kg_llm_results, kag_v1_results, kag_v2_results
        )
    else:
        # Partial comparison - create minimal dummy results for missing methods
        # This allows compare_five_methods to work but marks missing methods clearly
        if not has_kag_v1:
            kag_v1_results = {
                "metrics": {
                    "window_level": {f"window_{m}": 0.0 for m in ["accuracy", "precision", "recall", "f1"]},
                    "sensor_level": {f"sensor_{m}": 0.0 for m in ["accuracy", "precision", "recall", "f1"]},
                    "efficiency": {"total_processing_time_seconds": 0, "windows_per_second": 0}
                },
                "num_windows": 0
            }
        if not has_kag_v2:
            kag_v2_results = {
                "metrics": {
                    "window_level": {f"window_{m}": 0.0 for m in ["accuracy", "precision", "recall", "f1"]},
                    "sensor_level": {f"sensor_{m}": 0.0 for m in ["accuracy", "precision", "recall", "f1"]},
                    "efficiency": {"total_processing_time_seconds": 0, "windows_per_second": 0}
                },
                "num_windows": 0
            }
        comparison = compare_five_methods(
            llm_results, gdn_kg_results, gdn_kg_llm_results, kag_v1_results, kag_v2_results
        )

    report = format_comparison_report(comparison)
    print(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(comparison, output_path)
    print(f"\n✓ HTML report saved to: {output_path}")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ JSON comparison saved to: {json_path}")


if __name__ == "__main__":
    main()
