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
    with open(results_path, 'r') as f:
        return json.load(f)


def compare_metrics(
    llm_results: Dict,
    gdn_kg_results: Dict
) -> Dict[str, any]:
    """
    Compare metrics between two methods.
    
    Args:
        llm_results: Results from LLM baseline evaluation
        gdn_kg_results: Results from GDN->KG evaluation
        
    Returns:
        Dictionary with comparison metrics
    """
    llm_metrics = llm_results['metrics']
    gdn_kg_metrics = gdn_kg_results['metrics']
    
    comparison = {
        'window_level': {},
        'sensor_level': {},
        'efficiency': {}
    }
    
    # Window-level comparison
    wl_llm = llm_metrics['window_level']
    wl_gdn = gdn_kg_metrics['window_level']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        key = f'window_{metric}'
        llm_val = wl_llm.get(key, 0)
        gdn_val = wl_gdn.get(key, 0)
        diff = gdn_val - llm_val
        
        comparison['window_level'][metric] = {
            'llm': float(llm_val),
            'gdn_kg': float(gdn_val),
            'difference': float(diff),
            'improvement': float(diff / llm_val * 100) if llm_val > 0 else 0.0
        }
    
    # Sensor-level comparison
    sl_llm = llm_metrics['sensor_level']
    sl_gdn = gdn_kg_metrics['sensor_level']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        key = f'sensor_{metric}'
        llm_val = sl_llm.get(key, 0)
        gdn_val = sl_gdn.get(key, 0)
        diff = gdn_val - llm_val
        
        comparison['sensor_level'][metric] = {
            'llm': float(llm_val),
            'gdn_kg': float(gdn_val),
            'difference': float(diff),
            'improvement': float(diff / llm_val * 100) if llm_val > 0 else 0.0
        }
    
    # Efficiency comparison
    eff_llm = llm_metrics.get('efficiency', {})
    eff_gdn = gdn_kg_metrics.get('efficiency', {})
    
    comparison['efficiency'] = {
        'llm': {
            'avg_processing_time': eff_llm.get('avg_processing_time_seconds', 0),
            'total_processing_time': eff_llm.get('total_processing_time_seconds', 0),
            'windows_per_second': eff_llm.get('windows_per_second', 0)
        },
        'gdn_kg': {
            'avg_processing_time': eff_gdn.get('total_processing_time_seconds', 0) / gdn_kg_results['num_windows'],
            'total_processing_time': eff_gdn.get('total_processing_time_seconds', 0),
            'windows_per_second': eff_gdn.get('windows_per_second', 0),
            'gdn_time': eff_gdn.get('gdn_processing_time_seconds', 0),
            'kg_time': eff_gdn.get('kg_build_time_seconds', 0)
        }
    }
    
    # Per-fault-type comparison
    if 'per_fault_type' in llm_metrics and 'per_fault_type' in gdn_kg_metrics:
        comparison['per_fault_type'] = {}
        
        llm_ft = llm_metrics['per_fault_type']
        gdn_ft = gdn_kg_metrics['per_fault_type']
        
        all_fault_types = set(llm_ft.keys()) | set(gdn_ft.keys())
        
        for fault_type in all_fault_types:
            llm_ft_metrics = llm_ft.get(fault_type, {})
            gdn_ft_metrics = gdn_ft.get(fault_type, {})
            
            comparison['per_fault_type'][fault_type] = {
                'llm': {
                    'window_f1': llm_ft_metrics.get('window_f1', 0),
                    'sensor_f1': llm_ft_metrics.get('sensor_f1', 0)
                },
                'gdn_kg': {
                    'window_f1': gdn_ft_metrics.get('window_f1', 0),
                    'sensor_f1': gdn_ft_metrics.get('sensor_f1', 0)
                },
                'improvement': {
                    'window_f1': gdn_ft_metrics.get('window_f1', 0) - llm_ft_metrics.get('window_f1', 0),
                    'sensor_f1': gdn_ft_metrics.get('sensor_f1', 0) - llm_ft_metrics.get('sensor_f1', 0)
                }
            }
    
    return comparison


def compare_three_methods(
    llm_results: Dict,
    gdn_kg_results: Dict,
    gdn_kg_llm_results: Dict
) -> Dict[str, any]:
    """
    Compare metrics between three methods.
    
    Args:
        llm_results: Results from LLM baseline evaluation
        gdn_kg_results: Results from GDN->KG evaluation
        gdn_kg_llm_results: Results from GDN->KG->LLM evaluation
        
    Returns:
        Dictionary with comparison metrics
    """
    llm_metrics = llm_results['metrics']
    gdn_kg_metrics = gdn_kg_results['metrics']
    gdn_kg_llm_metrics = gdn_kg_llm_results['metrics']
    
    comparison = {
        'window_level': {},
        'sensor_level': {},
        'efficiency': {}
    }
    
    # Window-level comparison
    wl_llm = llm_metrics['window_level']
    wl_gdn = gdn_kg_metrics['window_level']
    wl_gdn_llm = gdn_kg_llm_metrics['window_level']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        key = f'window_{metric}'
        llm_val = wl_llm.get(key, 0)
        gdn_val = wl_gdn.get(key, 0)
        gdn_llm_val = wl_gdn_llm.get(key, 0)
        
        comparison['window_level'][metric] = {
            'llm_baseline': float(llm_val),
            'gdn_kg': float(gdn_val),
            'gdn_kg_llm': float(gdn_llm_val),
            'kg_llm_improvement_over_baseline': float(gdn_llm_val - llm_val),
            'kg_llm_improvement_over_gdn_kg': float(gdn_llm_val - gdn_val),
            'kg_llm_improvement_pct_over_baseline': float((gdn_llm_val - llm_val) / llm_val * 100) if llm_val > 0 else 0.0,
            'kg_llm_improvement_pct_over_gdn_kg': float((gdn_llm_val - gdn_val) / gdn_val * 100) if gdn_val > 0 else 0.0
        }
    
    # Sensor-level comparison
    sl_llm = llm_metrics['sensor_level']
    sl_gdn = gdn_kg_metrics['sensor_level']
    sl_gdn_llm = gdn_kg_llm_metrics['sensor_level']
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        key = f'sensor_{metric}'
        llm_val = sl_llm.get(key, 0)
        gdn_val = sl_gdn.get(key, 0)
        gdn_llm_val = sl_gdn_llm.get(key, 0)
        
        comparison['sensor_level'][metric] = {
            'llm_baseline': float(llm_val),
            'gdn_kg': float(gdn_val),
            'gdn_kg_llm': float(gdn_llm_val),
            'kg_llm_improvement_over_baseline': float(gdn_llm_val - llm_val),
            'kg_llm_improvement_over_gdn_kg': float(gdn_llm_val - gdn_val),
            'kg_llm_improvement_pct_over_baseline': float((gdn_llm_val - llm_val) / llm_val * 100) if llm_val > 0 else 0.0,
            'kg_llm_improvement_pct_over_gdn_kg': float((gdn_llm_val - gdn_val) / gdn_val * 100) if gdn_val > 0 else 0.0
        }
    
    # Efficiency comparison
    eff_llm = llm_metrics.get('efficiency', {})
    eff_gdn = gdn_kg_metrics.get('efficiency', {})
    eff_gdn_llm = gdn_kg_llm_metrics.get('efficiency', {})
    
    comparison['efficiency'] = {
        'llm_baseline': {
            'avg_processing_time': eff_llm.get('avg_processing_time_seconds', 0),
            'total_processing_time': eff_llm.get('total_processing_time_seconds', 0),
            'windows_per_second': eff_llm.get('windows_per_second', 0)
        },
        'gdn_kg': {
            'avg_processing_time': eff_gdn.get('total_processing_time_seconds', 0) / gdn_kg_results['num_windows'],
            'total_processing_time': eff_gdn.get('total_processing_time_seconds', 0),
            'windows_per_second': eff_gdn.get('windows_per_second', 0),
            'gdn_time': eff_gdn.get('gdn_processing_time_seconds', 0),
            'kg_time': eff_gdn.get('kg_build_time_seconds', 0)
        },
        'gdn_kg_llm': {
            'avg_processing_time': eff_gdn_llm.get('total_processing_time_seconds', 0) / gdn_kg_llm_results['num_windows'],
            'total_processing_time': eff_gdn_llm.get('total_processing_time_seconds', 0),
            'windows_per_second': eff_gdn_llm.get('windows_per_second', 0),
            'gdn_time': eff_gdn_llm.get('gdn_processing_time_seconds', 0),
            'kg_time': eff_gdn_llm.get('kg_build_time_seconds', 0),
            'llm_time': eff_gdn_llm.get('llm_processing_time_seconds', 0)
        }
    }
    
    # Per-fault-type comparison
    if 'per_fault_type' in llm_metrics and 'per_fault_type' in gdn_kg_metrics and 'per_fault_type' in gdn_kg_llm_metrics:
        comparison['per_fault_type'] = {}
        
        llm_ft = llm_metrics['per_fault_type']
        gdn_ft = gdn_kg_metrics['per_fault_type']
        gdn_llm_ft = gdn_kg_llm_metrics['per_fault_type']
        
        all_fault_types = set(llm_ft.keys()) | set(gdn_ft.keys()) | set(gdn_llm_ft.keys())
        
        for fault_type in all_fault_types:
            llm_ft_metrics = llm_ft.get(fault_type, {})
            gdn_ft_metrics = gdn_ft.get(fault_type, {})
            gdn_llm_ft_metrics = gdn_llm_ft.get(fault_type, {})
            
            comparison['per_fault_type'][fault_type] = {
                'llm_baseline': {
                    'window_f1': llm_ft_metrics.get('window_f1', 0),
                    'sensor_f1': llm_ft_metrics.get('sensor_f1', 0)
                },
                'gdn_kg': {
                    'window_f1': gdn_ft_metrics.get('window_f1', 0),
                    'sensor_f1': gdn_ft_metrics.get('sensor_f1', 0)
                },
                'gdn_kg_llm': {
                    'window_f1': gdn_llm_ft_metrics.get('window_f1', 0),
                    'sensor_f1': gdn_llm_ft_metrics.get('sensor_f1', 0)
                },
                'kg_llm_improvement': {
                    'window_f1_over_baseline': gdn_llm_ft_metrics.get('window_f1', 0) - llm_ft_metrics.get('window_f1', 0),
                    'sensor_f1_over_baseline': gdn_llm_ft_metrics.get('sensor_f1', 0) - llm_ft_metrics.get('sensor_f1', 0),
                    'window_f1_over_gdn_kg': gdn_llm_ft_metrics.get('window_f1', 0) - gdn_ft_metrics.get('window_f1', 0),
                    'sensor_f1_over_gdn_kg': gdn_llm_ft_metrics.get('sensor_f1', 0) - gdn_ft_metrics.get('sensor_f1', 0)
                }
            }
    
    return comparison


def format_comparison_report(comparison: Dict, three_way: bool = False) -> str:
    """Format comparison results as a human-readable report."""
    lines = []
    lines.append("="*80)
    if three_way:
        lines.append("METHOD COMPARISON REPORT: LLM-only vs GDN->KG vs GDN->KG->LLM")
    else:
        lines.append("METHOD COMPARISON REPORT: LLM-only vs GDN->KG")
    lines.append("="*80)
    lines.append("")
    
    if three_way:
        # Window-level comparison (three-way)
        lines.append("Window-Level Metrics:")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<15} {'LLM Baseline':<15} {'GDN->KG':<15} {'GDN->KG->LLM':<15} {'KG-LLM vs Baseline':<20} {'KG-LLM vs GDN->KG':<20}")
        lines.append("-" * 80)
        
        for metric, values in comparison['window_level'].items():
            lines.append(
                f"{metric.capitalize():<15} "
                f"{values['llm_baseline']:<15.4f} "
                f"{values['gdn_kg']:<15.4f} "
                f"{values['gdn_kg_llm']:<15.4f} "
                f"{values['kg_llm_improvement_over_baseline']:+.4f} ({values['kg_llm_improvement_pct_over_baseline']:+.2f}%)"
                f"{values['kg_llm_improvement_over_gdn_kg']:+.4f} ({values['kg_llm_improvement_pct_over_gdn_kg']:+.2f}%)"
            )
        lines.append("")
        
        # Sensor-level comparison (three-way)
        lines.append("Sensor-Level Metrics:")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<15} {'LLM Baseline':<15} {'GDN->KG':<15} {'GDN->KG->LLM':<15} {'KG-LLM vs Baseline':<20} {'KG-LLM vs GDN->KG':<20}")
        lines.append("-" * 80)
        
        for metric, values in comparison['sensor_level'].items():
            lines.append(
                f"{metric.capitalize():<15} "
                f"{values['llm_baseline']:<15.4f} "
                f"{values['gdn_kg']:<15.4f} "
                f"{values['gdn_kg_llm']:<15.4f} "
                f"{values['kg_llm_improvement_over_baseline']:+.4f} ({values['kg_llm_improvement_pct_over_baseline']:+.2f}%)"
                f"{values['kg_llm_improvement_over_gdn_kg']:+.4f} ({values['kg_llm_improvement_pct_over_gdn_kg']:+.2f}%)"
            )
        lines.append("")
        
        # KG Enhancement Impact section
        lines.append("KG Enhancement Impact:")
        lines.append("-" * 80)
        lines.append("This section shows how adding KG context improves LLM performance:")
        lines.append("")
        
        best_improvements = []
        for metric, values in comparison['window_level'].items():
            if values['kg_llm_improvement_over_baseline'] > 0:
                best_improvements.append((
                    f"Window {metric}",
                    values['kg_llm_improvement_over_baseline'],
                    values['kg_llm_improvement_pct_over_baseline']
                ))
        
        if best_improvements:
            best_improvements.sort(key=lambda x: x[1], reverse=True)
            lines.append("Top improvements from KG enhancement:")
            for metric_name, abs_imp, pct_imp in best_improvements[:5]:
                lines.append(f"  {metric_name}: +{abs_imp:.4f} ({pct_imp:+.2f}%)")
        lines.append("")
    else:
        # Two-way comparison (existing code)
        lines.append("Window-Level Metrics:")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<15} {'LLM':<12} {'GDN->KG':<12} {'Difference':<12} {'Improvement':<12}")
        lines.append("-" * 80)
        
        for metric, values in comparison['window_level'].items():
            lines.append(
                f"{metric.capitalize():<15} "
                f"{values['llm']:<12.4f} "
                f"{values['gdn_kg']:<12.4f} "
                f"{values['difference']:<12.4f} "
                f"{values['improvement']:<11.2f}%"
            )
        lines.append("")
        
        # Sensor-level comparison
        lines.append("Sensor-Level Metrics:")
        lines.append("-" * 80)
        lines.append(f"{'Metric':<15} {'LLM':<12} {'GDN->KG':<12} {'Difference':<12} {'Improvement':<12}")
        lines.append("-" * 80)
        
        for metric, values in comparison['sensor_level'].items():
            lines.append(
                f"{metric.capitalize():<15} "
                f"{values['llm']:<12.4f} "
                f"{values['gdn_kg']:<12.4f} "
                f"{values['difference']:<12.4f} "
                f"{values['improvement']:<11.2f}%"
            )
        lines.append("")
    
    # Efficiency comparison
    lines.append("Efficiency Metrics:")
    lines.append("-" * 80)
    
    if three_way:
        eff_llm = comparison['efficiency']['llm_baseline']
        eff_gdn = comparison['efficiency']['gdn_kg']
        eff_gdn_llm = comparison['efficiency']['gdn_kg_llm']
        
        lines.append(f"Processing Time:")
        lines.append(f"  LLM Baseline:     {eff_llm['total_processing_time']:.2f} seconds "
                    f"({eff_llm['windows_per_second']:.2f} windows/sec)")
        lines.append(f"  GDN->KG:          {eff_gdn['total_processing_time']:.2f} seconds "
                    f"({eff_gdn['windows_per_second']:.2f} windows/sec)")
        lines.append(f"    - GDN processing: {eff_gdn.get('gdn_time', 0):.2f} seconds")
        lines.append(f"    - KG building: {eff_gdn.get('kg_time', 0):.2f} seconds")
        lines.append(f"  GDN->KG->LLM:     {eff_gdn_llm['total_processing_time']:.2f} seconds "
                    f"({eff_gdn_llm['windows_per_second']:.2f} windows/sec)")
        lines.append(f"    - GDN processing: {eff_gdn_llm.get('gdn_time', 0):.2f} seconds")
        lines.append(f"    - KG building: {eff_gdn_llm.get('kg_time', 0):.2f} seconds")
        lines.append(f"    - LLM processing: {eff_gdn_llm.get('llm_time', 0):.2f} seconds")
    else:
        eff_llm = comparison['efficiency']['llm']
        eff_gdn = comparison['efficiency']['gdn_kg']
        
        lines.append(f"Processing Time:")
        lines.append(f"  LLM:     {eff_llm['total_processing_time']:.2f} seconds "
                    f"({eff_llm['windows_per_second']:.2f} windows/sec)")
        lines.append(f"  GDN->KG: {eff_gdn['total_processing_time']:.2f} seconds "
                    f"({eff_gdn['windows_per_second']:.2f} windows/sec)")
        lines.append(f"    - GDN processing: {eff_gdn.get('gdn_time', 0):.2f} seconds")
        lines.append(f"    - KG building: {eff_gdn.get('kg_time', 0):.2f} seconds")
    lines.append("")
    
    # Per-fault-type comparison
    if 'per_fault_type' in comparison:
        lines.append("Per-Fault-Type Comparison:")
        lines.append("-" * 80)
        
        for fault_type, ft_comp in comparison['per_fault_type'].items():
            lines.append(f"\n{fault_type}:")
            lines.append(f"  Window F1:")
            if three_way:
                lines.append(f"    LLM Baseline:     {ft_comp['llm_baseline']['window_f1']:.4f}")
                lines.append(f"    GDN->KG:          {ft_comp['gdn_kg']['window_f1']:.4f}")
                lines.append(f"    GDN->KG->LLM:     {ft_comp['gdn_kg_llm']['window_f1']:.4f}")
                lines.append(f"    KG-LLM vs Baseline: {ft_comp['kg_llm_improvement']['window_f1_over_baseline']:+.4f}")
                lines.append(f"    KG-LLM vs GDN->KG:  {ft_comp['kg_llm_improvement']['window_f1_over_gdn_kg']:+.4f}")
            else:
                lines.append(f"    LLM:     {ft_comp['llm']['window_f1']:.4f}")
                lines.append(f"    GDN->KG: {ft_comp['gdn_kg']['window_f1']:.4f}")
                lines.append(f"    Improvement: {ft_comp['improvement']['window_f1']:+.4f}")
            lines.append(f"  Sensor F1:")
            if three_way:
                lines.append(f"    LLM Baseline:     {ft_comp['llm_baseline']['sensor_f1']:.4f}")
                lines.append(f"    GDN->KG:          {ft_comp['gdn_kg']['sensor_f1']:.4f}")
                lines.append(f"    GDN->KG->LLM:     {ft_comp['gdn_kg_llm']['sensor_f1']:.4f}")
                lines.append(f"    KG-LLM vs Baseline: {ft_comp['kg_llm_improvement']['sensor_f1_over_baseline']:+.4f}")
                lines.append(f"    KG-LLM vs GDN->KG:  {ft_comp['kg_llm_improvement']['sensor_f1_over_gdn_kg']:+.4f}")
            else:
                lines.append(f"    LLM:     {ft_comp['llm']['sensor_f1']:.4f}")
                lines.append(f"    GDN->KG: {ft_comp['gdn_kg']['sensor_f1']:.4f}")
                lines.append(f"    Improvement: {ft_comp['improvement']['sensor_f1']:+.4f}")
        lines.append("")
    
    lines.append("="*80)
    
    return "\n".join(lines)


def generate_html_report(
    comparison: Dict,
    output_path: Path,
    three_way: bool = False
):
    """Generate HTML comparison report."""
    if three_way:
        title = "Method Comparison Report: LLM-only vs GDN->KG vs GDN->KG->LLM"
    else:
        title = "Method Comparison Report: LLM-only vs GDN->KG"
    
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
"""
    
    if three_way:
        html += """
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>GDN->KG->LLM</th>
            <th>KG-LLM vs Baseline</th>
            <th>KG-LLM vs GDN->KG</th>
        </tr>
"""
        for metric, values in comparison['window_level'].items():
            baseline_imp_class = 'improvement' if values['kg_llm_improvement_over_baseline'] > 0 else 'degradation'
            gdn_imp_class = 'improvement' if values['kg_llm_improvement_over_gdn_kg'] > 0 else 'degradation'
            html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values['llm_baseline']:.4f}</td>
            <td>{values['gdn_kg']:.4f}</td>
            <td>{values['gdn_kg_llm']:.4f}</td>
            <td class="{baseline_imp_class}">{values['kg_llm_improvement_over_baseline']:+.4f} ({values['kg_llm_improvement_pct_over_baseline']:+.2f}%)</td>
            <td class="{gdn_imp_class}">{values['kg_llm_improvement_over_gdn_kg']:+.4f} ({values['kg_llm_improvement_pct_over_gdn_kg']:+.2f}%)</td>
        </tr>
"""
    else:
        html += """
            <th>LLM</th>
            <th>GDN->KG</th>
            <th>Difference</th>
            <th>Improvement</th>
        </tr>
"""
        for metric, values in comparison['window_level'].items():
            improvement_class = 'improvement' if values['improvement'] > 0 else 'degradation'
            html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values['llm']:.4f}</td>
            <td>{values['gdn_kg']:.4f}</td>
            <td>{values['difference']:.4f}</td>
            <td class="{improvement_class}">{values['improvement']:.2f}%</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Sensor-Level Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
"""
    
    if three_way:
        html += """
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>GDN->KG->LLM</th>
            <th>KG-LLM vs Baseline</th>
            <th>KG-LLM vs GDN->KG</th>
        </tr>
"""
        for metric, values in comparison['sensor_level'].items():
            baseline_imp_class = 'improvement' if values['kg_llm_improvement_over_baseline'] > 0 else 'degradation'
            gdn_imp_class = 'improvement' if values['kg_llm_improvement_over_gdn_kg'] > 0 else 'degradation'
            html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values['llm_baseline']:.4f}</td>
            <td>{values['gdn_kg']:.4f}</td>
            <td>{values['gdn_kg_llm']:.4f}</td>
            <td class="{baseline_imp_class}">{values['kg_llm_improvement_over_baseline']:+.4f} ({values['kg_llm_improvement_pct_over_baseline']:+.2f}%)</td>
            <td class="{gdn_imp_class}">{values['kg_llm_improvement_over_gdn_kg']:+.4f} ({values['kg_llm_improvement_pct_over_gdn_kg']:+.2f}%)</td>
        </tr>
"""
    else:
        html += """
            <th>LLM</th>
            <th>GDN->KG</th>
            <th>Difference</th>
            <th>Improvement</th>
        </tr>
"""
        for metric, values in comparison['sensor_level'].items():
            improvement_class = 'improvement' if values['improvement'] > 0 else 'degradation'
            html += f"""
        <tr>
            <td>{metric.capitalize()}</td>
            <td>{values['llm']:.4f}</td>
            <td>{values['gdn_kg']:.4f}</td>
            <td>{values['difference']:.4f}</td>
            <td class="{improvement_class}">{values['improvement']:.2f}%</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Efficiency Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
"""
    
    if three_way:
        eff_llm = comparison['efficiency']['llm_baseline']
        eff_gdn = comparison['efficiency']['gdn_kg']
        eff_gdn_llm = comparison['efficiency']['gdn_kg_llm']
        
        html += """
            <th>LLM Baseline</th>
            <th>GDN->KG</th>
            <th>GDN->KG->LLM</th>
        </tr>
"""
        html += f"""
        <tr>
            <td>Total Processing Time (seconds)</td>
            <td>{eff_llm['total_processing_time']:.2f}</td>
            <td>{eff_gdn['total_processing_time']:.2f}</td>
            <td>{eff_gdn_llm['total_processing_time']:.2f}</td>
        </tr>
        <tr>
            <td>Windows per Second</td>
            <td>{eff_llm['windows_per_second']:.2f}</td>
            <td>{eff_gdn['windows_per_second']:.2f}</td>
            <td>{eff_gdn_llm['windows_per_second']:.2f}</td>
        </tr>
"""
        
        # Add KG Enhancement Impact section
        html += """
    </table>
    
    <div class="kg-impact">
        <h2>KG Enhancement Impact</h2>
        <p>This section highlights the improvements achieved by adding Knowledge Graph context to the LLM:</p>
        <ul>
"""
        
        # Find best improvements
        best_improvements = []
        for metric, values in comparison['window_level'].items():
            if values['kg_llm_improvement_over_baseline'] > 0:
                best_improvements.append((
                    f"Window {metric}",
                    values['kg_llm_improvement_over_baseline'],
                    values['kg_llm_improvement_pct_over_baseline']
                ))
        
        if best_improvements:
            best_improvements.sort(key=lambda x: x[1], reverse=True)
            html += "<li><strong>Top improvements from KG enhancement:</strong><ul>"
            for metric_name, abs_imp, pct_imp in best_improvements[:5]:
                html += f"<li>{metric_name}: +{abs_imp:.4f} ({pct_imp:+.2f}%)</li>"
            html += "</ul></li>"
        
        html += """
        </ul>
    </div>
"""
    else:
        html += """
            <th>LLM</th>
            <th>GDN->KG</th>
        </tr>
"""
        eff_llm = comparison['efficiency']['llm']
        eff_gdn = comparison['efficiency']['gdn_kg']
        
        html += f"""
        <tr>
            <td>Total Processing Time (seconds)</td>
            <td>{eff_llm['total_processing_time']:.2f}</td>
            <td>{eff_gdn['total_processing_time']:.2f}</td>
        </tr>
        <tr>
            <td>Windows per Second</td>
            <td>{eff_llm['windows_per_second']:.2f}</td>
            <td>{eff_gdn['windows_per_second']:.2f}</td>
        </tr>
"""
    
    html += """
    </table>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Compare evaluation results (2-way: LLM vs GDN->KG, or 3-way: LLM vs GDN->KG vs GDN->KG->LLM)'
    )
    parser.add_argument(
        '--llm-results',
        type=str,
        required=True,
        help='Path to LLM baseline results JSON'
    )
    parser.add_argument(
        '--gdn-kg-results',
        type=str,
        required=True,
        help='Path to GDN->KG results JSON'
    )
    parser.add_argument(
        '--gdn-kg-llm-results',
        type=str,
        default=None,
        help='Optional: Path to GDN->KG->LLM results JSON (enables 3-way comparison)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='comparison_report.html',
        help='Output path for comparison report (HTML)'
    )
    parser.add_argument(
        '--json-output',
        type=str,
        default=None,
        help='Optional: Also save comparison as JSON'
    )
    
    args = parser.parse_args()
    
    three_way = args.gdn_kg_llm_results is not None
    
    print("="*80)
    if three_way:
        print("Comparing Methods: LLM-only vs GDN->KG vs GDN->KG->LLM")
    else:
        print("Comparing Methods: LLM-only vs GDN->KG")
    print("="*80)
    print()
    
    # Load results
    print("Loading results...")
    llm_results = load_results(Path(args.llm_results))
    gdn_kg_results = load_results(Path(args.gdn_kg_results))
    print(f"  LLM results: {args.llm_results}")
    print(f"  GDN->KG results: {args.gdn_kg_results}")
    
    if three_way:
        gdn_kg_llm_results = load_results(Path(args.gdn_kg_llm_results))
        print(f"  GDN->KG->LLM results: {args.gdn_kg_llm_results}")
    print()
    
    # Compare metrics
    print("Computing comparison...")
    if three_way:
        comparison = compare_three_methods(llm_results, gdn_kg_results, gdn_kg_llm_results)
    else:
        comparison = compare_metrics(llm_results, gdn_kg_results)
    
    # Print report
    report = format_comparison_report(comparison, three_way=three_way)
    print(report)
    
    # Save HTML report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(comparison, output_path, three_way=three_way)
    print(f"\n✓ HTML report saved to: {output_path}")
    
    # Save JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ JSON comparison saved to: {json_path}")


if __name__ == '__main__':
    main()
