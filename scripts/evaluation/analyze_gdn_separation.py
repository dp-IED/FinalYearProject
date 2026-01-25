#!/usr/bin/env python3
"""
Analyze GDN model separation between normal and faulty windows.

Checks:
1. Distribution of anomaly scores for normal vs faulty windows
2. Separation metrics (mean difference, overlap, etc.)
3. Whether separation is sufficient (>= 0.3)
4. Correlation analysis - whether GDN learns good normal correlations
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add evaluation directory for metrics
eval_dir = str(project_root / "llm" / "evaluation")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# Import GDN processor
sys.path.insert(0, str(project_root / "anomaly-detection"))
from gdn_processor import GDNPredictor


def analyze_gdn_separation(
    dataset_path: str = "llm/evaluation/shared_dataset/test.npz",
    gdn_model_path: str = "anomaly-detection/best_multilabel_gdn_balanced.pt",
    limit: int = None
):
    """Analyze GDN separation between normal and faulty windows."""
    
    print("=" * 80)
    print("GDN SEPARATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Load dataset
    print("1. Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data['normalized_windows']
    sensor_labels_true = data['sensor_labels']
    window_labels_true = data['window_labels']
    
    sensor_names = ['ENGINE_RPM', 'VEHICLE_SPEED', 'THROTTLE', 'ENGINE_LOAD',
                    'COOLANT_TEMPERATURE', 'INTAKE_MANIFOLD_PRESSURE',
                    'SHORT_TERM_FUEL_TRIM_BANK_1', 'LONG_TERM_FUEL_TRIM_BANK_1']
    
    if limit:
        normalized_windows = normalized_windows[:limit]
        sensor_labels_true = sensor_labels_true[:limit]
        window_labels_true = window_labels_true[:limit]
    
    num_windows = len(normalized_windows)
    print(f"   Loaded {num_windows} windows")
    print()
    
    # Load GDN model
    print("2. Loading GDN model...")
    predictor = GDNPredictor(
        model_path=gdn_model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=32,
        top_k=3,
        hidden_dim=32,
        device='cpu'
    )
    
    # Get adjacency matrix (expected correlations)
    print("3. Computing adjacency matrix (expected correlations)...")
    adj_matrix = predictor.compute_adjacency_matrix()
    print(f"   Adjacency matrix shape: {adj_matrix.shape}")
    print(f"   Min correlation: {np.min(adj_matrix):.4f}")
    print(f"   Max correlation: {np.max(adj_matrix):.4f}")
    print(f"   Mean correlation: {np.mean(adj_matrix):.4f}")
    print()
    
    # Process through GDN
    print("4. Processing windows through GDN...")
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true,
        batch_size=32
    )
    
    gdn_predictions = kg_data['gdn_predictions']  # (num_windows, num_sensors)
    print(f"   GDN predictions shape: {gdn_predictions.shape}")
    print()
    
    # Identify normal vs faulty windows
    print("5. Analyzing separation...")
    print()
    
    # Binary: window is faulty if any sensor is faulty
    is_faulty_window = (sensor_labels_true.sum(axis=1) > 0)
    normal_mask = ~is_faulty_window
    faulty_mask = is_faulty_window
    
    num_normal = np.sum(normal_mask)
    num_faulty = np.sum(faulty_mask)
    
    print(f"   Normal windows: {num_normal}")
    print(f"   Faulty windows: {num_faulty}")
    print()
    
    # Analyze per-sensor separation
    print("=" * 80)
    print("PER-SENSOR SEPARATION ANALYSIS")
    print("=" * 80)
    print()
    
    separation_results = {}
    min_separation = float('inf')
    worst_sensor = None
    
    for sensor_idx, sensor_name in enumerate(sensor_names):
        normal_scores = gdn_predictions[normal_mask, sensor_idx]
        faulty_scores = gdn_predictions[faulty_mask, sensor_idx]
        
        if len(normal_scores) == 0 or len(faulty_scores) == 0:
            continue
        
        normal_mean = np.mean(normal_scores)
        faulty_mean = np.mean(faulty_scores)
        separation = faulty_mean - normal_mean
        
        normal_std = np.std(normal_scores)
        faulty_std = np.std(faulty_scores)
        
        # Calculate overlap (percentage of normal scores above faulty mean, etc.)
        normal_above_faulty_mean = np.sum(normal_scores > faulty_mean) / len(normal_scores) * 100
        faulty_below_normal_mean = np.sum(faulty_scores < normal_mean) / len(faulty_scores) * 100
        overlap_pct = (normal_above_faulty_mean + faulty_below_normal_mean) / 2
        
        # Percentiles
        normal_p95 = np.percentile(normal_scores, 95)
        faulty_p5 = np.percentile(faulty_scores, 5)
        
        separation_results[sensor_name] = {
            'normal_mean': float(normal_mean),
            'faulty_mean': float(faulty_mean),
            'separation': float(separation),
            'normal_std': float(normal_std),
            'faulty_std': float(faulty_std),
            'overlap_pct': float(overlap_pct),
            'normal_p95': float(normal_p95),
            'faulty_p5': float(faulty_p5),
            'normal_max': float(np.max(normal_scores)),
            'faulty_min': float(np.min(faulty_scores))
        }
        
        if separation < min_separation:
            min_separation = separation
            worst_sensor = sensor_name
        
        status = "✓" if separation >= 0.3 else "✗"
        print(f"{status} {sensor_name}:")
        print(f"   Normal mean: {normal_mean:.4f} ± {normal_std:.4f}")
        print(f"   Faulty mean: {faulty_mean:.4f} ± {faulty_std:.4f}")
        print(f"   Separation: {separation:.4f} {'(GOOD)' if separation >= 0.3 else '(INSUFFICIENT)'}")
        print(f"   Overlap: {overlap_pct:.1f}%")
        print(f"   Normal P95: {normal_p95:.4f}, Faulty P5: {faulty_p5:.4f}")
        print()
    
    # Overall separation
    print("=" * 80)
    print("OVERALL SEPARATION SUMMARY")
    print("=" * 80)
    print()
    
    # Max score per window (window-level anomaly)
    normal_max_scores = np.max(gdn_predictions[normal_mask], axis=1)
    faulty_max_scores = np.max(gdn_predictions[faulty_mask], axis=1)
    
    normal_max_mean = np.mean(normal_max_scores)
    faulty_max_mean = np.mean(faulty_max_scores)
    max_separation = faulty_max_mean - normal_max_mean
    
    print(f"Window-level (max sensor score):")
    print(f"   Normal mean max: {normal_max_mean:.4f}")
    print(f"   Faulty mean max: {faulty_max_mean:.4f}")
    print(f"   Separation: {max_separation:.4f} {'(GOOD)' if max_separation >= 0.3 else '(INSUFFICIENT)'}")
    print()
    
    # Check if separation is sufficient
    sufficient_separation = min_separation >= 0.3 and max_separation >= 0.3
    
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()
    
    if sufficient_separation:
        print("✓ Separation is SUFFICIENT (>= 0.3)")
        print(f"   Minimum sensor separation: {min_separation:.4f}")
        print(f"   Window-level separation: {max_separation:.4f}")
    else:
        print("✗ Separation is INSUFFICIENT (< 0.3)")
        print(f"   Minimum sensor separation: {min_separation:.4f} ({worst_sensor})")
        print(f"   Window-level separation: {max_separation:.4f}")
        print()
        print("RECOMMENDATION: Retrain GDN with:")
        print("  1. Correlation-based loss to learn normal correlation patterns")
        print("  2. Explicit separation objective (e.g., contrastive loss)")
        print("  3. Threshold learning to ensure >= 0.3 separation")
    
    print()
    
    # Analyze correlations
    print("=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Compute actual correlations in normal windows
    print("Computing actual correlations in normal windows...")
    normal_correlations = []
    for window_idx in np.where(normal_mask)[0][:min(100, num_normal)]:  # Sample up to 100 normal windows
        window_data = normalized_windows[window_idx]
        corr_matrix = np.corrcoef(window_data.T)
        normal_correlations.append(corr_matrix)
    
    if len(normal_correlations) > 0:
        avg_normal_corr = np.mean(normal_correlations, axis=0)
        
        print("Average correlations in normal windows:")
        print("Sensor pairs with largest deviations from GDN expectations:")
        deviations = []
        for i, name_i in enumerate(sensor_names):
            for j, name_j in enumerate(sensor_names):
                if i < j:
                    actual = avg_normal_corr[i, j]
                    expected = adj_matrix[i, j]
                    deviation = abs(actual - expected)
                    deviations.append((name_i, name_j, actual, expected, deviation))
        
        # Sort by deviation
        deviations.sort(key=lambda x: x[4], reverse=True)
        
        print("Top 10 largest deviations:")
        for name_i, name_j, actual, expected, deviation in deviations[:10]:
            print(f"  {name_i} <-> {name_j}:")
            print(f"    Actual (normal): {actual:.4f}")
            print(f"    Expected (GDN):  {expected:.4f}")
            print(f"    Deviation:        {deviation:.4f}")
            if deviation > 0.3:
                print(f"    ⚠️  EXCEEDS VIOLATION THRESHOLD (0.3)")
            print()
    
    # Save results
    results = {
        'separation_results': separation_results,
        'overall_separation': {
            'min_sensor_separation': float(min_separation),
            'worst_sensor': worst_sensor,
            'window_level_separation': float(max_separation),
            'sufficient': bool(sufficient_separation)
        },
        'correlation_analysis': {
            'adjacency_matrix': adj_matrix.tolist(),
            'avg_normal_correlations': avg_normal_corr.tolist() if len(normal_correlations) > 0 else None
        },
        'statistics': {
            'num_normal': int(num_normal),
            'num_faulty': int(num_faulty),
            'normal_max_scores_mean': float(normal_max_mean),
            'faulty_max_scores_mean': float(faulty_max_mean)
        }
    }
    
    output_path = Path("results/gdn_separation_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Create visualization
    print()
    print("Creating visualization...")
    create_separation_plots(
        gdn_predictions, normal_mask, faulty_mask, sensor_names, 
        separation_results, output_path.parent / "gdn_separation_plots.png"
    )
    print(f"✓ Plots saved to: {output_path.parent / 'gdn_separation_plots.png'}")
    
    return results


def create_separation_plots(gdn_predictions, normal_mask, faulty_mask, 
                            sensor_names, separation_results, output_path):
    """Create visualization plots for separation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Score distributions per sensor
    ax1 = axes[0, 0]
    for sensor_idx, sensor_name in enumerate(sensor_names):
        normal_scores = gdn_predictions[normal_mask, sensor_idx]
        faulty_scores = gdn_predictions[faulty_mask, sensor_idx]
        
        if len(normal_scores) > 0 and len(faulty_scores) > 0:
            ax1.hist(normal_scores, bins=20, alpha=0.5, label=f'{sensor_name} (normal)', density=True)
            ax1.hist(faulty_scores, bins=20, alpha=0.5, label=f'{sensor_name} (faulty)', density=True)
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distributions (All Sensors)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Separation by sensor
    ax2 = axes[0, 1]
    sensors = list(separation_results.keys())
    separations = [separation_results[s]['separation'] for s in sensors]
    colors = ['green' if s >= 0.3 else 'red' for s in separations]
    ax2.barh(sensors, separations, color=colors)
    ax2.axvline(x=0.3, color='red', linestyle='--', label='Target (0.3)')
    ax2.set_xlabel('Separation (Faulty Mean - Normal Mean)')
    ax2.set_title('Separation by Sensor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Window-level max scores
    ax3 = axes[1, 0]
    normal_max = np.max(gdn_predictions[normal_mask], axis=1)
    faulty_max = np.max(gdn_predictions[faulty_mask], axis=1)
    ax3.hist(normal_max, bins=30, alpha=0.6, label='Normal windows', density=True)
    ax3.hist(faulty_max, bins=30, alpha=0.6, label='Faulty windows', density=True)
    ax3.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax3.set_xlabel('Max Anomaly Score (Window-level)')
    ax3.set_ylabel('Density')
    ax3.set_title('Window-Level Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation deviation heatmap
    ax4 = axes[1, 1]
    # This would require loading adjacency matrix and normal correlations
    # For now, just show a placeholder
    ax4.text(0.5, 0.5, 'Correlation Analysis\n(See JSON output)', 
             ha='center', va='center', fontsize=14)
    ax4.set_title('Correlation Deviation Analysis')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GDN separation between normal and faulty windows')
    parser.add_argument('--dataset', type=str, 
                       default='llm/evaluation/shared_dataset/test.npz',
                       help='Path to test dataset')
    parser.add_argument('--model', type=str,
                       default='anomaly-detection/best_multilabel_gdn_balanced.pt',
                       help='Path to GDN model')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of windows to analyze')
    
    args = parser.parse_args()
    
    analyze_gdn_separation(args.dataset, args.model, args.limit)
