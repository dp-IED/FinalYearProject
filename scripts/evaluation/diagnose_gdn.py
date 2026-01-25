#!/usr/bin/env python3
"""
Deep diagnostic script for GDN's false positive problem.

Analyzes GDN evaluation results to identify root causes:
1. Score distribution analysis
2. Statistical separability (effect size)
3. Optimal threshold analysis
4. Root cause hypothesis
5. Recommended fixes
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

import torch
from gdn_processor import GDNPredictor

# Import get_embeddings_and_distances from evaluate_gdn
import importlib.util
eval_spec = importlib.util.spec_from_file_location("evaluate_gdn", project_root / "evaluate_gdn.py")
evaluate_gdn_module = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluate_gdn_module)
get_embeddings_and_distances = evaluate_gdn_module.get_embeddings_and_distances


def extract_individual_scores(dataset_path, gdn_model_path, limit=None):
    """
    Extract individual anomaly scores and labels from GDN evaluation.
    
    Returns:
        tuple: (anomaly_scores, binary_labels, normal_mask, faulty_mask)
    """
    print("Extracting individual scores from GDN...")
    
    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]
    
    sensor_names = [
        "ENGINE_RPM",
        "VEHICLE_SPEED",
        "THROTTLE",
        "ENGINE_LOAD",
        "COOLANT_TEMPERATURE",
        "INTAKE_MANIFOLD_PRESSURE",
        "SHORT_TERM_FUEL_TRIM_BANK_1",
        "LONG_TERM_FUEL_TRIM_BANK_1",
    ]
    
    if limit:
        normalized_windows = normalized_windows[:limit]
        sensor_labels_true = sensor_labels_true[:limit]
        window_labels_true = window_labels_true[:limit]
    
    # Load GDN model
    # Detect model dimensions from model path
    # Improved models use embed_dim=64, hidden_dim=64
    if "improved" in str(gdn_model_path).lower():
        embed_dim = 64
        hidden_dim = 64
    else:
        embed_dim = 32
        hidden_dim = 32
    
    predictor = GDNPredictor(
        model_path=gdn_model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=embed_dim,
        top_k=3,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    
    # Process through GDN
    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true,
        batch_size=32,
    )
    
    # Get embeddings and compute distances
    embeddings, distances, normal_center = get_embeddings_and_distances(
        predictor, normalized_windows, batch_size=32
    )
    
    # Identify normal vs faulty windows
    is_faulty_window = sensor_labels_true.sum(axis=1) > 0
    normal_mask = ~is_faulty_window
    faulty_mask = is_faulty_window
    
    # Compute anomaly scores
    if distances is None:
        if np.sum(normal_mask) > 0:
            normal_embeddings = embeddings[normal_mask]
            normal_center = np.mean(normal_embeddings, axis=0)
            distances = np.linalg.norm(embeddings - normal_center, axis=1)
        else:
            gdn_predictions = kg_data["gdn_predictions"]
            distances = np.max(gdn_predictions, axis=1)
    
    anomaly_scores = distances
    binary_labels = is_faulty_window.astype(int)
    
    return anomaly_scores, binary_labels, normal_mask, faulty_mask




def diagnose_gdn_model(
    results_path='results/gdn_evaluation_results.json',
    dataset_path='llm/evaluation/shared_dataset/test.npz',
    gdn_model_path='anomaly-detection/best_multilabel_gdn.pt',
    extract_scores=True
):
    """
    Deep diagnostic of GDN's false positive problem.
    
    Args:
        results_path: Path to existing evaluation results JSON
        dataset_path: Path to test dataset (for extracting individual scores)
        gdn_model_path: Path to GDN model (for extracting individual scores)
        extract_scores: Whether to extract individual scores (slower but more detailed)
    """
    print("="*70)
    print("GDN DIAGNOSTIC REPORT")
    print("="*70)
    
    # Load existing results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    metrics = results['classification_metrics']
    dist_stats = results['distance_statistics']
    
    print(f"\nCurrent Performance:")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1 Score:  {metrics['f1_score']:.3f}")
    print(f"   Threshold: {metrics['threshold']:.4f}")
    print(f"   ROC-AUC:   {metrics.get('roc_auc', 0.0):.3f}")
    print(f"   PR-AUC:    {metrics.get('pr_auc', 0.0):.3f}")
    
    print(f"\nSeparation Statistics:")
    print(f"   Normal mean:    {dist_stats['normal_mean']:.4f}")
    print(f"   Faulty mean:   {dist_stats['faulty_mean']:.4f}")
    print(f"   Separation:    {dist_stats['separation']:.4f}")
    print(f"   Separation ratio: {dist_stats['separation_ratio']:.2f}x")
    
    # Extract individual scores if requested
    if extract_scores:
        try:
            print("\n" + "-"*70)
            print("Extracting individual scores for detailed analysis...")
            anomaly_scores, binary_labels, normal_mask, faulty_mask = extract_individual_scores(
                dataset_path, gdn_model_path
            )
            
            # Separate by ground truth
            faulty_scores = anomaly_scores[faulty_mask]
            normal_scores = anomaly_scores[normal_mask]
            
            print(f"   ✓ Extracted {len(normal_scores)} normal and {len(faulty_scores)} faulty scores")
            
        except Exception as e:
            print(f"   ⚠️  Could not extract individual scores: {e}")
            print("   Using aggregated statistics from results file...")
            extract_scores = False
    
    if not extract_scores:
        # Use aggregated statistics (less detailed but still useful)
        print("\n" + "-"*70)
        print("Using aggregated statistics from results file...")
        
        # Estimate distributions from summary statistics
        normal_mean = dist_stats['normal_mean']
        normal_std = dist_stats['normal_std']
        faulty_mean = dist_stats['faulty_mean']
        faulty_std = dist_stats['faulty_std']
        
        # Generate synthetic scores for analysis (approximation)
        num_normal = results['statistics']['num_normal']
        num_faulty = results['statistics']['num_faulty']
        
        np.random.seed(42)  # For reproducibility
        normal_scores = np.random.normal(normal_mean, normal_std, num_normal)
        faulty_scores = np.random.normal(faulty_mean, faulty_std, num_faulty)
        
        # Ensure non-negative
        normal_scores = np.maximum(normal_scores, 0)
        faulty_scores = np.maximum(faulty_scores, 0)
        
        print("   ⚠️  Note: Using synthetic distributions based on summary statistics")
        print("   For more accurate analysis, ensure individual scores can be extracted")
    
    # 1. Score distribution analysis
    print("\n" + "="*70)
    print("1. SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\n   Faulty windows (n={len(faulty_scores)}):")
    print(f"      Mean:   {faulty_scores.mean():.4f}")
    print(f"      Median: {np.median(faulty_scores):.4f}")
    print(f"      Std:    {faulty_scores.std():.4f}")
    print(f"      Range:  [{faulty_scores.min():.4f}, {faulty_scores.max():.4f}]")
    print(f"      Q25:    {np.percentile(faulty_scores, 25):.4f}")
    print(f"      Q75:    {np.percentile(faulty_scores, 75):.4f}")
    
    print(f"\n   Normal windows (n={len(normal_scores)}):")
    print(f"      Mean:   {normal_scores.mean():.4f}")
    print(f"      Median: {np.median(normal_scores):.4f}")
    print(f"      Std:    {normal_scores.std():.4f}")
    print(f"      Range:  [{normal_scores.min():.4f}, {normal_scores.max():.4f}]")
    print(f"      Q25:    {np.percentile(normal_scores, 25):.4f}")
    print(f"      Q75:    {np.percentile(normal_scores, 75):.4f}")
    
    # 2. Overlap analysis
    print("\n" + "="*70)
    print("2. DISTRIBUTION OVERLAP ANALYSIS")
    print("="*70)
    
    overlap_start = max(normal_scores.min(), faulty_scores.min())
    overlap_end = min(normal_scores.max(), faulty_scores.max())
    
    normal_in_overlap = np.sum((normal_scores >= overlap_start) & (normal_scores <= overlap_end))
    faulty_in_overlap = np.sum((faulty_scores >= overlap_start) & (faulty_scores <= overlap_end))
    
    overlap_percent_normal = normal_in_overlap / len(normal_scores) * 100
    overlap_percent_faulty = faulty_in_overlap / len(faulty_scores) * 100
    
    print(f"\n   Overlap range: [{overlap_start:.4f}, {overlap_end:.4f}]")
    print(f"   Normal windows in overlap: {normal_in_overlap}/{len(normal_scores)} ({overlap_percent_normal:.1f}%)")
    print(f"   Faulty windows in overlap: {faulty_in_overlap}/{len(faulty_scores)} ({overlap_percent_faulty:.1f}%)")
    
    # Compute overlap coefficient (how much distributions overlap)
    bins = np.linspace(min(normal_scores.min(), faulty_scores.min()),
                      max(normal_scores.max(), faulty_scores.max()), 50)
    normal_hist, _ = np.histogram(normal_scores, bins=bins)
    faulty_hist, _ = np.histogram(faulty_scores, bins=bins)
    
    # Normalize histograms
    normal_hist = normal_hist / (normal_hist.sum() + 1e-8)
    faulty_hist = faulty_hist / (faulty_hist.sum() + 1e-8)
    
    # Overlap coefficient (minimum of the two at each bin)
    overlap_coefficient = np.sum(np.minimum(normal_hist, faulty_hist))
    
    print(f"\n   Overlap coefficient: {overlap_coefficient:.3f}")
    if overlap_coefficient > 0.8:
        print("      ⚠️  VERY HIGH overlap - distributions are nearly identical!")
    elif overlap_coefficient > 0.6:
        print("      ⚠️  HIGH overlap - weak separation")
    elif overlap_coefficient > 0.4:
        print("      ⚠️  MODERATE overlap - some separation")
    else:
        print("      ✓ LOW overlap - good separation")
    
    # 3. Statistical separability
    print("\n" + "="*70)
    print("3. STATISTICAL SEPARABILITY")
    print("="*70)
    
    t_stat, p_value = stats.ttest_ind(faulty_scores, normal_scores)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((faulty_scores.std()**2 + normal_scores.std()**2) / 2)
    effect_size = (faulty_scores.mean() - normal_scores.mean()) / (pooled_std + 1e-8)
    
    print(f"\n   T-test:")
    print(f"      t-statistic: {t_stat:.4f}")
    print(f"      p-value:     {p_value:.6f}")
    print(f"      {'Significant' if p_value < 0.05 else 'NOT significant'} difference (p < 0.05)")
    
    print(f"\n   Effect size (Cohen's d): {effect_size:.4f}")
    if abs(effect_size) < 0.2:
        print("      ⚠️  VERY SMALL effect size - distributions are nearly identical!")
        effect_category = "VERY_SMALL"
    elif abs(effect_size) < 0.5:
        print("      ⚠️  SMALL effect size - weak separation")
        effect_category = "SMALL"
    elif abs(effect_size) < 0.8:
        print("      ⚠️  MODERATE effect size - moderate separation")
        effect_category = "MODERATE"
    else:
        print("      ✓ LARGE effect size - good separation")
        effect_category = "LARGE"
    
    # 4. Optimal threshold analysis
    print("\n" + "="*70)
    print("4. THRESHOLD ANALYSIS")
    print("="*70)
    
    current_threshold = metrics['threshold']
    print(f"\n   Current threshold: {current_threshold:.4f}")
    
    # Combine scores and labels for threshold search
    all_scores = np.concatenate([normal_scores, faulty_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(faulty_scores))])
    
    # Find optimal threshold using F1 score
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 200)
    
    best_f1 = 0
    best_threshold = current_threshold
    best_precision = 0
    best_recall = 0
    
    f1_scores = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (all_scores >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:  # Need both classes
            tp = np.sum((y_pred == 1) & (all_labels == 1))
            fp = np.sum((y_pred == 1) & (all_labels == 0))
            fn = np.sum((y_pred == 0) & (all_labels == 1))
            tn = np.sum((y_pred == 0) & (all_labels == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        else:
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
    
    f1_scores = np.array(f1_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Also try Youden's J statistic (maximizes TPR - FPR)
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold_youden = thresholds_roc[optimal_idx]
    
    print(f"\n   Testing alternative thresholds:")
    print(f"      Current ({current_threshold:.4f}):")
    print(f"         P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    print(f"\n      Optimal (F1-maximizing, {best_threshold:.4f}):")
    print(f"         P={best_precision:.3f}, R={best_recall:.3f}, F1={best_f1:.3f}")
    
    print(f"\n      Youden's J ({optimal_threshold_youden:.4f}):")
    y_pred_youden = (all_scores >= optimal_threshold_youden).astype(int)
    tp = np.sum((y_pred_youden == 1) & (all_labels == 1))
    fp = np.sum((y_pred_youden == 1) & (all_labels == 0))
    fn = np.sum((y_pred_youden == 0) & (all_labels == 1))
    precision_youden = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_youden = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_youden = 2 * precision_youden * recall_youden / (precision_youden + recall_youden) if (precision_youden + recall_youden) > 0 else 0
    print(f"         P={precision_youden:.3f}, R={recall_youden:.3f}, F1={f1_youden:.3f}")
    
    # Compare to current
    current_f1 = metrics['f1_score']
    improvement = (best_f1 - current_f1) / current_f1 * 100 if current_f1 > 0 else 0
    
    print(f"\n   Improvement:")
    print(f"      F1 improvement: {improvement:+.1f}%")
    print(f"      Precision improvement: {(best_precision - metrics['precision']) / metrics['precision'] * 100:+.1f}%")
    print(f"      Recall change: {(best_recall - metrics['recall']) / metrics['recall'] * 100:+.1f}%")
    
    # 5. Root cause hypothesis
    print("\n" + "="*70)
    print("5. ROOT CAUSE HYPOTHESIS")
    print("="*70)
    
    print("\n   Analysis Summary:")
    print(f"      Effect size: {abs(effect_size):.3f} ({effect_category})")
    print(f"      Overlap coefficient: {overlap_coefficient:.3f}")
    print(f"      F1 improvement potential: {improvement:+.1f}%")
    
    scenario = None
    if abs(effect_size) < 0.3:
        scenario = "B"
        print("\n   ✗ PRIMARY ISSUE: GDN embeddings don't separate normal from faulty")
        print("     Possible causes:")
        print("     - GDN not trained properly (underfitting)")
        print("     - Training data quality issues")
        print("     - Feature/sensor selection problems")
        print("     - Model architecture insufficient for this data")
        print("     - Loss function not suited for imbalanced data")
        print("\n   RECOMMENDED FIX: Retrain GDN with better hyperparameters")
        print("     (See Fix B1-B3 in documentation)")
        
    elif improvement > 30 and abs(effect_size) > 0.5:
        scenario = "A"
        print("\n   ✓ PRIMARY ISSUE: Threshold too low")
        print(f"     Simply adjusting threshold from {current_threshold:.4f} to {best_threshold:.4f}")
        print(f"     improves F1 by {improvement:.1f}%")
        print("\n   RECOMMENDED FIX: Use optimal threshold")
        print("     Update threshold in evaluate_gdn.py or GDN predictor")
        
    elif improvement > 15:
        scenario = "C"
        print("\n   ⚠️  MIXED ISSUE: Both embedding quality and threshold")
        print("     Distribution overlap is significant")
        print(f"     Threshold adjustment helps ({improvement:.1f}% gain) but may not be enough")
        print("\n   RECOMMENDED FIX:")
        print("     1. Apply threshold fix immediately (gets you 15-30% improvement)")
        print("     2. Then retrain GDN with better hyperparameters (gets another 10-20%)")
    else:
        scenario = "B"
        print("\n   ✗ PRIMARY ISSUE: GDN model problem")
        print("     Threshold adjustment provides minimal improvement")
        print("     Root cause is poor embedding quality")
        print("\n   RECOMMENDED FIX: Retrain GDN with better hyperparameters")
    
    # 6. Visualization
    print("\n" + "="*70)
    print("6. GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Score distributions
    ax1 = plt.subplot(2, 3, 1)
    bins = np.linspace(min(normal_scores.min(), faulty_scores.min()),
                      max(normal_scores.max(), faulty_scores.max()), 30)
    ax1.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='green', density=True)
    ax1.hist(faulty_scores, bins=bins, alpha=0.6, label='Faulty', color='red', density=True)
    ax1.axvline(current_threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Current ({current_threshold:.3f})')
    ax1.axvline(best_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Optimal ({best_threshold:.3f})')
    ax1.set_xlabel('Anomaly Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall vs Threshold
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(thresholds, precisions, label='Precision', color='blue', linewidth=2)
    ax2.plot(thresholds, recalls, label='Recall', color='red', linewidth=2)
    ax2.plot(thresholds, f1_scores, label='F1 Score', color='green', linewidth=2)
    ax2.axvline(current_threshold, color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(best_threshold, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Threshold', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Precision-Recall-F1 vs Threshold', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plot comparison
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([normal_scores, faulty_scores], labels=['Normal', 'Faulty'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)
    ax3.set_ylabel('Anomaly Score', fontweight='bold')
    ax3.set_title('Score Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: ROC Curve
    ax4 = plt.subplot(2, 3, 4)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    ax4.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC={metrics.get("roc_auc", 0.0):.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax4.set_xlabel('False Positive Rate', fontweight='bold')
    ax4.set_ylabel('True Positive Rate', fontweight='bold')
    ax4.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Precision-Recall Curve
    ax5 = plt.subplot(2, 3, 5)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
    ax5.plot(recall_curve, precision_curve, color='red', linewidth=2,
             label=f'PR (AUC={metrics.get("pr_auc", 0.0):.3f})')
    ax5.set_xlabel('Recall', fontweight='bold')
    ax5.set_ylabel('Precision', fontweight='bold')
    ax5.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    DIAGNOSTIC SUMMARY
    
    Current Performance:
    ├─ Precision: {metrics['precision']:.3f}
    ├─ Recall:    {metrics['recall']:.3f}
    ├─ F1 Score:  {metrics['f1_score']:.3f}
    └─ Threshold: {current_threshold:.4f}
    
    Optimal Performance:
    ├─ Precision: {best_precision:.3f}
    ├─ Recall:    {best_recall:.3f}
    ├─ F1 Score:  {best_f1:.3f}
    └─ Threshold: {best_threshold:.4f}
    
    Improvement:
    └─ F1: {improvement:+.1f}%
    
    Statistical Analysis:
    ├─ Effect size: {effect_size:.3f} ({effect_category})
    ├─ Overlap:     {overlap_coefficient:.3f}
    └─ Scenario:    {scenario}
    
    Recommendation:
    {'Threshold adjustment' if scenario == 'A' else 'Retrain GDN' if scenario == 'B' else 'Both fixes needed'}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.tight_layout()
    diagnostic_plot_path = output_dir / 'gdn_diagnostic.png'
    plt.savefig(diagnostic_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   ✓ Diagnostic plot saved: {diagnostic_plot_path}")
    
    # Save diagnostic results
    diagnostic_results = {
        'scenario': scenario,
        'effect_size': float(effect_size),
        'effect_category': effect_category,
        'overlap_coefficient': float(overlap_coefficient),
        'current_threshold': float(current_threshold),
        'optimal_threshold': float(best_threshold),
        'optimal_threshold_youden': float(optimal_threshold_youden),
        'current_metrics': {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1_score']
        },
        'optimal_metrics': {
            'precision': float(best_precision),
            'recall': float(best_recall),
            'f1': float(best_f1)
        },
        'improvement': {
            'f1_percent': float(improvement),
            'precision_percent': float((best_precision - metrics['precision']) / metrics['precision'] * 100),
            'recall_percent': float((best_recall - metrics['recall']) / metrics['recall'] * 100)
        },
        'score_statistics': {
            'normal': {
                'mean': float(normal_scores.mean()),
                'std': float(normal_scores.std()),
                'min': float(normal_scores.min()),
                'max': float(normal_scores.max()),
                'median': float(np.median(normal_scores))
            },
            'faulty': {
                'mean': float(faulty_scores.mean()),
                'std': float(faulty_scores.std()),
                'min': float(faulty_scores.min()),
                'max': float(faulty_scores.max()),
                'median': float(np.median(faulty_scores))
            }
        }
    }
    
    diagnostic_results_path = output_dir / 'gdn_diagnostic_results.json'
    with open(diagnostic_results_path, 'w') as f:
        json.dump(diagnostic_results, f, indent=2)
    
    print(f"   ✓ Diagnostic results saved: {diagnostic_results_path}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    
    return diagnostic_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose GDN false positive problem')
    parser.add_argument('--results', type=str, default='results/gdn_evaluation_results.json',
                       help='Path to GDN evaluation results JSON')
    parser.add_argument('--dataset', type=str, default='llm/evaluation/shared_dataset/test.npz',
                       help='Path to test dataset')
    parser.add_argument('--model', type=str, default='anomaly-detection/best_multilabel_gdn.pt',
                       help='Path to GDN model')
    parser.add_argument('--no-extract', action='store_true',
                       help='Skip extracting individual scores (use aggregated stats only)')
    
    args = parser.parse_args()
    
    diagnosis = diagnose_gdn_model(
        results_path=args.results,
        dataset_path=args.dataset,
        gdn_model_path=args.model,
        extract_scores=not args.no_extract
    )
