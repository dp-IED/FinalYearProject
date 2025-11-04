#!/usr/bin/env python3
"""
Evaluation script for ML anomaly detector.

Tests the trained model at multiple fault injection rates and provides
comprehensive evaluation metrics and comparisons.

Usage:
    python eval.py --model-path anomaly-detection/models/ml_anomaly_detector_auto
    python eval.py --injection-rates 0.05 0.10 0.20 0.50
    python eval.py --save-results results.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from ml_anomaly_detector import (
    CarOBDMLDataLoader,
    MLAnomalyDetector,
    evaluate_anomaly_detection
)


def load_model(model_path: str) -> MLAnomalyDetector:
    """Load a trained model."""
    if not os.path.exists(f"{model_path}.pkl"):
        raise FileNotFoundError(f"Model not found: {model_path}.pkl")
    
    print(f"Loading model from {model_path}.pkl...")
    detector = MLAnomalyDetector.load_model(model_path)
    
    # Display model info
    print(f"✓ Model loaded successfully")
    print(f"  Algorithm: {detector.algorithm}")
    print(f"  Auto-detection: {'ENABLED' if detector.use_auto_detection else 'DISABLED'}")
    if detector.use_auto_detection:
        if 'isolation_forest' in detector.models:
            print(f"  Isolation Forest: contamination='auto'")
        if 'one_class_svm' in detector.models and detector.svm_auto_threshold:
            print(f"  One-Class SVM: threshold={detector.svm_auto_threshold:.4f} ({detector.svm_threshold_method})")
    print()
    
    return detector


def evaluate_at_rate(
    detector: MLAnomalyDetector,
    loader: CarOBDMLDataLoader,
    test_features: np.ndarray,
    injection_rate: float,
    random_seed: int = 42
) -> Dict[str, Dict]:
    """
    Evaluate model at a specific fault injection rate.
    
    Args:
        detector: Trained MLAnomalyDetector
        loader: CarOBDMLDataLoader instance
        test_features: Normal test features
        injection_rate: Fault injection rate (0.0 to 1.0)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of evaluation results per algorithm
    """
    np.random.seed(random_seed)
    
    # Create fault-injected data
    fault_features, fault_labels = loader.create_realistic_fault_data(
        test_features,
        fault_percentage=injection_rate
    )
    
    # Get predictions
    predictions = detector.predict(fault_features)
    
    # Evaluate
    results = evaluate_anomaly_detection(fault_labels, predictions)
    
    # Add injection rate info
    for algorithm in results:
        results[algorithm]['injection_rate'] = injection_rate
        results[algorithm]['n_samples'] = len(fault_labels)
        results[algorithm]['n_faults'] = int(np.sum(fault_labels))
    
    return results


def evaluate_multiple_rates(
    detector: MLAnomalyDetector,
    loader: CarOBDMLDataLoader,
    test_features: np.ndarray,
    injection_rates: List[float],
    random_seed: int = 42
) -> Dict[float, Dict[str, Dict]]:
    """
    Evaluate model at multiple fault injection rates.
    
    Returns:
        Dictionary mapping injection_rate -> algorithm -> metrics
    """
    print("=" * 70)
    print("EVALUATING AT MULTIPLE FAULT INJECTION RATES")
    print("=" * 70)
    print(f"\nInjection rates: {[f'{r*100:.0f}%' for r in injection_rates]}")
    print(f"Test samples: {len(test_features)}")
    print()
    
    all_results = {}
    
    for rate in injection_rates:
        print(f"\n{'='*60}")
        print(f"Injection Rate: {rate*100:.0f}%")
        print(f"{'='*60}")
        
        results = evaluate_at_rate(
            detector, loader, test_features, rate, random_seed
        )
        
        all_results[rate] = results
        
        # Print results
        for algorithm, metrics in results.items():
            print(f"\n  {algorithm.upper()}:")
            print(f"    F1-Score:      {metrics['f1_score']:.4f}")
            print(f"    Precision:     {metrics['precision']:.4f}")
            print(f"    Recall:        {metrics['recall']:.4f}")
            print(f"    Accuracy:      {metrics['accuracy']:.4f}")
            print(f"    Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
            print(f"    ROC-AUC:       {metrics['roc_auc']:.4f}")
            cm = metrics['confusion_matrix']
            print(f"    Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}")
    
    return all_results


def create_results_dataframe(all_results: Dict[float, Dict[str, Dict]]) -> pd.DataFrame:
    """Create a DataFrame from evaluation results for easy analysis."""
    rows = []
    
    for injection_rate, algorithms in all_results.items():
        for algorithm, metrics in algorithms.items():
            row = {
                'injection_rate': injection_rate,
                'injection_rate_pct': injection_rate * 100,
                'algorithm': algorithm,
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'specificity': metrics['specificity'],
                'roc_auc': metrics['roc_auc'],
                'tp': metrics['confusion_matrix']['tp'],
                'fp': metrics['confusion_matrix']['fp'],
                'fn': metrics['confusion_matrix']['fn'],
                'tn': metrics['confusion_matrix']['tn'],
                'n_samples': metrics.get('n_samples', 0),
                'n_faults': metrics.get('n_faults', 0)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    # Group by algorithm and injection rate
    summary_cols = ['injection_rate_pct', 'algorithm', 'f1_score', 'precision', 'recall', 'accuracy', 'roc_auc']
    summary_df = df[summary_cols].copy()
    summary_df['injection_rate_pct'] = summary_df['injection_rate_pct'].astype(int)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    print("\nResults by Injection Rate and Algorithm:")
    print(summary_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    
    for algorithm in df['algorithm'].unique():
        alg_df = df[df['algorithm'] == algorithm]
        print(f"\n{algorithm.upper()}:")
        print(f"  Mean F1-Score:      {alg_df['f1_score'].mean():.4f} ± {alg_df['f1_score'].std():.4f}")
        print(f"  Mean Precision:     {alg_df['precision'].mean():.4f} ± {alg_df['precision'].std():.4f}")
        print(f"  Mean Recall:        {alg_df['recall'].mean():.4f} ± {alg_df['recall'].std():.4f}")
        print(f"  Mean Accuracy:      {alg_df['accuracy'].mean():.4f} ± {alg_df['accuracy'].std():.4f}")
        print(f"  Mean Balanced Acc:  {alg_df['balanced_accuracy'].mean():.4f} ± {alg_df['balanced_accuracy'].std():.4f}")
        print(f"  Mean ROC-AUC:       {alg_df['roc_auc'].mean():.4f} ± {alg_df['roc_auc'].std():.4f}")


def save_results(df: pd.DataFrame, output_path: str):
    """Save results to CSV file."""
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate ML anomaly detector at multiple fault injection rates'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='anomaly-detection/models/ml_anomaly_detector_auto',
        help='Path to trained model (without .pkl)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/carOBD/obdiidata',
        help='Path to carOBD data directory'
    )
    parser.add_argument(
        '--injection-rates',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        help='Fault injection rates to test (default: 0.05 0.10 0.15 0.20 0.30 0.50)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=5000,
        help='Number of samples to use for testing (default: 5000)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save results CSV (optional)'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary table output'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    print("=" * 70)
    print("ML ANOMALY DETECTOR EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Data path: {args.data_path}")
    print(f"  Injection rates: {[f'{r*100:.0f}%' for r in args.injection_rates]}")
    print(f"  Test size: {args.test_size} samples")
    print(f"  Random seed: {args.random_seed}")
    print()
    
    # Load model
    detector = load_model(args.model_path)
    
    # Load test data
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)
    
    loader = CarOBDMLDataLoader(args.data_path)
    idle_data, motion_data = loader.load_all_data()
    
    # Sample test data
    print(f"\nLoaded {len(idle_data)} idle samples and {len(motion_data)} motion samples")
    
    # Sample balanced test data
    n_per_mode = args.test_size // 2
    test_idle = idle_data.sample(min(n_per_mode, len(idle_data)), random_state=args.random_seed)
    test_motion = motion_data.sample(min(n_per_mode, len(motion_data)), random_state=args.random_seed)
    test_data = pd.concat([test_idle, test_motion], ignore_index=True)
    
    print(f"Sampled {len(test_data)} test samples ({len(test_idle)} idle + {len(test_motion)} motion)")
    
    # Extract features
    test_features = loader.extract_features(test_data)
    print(f"Extracted features: shape {test_features.shape}")
    print()
    
    # Evaluate at multiple rates
    all_results = evaluate_multiple_rates(
        detector,
        loader,
        test_features,
        args.injection_rates,
        args.random_seed
    )
    
    # Create results DataFrame
    results_df = create_results_dataframe(all_results)
    
    # Print summary
    if not args.no_summary:
        print_summary_table(results_df)
    
    # Save results if requested
    if args.save_results:
        save_results(results_df, args.save_results)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    results_df = main()

