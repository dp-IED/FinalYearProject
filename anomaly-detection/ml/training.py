#!/usr/bin/env python3
"""
Training script for ML anomaly detector with auto-detection.

This script trains anomaly detection models using auto-detection to avoid
parameter leakage when working with synthetic fault injection.

Key Features:
- Auto-detection for both Isolation Forest and One-Class SVM
- No parameter leakage (thresholds from training data only)
- Test at multiple fault injection rates independently
- Proper train/validation split
"""

import os
import sys
import numpy as np
import pandas as pd
from ml_anomaly_detector import (
    CarOBDMLDataLoader,
    MLAnomalyDetector,
    evaluate_anomaly_detection
)


def train_with_auto_detection(
    data_path: str = "data/carOBD/obdiidata",
    model_path: str = "anomaly-detection/models/ml_anomaly_detector_auto",
    algorithm: str = 'both',
    validation_split: float = 0.2,
    svm_threshold_method: str = 'percentile',
    svm_threshold_percentile: int = 5
) -> MLAnomalyDetector:
    """
    Train ML anomaly detector with auto-detection enabled.
    
    Args:
        data_path: Path to carOBD data directory
        model_path: Path to save the trained model (without .pkl extension)
        algorithm: 'isolation_forest', 'one_class_svm', or 'both'
        validation_split: Fraction of data for validation
        svm_threshold_method: Method for SVM threshold detection ('percentile', 'iqr', 'std')
        svm_threshold_percentile: Percentile for percentile method (default: 5)
        
    Returns:
        Trained MLAnomalyDetector instance
    """
    print("=" * 70)
    print("ML Anomaly Detection Training with Auto-Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Validation split: {validation_split}")
    print(f"  Auto-detection: ENABLED")
    print(f"  SVM threshold method: {svm_threshold_method}")
    if svm_threshold_method == 'percentile':
        print(f"  SVM threshold percentile: {svm_threshold_percentile}%")
    print()
    
    # Create models directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize data loader
    loader = CarOBDMLDataLoader(data_path)
    
    try:
        # Load all data (normal data only - no faults)
        print("=" * 50)
        print("LOADING NORMAL DATA (No Faults)")
        print("=" * 50)
        
        idle_data, motion_data = loader.load_all_data()
        
        # Combine data for unified training
        all_data = pd.concat([idle_data, motion_data], ignore_index=True)
        print(f"Combined normal data: {len(all_data)} samples")
        
        # Extract features
        all_features = loader.extract_features(all_data)
        feature_names = loader.get_feature_names(all_data)
        
        print(f"Features shape: {all_features.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names[:5]}..." if len(feature_names) > 5 else f"Feature names: {feature_names}")
        
        # Train ML model with auto-detection
        print("\n" + "=" * 50)
        print("TRAINING WITH AUTO-DETECTION")
        print("=" * 50)
        print("\nNote: Models will determine thresholds from normal data only.")
        print("This ensures no parameter leakage when testing at different fault injection rates.\n")
        
        detector = MLAnomalyDetector(
            algorithm=algorithm,
            use_auto_detection=True,
            svm_threshold_method=svm_threshold_method,
            svm_threshold_percentile=svm_threshold_percentile
        )
        
        detector.fit_with_validation(
            all_features, 
            feature_names, 
            validation_split=validation_split
        )
        
        # Display training metadata
        print("\n" + "=" * 50)
        print("TRAINING METADATA")
        print("=" * 50)
        
        for model_name, metadata in detector.metadata.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Train score mean: {metadata.get('train_score_mean', 'N/A'):.4f}")
            print(f"  Train score std: {metadata.get('train_score_std', 'N/A'):.4f}")
            print(f"  Val score mean: {metadata.get('val_score_mean', 'N/A'):.4f}")
            print(f"  Val score std: {metadata.get('val_score_std', 'N/A'):.4f}")
            if 'use_auto_detection' in metadata and metadata['use_auto_detection']:
                print(f"  Auto-detection: ENABLED")
                if 'auto_threshold' in metadata and metadata['auto_threshold'] is not None:
                    print(f"  Auto-detected threshold: {metadata['auto_threshold']:.4f}")
                    print(f"  Threshold method: {metadata.get('threshold_method', 'N/A')}")
        
        # Save model
        print("\n" + "=" * 50)
        print("SAVING MODEL")
        print("=" * 50)
        
        detector.save_model(model_path)
        print(f"Model saved to: {model_path}.pkl")
        
        # Model summary
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        
        summary = detector.get_model_summary()
        for key, value in summary.items():
            if key != 'feature_names':  # Skip printing long feature list
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey features:")
        print("✓ Auto-detection enabled (no parameter leakage)")
        print("✓ Thresholds determined from normal data only")
        print("✓ Can test at any fault injection rate independently")
        print("✓ Unified model for both idle and motion data")
        print(f"✓ Model saved to: {model_path}.pkl")
        
        return detector
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_at_multiple_rates(
    detector: MLAnomalyDetector,
    loader: CarOBDMLDataLoader,
    test_data_sample: np.ndarray,
    injection_rates: list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
) -> dict:
    """
    Test model at multiple fault injection rates to demonstrate independence.
    
    Args:
        detector: Trained MLAnomalyDetector
        loader: CarOBDMLDataLoader instance
        test_data_sample: Sample of normal data for testing
        injection_rates: List of fault injection rates to test
        
    Returns:
        Dictionary of results for each injection rate
    """
    print("\n" + "=" * 70)
    print("TESTING AT MULTIPLE FAULT INJECTION RATES")
    print("=" * 70)
    print("\nThis demonstrates that auto-detection works independently")
    print("of the test injection rate (no parameter leakage).\n")
    
    results = {}
    
    for rate in injection_rates:
        print(f"\n{'='*50}")
        print(f"Testing at {rate*100:.0f}% fault injection")
        print(f"{'='*50}")
        
        # Create fault-injected data
        fault_features, fault_labels = loader.create_realistic_fault_data(
            test_data_sample, 
            fault_percentage=rate
        )
        
        # Predict
        predictions = detector.predict(fault_features)
        
        # Evaluate
        rate_results = evaluate_anomaly_detection(fault_labels, predictions)
        
        # Store results
        results[rate] = rate_results
        
        # Print summary
        for algorithm, metrics in rate_results.items():
            print(f"\n  {algorithm.upper()}:")
            print(f"    F1-Score: {metrics['f1_score']:.3f}")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    Accuracy: {metrics['accuracy']:.3f}")
    
    return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ML anomaly detector with auto-detection'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/carOBD/obdiidata',
        help='Path to carOBD data directory'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='anomaly-detection/models/ml_anomaly_detector_auto',
        help='Path to save model (without .pkl)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['isolation_forest', 'one_class_svm', 'both'],
        default='both',
        help='Algorithm to use'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--svm-threshold-method',
        type=str,
        choices=['percentile', 'iqr', 'std'],
        default='percentile',
        help='Method for SVM threshold detection'
    )
    parser.add_argument(
        '--svm-threshold-percentile',
        type=int,
        default=5,
        help='Percentile for percentile method (default: 5)'
    )
    parser.add_argument(
        '--test-multiple-rates',
        action='store_true',
        help='Test model at multiple injection rates after training'
    )
    
    args = parser.parse_args()
    
    # Train model
    detector = train_with_auto_detection(
        data_path=args.data_path,
        model_path=args.model_path,
        algorithm=args.algorithm,
        validation_split=args.validation_split,
        svm_threshold_method=args.svm_threshold_method,
        svm_threshold_percentile=args.svm_threshold_percentile
    )
    
    # Optional: Test at multiple rates
    if args.test_multiple_rates:
        loader = CarOBDMLDataLoader(args.data_path)
        idle_data, motion_data = loader.load_all_data()
        test_data = pd.concat([idle_data.head(1000), motion_data.head(1000)], ignore_index=True)
        test_features = loader.extract_features(test_data)
        
        test_at_multiple_rates(detector, loader, test_features)
    
    return detector


if __name__ == "__main__":
    detector = main()

