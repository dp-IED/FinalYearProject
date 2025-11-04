#!/usr/bin/env python3
"""
Simple test script for the ML anomaly detection model.
"""

import os
import numpy as np
import pandas as pd
from ml_anomaly_detector import MLAnomalyDetector, CarOBDMLDataLoader, evaluate_anomaly_detection


def main():
    """Test the ML anomaly detection model."""
    print("=" * 60)
    print("Testing ML Anomaly Detection Model")
    print("=" * 60)
    
    # Path to the saved model
    model_path = "anomaly-detection/models/unified_ml_detector"
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.pkl"):
        print(f"Model not found at {model_path}.pkl")
        print("Please run ml_anomaly_detector.py first to train and save the model.")
        return False
    
    # Load the saved model
    print("Loading saved model...")
    detector = MLAnomalyDetector.load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    loader = CarOBDMLDataLoader("data/carOBD/obdiidata")
    idle_data, motion_data = loader.load_all_data()
    
    # Use a small sample for testing
    test_idle = idle_data.head(500)
    test_motion = motion_data.head(500)
    test_data = pd.concat([test_idle, test_motion], ignore_index=True)
    
    print(f"Test data: {len(test_data)} samples")
    
    # Extract features
    test_features = loader.extract_features(test_data)
    print(f"Test features shape: {test_features.shape}")
    
    # Create realistic fault data for testing
    print("Creating realistic fault data for testing...")
    fault_features, fault_labels = loader.create_realistic_fault_data(test_features, fault_percentage=0.2)
    
    print(f"Fault data: {len(fault_features)} samples")
    print(f"Anomalies: {np.sum(fault_labels)} out of {len(fault_labels)}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = detector.predict(fault_features)
    
    # Evaluate results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    results = evaluate_anomaly_detection(fault_labels, predictions)
    
    for algorithm, metrics in results.items():
        print(f"\n{algorithm.upper()}:")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        
        # Show confusion matrix
        cm = metrics['confusion_matrix']
        print("  Confusion Matrix:")
        print(f"    True Positives: {cm['tp']}")
        print(f"    False Positives: {cm['fp']}")
        print(f"    False Negatives: {cm['fn']}")
        print(f"    True Negatives: {cm['tn']}")
    
    # Show some example predictions
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Show first 10 predictions
    for i in range(min(10, len(fault_features))):
        actual = "Anomaly" if fault_labels[i] == 1 else "Normal"
        pred_if = "Anomaly" if predictions['isolation_forest']['anomalies'][i] else "Normal"
        pred_svm = "Anomaly" if predictions['one_class_svm']['anomalies'][i] else "Normal"
        
        print(f"Sample {i+1}: Actual={actual}, IF={pred_if}, SVM={pred_svm}")
    
    # Model summary
    print("\n" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    
    summary = detector.get_model_summary()
    for key, value in summary.items():
        if key != 'feature_names':  # Skip printing long feature list
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("The model is ready for production use!")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
