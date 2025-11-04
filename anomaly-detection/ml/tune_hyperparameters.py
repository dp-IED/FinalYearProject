#!/usr/bin/env python3
"""
Hyperparameter tuning script for ML anomaly detector.

Uses grid search with cross-validation to find optimal parameters.
"""

import os
import numpy as np
import pandas as pd
from itertools import product
from ml_anomaly_detector import (
    CarOBDMLDataLoader,
    MLAnomalyDetector,
    evaluate_anomaly_detection
)


def grid_search_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list,
    param_grid: dict,
    test_fault_rate: float = 0.2
) -> dict:
    """
    Grid search for Isolation Forest hyperparameters.
    
    Args:
        X_train: Training features
        X_val: Validation features
        feature_names: Feature names
        param_grid: Dictionary of parameter grids
        test_fault_rate: Fault injection rate for testing
        
    Returns:
        Best parameters and score
    """
    print("=" * 70)
    print("GRID SEARCH: Isolation Forest")
    print("=" * 70)
    
    from sklearn.preprocessing import RobustScaler
    from ml_anomaly_detector import CarOBDMLDataLoader
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create fault-injected validation data
    loader = CarOBDMLDataLoader()
    fault_val, fault_labels = loader.create_realistic_fault_data(
        X_val, fault_percentage=test_fault_rate
    )
    fault_val_scaled = scaler.transform(fault_val)
    
    best_score = -np.inf
    best_params = None
    results = []
    
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    print(f"Using {test_fault_rate*100:.0f}% fault injection rate for evaluation\n")
    
    for i, params_tuple in enumerate(param_combinations, 1):
        params = dict(zip(param_grid.keys(), params_tuple))
        
        print(f"[{i}/{total_combinations}] Testing: {params}")
        
        # Create and train model
        detector = MLAnomalyDetector(algorithm='isolation_forest', use_auto_detection=True)
        detector.params['isolation_forest'].update(params)
        detector.scaler = scaler
        detector.feature_names = feature_names
        
        # Train
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(**detector.params['isolation_forest'])
        model.fit(X_train_scaled)
        detector.models['isolation_forest'] = model
        detector.is_trained = True
        
        # Evaluate
        predictions = detector.predict(fault_val_scaled)
        results_dict = evaluate_anomaly_detection(fault_labels, predictions)
        
        if 'isolation_forest' in results_dict:
            metrics = results_dict['isolation_forest']
            f1_score = metrics['f1_score']
            
            results.append({
                **params,
                'f1_score': f1_score,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy']
            })
            
            print(f"  F1-Score: {f1_score:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            if f1_score > best_score:
                best_score = f1_score
                best_params = params.copy()
                print(f"  ✓ New best! (F1: {best_score:.4f})")
    
    print(f"\n{'='*70}")
    print(f"Best Parameters: {best_params}")
    print(f"Best F1-Score: {best_score:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results)
    }


def grid_search_one_class_svm(
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list,
    param_grid: dict,
    test_fault_rate: float = 0.2,
    threshold_method: str = 'percentile'
) -> dict:
    """
    Grid search for One-Class SVM hyperparameters.
    
    Args:
        X_train: Training features
        X_val: Validation features
        feature_names: Feature names
        param_grid: Dictionary of parameter grids
        test_fault_rate: Fault injection rate for testing
        threshold_method: Method for threshold detection
        
    Returns:
        Best parameters and score
    """
    print("=" * 70)
    print("GRID SEARCH: One-Class SVM")
    print("=" * 70)
    
    from sklearn.preprocessing import RobustScaler
    from ml_anomaly_detector import CarOBDMLDataLoader
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create fault-injected validation data
    loader = CarOBDMLDataLoader()
    fault_val, fault_labels = loader.create_realistic_fault_data(
        X_val, fault_percentage=test_fault_rate
    )
    fault_val_scaled = scaler.transform(fault_val)
    
    best_score = -np.inf
    best_params = None
    results = []
    
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    print(f"Using {test_fault_rate*100:.0f}% fault injection rate for evaluation")
    print(f"Threshold method: {threshold_method}\n")
    
    for i, params_tuple in enumerate(param_combinations, 1):
        params = dict(zip(param_grid.keys(), params_tuple))
        
        print(f"[{i}/{total_combinations}] Testing: {params}")
        
        # Create and train model
        detector = MLAnomalyDetector(
            algorithm='one_class_svm',
            use_auto_detection=True,
            svm_threshold_method=threshold_method
        )
        detector.params['one_class_svm'].update(params)
        detector.scaler = scaler
        detector.feature_names = feature_names
        
        # Train
        from sklearn.svm import OneClassSVM
        model = OneClassSVM(**detector.params['one_class_svm'])
        model.fit(X_train_scaled)
        detector.models['one_class_svm'] = model
        
        # Auto-detect threshold
        val_scores = model.decision_function(X_val_scaled)
        detector.svm_auto_threshold = detector._detect_svm_threshold(
            val_scores,
            method=threshold_method,
            percentile=5
        )
        detector.is_trained = True
        
        # Evaluate
        predictions = detector.predict(fault_val_scaled)
        results_dict = evaluate_anomaly_detection(fault_labels, predictions)
        
        if 'one_class_svm' in results_dict:
            metrics = results_dict['one_class_svm']
            f1_score = metrics['f1_score']
            
            results.append({
                **params,
                'f1_score': f1_score,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'accuracy': metrics['accuracy']
            })
            
            print(f"  F1-Score: {f1_score:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            if f1_score > best_score:
                best_score = f1_score
                best_params = params.copy()
                print(f"  ✓ New best! (F1: {best_score:.4f})")
    
    print(f"\n{'='*70}")
    print(f"Best Parameters: {best_params}")
    print(f"Best F1-Score: {best_score:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': pd.DataFrame(results)
    }


def main():
    """Main hyperparameter tuning function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune hyperparameters for anomaly detection models')
    parser.add_argument('--data-path', type=str, default='data/carOBD/obdiidata')
    parser.add_argument('--algorithm', type=str, choices=['isolation_forest', 'one_class_svm', 'both'], default='both')
    parser.add_argument('--test-fault-rate', type=float, default=0.2, help='Fault injection rate for testing')
    parser.add_argument('--save-results', type=str, default=None, help='Path to save results CSV')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"\nData path: {args.data_path}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Test fault rate: {args.test_fault_rate*100:.0f}%")
    print()
    
    # Load data
    loader = CarOBDMLDataLoader(args.data_path)
    idle_data, motion_data = loader.load_all_data()
    all_data = pd.concat([idle_data, motion_data], ignore_index=True)
    
    # Extract features
    all_features = loader.extract_features(all_data)
    feature_names = loader.get_feature_names(all_data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val = train_test_split(
        all_features,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()
    
    all_results = {}
    
    # Tune Isolation Forest
    if args.algorithm in ['isolation_forest', 'both']:
        if_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.6, 0.8, 1.0]
        }
        
        if_results = grid_search_isolation_forest(
            X_train, X_val, feature_names, if_param_grid, args.test_fault_rate
        )
        all_results['isolation_forest'] = if_results
        
        if args.save_results:
            if_results['all_results'].to_csv(
                args.save_results.replace('.csv', '_isolation_forest.csv'),
                index=False
            )
    
    # Tune One-Class SVM
    if args.algorithm in ['one_class_svm', 'both']:
        svm_param_grid = {
            'nu': [0.05, 0.1, 0.15, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        svm_results = grid_search_one_class_svm(
            X_train, X_val, feature_names, svm_param_grid, args.test_fault_rate
        )
        all_results['one_class_svm'] = svm_results
        
        if args.save_results:
            svm_results['all_results'].to_csv(
                args.save_results.replace('.csv', '_one_class_svm.csv'),
                index=False
            )
    
    # Print summary
    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)
    
    for alg_name, results in all_results.items():
        print(f"\n{alg_name.upper()}:")
        print(f"  Best F1-Score: {results['best_score']:.4f}")
        print(f"  Best Parameters: {results['best_params']}")
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETED")
    print("=" * 70)
    print("\nTo use the best parameters, update training.py or create a new model with:")
    for alg_name, results in all_results.items():
        print(f"\n{alg_name.upper()}:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
    
    return all_results


if __name__ == "__main__":
    results = main()

