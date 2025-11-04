#!/usr/bin/env python3
"""
Ensemble anomaly detector combining Isolation Forest and One-Class SVM.

Provides multiple ensemble strategies:
- Voting (hard/soft)
- Score averaging
- Weighted combination
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ml_anomaly_detector import MLAnomalyDetector, evaluate_anomaly_detection


class EnsembleAnomalyDetector:
    """Ensemble detector combining multiple anomaly detection models."""
    
    def __init__(
        self,
        detector: MLAnomalyDetector,
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble detector.
        
        Args:
            detector: Trained MLAnomalyDetector with both models
            method: Ensemble method ('voting', 'average', 'weighted_average')
            weights: Optional weights for each model (default: based on F1-scores)
        """
        if not detector.is_trained:
            raise ValueError("Detector must be trained first")
        
        if 'isolation_forest' not in detector.models and 'one_class_svm' not in detector.models:
            raise ValueError("Detector must have at least one model")
        
        self.detector = detector
        self.method = method
        self.weights = weights or self._default_weights()
        self.is_trained = True
        
        print(f"Ensemble detector initialized:")
        print(f"  Method: {method}")
        print(f"  Models: {list(detector.models.keys())}")
        print(f"  Weights: {self.weights}")
    
    def _default_weights(self) -> Dict[str, float]:
        """Default weights based on typical performance."""
        # Isolation Forest typically performs better
        weights = {}
        if 'isolation_forest' in self.detector.models:
            weights['isolation_forest'] = 0.7
        if 'one_class_svm' in self.detector.models:
            weights['one_class_svm'] = 0.3
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict anomalies using ensemble method.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with ensemble predictions and individual model predictions
        """
        # Get individual model predictions
        individual_predictions = self.detector.predict(X)
        
        # Extract scores and predictions
        scores = {}
        predictions = {}
        
        for model_name in individual_predictions:
            scores[model_name] = individual_predictions[model_name]['scores']
            predictions[model_name] = individual_predictions[model_name]['predictions']
        
        # Apply ensemble method
        if self.method == 'voting':
            ensemble_pred, ensemble_scores = self._hard_voting(predictions)
        elif self.method == 'soft_voting':
            ensemble_pred, ensemble_scores = self._soft_voting(scores)
        elif self.method == 'average':
            ensemble_pred, ensemble_scores = self._score_average(scores)
        elif self.method == 'weighted_average':
            ensemble_pred, ensemble_scores = self._weighted_average(scores)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        # Combine results
        result = {
            'ensemble': {
                'predictions': ensemble_pred,
                'scores': ensemble_scores,
                'anomalies': ensemble_pred == -1
            },
            **individual_predictions
        }
        
        return result
    
    def _hard_voting(self, predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hard voting: Majority vote.
        
        Anomaly if majority of models predict anomaly.
        """
        if len(predictions) == 1:
            return list(predictions.values())[0], np.zeros(len(list(predictions.values())[0]))
        
        # Count votes (-1 for anomaly, 1 for normal)
        votes = np.zeros(len(list(predictions.values())[0]))
        for pred in predictions.values():
            votes += (pred == -1).astype(int)
        
        # Majority vote (anomaly if >50% vote anomaly)
        threshold = len(predictions) / 2
        ensemble_pred = np.where(votes > threshold, -1, 1)
        
        # Use vote count as score
        ensemble_scores = votes / len(predictions)
        
        return ensemble_pred, ensemble_scores
    
    def _soft_voting(self, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft voting: Average scores, then threshold.
        
        Average normalized scores from all models.
        """
        if len(scores) == 1:
            return np.where(list(scores.values())[0] < 0, -1, 1), list(scores.values())[0]
        
        # Normalize scores to [0, 1] range for each model
        normalized_scores = {}
        for model_name, score in scores.items():
            # Normalize to [0, 1], where 0 = most anomalous, 1 = most normal
            score_min, score_max = score.min(), score.max()
            if score_max > score_min:
                normalized = (score - score_min) / (score_max - score_min)
            else:
                normalized = np.ones_like(score)
            normalized_scores[model_name] = normalized
        
        # Average normalized scores
        ensemble_scores = np.zeros(len(list(scores.values())[0]))
        for model_name, norm_score in normalized_scores.items():
            ensemble_scores += self.weights.get(model_name, 1.0) * norm_score
        
        # Threshold at 0.5 (below average = anomaly)
        ensemble_pred = np.where(ensemble_scores < 0.5, -1, 1)
        
        return ensemble_pred, ensemble_scores
    
    def _score_average(self, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple score averaging.
        
        Average raw scores from all models.
        """
        if len(scores) == 1:
            return np.where(list(scores.values())[0] < 0, -1, 1), list(scores.values())[0]
        
        # Average scores
        ensemble_scores = np.zeros(len(list(scores.values())[0]))
        for model_name, score in scores.items():
            ensemble_scores += self.weights.get(model_name, 1.0) * score
        
        # Normalize by weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            ensemble_scores /= total_weight
        
        # Threshold at 0 (negative = anomaly for most models)
        ensemble_pred = np.where(ensemble_scores < 0, -1, 1)
        
        return ensemble_pred, ensemble_scores
    
    def _weighted_average(self, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted average of normalized scores.
        
        Similar to soft voting but with better normalization.
        """
        if len(scores) == 1:
            return np.where(list(scores.values())[0] < 0, -1, 1), list(scores.values())[0]
        
        # Normalize each model's scores using z-score
        normalized_scores = {}
        for model_name, score in scores.items():
            mean_score = np.mean(score)
            std_score = np.std(score)
            if std_score > 0:
                normalized = (score - mean_score) / std_score
            else:
                normalized = np.zeros_like(score)
            normalized_scores[model_name] = normalized
        
        # Weighted average
        ensemble_scores = np.zeros(len(list(scores.values())[0]))
        for model_name, norm_score in normalized_scores.items():
            ensemble_scores += self.weights.get(model_name, 1.0) * norm_score
        
        # Threshold at 0 (negative = anomaly)
        ensemble_pred = np.where(ensemble_scores < 0, -1, 1)
        
        return ensemble_pred, ensemble_scores
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate ensemble performance.
        
        Args:
            X: Feature matrix
            y_true: True labels (1 for anomaly, 0 for normal)
            
        Returns:
            Evaluation results for ensemble and individual models
        """
        predictions = self.predict(X)
        
        # Evaluate ensemble
        ensemble_results = {}
        ensemble_results['ensemble'] = {
            'predictions': predictions['ensemble']['anomalies'].astype(int),
            'scores': predictions['ensemble']['scores'],
            'anomalies': predictions['ensemble']['anomalies']
        }
        
        ensemble_metrics = evaluate_anomaly_detection(y_true, ensemble_results)
        
        # Evaluate individual models
        individual_metrics = evaluate_anomaly_detection(y_true, predictions)
        
        # Combine results
        results = {
            'ensemble': ensemble_metrics.get('ensemble', {}),
            **individual_metrics
        }
        
        return results


def main():
    """Test ensemble detector."""
    import argparse
    from ml_anomaly_detector import CarOBDMLDataLoader
    
    parser = argparse.ArgumentParser(description='Test ensemble anomaly detector')
    parser.add_argument('--model-path', type=str, default='anomaly-detection/models/ml_anomaly_detector_auto')
    parser.add_argument('--data-path', type=str, default='data/carOBD/obdiidata')
    parser.add_argument('--method', type=str, choices=['voting', 'soft_voting', 'average', 'weighted_average'], default='weighted_average')
    parser.add_argument('--test-size', type=int, default=5000)
    parser.add_argument('--injection-rate', type=float, default=0.2)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENSEMBLE ANOMALY DETECTOR")
    print("=" * 70)
    
    # Load model
    detector = MLAnomalyDetector.load_model(args.model_path)
    
    # Create ensemble
    ensemble = EnsembleAnomalyDetector(detector, method=args.method)
    
    # Load test data
    loader = CarOBDMLDataLoader(args.data_path)
    idle_data, motion_data = loader.load_all_data()
    
    test_idle = idle_data.sample(min(args.test_size // 2, len(idle_data)), random_state=42)
    test_motion = motion_data.sample(min(args.test_size // 2, len(motion_data)), random_state=42)
    test_data = pd.concat([test_idle, test_motion], ignore_index=True)
    
    test_features = loader.extract_features(test_data)
    
    # Create faults
    fault_features, fault_labels = loader.create_realistic_fault_data(
        test_features, fault_percentage=args.injection_rate
    )
    
    # Evaluate
    print(f"\nEvaluating at {args.injection_rate*100:.0f}% injection rate...")
    results = ensemble.evaluate(fault_features, fault_labels)
    
    # Print results
    print("\n" + "=" * 70)
    print("ENSEMBLE RESULTS")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  F1-Score:      {metrics['f1_score']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
    
    # Compare to individual models
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    if 'ensemble' in results and 'isolation_forest' in results:
        ensemble_f1 = results['ensemble']['f1_score']
        if_f1 = results['isolation_forest']['f1_score']
        improvement = ((ensemble_f1 - if_f1) / if_f1) * 100
        print(f"\nEnsemble vs Isolation Forest:")
        print(f"  F1 improvement: {improvement:+.2f}%")
        print(f"  ({if_f1:.4f} → {ensemble_f1:.4f})")
    
    if 'ensemble' in results and 'one_class_svm' in results:
        ensemble_f1 = results['ensemble']['f1_score']
        svm_f1 = results['one_class_svm']['f1_score']
        improvement = ((ensemble_f1 - svm_f1) / svm_f1) * 100
        print(f"\nEnsemble vs One-Class SVM:")
        print(f"  F1 improvement: {improvement:+.2f}%")
        print(f"  ({svm_f1:.4f} → {ensemble_f1:.4f})")
    
    return ensemble, results


if __name__ == "__main__":
    import pandas as pd
    ensemble, results = main()

