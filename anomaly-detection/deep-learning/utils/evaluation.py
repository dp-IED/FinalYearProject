"""
Evaluation utilities for deep learning anomaly detection models.
Includes threshold optimization and metrics compatible with ML models.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import sys
import os

# Import evaluation function from ML models
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ml_path = os.path.join(project_root, 'anomaly-detection', 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)
from ml_anomaly_detector import evaluate_anomaly_detection


def optimize_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = 'percentile',
    percentile: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Optimize threshold for anomaly detection using different methods.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: True labels (1 = anomaly, 0 = normal)
        method: Method to use ('percentile', 'iqr', 'std', 'f1_optimal')
        percentile: Percentile for percentile method (default: 5)
        
    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    if method == 'percentile':
        # Bottom X% are considered anomalies
        threshold = np.percentile(scores, percentile)
    
    elif method == 'iqr':
        # Interquartile Range method
        Q1 = np.percentile(scores, 25)
        Q3 = np.percentile(scores, 75)
        IQR = Q3 - Q1
        threshold = Q1 - 1.5 * IQR  # Standard outlier detection
    
    elif method == 'std':
        # Standard deviation method (2 sigma rule)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score - 2 * std_score
    
    elif method == 'f1_optimal':
        # Find threshold that maximizes F1 score
        # Note: For scores where higher = more anomalous, we need to invert
        # For deep learning models, typically higher reconstruction error = more anomalous
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.max(scores)
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    # Calculate metrics at this threshold
    predictions = (scores >= threshold).astype(int)
    metrics = calculate_metrics(labels, predictions, scores)
    
    return float(threshold), metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels (1 = anomaly, 0 = normal)
        scores: Optional anomaly scores for AUC calculation
        
    Returns:
        Dictionary of metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn = fp = fn = tp = 0
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    
    # Calculate ROC-AUC if scores provided
    if scores is not None:
        try:
            if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                roc_auc = roc_auc_score(y_true, scores)
                fpr, tpr, _ = roc_curve(y_true, scores)
                pr_auc = average_precision_score(y_true, scores)
                
                metrics['roc_auc'] = roc_auc
                metrics['pr_auc'] = pr_auc
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            else:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        except ValueError:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    return metrics


def evaluate_deep_learning_model(
    model,
    sequences: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    threshold_method: str = 'percentile',
    threshold_percentile: int = 5,
    device: Optional[object] = None
) -> Dict[str, any]:
    """
    Evaluate deep learning model on sequences with automatic threshold optimization.
    
    Args:
        model: Trained model (LSTM, CNN-LSTM, or Autoencoder)
        sequences: Input sequences of shape (n_samples, sequence_length, n_features)
        labels: True labels (1 = anomaly, 0 = normal)
        threshold: Optional fixed threshold (if None, will optimize)
        threshold_method: Method for threshold optimization
        threshold_percentile: Percentile for percentile method
        device: Device to run model on
        
    Returns:
        Dictionary containing threshold, predictions, and metrics
    """
    # Get anomaly scores from model
    if hasattr(model, 'get_anomaly_scores'):
        scores = model.get_anomaly_scores(sequences, device=device)
    else:
        # Fallback: use predict method
        import torch
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.to(device)
        model.eval()
        
        x = torch.FloatTensor(sequences).to(device)
        batch_size = 64
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size]
                batch_scores = model.predict(batch)
                scores.append(batch_scores)
        
        scores = np.concatenate(scores, axis=0)
    
    # For LSTM models, prediction errors are already anomaly scores (higher = more anomalous)
    # For autoencoder, reconstruction errors are also anomaly scores (higher = more anomalous)
    # Check model type to determine if inversion is needed
    # Only invert if the model outputs scores where lower = more anomalous
    if hasattr(model, 'model_type'):
        # LSTM and autoencoder both use errors where higher = more anomalous, so no inversion needed
        if model.model_type in ['lstm', 'autoencoder']:
            pass  # No inversion needed
    else:
        # Fallback: check correlation if model type unknown
        # But be careful - LSTM prediction errors should already be higher = more anomalous
        if len(np.unique(labels)) > 1:
            correlation = np.corrcoef(scores, labels)[0, 1]
            if correlation < -0.1:  # Only invert if strongly negative correlation
                scores = -scores  # Invert so higher = more anomalous
                print(f"Warning: Inverted scores based on negative correlation ({correlation:.3f})")
    
    # Optimize threshold if not provided
    if threshold is None:
        # Use validation scores to optimize threshold
        # For auto-detection, we should use normal data only
        # But here we have labels, so we can optimize on them
        threshold, _ = optimize_threshold(
            scores,
            labels,
            method=threshold_method,
            percentile=threshold_percentile
        )
    else:
        # Calculate metrics at fixed threshold
        predictions = (scores >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions, scores)
    
    # Get final predictions
    predictions = (scores >= threshold).astype(int)
    metrics = calculate_metrics(labels, predictions, scores)
    
    return {
        'threshold': threshold,
        'predictions': predictions,
        'scores': scores,
        'metrics': metrics,
        'labels': labels
    }


def convert_to_ml_format(predictions_dict: Dict) -> Dict[str, Dict]:
    """
    Convert deep learning evaluation results to ML evaluation format.
    This allows using the same evaluate_anomaly_detection function.
    
    Args:
        predictions_dict: Results from evaluate_deep_learning_model
        
    Returns:
        Dictionary in format compatible with evaluate_anomaly_detection
    """
    return {
        'deep_learning': {
            'predictions': predictions_dict['predictions'],
            'scores': predictions_dict['scores'],
            'anomalies': predictions_dict['predictions'] == 1
        }
    }

