"""
Evaluation metrics for comparing diagnostic methods.

Provides metrics for:
- Window-level accuracy
- Sensor-level precision/recall/F1
- Per-fault-type metrics
- Confusion matrices
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_window_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute window-level metrics (multi-class classification: 0-8).
    
    Window labels are sensor-indexed:
    - 0 = no fault
    - 1-8 = anomalous sensor index (1-indexed: sensor 0 -> label 1, sensor 7 -> label 8)
    
    Args:
        y_true: (N,) array - true window labels (0-8)
        y_pred: (N,) array - predicted window labels (0-8)
        sensor_names: Optional list of sensor names for per-class metrics
        
    Returns:
        Dictionary with accuracy, precision, recall, F1 (weighted and macro),
        and optionally per-class metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted metrics (accounts for class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Macro metrics (unweighted average across classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'window_accuracy': float(accuracy),
        'window_precision_weighted': float(precision_weighted),
        'window_recall_weighted': float(recall_weighted),
        'window_f1_weighted': float(f1_weighted),
        'window_precision_macro': float(precision_macro),
        'window_recall_macro': float(recall_macro),
        'window_f1_macro': float(f1_macro),
        # Keep backward compatibility aliases (using weighted as default)
        'window_precision': float(precision_weighted),
        'window_recall': float(recall_weighted),
        'window_f1': float(f1_weighted)
    }
    
    # Per-class metrics (for each sensor class 0-8)
    if sensor_names is not None:
        num_classes = len(sensor_names) + 1  # 0 (no fault) + sensors
        per_class_metrics = {}
        
        # Class 0: No fault
        class_0_mask_true = (y_true == 0)
        class_0_mask_pred = (y_pred == 0)
        per_class_metrics['no_fault'] = {
            'precision': float(precision_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'recall': float(recall_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'f1': float(f1_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'support': int(class_0_mask_true.sum())
        }
        
        # Classes 1-8: Each sensor
        for sensor_idx in range(len(sensor_names)):
            class_label = sensor_idx + 1  # 1-indexed
            class_mask_true = (y_true == class_label)
            class_mask_pred = (y_pred == class_label)
            
            per_class_metrics[sensor_names[sensor_idx]] = {
                'precision': float(precision_score(class_mask_true, class_mask_pred, zero_division=0)),
                'recall': float(recall_score(class_mask_true, class_mask_pred, zero_division=0)),
                'f1': float(f1_score(class_mask_true, class_mask_pred, zero_division=0)),
                'support': int(class_mask_true.sum())
            }
        
        metrics['per_class'] = per_class_metrics
    
    return metrics


def compute_sensor_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Compute sensor-level metrics (multi-label classification).
    
    Args:
        y_true: (N, num_sensors) binary array - true sensor labels
        y_pred: (N, num_sensors) binary array - predicted sensor labels
        sensor_names: Optional list of sensor names for per-sensor metrics
        
    Returns:
        Dictionary with overall and per-sensor metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    metrics = {
        'sensor_accuracy': float(accuracy),
        'sensor_precision': float(precision),
        'sensor_recall': float(recall),
        'sensor_f1': float(f1)
    }
    
    # Per-sensor metrics
    if sensor_names:
        per_sensor = {}
        num_sensors = len(sensor_names)
        
        for i, sensor_name in enumerate(sensor_names):
            if i < y_true.shape[1]:
                sensor_true = y_true[:, i]
                sensor_pred = y_pred[:, i]
                
                per_sensor[sensor_name] = {
                    'accuracy': float(accuracy_score(sensor_true, sensor_pred)),
                    'precision': float(precision_score(sensor_true, sensor_pred, zero_division=0)),
                    'recall': float(recall_score(sensor_true, sensor_pred, zero_division=0)),
                    'f1': float(f1_score(sensor_true, sensor_pred, zero_division=0)),
                    'true_positives': int(np.sum((sensor_true == 1) & (sensor_pred == 1))),
                    'false_positives': int(np.sum((sensor_true == 0) & (sensor_pred == 1))),
                    'false_negatives': int(np.sum((sensor_true == 1) & (sensor_pred == 0))),
                    'true_negatives': int(np.sum((sensor_true == 0) & (sensor_pred == 0)))
                }
        
        metrics['per_sensor'] = per_sensor
    
    return metrics


def compute_per_fault_type_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_types: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute metrics per fault type.
    
    Args:
        y_true: (N, num_sensors) binary array - true sensor labels
        y_pred: (N, num_sensors) binary array - predicted sensor labels
        fault_types: (N,) array of fault type strings
        sensor_names: Optional list of sensor names
        
    Returns:
        Dictionary with metrics per fault type
    """
    unique_fault_types = np.unique(fault_types)
    unique_fault_types = [ft for ft in unique_fault_types if ft is not None and ft != '']
    
    per_fault_metrics = {}
    
    for fault_type in unique_fault_types:
        # Find windows with this fault type
        mask = fault_types == fault_type
        if mask.sum() == 0:
            continue
        
        fault_y_true = y_true[mask]
        fault_y_pred = y_pred[mask]
        
        # Window-level metrics for this fault type
        window_true = (fault_y_true.sum(axis=1) > 0).astype(int)
        window_pred = (fault_y_pred.sum(axis=1) > 0).astype(int)
        
        per_fault_metrics[fault_type] = {
            'num_windows': int(mask.sum()),
            'window_accuracy': float(accuracy_score(window_true, window_pred)),
            'window_precision': float(precision_score(window_true, window_pred, zero_division=0)),
            'window_recall': float(recall_score(window_true, window_pred, zero_division=0)),
            'window_f1': float(f1_score(window_true, window_pred, zero_division=0)),
            'sensor_accuracy': float(accuracy_score(fault_y_true.flatten(), fault_y_pred.flatten())),
            'sensor_precision': float(precision_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_recall': float(recall_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_f1': float(f1_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0))
        }
    
    return per_fault_metrics


def compute_confusion_matrices(
    y_true_sensor: np.ndarray,
    y_pred_sensor: np.ndarray,
    y_true_window: Optional[np.ndarray] = None,
    y_pred_window: Optional[np.ndarray] = None,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrices for window-level and sensor-level predictions.
    
    Args:
        y_true_sensor: (N, num_sensors) binary array - true sensor labels
        y_pred_sensor: (N, num_sensors) binary array - predicted sensor labels
        y_true_window: (N,) array - true window labels (0-8, sensor-indexed)
        y_pred_window: (N,) array - predicted window labels (0-8, sensor-indexed)
        sensor_names: Optional list of sensor names
        
    Returns:
        Dictionary with confusion matrices
    """
    y_true_sensor = y_true_sensor.astype(int)
    y_pred_sensor = y_pred_sensor.astype(int)
    
    # Window-level confusion matrix (multi-class: 0-8)
    if y_true_window is not None and y_pred_window is not None:
        y_true_window = y_true_window.astype(int)
        y_pred_window = y_pred_window.astype(int)
        # Multi-class confusion matrix (9 classes: 0-8)
        window_cm = confusion_matrix(y_true_window, y_pred_window, labels=list(range(9)))
    else:
        # Fallback: binary window-level (for backward compatibility)
        window_true = (y_true_sensor.sum(axis=1) > 0).astype(int)
        window_pred = (y_pred_sensor.sum(axis=1) > 0).astype(int)
        window_cm = confusion_matrix(window_true, window_pred)
    
    # Sensor-level confusion matrix (flattened)
    sensor_true_flat = y_true_sensor.flatten()
    sensor_pred_flat = y_pred_sensor.flatten()
    sensor_cm = confusion_matrix(sensor_true_flat, sensor_pred_flat)
    
    matrices = {
        'window_confusion_matrix': window_cm.tolist(),
        'sensor_confusion_matrix': sensor_cm.tolist()
    }
    
    # Per-sensor confusion matrices
    if sensor_names:
        per_sensor_cm = {}
        for i, sensor_name in enumerate(sensor_names):
            if i < y_true_sensor.shape[1]:
                sensor_true = y_true_sensor[:, i]
                sensor_pred = y_pred_sensor[:, i]
                per_sensor_cm[sensor_name] = confusion_matrix(sensor_true, sensor_pred).tolist()
        
        matrices['per_sensor_confusion_matrices'] = per_sensor_cm
    
    return matrices


def compute_all_metrics(
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
    y_true_sensor: np.ndarray,
    y_pred_sensor: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    fault_types: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true_window: (N,) binary array - true window labels
        y_pred_window: (N,) binary array - predicted window labels
        y_true_sensor: (N, num_sensors) binary array - true sensor labels
        y_pred_sensor: (N, num_sensors) binary array - predicted sensor labels
        sensor_names: Optional list of sensor names
        fault_types: Optional (N,) array of fault type strings
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Window-level metrics (multi-class: 0-8)
    metrics['window_level'] = compute_window_level_metrics(
        y_true_window, y_pred_window, sensor_names
    )
    
    # Sensor-level metrics
    metrics['sensor_level'] = compute_sensor_level_metrics(
        y_true_sensor, y_pred_sensor, sensor_names
    )
    
    # Confusion matrices
    metrics['confusion_matrices'] = compute_confusion_matrices(
        y_true_sensor, y_pred_sensor, y_true_window, y_pred_window, sensor_names
    )
    
    # Per-fault-type metrics
    if fault_types is not None:
        metrics['per_fault_type'] = compute_per_fault_type_metrics(
            y_true_sensor, y_pred_sensor, fault_types, sensor_names
        )
    
    return metrics


def format_metrics_report(metrics: Dict[str, any]) -> str:
    """
    Format metrics as a human-readable report string.
    
    Args:
        metrics: Dictionary from compute_all_metrics
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("="*80)
    lines.append("EVALUATION METRICS REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Window-level metrics
    if 'window_level' in metrics:
        lines.append("Window-Level Metrics (Sensor-Indexed: 0-8):")
        lines.append("-" * 40)
        wl = metrics['window_level']
        lines.append(f"  Accuracy:  {wl['window_accuracy']:.4f}")
        lines.append(f"  Precision (weighted): {wl['window_precision_weighted']:.4f}")
        lines.append(f"  Recall (weighted):    {wl['window_recall_weighted']:.4f}")
        lines.append(f"  F1 Score (weighted):  {wl['window_f1_weighted']:.4f}")
        lines.append(f"  Precision (macro): {wl['window_precision_macro']:.4f}")
        lines.append(f"  Recall (macro):    {wl['window_recall_macro']:.4f}")
        lines.append(f"  F1 Score (macro):  {wl['window_f1_macro']:.4f}")
        
        # Per-class metrics if available
        if 'per_class' in wl:
            lines.append("\n  Per-Class Metrics:")
            for class_name, class_metrics in wl['per_class'].items():
                if class_metrics['support'] > 0:  # Only show classes with support
                    lines.append(f"    {class_name}:")
                    lines.append(f"      Precision: {class_metrics['precision']:.4f}")
                    lines.append(f"      Recall:    {class_metrics['recall']:.4f}")
                    lines.append(f"      F1:        {class_metrics['f1']:.4f}")
                    lines.append(f"      Support:   {class_metrics['support']}")
        lines.append("")
    
    # Sensor-level metrics
    if 'sensor_level' in metrics:
        lines.append("Sensor-Level Metrics:")
        lines.append("-" * 40)
        sl = metrics['sensor_level']
        lines.append(f"  Accuracy:  {sl['sensor_accuracy']:.4f}")
        lines.append(f"  Precision: {sl['sensor_precision']:.4f}")
        lines.append(f"  Recall:    {sl['sensor_recall']:.4f}")
        lines.append(f"  F1 Score:  {sl['sensor_f1']:.4f}")
        lines.append("")
        
        # Per-sensor metrics
        if 'per_sensor' in sl:
            lines.append("Per-Sensor Metrics:")
            lines.append("-" * 40)
            for sensor_name, sensor_metrics in sl['per_sensor'].items():
                lines.append(f"  {sensor_name}:")
                lines.append(f"    Accuracy:  {sensor_metrics['accuracy']:.4f}")
                lines.append(f"    Precision: {sensor_metrics['precision']:.4f}")
                lines.append(f"    Recall:    {sensor_metrics['recall']:.4f}")
                lines.append(f"    F1 Score:  {sensor_metrics['f1']:.4f}")
                lines.append(f"    TP: {sensor_metrics['true_positives']}, "
                           f"FP: {sensor_metrics['false_positives']}, "
                           f"FN: {sensor_metrics['false_negatives']}, "
                           f"TN: {sensor_metrics['true_negatives']}")
            lines.append("")
    
    # Per-fault-type metrics
    if 'per_fault_type' in metrics:
        lines.append("Per-Fault-Type Metrics:")
        lines.append("-" * 40)
        for fault_type, ft_metrics in metrics['per_fault_type'].items():
            lines.append(f"  {fault_type} (N={ft_metrics['num_windows']}):")
            lines.append(f"    Window F1: {ft_metrics['window_f1']:.4f}")
            lines.append(f"    Sensor F1: {ft_metrics['sensor_f1']:.4f}")
        lines.append("")
    
    lines.append("="*80)
    
    return "\n".join(lines)
