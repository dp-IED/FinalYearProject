"""
Evaluation metrics for comparing diagnostic methods.

Provides metrics for:
- Window-level accuracy
- Sensor-level precision/recall/F1
- Per-fault-type metrics
- Confusion matrices
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_window_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
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
        'window_precision': float(precision_weighted),
        'window_recall': float(recall_weighted),
        'window_f1': float(f1_weighted)
    }

    if sensor_names is not None:
        per_class_metrics = {}
        class_0_mask_true = (y_true == 0)
        class_0_mask_pred = (y_pred == 0)
        per_class_metrics['no_fault'] = {
            'precision': float(precision_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'recall': float(recall_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'f1': float(f1_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'support': int(class_0_mask_true.sum())
        }
        for sensor_idx in range(len(sensor_names)):
            class_label = sensor_idx + 1
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
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

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

    if sensor_names:
        per_sensor = {}
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
    valid_fault_types = [ft for ft in fault_types if ft is not None and ft != '']
    unique_fault_types = list(set(valid_fault_types)) if valid_fault_types else []

    per_fault_metrics = {}
    for fault_type in unique_fault_types:
        mask = fault_types == fault_type
        if mask.sum() == 0:
            continue

        fault_y_true = y_true[mask]
        fault_y_pred = y_pred[mask]
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
    y_true_sensor = y_true_sensor.astype(int)
    y_pred_sensor = y_pred_sensor.astype(int)

    if y_true_window is not None and y_pred_window is not None:
        y_true_window = y_true_window.astype(int)
        y_pred_window = y_pred_window.astype(int)
        unique_true_labels = sorted([int(l) for l in set(y_true_window)])
        all_possible_labels = list(range(9))
        labels_to_use = [l for l in all_possible_labels if l in unique_true_labels]

        if len(labels_to_use) > 0:
            labels_to_use = [int(l) for l in labels_to_use]
            window_cm = confusion_matrix(y_true_window, y_pred_window, labels=labels_to_use)
        else:
            if len(unique_true_labels) > 0:
                unique_true_labels = [int(l) for l in unique_true_labels]
                window_cm = confusion_matrix(y_true_window, y_pred_window, labels=unique_true_labels)
            else:
                window_cm = confusion_matrix(y_true_window, y_pred_window)
    else:
        window_true = (y_true_sensor.sum(axis=1) > 0).astype(int)
        window_pred = (y_pred_sensor.sum(axis=1) > 0).astype(int)
        window_cm = confusion_matrix(window_true, window_pred)

    sensor_true_flat = y_true_sensor.flatten()
    sensor_pred_flat = y_pred_sensor.flatten()
    sensor_cm = confusion_matrix(sensor_true_flat, sensor_pred_flat)

    matrices = {
        'window_confusion_matrix': window_cm.tolist(),
        'sensor_confusion_matrix': sensor_cm.tolist()
    }

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
    metrics = {}
    metrics['window_level'] = compute_window_level_metrics(
        y_true_window, y_pred_window, sensor_names
    )
    metrics['sensor_level'] = compute_sensor_level_metrics(
        y_true_sensor, y_pred_sensor, sensor_names
    )
    metrics['confusion_matrices'] = compute_confusion_matrices(
        y_true_sensor, y_pred_sensor, y_true_window, y_pred_window, sensor_names
    )

    if fault_types is not None and len(fault_types) > 0:
        try:
            metrics['per_fault_type'] = compute_per_fault_type_metrics(
                y_true_sensor, y_pred_sensor, fault_types, sensor_names
            )
        except Exception:
            pass

    return metrics


def format_metrics_report(metrics: Dict[str, any]) -> str:
    lines = []
    lines.append("="*80)
    lines.append("EVALUATION METRICS REPORT")
    lines.append("="*80)
    lines.append("")

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
        if 'per_class' in wl:
            lines.append("\n  Per-Class Metrics:")
            for class_name, class_metrics in wl['per_class'].items():
                if class_metrics['support'] > 0:
                    lines.append(f"    {class_name}:")
                    lines.append(f"      Precision: {class_metrics['precision']:.4f}")
                    lines.append(f"      Recall:    {class_metrics['recall']:.4f}")
                    lines.append(f"      F1:        {class_metrics['f1']:.4f}")
                    lines.append(f"      Support:   {class_metrics['support']}")
        lines.append("")

    if 'sensor_level' in metrics:
        lines.append("Sensor-Level Metrics:")
        lines.append("-" * 40)
        sl = metrics['sensor_level']
        lines.append(f"  Accuracy:  {sl['sensor_accuracy']:.4f}")
        lines.append(f"  Precision: {sl['sensor_precision']:.4f}")
        lines.append(f"  Recall:    {sl['sensor_recall']:.4f}")
        lines.append(f"  F1 Score:  {sl['sensor_f1']:.4f}")
        lines.append("")
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


def compute_embedding_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances_to_normal: np.ndarray,
    distances_to_anomalous: np.ndarray
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    normal_mask = y_true == 0
    anomalous_mask = y_true == 1

    if normal_mask.any() and anomalous_mask.any():
        mean_dist_normal = float(np.mean(distances_to_normal[normal_mask]))
        mean_dist_anomalous = float(np.mean(distances_to_anomalous[anomalous_mask]))
        distance_separation = mean_dist_anomalous - mean_dist_normal
    else:
        distance_separation = 0.0
        mean_dist_normal = 0.0
        mean_dist_anomalous = 0.0

    try:
        if len(np.unique(y_true)) > 1:
            distance_auc = float(roc_auc_score(y_true, -distances_to_normal))
        else:
            distance_auc = 0.0
    except Exception:
        distance_auc = 0.0

    confidence_scores = 1.0 / (1.0 + np.exp(distances_to_normal - distances_to_anomalous))
    correctness = (y_true == y_pred).astype(float)

    try:
        confidence_calibration = float(np.corrcoef(confidence_scores, correctness)[0, 1])
        if np.isnan(confidence_calibration):
            confidence_calibration = 0.0
    except Exception:
        confidence_calibration = 0.0

    return {
        'distance_separation': float(distance_separation),
        'mean_dist_normal': float(mean_dist_normal),
        'mean_dist_anomalous': float(mean_dist_anomalous),
        'distance_auc': float(distance_auc),
        'confidence_calibration': float(confidence_calibration)
    }


def analyze_embedding_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    centers: np.ndarray
) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)

    normal_center = centers[0]
    anomalous_center = centers[1]

    fp_distances_normal = []
    fp_distances_anomalous = []
    fn_distances_normal = []
    fn_distances_anomalous = []

    if false_positives.any():
        fp_embeddings = embeddings[false_positives]
        fp_distances_normal = [float(np.linalg.norm(emb - normal_center)) for emb in fp_embeddings]
        fp_distances_anomalous = [float(np.linalg.norm(emb - anomalous_center)) for emb in fp_embeddings]

    if false_negatives.any():
        fn_embeddings = embeddings[false_negatives]
        fn_distances_normal = [float(np.linalg.norm(emb - normal_center)) for emb in fn_embeddings]
        fn_distances_anomalous = [float(np.linalg.norm(emb - anomalous_center)) for emb in fn_embeddings]

    fp_closer_to_normal = sum(1 for d_n, d_a in zip(fp_distances_normal, fp_distances_anomalous) if d_n < d_a)
    fp_closer_to_anomalous = len(fp_distances_normal) - fp_closer_to_normal
    fn_closer_to_normal = sum(1 for d_n, d_a in zip(fn_distances_normal, fn_distances_anomalous) if d_n < d_a)
    fn_closer_to_anomalous = len(fn_distances_normal) - fn_closer_to_normal

    return {
        'num_false_positives': int(false_positives.sum()),
        'num_false_negatives': int(false_negatives.sum()),
        'fp_mean_dist_normal': float(np.mean(fp_distances_normal)) if fp_distances_normal else 0.0,
        'fp_mean_dist_anomalous': float(np.mean(fp_distances_anomalous)) if fp_distances_anomalous else 0.0,
        'fn_mean_dist_normal': float(np.mean(fn_distances_normal)) if fn_distances_normal else 0.0,
        'fn_mean_dist_anomalous': float(np.mean(fn_distances_anomalous)) if fn_distances_anomalous else 0.0,
        'fp_closer_to_normal': int(fp_closer_to_normal),
        'fp_closer_to_anomalous': int(fp_closer_to_anomalous),
        'fn_closer_to_normal': int(fn_closer_to_normal),
        'fn_closer_to_anomalous': int(fn_closer_to_anomalous),
        'error_analysis': {
            'false_positives': {
                'closer_to_normal': fp_closer_to_normal > fp_closer_to_anomalous,
                'mean_dist_normal': float(np.mean(fp_distances_normal)) if fp_distances_normal else 0.0,
                'mean_dist_anomalous': float(np.mean(fp_distances_anomalous)) if fp_distances_anomalous else 0.0
            },
            'false_negatives': {
                'closer_to_normal': fn_closer_to_normal > fn_closer_to_anomalous,
                'mean_dist_normal': float(np.mean(fn_distances_normal)) if fn_distances_normal else 0.0,
                'mean_dist_anomalous': float(np.mean(fn_distances_anomalous)) if fn_distances_anomalous else 0.0
            }
        }
    }


def plot_distance_distributions(
    distances_normal_class: np.ndarray,
    distances_anomalous_class: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        distances_normal_class,
        bins=30,
        alpha=0.6,
        color='blue',
        label=f'Normal Windows (N={len(distances_normal_class)})',
        edgecolor='darkblue'
    )
    ax.hist(
        distances_anomalous_class,
        bins=30,
        alpha=0.6,
        color='red',
        label=f'Anomalous Windows (N={len(distances_anomalous_class)})',
        edgecolor='darkred'
    )

    mean_normal = np.mean(distances_normal_class) if len(distances_normal_class) > 0 else 0.0
    mean_anomalous = np.mean(distances_anomalous_class) if len(distances_anomalous_class) > 0 else 0.0

    ax.axvline(mean_normal, color='blue', linestyle='--', linewidth=2, label=f'Mean Normal: {mean_normal:.3f}')
    ax.axvline(mean_anomalous, color='red', linestyle='--', linewidth=2, label=f'Mean Anomalous: {mean_anomalous:.3f}')

    ax.set_xlabel('Distance to Class Center', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Distances to Class Centers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved distance distribution plot to {save_path}")

    return fig
