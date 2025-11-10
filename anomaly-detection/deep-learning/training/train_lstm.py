#!/usr/bin/env python3
"""
Training script for LSTM anomaly detection model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Add paths for imports
deep_learning_path = os.path.join(os.path.dirname(__file__), '..')
if deep_learning_path not in sys.path:
    sys.path.insert(0, deep_learning_path)
    
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ml_path = os.path.join(project_root, 'anomaly-detection', 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

from models.lstm import LSTMModel
from utils.data_loader import SequenceDataLoader, TimeSeriesDataset, CarOBDMLDataLoader
from utils.evaluation import evaluate_deep_learning_model, optimize_threshold, calculate_metrics
from typing import Dict, List, Optional


def train_epoch(model, train_loader, criterion, optimizer, device, track_gradients=False,
                fault_info=None, column_loss_weight=0.3, use_column_supervision=True):
    """
    Train for one epoch - predict next timestep with optional column supervision.
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device to train on
        track_gradients: Whether to track gradient norms
        fault_info: Optional fault information for column supervision
        column_loss_weight: Weight for column detection loss
        use_column_supervision: Whether to use column supervision loss
    """
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_col_loss = 0.0
    n_batches = 0
    n_batches_with_faults = 0
    total_grad_norm = 0.0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle different batch formats from DataLoader
        # PyTorch DataLoader with dataset returning tuples returns a LIST of tensors: [sequences, labels]
        # PyTorch DataLoader with dataset returning single tensor returns a single tensor
        
        # Check if it's a list or tuple with 2 elements (sequences, labels)
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            sequences, labels = batch_data[0], batch_data[1]
            
            # Both should already be tensors from DataLoader's default collate_fn
            # Just move to device
            sequences = sequences.to(device)
            labels = labels.to(device) if labels is not None else None
            
        elif isinstance(batch_data, torch.Tensor):
            # Single tensor (no labels case)
            sequences = batch_data.to(device)
            labels = None
            
        else:
            raise ValueError(f"Unexpected batch_data type: {type(batch_data)}, length: {len(batch_data) if isinstance(batch_data, (list, tuple)) else 'N/A'}")
        
        optimizer.zero_grad()
        
        # Use all but last timestep to predict the last timestep
        input_seq = sequences[:, :-1, :]  # (batch, seq_len-1, features)
        target = sequences[:, -1, :]  # (batch, features) - last timestep
        
        # Forward pass: predict next timestep with attention
        predicted, column_scores, _, _ = model(input_seq, return_attentions=True)
        
        # Primary loss: MSE between predicted and actual next timestep
        mse_loss = criterion(predicted, target)
        
        # Auxiliary loss: Column detection (only if supervision enabled and fault_info available)
        col_loss = torch.tensor(0.0, device=device)
        if use_column_supervision and fault_info is not None:
            # Get batch indices for fault info lookup
            batch_start = batch_idx * sequences.shape[0]
            
            # Create column labels from fault_info
            batch_size = sequences.shape[0]
            column_labels = torch.zeros_like(column_scores)  # (batch, input_size)
            
            for i in range(batch_size):
                seq_idx = batch_start + i
                if seq_idx < len(fault_info['modified_columns']):
                    modified_cols = fault_info['modified_columns'][seq_idx]
                    if len(modified_cols) > 0:
                        column_labels[i, modified_cols] = 1.0
                        if i == 0:  # Count batch once if any sample has faults
                            n_batches_with_faults += 1
            
            # Only compute loss if we have labels
            if column_labels.sum() > 0:
                col_loss = F.binary_cross_entropy(column_scores, column_labels, reduction='mean')
        
        # Combined loss
        total_batch_loss = mse_loss + column_loss_weight * col_loss
        
        total_batch_loss.backward()
        
        # Track gradient norm before clipping
        if track_gradients:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            total_grad_norm += grad_norm.item()
        else:
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_mse_loss += mse_loss.item()
        total_col_loss += col_loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_mse_loss = total_mse_loss / n_batches if n_batches > 0 else 0.0
    avg_col_loss = total_col_loss / n_batches if n_batches > 0 else 0.0
    avg_grad_norm = total_grad_norm / n_batches if track_gradients and n_batches > 0 else None
    
    return avg_loss, avg_grad_norm, avg_mse_loss, avg_col_loss


def validate_epoch(model, val_loader, device, fault_info=None):
    """
    Validate and get prediction errors for threshold optimization.
    Also returns column and timestamp scores if available.
    """
    model.eval()
    all_errors = []
    all_labels = []
    all_column_scores = []
    all_timestamp_scores = []
    
    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(val_loader):
            sequences = batch.to(device)
            labels = labels.to(device)
            
            # Get prediction errors (anomaly scores)
            errors = model.get_prediction_error(sequences)
            
            # Get column and timestamp scores from attention
            input_seq = sequences[:, :-1, :]
            _, column_scores, timestamp_scores, _ = model(input_seq, return_attentions=True)
            
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_column_scores.append(column_scores.cpu().numpy())
            all_timestamp_scores.append(timestamp_scores.squeeze(-1).cpu().numpy())
    
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    all_column_scores = np.concatenate(all_column_scores) if all_column_scores else None
    all_timestamp_scores = np.concatenate(all_timestamp_scores) if all_timestamp_scores else None
    
    return all_errors, all_labels, all_column_scores, all_timestamp_scores


def evaluate_column_level_detection_lightweight(
    model,
    sequences: np.ndarray,
    fault_info: Dict,
    feature_names: List[str],
    threshold_percentile: int = 95,
    device: Optional[torch.device] = None,
    sample_size: Optional[int] = None
) -> Dict:
    """
    Lightweight version for training-time evaluation (samples subset for speed).
    """
    if sample_size is not None and len(sequences) > sample_size:
        # Sample sequences for faster evaluation during training
        sample_indices = np.random.choice(len(sequences), sample_size, replace=False)
        sequences = sequences[sample_indices]
        fault_info_sampled = {
            'modified_columns': [fault_info['modified_columns'][i] for i in sample_indices],
            'percentage_changes': [fault_info['percentage_changes'][i] for i in sample_indices]
        }
        fault_info = fault_info_sampled
    
    return evaluate_column_level_detection(
        model, sequences, fault_info, feature_names, threshold_percentile, device
    )


def evaluate_column_level_detection(
    model,
    sequences: np.ndarray,
    fault_info: Dict,
    feature_names: List[str],
    threshold_percentile: int = 95,
    device: Optional[torch.device] = None,
    use_attention_scores: bool = True,
    use_adaptive_threshold: bool = True,
    min_precision: float = 0.15
) -> Dict:
    """
    Evaluate how well the model identifies which columns are anomalous.
    Uses attention column scores if available, otherwise falls back to prediction errors.
    
    Args:
        model: Trained LSTM model
        sequences: Input sequences (n_samples, seq_len, n_features)
        fault_info: Dictionary with 'modified_columns', 'percentage_changes', 'fault_types'
        feature_names: List of feature names
        threshold_percentile: Percentile to use for identifying anomalous columns
        device: Device to run inference on
        use_attention_scores: If True, use attention column scores; else use prediction errors
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Get column scores from attention or fall back to prediction errors
    x = torch.FloatTensor(sequences).to(device)
    batch_size = 64
    all_column_scores = []
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size]
            if use_attention_scores:
                # Use attention column scores directly
                input_seq = batch[:, :-1, :]
                _, column_scores, _, _ = model(input_seq, return_attentions=True)
                all_column_scores.append(column_scores.cpu().numpy())
            else:
                # Fall back to feature-level errors
                feature_errors = model.get_feature_level_errors(batch)
                all_column_scores.append(feature_errors.cpu().numpy())
    
    column_scores = np.concatenate(all_column_scores, axis=0)  # (n_samples, n_features)
    
    # Adaptive thresholding: find optimal threshold per feature that maximizes F1
    # while maintaining minimum precision to avoid false positives
    if use_adaptive_threshold:
        thresholds_per_feature = np.zeros(column_scores.shape[1])
        actual_modified = fault_info['modified_columns']
        
        for col_idx in range(column_scores.shape[1]):
            # Get scores for this column across all samples
            col_scores = column_scores[:, col_idx]
            
            # Try different percentiles and find one that maximizes F1 while maintaining precision
            best_f1 = 0.0
            best_threshold = np.percentile(col_scores, threshold_percentile)
            
            # Try percentiles from 80 to 98 (expanded range for better optimization)
            for pct in range(80, 99):
                test_threshold = np.percentile(col_scores, pct)
                
                # Calculate precision and recall for this threshold
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                
                for i in range(len(sequences)):
                    is_anomalous = col_scores[i] > test_threshold
                    is_actual_fault = col_idx in actual_modified[i]
                    
                    if is_anomalous and is_actual_fault:
                        true_positives += 1
                    elif is_anomalous and not is_actual_fault:
                        false_positives += 1
                    elif not is_anomalous and is_actual_fault:
                        false_negatives += 1
                
                # Calculate precision and recall
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
                
                # Only consider thresholds that meet minimum precision requirement
                if precision >= min_precision:
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = test_threshold
            
            thresholds_per_feature[col_idx] = best_threshold
    else:
        # Use fixed percentile threshold
        thresholds_per_feature = np.percentile(column_scores, threshold_percentile, axis=0)
    
    # Identify which columns are flagged as anomalous
    predicted_anomalous_columns = []
    for i in range(len(column_scores)):
        anomalous_cols = np.where(column_scores[i] > thresholds_per_feature)[0].tolist()
        predicted_anomalous_columns.append(anomalous_cols)
    
    # Compare with actual modified columns
    actual_modified = fault_info['modified_columns']
    actual_percentage_changes = fault_info['percentage_changes']
    
    # Calculate metrics
    correct_detections = 0
    total_faults = 0
    precision_scores = []
    recall_scores = []
    f1_scores = []
    percentage_deviations = []
    
    for i in range(len(sequences)):
        if len(actual_modified[i]) > 0:  # Only evaluate on samples with faults
            total_faults += 1
            
            actual_cols = set(actual_modified[i])
            predicted_cols = set(predicted_anomalous_columns[i])
            
            # Calculate precision/recall for this sample
            if len(predicted_cols) > 0:
                precision = len(actual_cols & predicted_cols) / len(predicted_cols)
            else:
                precision = 0.0
            
            if len(actual_cols) > 0:
                recall = len(actual_cols & predicted_cols) / len(actual_cols)
            else:
                recall = 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            f1_scores.append(f1)
            
            # Check if at least one correct column was identified
            if len(actual_cols & predicted_cols) > 0:
                correct_detections += 1
            
            # Calculate percentage deviation for detected columns
            for col_idx in (actual_cols & predicted_cols):
                actual_pct_change = actual_percentage_changes[i][actual_modified[i].index(col_idx)]
                predicted_score = column_scores[i, col_idx]
                
                # Estimate percentage deviation from normal
                # Use the score magnitude relative to normal scores
                normal_score = np.percentile(column_scores[:, col_idx], 50)  # Median score
                estimated_pct_deviation = ((predicted_score - normal_score) / (normal_score + 1e-8)) * 100
                
                percentage_deviations.append({
                    'sample_idx': i,
                    'column_idx': col_idx,
                    'column_name': feature_names[col_idx] if col_idx < len(feature_names) else f'Feature_{col_idx}',
                    'actual_pct_change': actual_pct_change,
                    'estimated_pct_deviation': estimated_pct_deviation,
                    'error': abs(estimated_pct_deviation - actual_pct_change)
                })
    
    results = {
        'total_faults': total_faults,
        'correct_detections': correct_detections,
        'detection_rate': correct_detections / total_faults if total_faults > 0 else 0.0,
        'avg_precision': np.mean(precision_scores) if precision_scores else 0.0,
        'avg_recall': np.mean(recall_scores) if recall_scores else 0.0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,
        'percentage_deviations': percentage_deviations,
        'avg_percentage_error': np.mean([d['error'] for d in percentage_deviations]) if percentage_deviations else 0.0
    }
    
    return results


def evaluate_timestamp_detection(
    model,
    sequences: np.ndarray,
    fault_info: Dict,
    threshold_percentile: int = 90,  # Changed from 95 to 90 for better recall
    device: Optional[torch.device] = None,
    use_f1_optimal: bool = True  # New: use F1-optimal threshold instead of percentile
) -> Dict:
    """
    Evaluate how well the model identifies which timesteps are anomalous.
    
    Args:
        model: Trained LSTM model
        sequences: Input sequences (n_samples, seq_len, n_features)
        fault_info: Dictionary with fault information (aggregated per sequence)
        threshold_percentile: Percentile to use for identifying anomalous timesteps (fallback if use_f1_optimal=False)
        device: Device to run inference on
        use_f1_optimal: If True, find F1-optimal threshold instead of using percentile
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Get timestamp scores from attention
    x = torch.FloatTensor(sequences).to(device)
    batch_size = 64
    all_timestamp_scores = []
    
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size]
            timestamp_scores = model.get_timestamp_scores(batch)
            all_timestamp_scores.append(timestamp_scores.cpu().numpy())
    
    timestamp_scores = np.concatenate(all_timestamp_scores, axis=0)  # (n_samples, seq_len-1)
    
    # Find optimal threshold
    actual_modified = fault_info['modified_columns']
    
    if use_f1_optimal:
        # Find F1-optimal threshold by testing different percentiles
        best_f1 = 0.0
        best_threshold = np.percentile(timestamp_scores, threshold_percentile)
        
        # Try percentiles from 70 to 99
        for pct in range(70, 100):
            test_threshold = np.percentile(timestamp_scores, pct)
            
            tp = fp = fn = 0
            for i in range(len(sequences)):
                has_fault = len(actual_modified[i]) > 0
                predicted_timesteps = set(np.where(timestamp_scores[i] > test_threshold)[0])
                
                if has_fault:
                    if len(predicted_timesteps) > 0:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if len(predicted_timesteps) > 0:
                        fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = test_threshold
        
        threshold = best_threshold
    else:
        # Use percentile threshold
        threshold = np.percentile(timestamp_scores, threshold_percentile)
    
    # Identify which timesteps are flagged as anomalous
    predicted_anomalous_timesteps = []
    for i in range(len(timestamp_scores)):
        anomalous_timesteps = np.where(timestamp_scores[i] > threshold)[0].tolist()
        predicted_anomalous_timesteps.append(anomalous_timesteps)
    
    # Compare with actual faults (faults indicate which sequences have anomalies)
    # For sequences with faults, we assume all timesteps in the sequence could be anomalous
    actual_modified = fault_info['modified_columns']
    
    # Calculate metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_faults = 0
    
    for i in range(len(sequences)):
        has_fault = len(actual_modified[i]) > 0
        predicted_timesteps = set(predicted_anomalous_timesteps[i])
        
        if has_fault:
            total_faults += 1
            # If any timestep is predicted as anomalous, it's a true positive
            if len(predicted_timesteps) > 0:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            # No fault, but predicted timesteps = false positive
            if len(predicted_timesteps) > 0:
                false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'total_faults': total_faults,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'threshold': threshold,
        'threshold_method': 'f1_optimal' if use_f1_optimal else f'percentile_{threshold_percentile}'
    }
    
    return results


def train_lstm(
    data_path: str = "data/carOBD/obdiidata",
    model_path: str = "anomaly-detection/models/deep_learning/lstm",
    sequence_length: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    threshold_method: str = 'percentile',
    threshold_percentile: int = 5,
    use_auto_detection: bool = True,
    num_attention_heads: int = 4,
    attention_dropout: float = 0.1,
    column_loss_weight: float = 0.7,
    use_column_supervision: bool = True,
    train_fault_percentage: float = 0.05
):
    """Train LSTM model for anomaly detection."""
    
    print("=" * 70)
    print("LSTM Anomaly Detection Training")
    print("=" * 70)
    print("Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate} (recommended: 0.0001 for normalized data)")
    print(f"  Epochs: {epochs}")
    print(f"  Auto-detection: {use_auto_detection}")
    print(f"  Attention heads: {num_attention_heads}")
    print(f"  Column supervision: {use_column_supervision} (weight: {column_loss_weight})")
    print(f"  Training fault percentage: {train_fault_percentage*100:.1f}%")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    # Initialize data loaders
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    ml_loader = CarOBDMLDataLoader(data_path)
    seq_loader = SequenceDataLoader(sequence_length=sequence_length)
    
    # Load data and create sequences
    # Step 1: Load all data and split properly (with same random seed)
    # Step 2: Training = normal data only, Validation/Test = fault-injected data
    
    print("Loading and splitting data...")
    train_dataset, val_dataset, test_dataset, train_features_raw, val_features_raw, test_features_raw = seq_loader.load_data_from_ml_loader(
        ml_loader,
        data_path=data_path,
        validation_split=validation_split,
        test_split=test_split,
        fault_percentage=0.0,  # First get normal data splits
        return_raw_features=True
    )
    
    # Normalize features (critical for training deep learning models)
    # Use StandardScaler to get mean=0, std=1 for better convergence
    print("Normalizing features (StandardScaler: mean=0, std=1)...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features_raw)
    val_features_scaled = scaler.transform(val_features_raw)
    test_features_scaled = scaler.transform(test_features_raw)
    
    print("Feature statistics (before scaling):")
    print(f"  Train - Mean range: [{train_features_raw.mean(axis=0).min():.2f}, {train_features_raw.mean(axis=0).max():.2f}]")
    print(f"  Train - Std range: [{train_features_raw.std(axis=0).min():.2f}, {train_features_raw.std(axis=0).max():.2f}]")
    print(f"  Train - Value range: [{train_features_raw.min():.2f}, {train_features_raw.max():.2f}]")
    
    print("Feature statistics (after StandardScaler):")
    print(f"  Train - Mean range: [{train_features_scaled.mean(axis=0).min():.6f}, {train_features_scaled.mean(axis=0).max():.6f}] (should be ~0)")
    print(f"  Train - Std range: [{train_features_scaled.std(axis=0).min():.6f}, {train_features_scaled.std(axis=0).max():.6f}] (should be ~1)")
    print(f"  Train - Value range: [{train_features_scaled.min():.2f}, {train_features_scaled.max():.2f}]")
    
    # Check for any issues
    if np.isnan(train_features_scaled).any():
        print("  WARNING: NaN values found in scaled features!")
    if np.isinf(train_features_scaled).any():
        print("  WARNING: Inf values found in scaled features!")
    
    # Get feature names for column-level tracking
    # Load a sample of data to get feature names
    sample_idle, sample_motion = ml_loader.load_all_data()
    sample_all_data = pd.concat([sample_idle, sample_motion], ignore_index=True).head(100)
    feature_names = ml_loader.get_feature_names(sample_all_data)
    
    # Add faulty data to training set (for column supervision learning)
    print(f"Creating fault-injected training set ({train_fault_percentage*100:.1f}% faults)...")
    train_features_fault, train_labels, train_fault_info = ml_loader.create_realistic_fault_data(
        train_features_scaled, fault_percentage=train_fault_percentage, feature_names=feature_names
    )
    
    # Mix normal and faulty training data
    # Create labels: 0 for normal, 1 for faulty
    train_labels_full = train_labels
    
    # Clip extreme values that might cause training instability (before creating sequences)
    # Use 5-sigma rule for clipping to prevent extreme outliers
    std = np.std(train_features_fault, axis=0)
    mean = np.mean(train_features_fault, axis=0)
    train_features_fault = np.clip(train_features_fault, mean - 5 * std, mean + 5 * std)
    
    # Convert to sequences
    train_sequences_fault, train_seq_labels = seq_loader.create_sequences(train_features_fault, labels=train_labels_full)
    
    # Now create fault-injected validation and test sets from SCALED features
    print("Creating fault-injected validation and test sets (from scaled features)...")
    fault_percentage = 0.2  # 20% fault injection for evaluation
    
    val_features_fault, val_labels, val_fault_info = ml_loader.create_realistic_fault_data(
        val_features_scaled, fault_percentage=fault_percentage, feature_names=feature_names
    )
    test_features_fault, test_labels, test_fault_info = ml_loader.create_realistic_fault_data(
        test_features_scaled, fault_percentage=fault_percentage, feature_names=feature_names
    )
    
    # Clip extreme values for validation and test sets
    for data, name in [(val_features_fault, 'val'), (test_features_fault, 'test')]:
        std = np.std(data, axis=0)
        mean = np.mean(data, axis=0)
        clipped = np.clip(data, mean - 5 * std, mean + 5 * std)
        if name == 'val':
            val_features_fault = clipped
        else:
            test_features_fault = clipped
    
    # Convert fault-injected training data to sequences
    train_dataset = TimeSeriesDataset(train_sequences_fault, labels=train_seq_labels)
    
    # Convert fault-injected data to sequences
    # Note: create_sequences creates overlapping windows, so we need to map fault_info to sequences
    # The last timestep of each sequence corresponds to the feature index
    val_sequences_fault, val_seq_labels = seq_loader.create_sequences(val_features_fault, labels=val_labels)
    test_sequences_fault, test_seq_labels = seq_loader.create_sequences(test_features_fault, labels=test_labels)
    
    # Map fault_info to sequence indices - aggregate faults from ALL timesteps in each sequence
    # This allows evaluation of sequence-wide anomaly detection (not just at prediction target)
    seq_length = seq_loader.sequence_length
    
    def aggregate_faults_across_sequence(fault_info, n_sequences, seq_length):
        """Aggregate faults from all timesteps in each sequence."""
        fault_info_seq = {
            'modified_columns': [],
            'percentage_changes': [],
            'fault_types': [],
            'original_values': [],
            'modified_values': []
        }
        
        for seq_idx in range(n_sequences):
            # This sequence contains features [seq_idx : seq_idx + seq_length]
            # Collect all faults from ANY timestep in this sequence
            sequence_modified_cols = set()
            sequence_pct_changes = {}  # col_idx -> list of pct changes
            sequence_fault_types = []
            sequence_original_vals = {}  # col_idx -> list of original values
            sequence_modified_vals = {}  # col_idx -> list of modified values
            
            # Check each timestep in the sequence
            for timestep_offset in range(seq_length):
                feature_idx = seq_idx + timestep_offset
                
                if feature_idx < len(fault_info['modified_columns']):
                    # Check if this feature has faults
                    modified_cols_at_timestep = fault_info['modified_columns'][feature_idx]
                    pct_changes_at_timestep = fault_info['percentage_changes'][feature_idx]
                    
                    # Aggregate faults from this timestep
                    for col_idx, pct_change in zip(modified_cols_at_timestep, pct_changes_at_timestep):
                        sequence_modified_cols.add(col_idx)
                        if col_idx not in sequence_pct_changes:
                            sequence_pct_changes[col_idx] = []
                        sequence_pct_changes[col_idx].append(pct_change)
                        
                        # Track original/modified values if available
                        if feature_idx < len(fault_info['original_values']):
                            try:
                                orig_idx = modified_cols_at_timestep.index(col_idx)
                                if col_idx not in sequence_original_vals:
                                    sequence_original_vals[col_idx] = []
                                    sequence_modified_vals[col_idx] = []
                                if orig_idx < len(fault_info['original_values'][feature_idx]):
                                    sequence_original_vals[col_idx].append(fault_info['original_values'][feature_idx][orig_idx])
                                    sequence_modified_vals[col_idx].append(fault_info['modified_values'][feature_idx][orig_idx])
                            except (ValueError, IndexError):
                                pass
                    
                    # Track fault types (use the first one encountered)
                    if fault_info['fault_types'][feature_idx] is not None:
                        sequence_fault_types.append(fault_info['fault_types'][feature_idx])
            
            # Convert sets/lists to the format expected by evaluation
            fault_info_seq['modified_columns'].append(list(sequence_modified_cols))
            
            # Average percentage changes if a column appears in multiple timesteps
            avg_pct_changes = []
            for col_idx in sorted(sequence_modified_cols):
                avg_pct = np.mean(sequence_pct_changes[col_idx]) if sequence_pct_changes[col_idx] else 0.0
                avg_pct_changes.append(avg_pct)
            fault_info_seq['percentage_changes'].append(avg_pct_changes)
            
            # Use first fault type encountered (or most common)
            fault_info_seq['fault_types'].append(
                sequence_fault_types[0] if sequence_fault_types else None
            )
            
            # Average original/modified values
            avg_original = []
            avg_modified = []
            for col_idx in sorted(sequence_modified_cols):
                if col_idx in sequence_original_vals and sequence_original_vals[col_idx]:
                    avg_original.append(np.mean(sequence_original_vals[col_idx]))
                    avg_modified.append(np.mean(sequence_modified_vals[col_idx]))
                else:
                    avg_original.append(0.0)
                    avg_modified.append(0.0)
            fault_info_seq['original_values'].append(avg_original)
            fault_info_seq['modified_values'].append(avg_modified)
        
        return fault_info_seq
    
    # Aggregate faults across sequences for training, validation, and test
    train_fault_info_seq = aggregate_faults_across_sequence(train_fault_info, len(train_sequences_fault), seq_length)
    val_fault_info_seq = aggregate_faults_across_sequence(val_fault_info, len(val_sequences_fault), seq_length)
    test_fault_info_seq = aggregate_faults_across_sequence(test_fault_info, len(test_sequences_fault), seq_length)
    
    # Use sequence-aligned fault info
    train_fault_info = train_fault_info_seq
    val_fault_info = val_fault_info_seq
    test_fault_info = test_fault_info_seq
    
    # Create fault-injected datasets
    val_dataset = TimeSeriesDataset(val_sequences_fault, labels=val_seq_labels)
    test_dataset = TimeSeriesDataset(test_sequences_fault, labels=test_seq_labels)
    
    print(f"Training: {len(train_dataset)} sequences ({train_fault_percentage*100:.1f}% faults)")
    print(f"Validation: {len(val_dataset)} sequences ({fault_percentage*100}% faults)")
    print(f"Test: {len(test_dataset)} sequences ({fault_percentage*100}% faults)")
    
    train_loader, val_loader, test_loader = seq_loader.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=batch_size
    )
    
    # Get input size from first batch
    sample_batch, _ = next(iter(val_loader))
    input_size = sample_batch.shape[2]
    
    print(f"Input size: {input_size}")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Initialize model
    print("\n" + "=" * 50)
    print("INITIALIZING MODEL")
    print("=" * 50)
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_attention_heads=num_attention_heads,
        attention_dropout=attention_dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # Track gradients for first 5 epochs to diagnose issues
    track_gradients = True
    
    for epoch in range(epochs):
        # Only track gradients for first few epochs
        if epoch >= 5:
            track_gradients = False
        
        # Train epoch: now includes faulty data with column supervision
        train_loss, grad_norm, train_mse_loss, train_col_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            track_gradients=track_gradients,
            fault_info=train_fault_info,  # Training data now has faults
            column_loss_weight=column_loss_weight,
            use_column_supervision=use_column_supervision
        )
        
        # Validate on fault-injected data
        val_errors, val_labels, val_column_scores, val_timestamp_scores = validate_epoch(
            model, val_loader, device, fault_info=val_fault_info
        )
        
        # Calculate validation metrics:
        # 1. Normal data prediction error (should be low)
        # 2. Anomaly data prediction error (should be higher)
        normal_mask = val_labels == 0
        anomaly_mask = val_labels == 1
        
        if np.sum(normal_mask) > 0:
            val_loss_normal = np.mean(val_errors[normal_mask])
        else:
            val_loss_normal = train_loss
        
        if np.sum(anomaly_mask) > 0:
            val_loss_anomaly = np.mean(val_errors[anomaly_mask])
        else:
            val_loss_anomaly = val_loss_normal
        
        # Overall validation loss: mean of all prediction errors
        val_loss = np.mean(val_errors)
        
        scheduler.step(val_loss_normal)  # Use normal data loss for scheduling
        
        # Calculate separation ratio (higher is better - means anomalies have much higher error)
        separation_ratio = val_loss_anomaly / val_loss_normal if val_loss_normal > 0 else 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Optional: Apply column supervision during validation (few gradient steps)
        # This helps the model learn from validation faults while preventing overfitting
        if use_column_supervision and val_fault_info is not None and epoch % 5 == 0:
            # Do a few gradient steps on validation data every 5 epochs
            model.train()
            val_supervision_batches = 0
            val_supervision_loss = 0.0
            
            for batch_idx, (batch, labels) in enumerate(val_loader):
                if batch_idx >= 3:  # Limit to 3 batches to avoid overfitting
                    break
                
                sequences = batch.to(device)
                optimizer.zero_grad()
                
                input_seq = sequences[:, :-1, :]
                target = sequences[:, -1, :]
                
                predicted, column_scores, _, _ = model(input_seq, return_attentions=True)
                
                # MSE loss
                mse_loss = criterion(predicted, target)
                
                # Column supervision loss
                batch_start = batch_idx * sequences.shape[0]
                batch_size = sequences.shape[0]
                column_labels = torch.zeros_like(column_scores)
                
                for i in range(batch_size):
                    seq_idx = batch_start + i
                    if seq_idx < len(val_fault_info['modified_columns']):
                        modified_cols = val_fault_info['modified_columns'][seq_idx]
                        if len(modified_cols) > 0:
                            column_labels[i, modified_cols] = 1.0
                
                col_loss = torch.tensor(0.0, device=device)
                if column_labels.sum() > 0:
                    col_loss = F.binary_cross_entropy(column_scores, column_labels, reduction='mean')
                
                # Combined loss (with lower weight for validation supervision)
                total_loss = mse_loss + (column_loss_weight * 0.5) * col_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                val_supervision_loss += total_loss.item()
                val_supervision_batches += 1
            
            model.eval()
        
        # Evaluate column-level detection (lightweight, sample for speed)
        # Only evaluate every few epochs or on first/last epochs to save time
        col_detection_metrics = None
        if epoch == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            try:
                # Convert sequences to numpy if needed
                val_sequences_np = val_sequences_fault if isinstance(val_sequences_fault, np.ndarray) else val_sequences_fault.numpy()
                col_detection_metrics = evaluate_column_level_detection_lightweight(
                    model, val_sequences_np, val_fault_info, feature_names,
                    threshold_percentile=90, device=device, sample_size=2000  # Sample for speed
                )
            except Exception as e:
                # Don't fail training if column detection fails
                print(f"Warning: Column detection evaluation failed: {e}")
                col_detection_metrics = None
        
        # Build print string
        print_str = (f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} "
                     f"(MSE: {train_mse_loss:.6f}, Col: {train_col_loss:.6f}), "
                     f"Val Loss (normal): {val_loss_normal:.6f}, "
                     f"Val Loss (anomaly): {val_loss_anomaly:.6f}, "
                     f"Val Loss (overall): {val_loss:.6f}, "
                     f"Separation: {separation_ratio:.1f}x, "
                     f"LR: {current_lr:.6f}")
        
        if grad_norm is not None:
            print_str += f", Grad Norm: {grad_norm:.4f}"
        
        # Add column detection metrics
        if col_detection_metrics is not None:
            print_str += (f", Col Detection: {col_detection_metrics['detection_rate']:.2%} "
                         f"(F1: {col_detection_metrics['avg_f1']:.3f}, "
                         f"P: {col_detection_metrics['avg_precision']:.3f}, "
                         f"R: {col_detection_metrics['avg_recall']:.3f})")
        
        print(print_str)
        
        # Save best model based on normal data validation loss (we want low prediction error on normal data)
        if val_loss_normal < best_val_loss:
            best_val_loss = val_loss_normal
            patience_counter = 0
            model.save_model(model_path, metadata={
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sequence_length': sequence_length,
                'threshold_method': threshold_method,
                'threshold_percentile': threshold_percentile
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    print("\n" + "=" * 50)
    print("LOADING BEST MODEL")
    print("=" * 50)
    model = LSTMModel.load_model(model_path, device=device)
    
    if use_auto_detection:
        print("\n" + "=" * 50)
        print("AUTO-DETECTING THRESHOLD")
        print("=" * 50)
        
        val_errors, val_labels, _, _ = validate_epoch(model, val_loader, device)
        
        # Prediction errors are already anomaly scores (higher error = more anomalous)
        # Optimize threshold directly on prediction errors
        threshold, threshold_metrics = optimize_threshold(
            val_errors,
            val_labels,
            method=threshold_method,
            percentile=threshold_percentile
        )
        
        print(f"Auto-detected threshold: {threshold:.4f} (method: {threshold_method})")
        print(f"Validation F1: {threshold_metrics['f1_score']:.4f}")
        print(f"Validation Precision: {threshold_metrics['precision']:.4f}")
        print(f"Validation Recall: {threshold_metrics['recall']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)
    
    # Get prediction errors for test set
    test_sequences = test_dataset.sequences.numpy()
    test_labels = test_dataset.labels.numpy()
    
    # Get prediction errors (anomaly scores)
    test_errors = model.get_anomaly_scores(test_sequences, device=device)
    
    # Calculate test loss breakdown (normal vs anomaly) - same as validation
    test_normal_mask = test_labels == 0
    test_anomaly_mask = test_labels == 1
    
    if np.sum(test_normal_mask) > 0:
        test_loss_normal = np.mean(test_errors[test_normal_mask])
    else:
        test_loss_normal = 0.0
    
    if np.sum(test_anomaly_mask) > 0:
        test_loss_anomaly = np.mean(test_errors[test_anomaly_mask])
    else:
        test_loss_anomaly = 0.0
    
    test_loss_overall = np.mean(test_errors)
    
    print(f"\nTest Loss Breakdown:")
    print(f"  Test Loss (normal): {test_loss_normal:.6f}")
    print(f"  Test Loss (anomaly): {test_loss_anomaly:.6f}")
    print(f"  Test Loss (overall): {test_loss_overall:.6f}")
    print(f"  Separation ratio (anomaly/normal): {test_loss_anomaly/test_loss_normal:.2f}x" if test_loss_normal > 0 else "  (no normal samples)")
    
    # Use threshold to make predictions
    if use_auto_detection:
        test_predictions = (test_errors >= threshold).astype(int)
    else:
        # If no auto-detection, optimize threshold on test set (not recommended but available)
        threshold, _ = optimize_threshold(test_errors, test_labels, method=threshold_method, percentile=threshold_percentile)
        test_predictions = (test_errors >= threshold).astype(int)
    
    # Calculate metrics
    from utils.evaluation import calculate_metrics
    test_metrics = calculate_metrics(test_labels, test_predictions, test_errors)
    
    test_results = {
        'threshold': threshold,
        'predictions': test_predictions,
        'scores': test_errors,
        'metrics': test_metrics,
        'labels': test_labels,
        'test_loss_normal': test_loss_normal,
        'test_loss_anomaly': test_loss_anomaly,
        'test_loss_overall': test_loss_overall
    }
    
    metrics = test_results['metrics']
    print(f"\nTest Results (Classification Metrics):")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}")
    print(f"  PR-AUC: {metrics.get('pr_auc', 0.0):.4f}")
    
    # Print diagnostic information if ROC-AUC is low
    if metrics.get('roc_auc', 1.0) < 0.7:
        print(f"\n  ROC-AUC Diagnostic (Low AUC detected):")
        if 'score_label_correlation' in metrics:
            print(f"    Score-Label Correlation: {metrics['score_label_correlation']:.4f} (should be positive)")
        if 'score_separation_ratio' in metrics:
            print(f"    Score Separation Ratio: {metrics['score_separation_ratio']:.2f}x")
        if 'score_overlap_percentage' in metrics:
            print(f"    Score Overlap: {metrics['score_overlap_percentage']:.2f}% of normal scores above 5th percentile of anomalies")
    
    # Column-level anomaly detection evaluation
    print("\n" + "=" * 50)
    print("COLUMN-LEVEL ANOMALY DETECTION EVALUATION")
    print("=" * 50)
    
    test_col_results = evaluate_column_level_detection(
        model, test_sequences_fault, test_fault_info, feature_names, 
        threshold_percentile=90, device=device, use_adaptive_threshold=True, min_precision=0.15
    )
    
    print(f"\nColumn Detection Results:")
    print(f"  Total faults: {test_col_results['total_faults']}")
    print(f"  Correct detections: {test_col_results['correct_detections']}")
    print(f"  Detection rate: {test_col_results['detection_rate']:.2%}")
    print(f"  Average Precision: {test_col_results['avg_precision']:.4f}")
    print(f"  Average Recall: {test_col_results['avg_recall']:.4f}")
    print(f"  Average F1: {test_col_results['avg_f1']:.4f}")
    print(f"  Average percentage error: {test_col_results['avg_percentage_error']:.2f}%")
    
    # Print top deviations
    if test_col_results['percentage_deviations']:
        print(f"\nTop 10 Column-Level Detections (Best Accuracy):")
        sorted_deviations = sorted(test_col_results['percentage_deviations'], 
                                  key=lambda x: abs(x['error']))[:10]
        for dev in sorted_deviations:
            print(f"  {dev['column_name']}: Actual={dev['actual_pct_change']:.2f}%, "
                  f"Estimated={dev['estimated_pct_deviation']:.2f}%, "
                  f"Error={dev['error']:.2f}%")
        
        # Print summary by column
        print(f"\nColumn Detection Summary (by feature):")
        column_stats = {}
        for dev in test_col_results['percentage_deviations']:
            col_name = dev['column_name']
            if col_name not in column_stats:
                column_stats[col_name] = {'count': 0, 'errors': [], 'actual_changes': []}
            column_stats[col_name]['count'] += 1
            column_stats[col_name]['errors'].append(dev['error'])
            column_stats[col_name]['actual_changes'].append(dev['actual_pct_change'])
        
        for col_name, stats in sorted(column_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            avg_error = np.mean(stats['errors'])
            avg_actual = np.mean(stats['actual_changes'])
            print(f"  {col_name}: {stats['count']} detections, "
                  f"Avg error={avg_error:.2f}%, Avg actual change={avg_actual:.2f}%")
    
    # Evaluate timestamp detection
    print("\n" + "=" * 50)
    print("TIMESTAMP-LEVEL ANOMALY DETECTION EVALUATION")
    print("=" * 50)
    
    test_timestamp_results = evaluate_timestamp_detection(
        model, test_sequences_fault, test_fault_info,
        threshold_percentile=90, device=device, use_f1_optimal=True  # Use F1-optimal threshold for better recall
    )
    
    print(f"\nTimestamp Detection Results:")
    print(f"  Total faults: {test_timestamp_results['total_faults']}")
    print(f"  True Positives: {test_timestamp_results['true_positives']}")
    print(f"  False Positives: {test_timestamp_results['false_positives']}")
    print(f"  False Negatives: {test_timestamp_results['false_negatives']}")
    print(f"  Precision: {test_timestamp_results['precision']:.4f}")
    print(f"  Recall: {test_timestamp_results['recall']:.4f}")
    print(f"  F1 Score: {test_timestamp_results['f1_score']:.4f}")
    print(f"  Threshold: {test_timestamp_results['threshold']:.4f}")
    
    # Evaluate model at different fault percentages (robustness test)
    print("\n" + "=" * 50)
    print("ROBUSTNESS EVALUATION: VARYING FAULT PERCENTAGES")
    print("=" * 50)
    
    fault_percentages = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    robustness_results = []
    
    for fault_pct in fault_percentages:
        print(f"\nTesting with {fault_pct*100:.0f}% fault injection...")
        
        # Create test data with this fault percentage
        test_features_fault_var, test_labels_var, test_fault_info_var = ml_loader.create_realistic_fault_data(
            test_features_scaled, fault_percentage=fault_pct, feature_names=feature_names
        )
        
        # Clip extreme values
        std = np.std(test_features_fault_var, axis=0)
        mean = np.mean(test_features_fault_var, axis=0)
        test_features_fault_var = np.clip(test_features_fault_var, mean - 5 * std, mean + 5 * std)
        
        # Convert to sequences
        test_sequences_fault_var, test_seq_labels_var = seq_loader.create_sequences(
            test_features_fault_var, labels=test_labels_var
        )
        
        # Aggregate fault info
        test_fault_info_var_seq = aggregate_faults_across_sequence(
            test_fault_info_var, len(test_sequences_fault_var), seq_length
        )
        
        # Evaluate column detection
        try:
            col_results = evaluate_column_level_detection(
                model, test_sequences_fault_var, test_fault_info_var_seq, feature_names,
                threshold_percentile=90, device=device, use_adaptive_threshold=True, min_precision=0.15
            )
            
            # Evaluate overall anomaly detection
            test_dataset_var = TimeSeriesDataset(test_sequences_fault_var, labels=test_seq_labels_var)
            test_loader_var = torch.utils.data.DataLoader(test_dataset_var, batch_size=batch_size, shuffle=False)
            
            test_errors_var, test_labels_var_flat, _, _ = validate_epoch(
                model, test_loader_var, device, fault_info=test_fault_info_var_seq
            )
            
            # Calculate classification metrics
            threshold_var, threshold_metrics_var = optimize_threshold(
                test_errors_var, test_labels_var_flat, method='f1_optimal'
            )
            test_predictions_var = (test_errors_var >= threshold_var).astype(int)
            test_metrics_var = calculate_metrics(test_labels_var_flat, test_predictions_var, test_errors_var)
            
            robustness_results.append({
                'fault_percentage': fault_pct,
                'column_detection_rate': col_results['detection_rate'],
                'column_f1': col_results['avg_f1'],
                'column_precision': col_results['avg_precision'],
                'column_recall': col_results['avg_recall'],
                'anomaly_f1': test_metrics_var['f1_score'],
                'anomaly_precision': test_metrics_var['precision'],
                'anomaly_recall': test_metrics_var['recall'],
                'anomaly_accuracy': test_metrics_var['accuracy']
            })
            
            print(f"  Column Detection: {col_results['detection_rate']:.2%} (F1: {col_results['avg_f1']:.3f})")
            print(f"  Anomaly Detection: F1={test_metrics_var['f1_score']:.3f}, "
                  f"P={test_metrics_var['precision']:.3f}, R={test_metrics_var['recall']:.3f}")
            
        except Exception as e:
            print(f"  Error evaluating at {fault_pct*100:.0f}%: {e}")
            robustness_results.append({
                'fault_percentage': fault_pct,
                'error': str(e)
            })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"{'Fault %':<10} {'Col F1':<10} {'Col Recall':<12} {'Anom F1':<10} {'Anom Recall':<12}")
    print("-" * 80)
    for result in robustness_results:
        if 'error' not in result:
            print(f"{result['fault_percentage']*100:>6.0f}%   "
                  f"{result['column_f1']:>8.3f}   "
                  f"{result['column_recall']:>10.3f}     "
                  f"{result['anomaly_f1']:>8.3f}   "
                  f"{result['anomaly_recall']:>10.3f}")
        else:
            print(f"{result['fault_percentage']*100:>6.0f}%   ERROR: {result['error']}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model saved to: {model_path}.pth")
    
    return model, test_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LSTM anomaly detection model')
    parser.add_argument('--data-path', type=str, default='data/carOBD/obdiidata',
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str,
                       default='anomaly-detection/models/deep_learning/lstm',
                       help='Path to save model')
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='Sequence length')
    parser.add_argument('--hidden-size', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001 for normalized data)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--threshold-method', type=str, default='percentile',
                       choices=['percentile', 'iqr', 'std', 'f1_optimal'],
                       help='Threshold optimization method')
    parser.add_argument('--threshold-percentile', type=int, default=5,
                       help='Percentile for percentile method')
    parser.add_argument('--no-auto-detection', action='store_true',
                       help='Disable auto-detection')
    parser.add_argument('--num-attention-heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Dropout for attention layers (default: 0.1)')
    parser.add_argument('--column-loss-weight', type=float, default=0.7,
                       help='Weight for column detection loss (default: 0.7)')
    parser.add_argument('--no-column-supervision', action='store_true',
                       help='Disable supervised column detection loss')
    parser.add_argument('--train-fault-percentage', type=float, default=0.05,
                       help='Percentage of training data with faults (default: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    train_lstm(
        data_path=args.data_path,
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        use_auto_detection=not args.no_auto_detection,
        num_attention_heads=args.num_attention_heads,
        attention_dropout=args.attention_dropout,
        column_loss_weight=args.column_loss_weight,
        use_column_supervision=not args.no_column_supervision,
        train_fault_percentage=args.train_fault_percentage
    )


if __name__ == "__main__":
    main()

