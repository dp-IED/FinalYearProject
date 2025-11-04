#!/usr/bin/env python3
"""
Training script for Autoencoder anomaly detection model.
Uses reconstruction error for anomaly detection.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add paths for imports
deep_learning_path = os.path.join(os.path.dirname(__file__), '..')
if deep_learning_path not in sys.path:
    sys.path.insert(0, deep_learning_path)
    
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ml_path = os.path.join(project_root, 'anomaly-detection', 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)

from models.autoencoder import AutoencoderModel
from utils.data_loader import SequenceDataLoader, CarOBDMLDataLoader
from utils.evaluation import evaluate_deep_learning_model, optimize_threshold


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in train_loader:
        sequences = batch.to(device)  # (batch, seq_len, features)
        
        optimizer.zero_grad()
        
        # Forward pass: reconstruct input
        reconstructed = model(sequences)
        
        # Reconstruction error (MSE)
        loss = criterion(reconstructed, sequences)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def validate_epoch(model, val_loader, device):
    """Validate and get reconstruction errors for threshold optimization."""
    model.eval()
    all_errors = []
    all_labels = []
    
    with torch.no_grad():
        for batch, labels in val_loader:
            sequences = batch.to(device)
            labels = labels.to(device)
            
            # Get reconstruction errors
            errors = model.get_reconstruction_error(sequences)
            
            all_errors.append(errors.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    
    return all_errors, all_labels


def train_autoencoder(
    data_path: str = "data/carOBD/obdiidata",
    model_path: str = "anomaly-detection/models/deep_learning/autoencoder",
    sequence_length: int = 30,
    encoder_dims: list = [64, 32, 16],
    dropout: float = 0.2,
    activation: str = 'relu',
    batch_size: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    threshold_method: str = 'percentile',
    threshold_percentile: int = 5,
    use_auto_detection: bool = True
):
    """Train Autoencoder model for anomaly detection."""
    
    print("=" * 70)
    print("Autoencoder Anomaly Detection Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Encoder dimensions: {encoder_dims}")
    print(f"  Activation: {activation}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Auto-detection: {use_auto_detection}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    train_dataset, val_dataset, test_dataset = seq_loader.load_data_from_ml_loader(
        ml_loader,
        data_path=data_path,
        validation_split=validation_split,
        test_split=test_split,
        fault_percentage=0.2 if use_auto_detection else 0.0
    )
    
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
    
    model = AutoencoderModel(
        input_size=input_size,
        sequence_length=sequence_length,
        encoder_dims=encoder_dims,
        dropout=dropout,
        activation=activation
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
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_errors, val_labels = validate_epoch(model, val_loader, device)
        val_loss = np.mean(val_errors)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    model = AutoencoderModel.load_model(model_path, device=device)
    
    # Auto-detection: Optimize threshold from validation set
    if use_auto_detection:
        print("\n" + "=" * 50)
        print("AUTO-DETECTING THRESHOLD")
        print("=" * 50)
        
        val_errors, val_labels = validate_epoch(model, val_loader, device)
        
        # Reconstruction errors are already anomaly scores (higher = more anomalous)
        threshold, threshold_metrics = optimize_threshold(
            val_errors,
            val_labels,
            method=threshold_method,
            percentile=threshold_percentile
        )
        
        print(f"Auto-detected threshold: {threshold:.4f} (method: {threshold_method})")
        print(f"Validation F1: {threshold_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)
    
    test_results = evaluate_deep_learning_model(
        model,
        test_dataset.sequences.numpy(),
        test_dataset.labels.numpy(),
        threshold=threshold if use_auto_detection else None,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        device=device
    )
    
    metrics = test_results['metrics']
    print(f"\nTest Results:")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Model saved to: {model_path}.pth")
    
    return model, test_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Autoencoder anomaly detection model')
    parser.add_argument('--data-path', type=str, default='data/carOBD/obdiidata',
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str,
                       default='anomaly-detection/models/deep_learning/autoencoder',
                       help='Path to save model')
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='Sequence length')
    parser.add_argument('--encoder-dims', type=int, nargs='+', default=[64, 32, 16],
                       help='Encoder layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'tanh'],
                       help='Activation function')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--threshold-method', type=str, default='percentile',
                       choices=['percentile', 'iqr', 'std', 'f1_optimal'],
                       help='Threshold optimization method')
    parser.add_argument('--threshold-percentile', type=int, default=5,
                       help='Percentile for percentile method')
    parser.add_argument('--no-auto-detection', action='store_true',
                       help='Disable auto-detection')
    
    args = parser.parse_args()
    
    train_autoencoder(
        data_path=args.data_path,
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        encoder_dims=args.encoder_dims,
        dropout=args.dropout,
        activation=args.activation,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        use_auto_detection=not args.no_auto_detection
    )


if __name__ == "__main__":
    main()

