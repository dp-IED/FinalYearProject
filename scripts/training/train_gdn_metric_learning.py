#!/usr/bin/env python3
"""
Train MetricLearningGDN with Proxy-NCA loss as primary objective.

Key differences from classification-focused training:
1. PRIMARY: Proxy-NCA loss (metric learning) - optimizes embedding separation
2. AUXILIARY: Classification loss (small weight) - for regularization
3. Larger embedding dimensions (128-256) for better separation
4. Focus on separation ratio and embedding quality, not classification accuracy

Usage:
    python train_gdn_metric_learning.py --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_metric_learning.pt
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from pathlib import Path
import json

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

from models.gdn_model import MetricLearningGDN, ProxyNCALoss
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE
)


def train_metric_learning_gdn(
    train_loader, val_loader, test_loader,
    num_sensors, window_size,
    num_epochs=150,
    device="cpu",
    lambda_classification=0.1,  # Small weight for classification (auxiliary)
    learning_rate=0.001,
    weight_decay=1e-5,
    model_save_path="anomaly-detection/best_multilabel_gdn_metric_learning.pt",
    embed_dim=128,
    hidden_dim=128,
    proxy_alpha=32.0,
):
    """
    Train MetricLearningGDN with Proxy-NCA loss as primary objective.
    
    Args:
        train_loader, val_loader, test_loader: DataLoaders
        num_sensors: Number of sensors
        window_size: Window size
        num_epochs: Number of training epochs
        device: Device to train on
        lambda_classification: Weight for classification loss (auxiliary)
        learning_rate: Learning rate
        weight_decay: Weight decay
        model_save_path: Path to save best model
        embed_dim: Embedding dimension (larger for better separation)
        hidden_dim: Hidden dimension
        proxy_alpha: Temperature parameter for Proxy-NCA
    
    Returns:
        model: Trained model
        proxy_loss: Proxy-NCA loss module (contains learned proxies)
        history: Training history
    """
    # Initialize model
    model = MetricLearningGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=3,
        hidden_dim=hidden_dim,
    ).to(device)
    
    # Initialize Proxy-NCA loss (learnable proxies)
    proxy_loss = ProxyNCALoss(
        embed_dim=hidden_dim,  # Use hidden_dim for proxy dimension
        num_classes=2,
        alpha=proxy_alpha,
    ).to(device)
    
    # Classification loss (auxiliary, small weight)
    sensor_criterion = nn.BCEWithLogitsLoss()
    global_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer (includes model params + proxy parameters)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proxy_loss.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_metric_loss': [],
        'train_class_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': [],
        'separation_ratio': [],
        'proxy_separation': [],
        'learning_rates': []
    }
    
    best_val_f1 = 0.0
    best_separation = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    print(f"\n{'='*70}")
    print("Training Metric Learning GDN (Proxy-NCA Primary)")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Model: MetricLearningGDN")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Proxy-NCA alpha: {proxy_alpha}")
    print(f"  Lambda classification (auxiliary): {lambda_classification}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  Device: {device}\n")
    
    for epoch in range(num_epochs):
        model.train()
        proxy_loss.train()
        
        train_loss_metric = 0.0
        train_loss_class = 0.0
        train_loss_total = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
            for X_batch, _, sensor_labels_batch, window_labels_batch in pbar:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - get embeddings and logits
                sensor_logits, global_logits, embeddings = model.forward_logits(X_batch)
                
                # PRIMARY: Proxy-NCA loss (metric learning)
                metric_loss = proxy_loss(embeddings, window_labels_batch)
                
                # AUXILIARY: Classification loss (small weight)
                class_loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                class_loss_global = global_criterion(global_logits, window_labels_batch.float())
                class_loss = class_loss_sensor + class_loss_global
                
                # Combined loss: metric learning primary, classification auxiliary
                loss = metric_loss + lambda_classification * class_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(proxy_loss.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss_metric += metric_loss.item() * X_batch.size(0)
                train_loss_class += class_loss.item() * X_batch.size(0)
                train_loss_total += loss.item() * X_batch.size(0)
        
        train_loss_metric /= len(train_loader.dataset)
        train_loss_class /= len(train_loader.dataset)
        train_loss_total /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        proxy_loss.eval()
        
        val_loss = 0.0
        all_val_probs = []
        all_val_labels = []
        all_val_distances = []
        all_val_window_labels = []
        all_val_embeddings = []
        
        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)
                
                # Forward pass
                sensor_logits, global_logits, embeddings = model.forward_logits(X_batch)
                
                # Compute losses
                metric_loss = proxy_loss(embeddings, window_labels_batch)
                class_loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                class_loss_global = global_criterion(global_logits, window_labels_batch.float())
                class_loss = class_loss_sensor + class_loss_global
                batch_loss = metric_loss + lambda_classification * class_loss
                
                val_loss += batch_loss.item() * X_batch.size(0)
                
                # Get probabilities for metrics
                sensor_probs = torch.sigmoid(sensor_logits)
                global_probs = torch.sigmoid(global_logits)
                
                all_val_probs.append(global_probs.cpu().numpy())
                all_val_labels.append(window_labels_batch.cpu().numpy())
                all_val_embeddings.append(embeddings.cpu().numpy())
                all_val_window_labels.append(window_labels_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        
        # Compute metrics
        val_probs = np.concatenate(all_val_probs)
        val_labels = np.concatenate(all_val_labels)
        val_embeddings = np.concatenate(all_val_embeddings)
        val_window_labels = np.concatenate(all_val_window_labels)
        
        # Classification metrics
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        
        # Separation metrics (using proxy centers)
        proxy_separation = proxy_loss.get_proxy_separation()
        
        # Compute separation ratio using embeddings
        normal_mask = val_window_labels == 0
        anomalous_mask = val_window_labels == 1
        
        # Initialize defaults
        normal_dist = 0.0
        anomalous_dist = 0.0
        separation_ratio = 1.0
        
        if normal_mask.any() and anomalous_mask.any():
            # Compute distances to proxy centers
            proxies_norm = F.normalize(proxy_loss.proxies, p=2, dim=1).detach().cpu().numpy()
            normal_proxy = proxies_norm[0]
            anomalous_proxy = proxies_norm[1]
            
            normal_emb = val_embeddings[normal_mask]
            anomalous_emb = val_embeddings[anomalous_mask]
            
            # Distance to normal proxy
            normal_dist = np.linalg.norm(normal_emb - normal_proxy, axis=1).mean()
            anomalous_dist = np.linalg.norm(anomalous_emb - normal_proxy, axis=1).mean()
            
            separation_ratio = anomalous_dist / (normal_dist + 1e-8)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss_total)
        history['train_metric_loss'].append(train_loss_metric)
        history['train_class_loss'].append(train_loss_class)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['separation_ratio'].append(separation_ratio)
        history['proxy_separation'].append(proxy_separation)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss_total:.4f} (Metric: {train_loss_metric:.4f}, Class: {train_loss_class:.4f}) | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Sep Ratio: {separation_ratio:.2f}× | "
              f"Proxy Sep: {proxy_separation:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model (based on separation ratio, not F1)
        if separation_ratio > best_separation:
            best_separation = separation_ratio
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "proxy_state_dict": proxy_loss.state_dict(),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_sensors": num_sensors,
                "window_size": window_size,
                "best_val_f1": val_f1,
                "best_separation_ratio": separation_ratio,
                "proxy_separation": proxy_separation,
                "final_separation_ratio": separation_ratio,
                "normal_mean_distance": normal_dist if normal_mask.any() else 0.0,
                "anomalous_mean_distance": anomalous_dist if anomalous_mask.any() else 0.0,
            }, model_save_path)
            print(f"  ✓ New best model saved (Sep Ratio: {separation_ratio:.2f}×)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best checkpoint
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    proxy_loss.load_state_dict(checkpoint["proxy_state_dict"])
    
    print(f"\n✓ Training complete. Best model saved to: {model_save_path}")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Best Separation Ratio: {checkpoint['best_separation_ratio']:.2f}×")
    print(f"  Proxy Separation: {checkpoint['proxy_separation']:.4f}")
    
    return model, proxy_loss, history


def main():
    parser = argparse.ArgumentParser(
        description="Train MetricLearningGDN with Proxy-NCA loss"
    )
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to carOBD data directory")
    parser.add_argument("--output", type=str,
                       default="anomaly-detection/best_multilabel_gdn_metric_learning.pt",
                       help="Output model path")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lambda-class", type=float, default=0.1,
                       help="Classification loss weight (auxiliary)")
    parser.add_argument("--proxy-alpha", type=float, default=32.0,
                       help="Proxy-NCA temperature parameter")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cpu/cuda/mps)")
    parser.add_argument("--fault-percentage", type=float, default=0.15,
                       help="Fault injection percentage")
    
    args = parser.parse_args()
    
    # Device detection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_path = Path(args.data_path)
    
    # Load all CSV files
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    
    print(f"  Found {len(csv_files)} CSV files")
    
    # Load and combine data
    dfs = []
    for csv_file in csv_files[:50]:  # Limit for faster processing
        try:
            df = pd.read_csv(csv_file)
            # Add ID_COL if missing (for compatibility)
            if ID_COL not in df.columns:
                df[ID_COL] = csv_file.stem  # Use filename as ID
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined dataset shape: {combined_df.shape}")
    
    # Preprocessing
    print("2. Preprocessing data...")
    df_clean = remove_zero_variance_columns(combined_df)
    df_clean = mean_fill_missing_timestamps_and_remove_duplicates(df_clean, TIME_COL, id_cols=[ID_COL])
    df_clean = downsample(df_clean, TIME_COL, ID_COL, downsample_factor=2)
    df_clean = filter_long_drives(df_clean, ID_COL, min_length=WINDOW_SIZE)
    df_clean = add_cross_channel_features(df_clean)
    
    # Build windows
    print("3. Building windows...")
    windows, y_targets, scaler = build_clean_windows(
        df_clean, 
        SENSOR_COLS,
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        scaler=None
    )
    print(f"  Created {len(windows)} windows")
    print(f"  Windows shape: {windows.shape}")
    
    # Convert to numpy for sensor_names
    sensor_names = SENSOR_COLS
    
    # Inject faults
    print("4. Injecting faults...")
    windows_with_faults, _, sensor_labels, window_labels = inject_faults_with_sensor_labels(
        windows, y_targets, sensor_names, fault_percentage=args.fault_percentage, random_state=42
    )
    
    num_sensors = len(sensor_names)
    num_windows = len(windows_with_faults)
    
    print(f"  Total windows: {num_windows}")
    print(f"  Normal windows: {(window_labels == 0).sum()}")
    print(f"  Faulty windows: {(window_labels == 1).sum()}")
    
    # Split data (STRATIFIED to ensure each split has faulty windows)
    print("5. Splitting data (stratified)...")
    np.random.seed(42)
    
    # Get indices for normal and faulty windows
    normal_indices = np.where(window_labels == 0)[0]
    faulty_indices = np.where(window_labels == 1)[0]
    
    # Shuffle each group
    np.random.shuffle(normal_indices)
    np.random.shuffle(faulty_indices)
    
    # Split each group proportionally
    train_size_normal = int(0.7 * len(normal_indices))
    val_size_normal = int(0.15 * len(normal_indices))
    
    train_size_faulty = int(0.7 * len(faulty_indices))
    val_size_faulty = int(0.15 * len(faulty_indices))
    
    # Combine indices
    train_idx = np.concatenate([
        normal_indices[:train_size_normal],
        faulty_indices[:train_size_faulty]
    ])
    val_idx = np.concatenate([
        normal_indices[train_size_normal:train_size_normal + val_size_normal],
        faulty_indices[train_size_faulty:train_size_faulty + val_size_faulty]
    ])
    test_idx = np.concatenate([
        normal_indices[train_size_normal + val_size_normal:],
        faulty_indices[train_size_faulty + val_size_faulty:]
    ])
    
    # Shuffle each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    X_train = windows_with_faults[train_idx]
    y_train = window_labels[train_idx]
    sensor_labels_train = sensor_labels[train_idx]
    
    X_val = windows_with_faults[val_idx]
    y_val = window_labels[val_idx]
    sensor_labels_val = sensor_labels[val_idx]
    
    X_test = windows_with_faults[test_idx]
    y_test = window_labels[test_idx]
    sensor_labels_test = sensor_labels[test_idx]
    
    print(f"  Train: {len(X_train)} windows ({(y_train == 1).sum()} faulty)")
    print(f"  Val: {len(X_val)} windows ({(y_val == 1).sum()} faulty)")
    print(f"  Test: {len(X_test)} windows ({(y_test == 1).sum()} faulty)")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train),
        torch.FloatTensor(y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train),
        torch.FloatTensor(sensor_labels_train.numpy() if isinstance(sensor_labels_train, torch.Tensor) else sensor_labels_train),
        torch.FloatTensor(y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val),
        torch.FloatTensor(y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val),
        torch.FloatTensor(sensor_labels_val.numpy() if isinstance(sensor_labels_val, torch.Tensor) else sensor_labels_val),
        torch.FloatTensor(y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test),
        torch.FloatTensor(y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test),
        torch.FloatTensor(sensor_labels_test.numpy() if isinstance(sensor_labels_test, torch.Tensor) else sensor_labels_test),
        torch.FloatTensor(y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Train model
    print("\n6. Training model...")
    model, proxy_loss, history = train_metric_learning_gdn(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        num_epochs=args.epochs,
        device=device,
        lambda_classification=args.lambda_class,
        learning_rate=args.lr,
        model_save_path=args.output,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        proxy_alpha=args.proxy_alpha,
    )
    
    print("\n✓ Training complete!")
    print(f"  Model saved to: {args.output}")
    print(f"  Best separation ratio: {max(history['separation_ratio']):.2f}×")
    print(f"  Best proxy separation: {max(history['proxy_separation']):.4f}")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
