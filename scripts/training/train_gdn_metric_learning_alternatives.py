#!/usr/bin/env python3
"""
Train MetricLearningGDN with alternative metric learning losses:
- Triplet Loss
- Contrastive Loss
- Fixed Proxy-NCA Loss (with squared Euclidean distance)

Usage:
    python train_gdn_metric_learning_alternatives.py --loss triplet --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_triplet.pt
    python train_gdn_metric_learning_alternatives.py --loss contrastive --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_contrastive.pt
    python train_gdn_metric_learning_alternatives.py --loss proxynca --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_proxynca_fixed.pt
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import json

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

from models.gdn_model import MetricLearningGDN, ProxyNCALoss, TripletLoss, ContrastiveLoss
import torch.nn.functional as F
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    WINDOW_SIZE
)


def train_metric_learning_gdn(
    train_loader, val_loader, test_loader,
    num_sensors, window_size,
    loss_type="triplet",  # "triplet", "contrastive", "proxynca"
    num_epochs=150,
    device="cpu",
    lambda_classification=0.1,
    learning_rate=0.001,
    weight_decay=1e-5,
    model_save_path="anomaly-detection/best_multilabel_gdn_triplet.pt",
    embed_dim=128,
    hidden_dim=128,
    proxy_alpha=32.0,
    triplet_margin=1.0,
    contrastive_margin=1.0,
):
    """
    Train MetricLearningGDN with specified metric learning loss.
    """
    # Initialize model
    model = MetricLearningGDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        top_k=3,
        num_gat_layers=2,
        num_temporal_layers=2,
        bidirectional_temporal=True,
        dropout=0.2,
    ).to(device)
    
    # Initialize metric learning loss
    if loss_type == "proxynca":
        metric_loss = ProxyNCALoss(embed_dim=embed_dim, num_classes=2, alpha=proxy_alpha).to(device)
    elif loss_type == "triplet":
        metric_loss = TripletLoss(margin=triplet_margin).to(device)
    elif loss_type == "contrastive":
        metric_loss = ContrastiveLoss(margin=contrastive_margin).to(device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Classification losses (auxiliary)
    sensor_criterion = nn.BCEWithLogitsLoss()
    global_criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(metric_loss.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early stopping
    best_sep_ratio = 0.0
    best_proxy_sep = 0.0
    patience_counter = 0
    patience = 20
    
    print(f"\n{'='*70}")
    print(f"Training Metric Learning GDN ({loss_type.upper()} Loss)")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Model: MetricLearningGDN")
    print(f"  Loss: {loss_type.upper()}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    if loss_type == "proxynca":
        print(f"  Proxy-NCA alpha: {proxy_alpha}")
    elif loss_type == "triplet":
        print(f"  Triplet margin: {triplet_margin}")
    elif loss_type == "contrastive":
        print(f"  Contrastive margin: {contrastive_margin}")
    print(f"  Lambda classification (auxiliary): {lambda_classification}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stop patience: {patience}")
    print(f"  Device: {device}\n")
    
    for epoch in range(num_epochs):
        model.train()
        metric_loss.train()
        
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
                
                # PRIMARY: Metric learning loss
                metric_loss_val = metric_loss(embeddings, window_labels_batch)
                
                # AUXILIARY: Classification loss (small weight)
                class_loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                class_loss_global = global_criterion(global_logits, window_labels_batch.float())
                class_loss = class_loss_sensor + class_loss_global
                
                # Combined loss: metric learning primary, classification auxiliary
                loss = metric_loss_val + lambda_classification * class_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if hasattr(metric_loss, 'parameters'):
                    torch.nn.utils.clip_grad_norm_(metric_loss.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss_metric += metric_loss_val.item() * X_batch.size(0)
                train_loss_class += class_loss.item() * X_batch.size(0)
                train_loss_total += loss.item() * X_batch.size(0)
        
        train_loss_metric /= len(train_loader.dataset)
        train_loss_class /= len(train_loader.dataset)
        train_loss_total /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        metric_loss.eval()
        
        val_loss_total = 0.0
        val_embeddings = []
        val_labels = []
        val_global_logits = []
        
        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)
                
                sensor_logits, global_logits, embeddings = model.forward_logits(X_batch)
                
                metric_loss_val = metric_loss(embeddings, window_labels_batch)
                class_loss_sensor = sensor_criterion(sensor_logits, sensor_labels_batch)
                class_loss_global = global_criterion(global_logits, window_labels_batch.float())
                class_loss = class_loss_sensor + class_loss_global
                loss = metric_loss_val + lambda_classification * class_loss
                
                val_loss_total += loss.item() * X_batch.size(0)
                val_embeddings.append(embeddings.cpu())
                val_labels.append(window_labels_batch.cpu())
                val_global_logits.append(global_logits.cpu())
        
        val_loss_total /= len(val_loader.dataset)
        val_embeddings = torch.cat(val_embeddings, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        val_global_logits = torch.cat(val_global_logits, dim=0)
        
        # Compute separation metrics
        normal_emb = val_embeddings[val_labels == 0]
        anomalous_emb = val_embeddings[val_labels == 1]
        
        if len(normal_emb) > 0 and len(anomalous_emb) > 0:
            normal_center = normal_emb.mean(axis=0)
            normal_dists = np.linalg.norm(normal_emb - normal_center, axis=1)
            anomalous_dists = np.linalg.norm(anomalous_emb - normal_center, axis=1)
            
            sep_ratio = anomalous_dists.mean() / (normal_dists.mean() + 1e-8)
            
            # Proxy separation (for Proxy-NCA)
            proxy_sep = 0.0
            if loss_type == "proxynca":
                proxies_norm = F.normalize(metric_loss.proxies, p=2, dim=1).detach().cpu().numpy()
                proxy_sep = np.linalg.norm(proxies_norm[0] - proxies_norm[1])
        else:
            sep_ratio = 1.0
            proxy_sep = 0.0
        
        # Classification metrics
        val_pred_probs = torch.sigmoid(val_global_logits).numpy()
        val_f1 = f1_score(val_labels, (val_pred_probs > 0.5).astype(int), zero_division=0)
        val_auc = roc_auc_score(val_labels, val_pred_probs) if len(np.unique(val_labels)) > 1 else 0.5
        
        # Learning rate scheduling
        scheduler.step(val_loss_total)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_total:.4f} (Metric: {train_loss_metric:.4f}, Class: {train_loss_class:.4f}) | "
              f"Val Loss: {val_loss_total:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | "
              f"Sep Ratio: {sep_ratio:.2f}× | Proxy Sep: {proxy_sep:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping and model saving
        if sep_ratio > best_sep_ratio:
            best_sep_ratio = sep_ratio
            best_proxy_sep = proxy_sep
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "metric_loss_state_dict": metric_loss.state_dict() if hasattr(metric_loss, 'state_dict') else {},
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_sep_ratio": best_sep_ratio,
                "best_proxy_sep": best_proxy_sep,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "loss_type": loss_type,
            }
            torch.save(checkpoint, model_save_path)
            print(f"  ✓ New best model saved (Sep Ratio: {sep_ratio:.2f}×)")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete. Best model saved to: {model_save_path}")
    print(f"  Best Val F1: {val_f1:.4f}")
    print(f"  Best Separation Ratio: {best_sep_ratio:.2f}×")
    if loss_type == "proxynca":
        print(f"  Proxy Separation: {best_proxy_sep:.4f}")
    
    return model, metric_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MetricLearningGDN with alternative losses")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output", type=str, required=True, help="Output model path")
    parser.add_argument("--loss", type=str, choices=["triplet", "contrastive", "proxynca"], default="triplet",
                       help="Metric learning loss type")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lambda-class", type=float, default=0.1, help="Classification loss weight")
    parser.add_argument("--proxy-alpha", type=float, default=32.0, help="Proxy-NCA alpha (only for proxynca)")
    parser.add_argument("--triplet-margin", type=float, default=1.0, help="Triplet loss margin (only for triplet)")
    parser.add_argument("--contrastive-margin", type=float, default=1.0, help="Contrastive loss margin (only for contrastive)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    # Load and preprocess data (same as train_gdn_metric_learning.py)
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
            if "ID_COL" not in df.columns:
                df["ID_COL"] = csv_file.stem
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined dataset: {len(df_combined)} rows")
    
    # Preprocess
    from train_gdn_center_loss import (
        SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE,
        remove_zero_variance_columns,
        mean_fill_missing_timestamps_and_remove_duplicates,
        downsample,
        filter_long_drives,
        add_cross_channel_features,
        build_clean_windows,
        inject_faults_with_sensor_labels,
    )
    
    df_clean = remove_zero_variance_columns(df_combined)
    df_clean = mean_fill_missing_timestamps_and_remove_duplicates(df_clean, TIME_COL, id_cols=[ID_COL])
    df_clean = downsample(df_clean, TIME_COL)
    df_clean = filter_long_drives(df_clean, ID_COL, TIME_COL)
    df_clean = add_cross_channel_features(df_clean, SENSOR_COLS)
    
    windows, y_targets, scaler = build_clean_windows(df_clean, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE)
    sensor_names = SENSOR_COLS
    
    windows_with_faults, _, sensor_labels, window_labels = inject_faults_with_sensor_labels(
        windows, y_targets, sensor_names, fault_percentage=0.15, random_state=42
    )
    
    # Convert to tensors
    X = torch.FloatTensor(windows_with_faults)
    y_sensor = torch.FloatTensor(sensor_labels)
    y_window = torch.FloatTensor(window_labels)
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp, y_sensor_train, y_sensor_temp = train_test_split(
        X, y_window, y_sensor, test_size=0.3, random_state=42, stratify=y_window
    )
    X_val, X_test, y_val, y_test, y_sensor_val, y_sensor_test = train_test_split(
        X_temp, y_temp, y_sensor_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create datasets (format: X, y_sensor, sensor_labels, window_labels)
    train_dataset = TensorDataset(X_train, y_train, y_sensor_train, y_train)
    val_dataset = TensorDataset(X_val, y_val, y_sensor_val, y_val)
    test_dataset = TensorDataset(X_test, y_test, y_sensor_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"  Train: {len(X_train)} windows ({y_train.sum().int().item()} faulty)")
    print(f"  Val: {len(X_val)} windows ({y_val.sum().int().item()} faulty)")
    print(f"  Test: {len(X_test)} windows ({y_test.sum().int().item()} faulty)")
    
    # Train
    model, metric_loss = train_metric_learning_gdn(
        train_loader, val_loader, test_loader,
        num_sensors=len(sensor_names),
        window_size=WINDOW_SIZE,
        loss_type=args.loss,
        num_epochs=args.epochs,
        device=device,
        lambda_classification=args.lambda_class,
        learning_rate=args.lr,
        model_save_path=args.output,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        proxy_alpha=args.proxy_alpha,
        triplet_margin=args.triplet_margin,
        contrastive_margin=args.contrastive_margin,
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {args.output}")
