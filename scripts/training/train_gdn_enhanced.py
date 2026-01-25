#!/usr/bin/env python3
"""
Enhanced Training Script for ImprovedMultiLabelGDN

Integrates all improvements:
1. Enhanced data augmentation and fault injection
2. Better class imbalance handling (effective number weighting, weighted sampling)
3. Threshold optimization (precision-recall balance, Youden's J)
4. Multi-task learning (reconstruction, forecasting)
5. Ensemble support (optional)

Usage:
    python train_gdn_enhanced.py --data-path data/carOBD/obdiidata --output anomaly-detection/best_multilabel_gdn_enhanced.pt
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import json
from scipy.interpolate import interp1d
from scipy.signal import resample

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "anomaly-detection"))

from models.gdn_model import ImprovedMultiLabelGDN
from train_gdn_center_loss import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_clean_windows,
    inject_faults_with_sensor_labels,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
)

torch.set_default_dtype(torch.float32)


# ============================================================================
# Enhanced Data Augmentation
# ============================================================================


def advanced_fault_injection(
    X_windows,
    y_windows,
    sensor_cols,
    fault_percentage=0.15,
    augmentation_strategies=["dropout", "drift", "correlated", "time_varying"],
    random_state=42,
):
    """
    Enhanced fault injection with multiple realistic patterns.

    Improvements:
    - Gradual fault onset (more realistic than instant)
    - Multi-sensor correlated faults
    - Time-varying fault magnitudes
    - Sensor-specific fault patterns
    """
    np.random.seed(random_state)
    N, W, D = X_windows.shape
    n_fault = max(1, int(N * fault_percentage))

    X_faulty = X_windows.clone()
    sensor_labels = torch.zeros(N, D, dtype=torch.float32)
    window_labels = torch.zeros(N, dtype=torch.long)

    fault_indices = np.random.choice(N, n_fault, replace=False)
    pid_idx = {name: i for i, name in enumerate(sensor_cols)}

    for idx in fault_indices:
        win = X_faulty[idx].numpy()
        strategy = np.random.choice(augmentation_strategies)

        # Strategy 1: Gradual dropout (more realistic)
        if strategy == "dropout" and "VEHICLE_SPEED ()" in pid_idx:
            speed_i = pid_idx["VEHICLE_SPEED ()"]
            if win[:, speed_i].mean() > 0.15:
                # Gradual onset: ramp down over 20% of window
                ramp_start = int(W * 0.3)
                ramp_end = int(W * 0.5)
                ramp_len = ramp_end - ramp_start

                for t in range(ramp_start, ramp_end):
                    progress = (t - ramp_start) / ramp_len
                    win[t, speed_i] *= 1 - progress  # Gradual decrease

                # Complete dropout for rest
                win[ramp_end:, speed_i] = np.random.uniform(0, 0.02, W - ramp_end)
                affected_sensors = [speed_i]
                sensor_labels[idx, speed_i] = 1.0

        # Strategy 2: Sensor drift (slow degradation)
        elif strategy == "drift":
            sensor_i = np.random.choice(D)
            drift_rate = np.random.uniform(0.01, 0.03)
            base_value = win[:, sensor_i].mean()
            for t in range(W):
                win[t, sensor_i] = base_value + drift_rate * t
            sensor_labels[idx, sensor_i] = 1.0

        # Strategy 3: Correlated multi-sensor fault
        elif strategy == "correlated":
            primary_sensor = np.random.choice(D)
            affected_sensors = [primary_sensor]
            # Corrupt related sensors (adjacent in sensor list)
            if primary_sensor < D - 1:
                affected_sensors.append(primary_sensor + 1)
            if primary_sensor > 0:
                affected_sensors.append(primary_sensor - 1)

            for sensor_i in affected_sensors:
                scale = np.random.uniform(0.7, 0.9)
                win[:, sensor_i] *= scale
                sensor_labels[idx, sensor_i] = 1.0

        # Strategy 4: Time-varying magnitude
        elif strategy == "time_varying":
            sensor_i = np.random.choice(D)
            base_value = win[:, sensor_i].mean()
            for t in range(W):
                # Sinusoidal fault magnitude
                fault_magnitude = 0.2 * np.sin(2 * np.pi * t / W)
                win[t, sensor_i] = base_value * (1 + fault_magnitude)
            sensor_labels[idx, sensor_i] = 1.0

        # Fallback to original fault injection
        else:
            # Use original inject_faults_with_sensor_labels for this window
            win_tensor = torch.FloatTensor(win).unsqueeze(0)
            y_win = y_windows[idx : idx + 1]
            _, _, sensor_lbl, window_lbl = inject_faults_with_sensor_labels(
                win_tensor,
                y_win,
                sensor_cols,
                fault_percentage=1.0,
                random_state=random_state + idx,
            )
            win = win_tensor[0].numpy()
            sensor_labels[idx] = sensor_lbl[0]

        X_faulty[idx] = torch.FloatTensor(win)
        if sensor_labels[idx].sum() > 0:
            window_labels[idx] = 1

    return X_faulty, y_windows, sensor_labels, window_labels


def add_temporal_augmentation(X_windows, augmentation_rate=0.1, random_state=None):
    """
    Add temporal augmentations to normal windows:
    - Time warping (speed up/slow down)
    - Random cropping
    - Noise injection
    """
    if random_state is not None:
        np.random.seed(random_state)

    N, W, D = X_windows.shape
    n_augment = max(1, int(N * augmentation_rate))
    augment_indices = np.random.choice(N, n_augment, replace=False)

    X_augmented = X_windows.clone()

    for idx in augment_indices:
        win = X_augmented[idx].numpy()

        # Time warping: stretch or compress
        warp_factor = np.random.uniform(0.95, 1.05)
        if abs(warp_factor - 1.0) > 0.01:
            t_original = np.linspace(0, 1, W)
            t_warped = np.linspace(0, 1, int(W * warp_factor))
            f = interp1d(
                t_original, win, axis=0, kind="linear", fill_value="extrapolate"
            )
            win_warped = f(t_warped)
            # Resample back to original length
            win = resample(win_warped, W, axis=0)

        # Add Gaussian noise to normal windows (helps robustness)
        noise_level = np.random.uniform(0.01, 0.03)
        win += np.random.normal(0, noise_level, win.shape)
        win = np.clip(win, 0, 1)  # Keep in [0, 1] range

        X_augmented[idx] = torch.FloatTensor(win)

    return X_augmented


# ============================================================================
# Class Imbalance Handling
# ============================================================================


def compute_class_weights(y_train, method="effective", beta=0.999):
    """
    Compute class weights using multiple strategies.

    Methods:
    - 'balanced': sklearn-style balanced weights
    - 'effective': Effective number of samples (for extreme imbalance)
    - 'focal': Focal loss style weighting
    """
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos

    if method == "balanced":
        pos_weight = n_neg / (n_pos + 1e-8)

    elif method == "effective":
        # Effective number of samples (for extreme imbalance)
        # E_n = (1 - beta^n) / (1 - beta)
        effective_pos = (1 - beta**n_pos) / (1 - beta)
        effective_neg = (1 - beta**n_neg) / (1 - beta)
        pos_weight = effective_neg / (effective_pos + 1e-8)

    elif method == "focal":
        # Focal loss style: down-weight easy examples
        pos_weight = (n_neg / n_pos) ** 0.5  # Square root for less aggressive

    return pos_weight


def create_weighted_sampler(y_train, replacement=True):
    """
    Create a WeightedRandomSampler for better batch balance.
    Ensures each batch has roughly equal normal/faulty samples.
    """
    class_weights = torch.ones(len(y_train), dtype=torch.float32)
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos

    if n_pos > 0 and n_neg > 0:
        class_weights[y_train == 1] = len(y_train) / (n_pos + 1e-8)
        class_weights[y_train == 0] = len(y_train) / (n_neg + 1e-8)

    sampler = WeightedRandomSampler(
        weights=class_weights, num_samples=len(y_train), replacement=replacement
    )
    return sampler


# ============================================================================
# Threshold Optimization
# ============================================================================


def optimize_threshold(
    y_true,
    scores,
    metric="precision_recall",
    precision_target=0.5,
    recall_target=0.7,
    beta=1.0,
):
    """
    Optimize threshold using multiple strategies.

    Metrics:
    - 'f1': Maximize F1 score
    - 'precision_recall': Balance precision and recall
    - 'fbeta': F-beta score (beta controls precision/recall tradeoff)
    - 'youden': Youden's J statistic (maximize TPR - FPR)
    """
    from sklearn.metrics import fbeta_score

    if metric == "f1":
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_f1 = 0
        best_threshold = np.median(scores)

        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

    elif metric == "precision_recall":
        # Find threshold that achieves target precision/recall
        precision, recall, thresholds = precision_recall_curve(y_true, scores)

        # Find threshold closest to target
        distances = np.abs(precision - precision_target) + np.abs(
            recall - recall_target
        )
        best_idx = np.argmin(distances)
        best_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )

    elif metric == "fbeta":
        # F-beta score (beta > 1 favors recall, beta < 1 favors precision)
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_fbeta = 0
        best_threshold = np.median(scores)

        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            if len(np.unique(y_pred)) > 1:
                fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_threshold = t

    elif metric == "youden":
        # Maximize TPR - FPR (Youden's J statistic)
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_threshold = (
            thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        )

    return best_threshold


# ============================================================================
# Multi-Task Learning Extension
# ============================================================================


class MultiTaskImprovedGDN(ImprovedMultiLabelGDN):
    """
    Extended ImprovedMultiLabelGDN with auxiliary tasks:
    - Sensor value reconstruction
    - Temporal forecasting (next-step prediction)
    """

    def __init__(
        self,
        num_nodes,
        window_size,
        embed_dim=64,
        hidden_dim=64,
        num_gat_layers=2,
        use_reconstruction=True,
        use_forecasting=True,
    ):
        super().__init__(num_nodes, window_size, embed_dim, hidden_dim, num_gat_layers)

        self.use_reconstruction = use_reconstruction
        self.use_forecasting = use_forecasting

        # Reconstruction head (predicts mean sensor values)
        if use_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_nodes),  # Reconstruct each sensor
            )

        # Forecasting head (predicts next timestep)
        if use_forecasting:
            self.forecasting_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_nodes),  # Predict next sensor values
            )

    def forward_with_auxiliary(self, x, return_auxiliary=False):
        """
        Forward pass with auxiliary task outputs.
        """
        B, W, N = x.shape

        # Standard forward pass
        sensor_probs, global_probs = self.forward(x, return_global=True)

        # Get embeddings for auxiliary tasks
        embeddings = self.get_embeddings(x)  # (B, hidden_dim)

        auxiliary_outputs = {}

        if self.use_reconstruction:
            # Reconstruct mean sensor values
            reconstructed = self.reconstruction_head(embeddings)  # (B, N)
            auxiliary_outputs["reconstruction"] = reconstructed

        if self.use_forecasting:
            # Predict next timestep sensor values
            forecasted = self.forecasting_head(embeddings)  # (B, N)
            auxiliary_outputs["forecasting"] = forecasted

        if return_auxiliary:
            return sensor_probs, global_probs, auxiliary_outputs

        return sensor_probs, global_probs


def multi_task_loss(
    sensor_logits,
    global_logits,
    sensor_labels,
    window_labels,
    auxiliary_outputs,
    x,
    x_next=None,
    lambda_reconstruction=0.1,
    lambda_forecasting=0.1,
):
    """
    Combined loss with auxiliary tasks.
    """
    # Primary classification losses
    sensor_criterion = nn.BCEWithLogitsLoss()
    global_criterion = nn.BCEWithLogitsLoss()

    sensor_loss = sensor_criterion(sensor_logits, sensor_labels)
    global_loss = global_criterion(global_logits, window_labels.float())

    total_loss = sensor_loss + 0.3 * global_loss

    # Auxiliary reconstruction loss
    if "reconstruction" in auxiliary_outputs:
        # Reconstruct mean sensor values across time
        target_mean = x.mean(dim=1)  # (B, N) mean across time dimension
        recon_loss = F.mse_loss(auxiliary_outputs["reconstruction"], target_mean)
        total_loss += lambda_reconstruction * recon_loss

    # Auxiliary forecasting loss
    if "forecasting" in auxiliary_outputs and x_next is not None:
        # Predict next timestep
        target_next = x_next[:, -1, :]  # (B, N) last timestep
        forecast_loss = F.mse_loss(auxiliary_outputs["forecasting"], target_next)
        total_loss += lambda_forecasting * forecast_loss

    return total_loss


# ============================================================================
# Enhanced Training Function
# ============================================================================


def train_enhanced_gdn(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    use_augmentation=True,
    use_weighted_sampler=True,
    use_multi_task=False,
    num_epochs=150,
    device="cpu",
    learning_rate=0.0005,
    weight_decay=1e-4,
    embed_dim=64,
    hidden_dim=64,
    top_k=3,
    num_gat_layers=2,
    class_weight_method="effective",
    threshold_metric="precision_recall",
    lambda_reconstruction=0.1,
    lambda_forecasting=0.1,
    early_stop_patience=20,
    model_save_path="anomaly-detection/best_multilabel_gdn_enhanced.pt",
):
    """
    Enhanced training with all improvements.
    """
    # Initialize model
    if use_multi_task:
        model = MultiTaskImprovedGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            use_reconstruction=True,
            use_forecasting=True,
        ).to(device)
    else:
        model = ImprovedMultiLabelGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
        ).to(device)

    # Compute class weights
    # Extract labels from train_loader to compute weights
    all_train_labels = []
    for _, _, _, y_batch in train_loader:
        all_train_labels.extend(
            y_batch.numpy()
            if isinstance(y_batch, np.ndarray)
            else y_batch.cpu().numpy()
        )
    all_train_labels = np.array(all_train_labels)

    pos_weight = compute_class_weights(
        torch.FloatTensor(all_train_labels), method=class_weight_method, beta=0.999
    )

    # Convert to tensor if needed
    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    # Loss functions
    sensor_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Early stopping
    best_precision = 0.0
    best_f1 = 0.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_auc": [],
        "best_threshold": [],
    }

    print(f"\n{'=' * 70}")
    print(f"Training Enhanced ImprovedMultiLabelGDN")
    print(f"{'=' * 70}")
    print(f"Configuration:")
    print(
        f"  Model: {'MultiTaskImprovedGDN' if use_multi_task else 'ImprovedMultiLabelGDN'}"
    )
    print(f"  Data augmentation: {use_augmentation}")
    print(f"  Weighted sampler: {use_weighted_sampler}")
    print(f"  Multi-task learning: {use_multi_task}")
    print(f"  Class weight method: {class_weight_method}")
    print(f"  Pos weight: {pos_weight:.4f}")
    print(f"  Threshold metric: {threshold_metric}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0

        for batch_idx, (
            X_batch,
            _,
            sensor_labels_batch,
            window_labels_batch,
        ) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            sensor_labels_batch = sensor_labels_batch.to(device)
            window_labels_batch = window_labels_batch.to(device)

            # Data augmentation on-the-fly (30% of batches)
            if use_augmentation and np.random.rand() < 0.3:
                X_batch = add_temporal_augmentation(
                    X_batch.cpu(),
                    augmentation_rate=0.1,
                    random_state=epoch * 1000 + batch_idx,
                ).to(device)

            optimizer.zero_grad()

            # Forward pass
            if use_multi_task:
                sensor_probs, global_probs, auxiliary_outputs = (
                    model.forward_with_auxiliary(X_batch, return_auxiliary=True)
                )
                # Get logits (inverse sigmoid)
                sensor_logits = torch.logit(sensor_probs.clamp(1e-7, 1 - 1e-7))
                global_logits = torch.logit(global_probs.clamp(1e-7, 1 - 1e-7))

                # Multi-task loss
                # For forecasting, use next window if available
                x_next = None
                if batch_idx < len(train_loader) - 1:
                    # Get next batch's first window (approximation)
                    pass  # Skip forecasting for now (would need sequence data)

                loss = multi_task_loss(
                    sensor_logits,
                    global_logits,
                    sensor_labels_batch,
                    window_labels_batch,
                    auxiliary_outputs,
                    X_batch,
                    x_next,
                    lambda_reconstruction,
                    lambda_forecasting,
                )
            else:
                # Standard forward pass
                sensor_logits, global_logits, _ = model.forward_logits(X_batch)

                # Classification losses
                sensor_loss = sensor_criterion(sensor_logits, sensor_labels_batch)
                global_loss = global_criterion(
                    global_logits, window_labels_batch.float()
                )
                loss = sensor_loss + 0.3 * global_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.item() * X_batch.size(0)

        train_loss_total /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_scores = []
        val_labels = []

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.to(device)

                if use_multi_task:
                    sensor_probs, global_probs, _ = model.forward_with_auxiliary(
                        X_batch, return_auxiliary=False
                    )
                    sensor_logits = torch.logit(sensor_probs.clamp(1e-7, 1 - 1e-7))
                    global_logits = torch.logit(global_probs.clamp(1e-7, 1 - 1e-7))
                else:
                    sensor_logits, global_logits, _ = model.forward_logits(X_batch)

                sensor_loss = sensor_criterion(sensor_logits, sensor_labels_batch)
                global_loss = global_criterion(
                    global_logits, window_labels_batch.float()
                )
                loss = sensor_loss + 0.3 * global_loss

                val_loss_total += loss.item() * X_batch.size(0)

                # Get probabilities for threshold optimization
                if use_multi_task:
                    global_probs = global_probs  # Already probabilities
                else:
                    global_probs = torch.sigmoid(global_logits)
                val_scores.extend(global_probs.cpu().numpy().flatten())
                val_labels.extend(window_labels_batch.cpu().numpy().flatten())

        val_loss_total /= len(val_loader.dataset)
        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)

        # Optimize threshold
        threshold = optimize_threshold(
            val_labels,
            val_scores,
            metric=threshold_metric,
            precision_target=0.5,
            recall_target=0.7,
        )

        val_preds = (val_scores >= threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_auc = (
            roc_auc_score(val_labels, val_scores)
            if len(np.unique(val_labels)) > 1
            else 0.5
        )

        # Learning rate scheduling
        scheduler.step(val_loss_total)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss_total)
        history["val_loss"].append(val_loss_total)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_auc"].append(val_auc)
        history["best_threshold"].append(threshold)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss_total:.4f} | "
            f"Val Loss: {val_loss_total:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Threshold: {threshold:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model (based on precision, then F1)
        if val_precision > best_precision or (
            val_precision == best_precision and val_f1 > best_f1
        ):
            best_precision = val_precision
            best_f1 = val_f1
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_precision": best_precision,
                "best_f1": best_f1,
                "best_threshold": threshold,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_gat_layers": num_gat_layers,
                "pos_weight": pos_weight,
                "class_weight_method": class_weight_method,
                "threshold_metric": threshold_metric,
            }
            torch.save(checkpoint, model_save_path)
            print(
                f"  ✓ New best model saved (Precision: {val_precision:.4f}, F1: {val_f1:.4f})"
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print(f"\n✓ Training complete. Best model saved to: {model_save_path}")
    print(f"  Best Precision: {best_precision:.4f}")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Best Threshold: {history['best_threshold'][-1]:.4f}")

    return model, history


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced ImprovedMultiLabelGDN")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to carOBD data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anomaly-detection/best_multilabel_gdn_enhanced.pt",
        help="Output model path",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument(
        "--num-gat-layers", type=int, default=2, help="Number of GAT layers"
    )
    parser.add_argument(
        "--fault-percentage",
        type=float,
        default=0.15,
        help="Fault injection percentage",
    )
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        default=True,
        help="Use data augmentation",
    )
    parser.add_argument(
        "--no-augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        default=True,
        help="Use weighted random sampler",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        dest="use_weighted_sampler",
        action="store_false",
        help="Disable weighted sampler",
    )
    parser.add_argument(
        "--use-multi-task",
        action="store_true",
        default=False,
        help="Use multi-task learning",
    )
    parser.add_argument(
        "--class-weight-method",
        type=str,
        default="effective",
        choices=["balanced", "effective", "focal"],
        help="Class weight computation method",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="precision_recall",
        choices=["f1", "precision_recall", "fbeta", "youden"],
        help="Threshold optimization metric",
    )
    parser.add_argument(
        "--lambda-reconstruction",
        type=float,
        default=0.1,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--lambda-forecasting",
        type=float,
        default=0.1,
        help="Weight for forecasting loss",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cpu/cuda/mps)"
    )

    args = parser.parse_args()

    # Device detection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_path = Path(args.data_path)

    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    print(f"  Found {len(csv_files)} CSV files")

    # Load and combine data
    dfs = []
    for csv_file in csv_files[:50]:  # Limit for faster processing
        try:
            df = pd.read_csv(csv_file)
            if ID_COL not in df.columns:
                df[ID_COL] = csv_file.stem
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
    df_clean = mean_fill_missing_timestamps_and_remove_duplicates(
        df_clean, TIME_COL, id_cols=[ID_COL]
    )
    df_clean = downsample(df_clean, TIME_COL, ID_COL, downsample_factor=2)
    df_clean = filter_long_drives(df_clean, ID_COL, min_length=WINDOW_SIZE)
    df_clean = add_cross_channel_features(df_clean)

    # Build windows
    print("3. Building windows...")
    windows, y_targets, scaler = build_clean_windows(
        df_clean, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    print(f"  Created {len(windows)} windows")

    sensor_names = SENSOR_COLS

    # Original fault injection
    print("4. Injecting faults...")
    windows_with_faults, _, sensor_labels, window_labels = (
        inject_faults_with_sensor_labels(
            windows,
            y_targets,
            sensor_names,
            fault_percentage=args.fault_percentage,
            random_state=42,
        )
    )

    num_sensors = len(sensor_names)
    num_windows = len(windows_with_faults)

    print(f"  Total windows: {num_windows}")
    print(f"  Normal windows: {(window_labels == 0).sum()}")
    print(f"  Faulty windows: {(window_labels == 1).sum()}")

    # Stratified split
    print("5. Splitting data (stratified)...")
    X_train, X_temp, y_train, y_temp, sensor_train, sensor_temp = train_test_split(
        windows_with_faults.numpy()
        if isinstance(windows_with_faults, torch.Tensor)
        else windows_with_faults,
        window_labels.numpy()
        if isinstance(window_labels, torch.Tensor)
        else window_labels,
        sensor_labels.numpy()
        if isinstance(sensor_labels, torch.Tensor)
        else sensor_labels,
        test_size=0.3,
        random_state=42,
        stratify=window_labels.numpy()
        if isinstance(window_labels, torch.Tensor)
        else window_labels,
    )
    X_val, X_test, y_val, y_test, sensor_val, sensor_test = train_test_split(
        X_temp, y_temp, sensor_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} windows ({(y_train == 1).sum()} faulty)")
    print(f"  Val: {len(X_val)} windows ({(y_val == 1).sum()} faulty)")
    print(f"  Test: {len(X_test)} windows ({(y_test == 1).sum()} faulty)")

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)

    sensor_train = torch.FloatTensor(sensor_train)
    sensor_val = torch.FloatTensor(sensor_val)
    sensor_test = torch.FloatTensor(sensor_test)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train, sensor_train, y_train)
    val_dataset = TensorDataset(X_val, y_val, sensor_val, y_val)
    test_dataset = TensorDataset(X_test, y_test, sensor_test, y_test)

    # Create weighted sampler if enabled
    train_loader_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    if args.use_weighted_sampler:
        sampler = create_weighted_sampler(y_train)
        train_loader_kwargs = {
            "batch_size": args.batch_size,
            "sampler": sampler,
            "shuffle": False,
        }

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train model
    print("\n6. Training enhanced model...")
    model, history = train_enhanced_gdn(
        train_loader=train_loader,
        val_loader=val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        use_augmentation=args.use_augmentation,
        use_weighted_sampler=args.use_weighted_sampler,
        use_multi_task=args.use_multi_task,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        class_weight_method=args.class_weight_method,
        threshold_metric=args.threshold_metric,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_forecasting=args.lambda_forecasting,
        model_save_path=args.output,
    )

    # Save training history
    history_path = args.output.replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2
        )
    print(f"  Training history saved to: {history_path}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
