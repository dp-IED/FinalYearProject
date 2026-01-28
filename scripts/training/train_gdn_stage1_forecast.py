#!/usr/bin/env python3
"""
Stage 1: Graph Structure Learning (Self-Supervised)
Training script for GDN with forecasting loss only.

Objective: Learn meaningful sensor embeddings and graph structure through
self-supervised forecasting task. No labels needed.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "anomaly-detection"))
from models.gdn_model import MultiLabelGDN, KAGOptimizedGDN

torch.set_default_dtype(torch.float32)

# ============================================================================
# Constants
# ============================================================================

DATA_PATH = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata"
SENSOR_COLS = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
]
ID_COL = "drive_id"
TIME_COL = "ENGINE_RUN_TINE ()"
WINDOW_SIZE = 300
FORECAST_HORIZONS = [1, 5, 10]  # Multi-horizon forecasting

# Training hyperparameters
NUM_EPOCHS = 75  # Default: 50-100 epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Model architecture
EMBED_DIM = 32  # Reduced from 64 for better generalization
TOP_K = 5  # Increased from 3 for better connectivity
HIDDEN_DIM = 64

# ============================================================================
# Model with Forecasting Head
# ============================================================================


class GDNWithForecasting(nn.Module):
    """
    MultiLabelGDN extended with forecasting and reconstruction heads for self-supervised learning.
    - Multi-horizon forecasting: predicts t+1, t+5, t+10
    - Reconstruction: reconstructs full window from embeddings
    """

    def __init__(self, base_model: MultiLabelGDN, num_horizons=3):
        super().__init__()
        self.base_model = base_model
        num_nodes = base_model.num_nodes
        hidden_dim = base_model.hidden_dim
        window_size = base_model.window_size
        self.num_horizons = num_horizons

        # Multi-horizon forecasting head: predicts multiple future timesteps
        self.forecasting_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_horizons * num_nodes),  # Predict num_horizons * num_sensors
        )
        
        # Reconstruction head: reconstructs full window from sensor embeddings
        self.reconstruction_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, window_size),  # Reconstruct full window for each sensor
            )
            for _ in range(num_nodes)
        ])

    def forward(self, x, return_forecast=False, return_reconstruction=False):
        """
        Forward pass through base model.

        Args:
            x: (B, W, N) input tensor
            return_forecast: If True, also return forecasted future timesteps
            return_reconstruction: If True, also return reconstructed window

        Returns:
            - sensor_logits: (B, N) logits for each sensor (from base model)
            - forecast: (B, num_horizons, N) predicted future timesteps (optional)
            - reconstruction: (B, W, N) reconstructed window (optional)
        """
        # Get embeddings from base model
        embeddings = self.base_model.get_embeddings(x)  # (B, hidden_dim)
        
        # Get sensor embeddings for reconstruction
        sensor_embeddings = self.base_model.get_sensor_embeddings(x)  # (B, N, hidden_dim)

        # Get base model outputs
        sensor_logits = self.base_model(x)
        
        outputs = [sensor_logits]

        if return_forecast:
            # Predict multiple horizons
            forecast_flat = self.forecasting_head(embeddings)  # (B, num_horizons * num_nodes)
            forecast = forecast_flat.reshape(-1, self.num_horizons, self.base_model.num_nodes)  # (B, num_horizons, N)
            outputs.append(forecast)
        
        if return_reconstruction:
            # Reconstruct each sensor's time series
            reconstructions = []
            for i in range(self.base_model.num_nodes):
                recon = self.reconstruction_head[i](sensor_embeddings[:, i, :])  # (B, window_size)
                reconstructions.append(recon)
            # Stack along sensor dimension: (B, window_size, N)
            reconstruction = torch.stack(reconstructions, dim=2)  # (B, window_size, num_nodes)
            # This matches input format (B, W, N) where W=window_size, N=num_nodes
            outputs.append(reconstruction)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def get_embeddings(self, x):
        """Get embeddings from base model."""
        return self.base_model.get_embeddings(x)


# ============================================================================
# Data Preprocessing Functions (reused from train_gdn_triplet_center.py)
# ============================================================================


def remove_zero_variance_columns(
    df: pd.DataFrame, exclude_cols: list[str] = None
) -> pd.DataFrame:
    """Remove columns with zero variance."""
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]

    std_df = df[cols_to_check].std()
    zero_variance_cols = std_df[std_df == 0].index.tolist()

    print(f"{len(zero_variance_cols)} columns with zero variance: {zero_variance_cols}")

    if len(zero_variance_cols) > 0:
        df = df.drop(columns=zero_variance_cols)

    return df


def mean_fill_missing_timestamps_and_remove_duplicates(
    df: pd.DataFrame, time_col: str, id_cols: list[str] = None
) -> pd.DataFrame:
    """Remove duplicate timestamps by averaging all numeric columns."""
    if id_cols is None:
        id_cols = []

    existing_id_cols = [col for col in id_cols if col in df.columns]
    group_cols = [time_col] + existing_id_cols

    agg_dict = {}
    for col in df.columns:
        if col not in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "mean"
            else:
                agg_dict[col] = "first"

    df_clean = df.groupby(group_cols, as_index=False).agg(agg_dict)
    return df_clean


def downsample(df, time_col, source_file_col, downsample_factor=2):
    """Downsample data by factor."""
    result_dfs = []

    for source_file in df[source_file_col].unique():
        file_df = df[df[source_file_col] == source_file].copy()

        if len(file_df) < downsample_factor * 2:
            continue

        file_df = file_df.sort_values(time_col).reset_index(drop=True)
        downsampled = file_df.iloc[::downsample_factor].copy()
        downsampled[time_col] = np.arange(len(downsampled)) * downsample_factor

        result_dfs.append(downsampled.reset_index(drop=True))

    return pd.concat(result_dfs, ignore_index=True)


def filter_long_drives(df, id_col="drive_id", min_length=608):
    """Keep only drives long enough for context window."""
    drive_lengths = df.groupby(id_col).size()
    valid_drives = drive_lengths[drive_lengths >= min_length].index

    print(f"Keeping {len(valid_drives)}/{df[id_col].nunique()} drives")
    print(f"Dropped {len(df) - df[df[id_col].isin(valid_drives)].shape[0]} timesteps")

    return df[df[id_col].isin(valid_drives)].reset_index(drop=True)


def add_cross_channel_features(data):
    """Engineer features that capture cross-channel relationships."""
    if "ENGINE_RPM ()" in data.columns and "VEHICLE_SPEED ()" in data.columns:
        data["RPM_SPEED_RATIO"] = data["ENGINE_RPM ()"] / (data["VEHICLE_SPEED ()"] + 1)

    if "THROTTLE ()" in data.columns and "ENGINE_LOAD ()" in data.columns:
        data["THROTTLE_LOAD_RATIO"] = data["THROTTLE ()"] / (data["ENGINE_LOAD ()"] + 1)

    if "VEHICLE_SPEED ()" in data.columns:
        data["IS_IDLE"] = (data["VEHICLE_SPEED ()"] < 5).astype(float)
        data["IS_HIGHWAY"] = (data["VEHICLE_SPEED ()"] > 60).astype(float)

    if "ENGINE_RPM ()" in data.columns:
        data["RPM_ACCEL"] = data.groupby("drive_id")["ENGINE_RPM ()"].diff().fillna(0)

    return data


def build_forecast_windows(
    df, sensor_cols, id_col, time_col, window_size, horizons=[1, 5, 10], scaler=None
):
    """
    Build windows for multi-horizon forecasting: window[t] predicts window[t+h] for h in horizons.
    Returns consecutive windows for self-supervised learning with drive IDs.
    
    Args:
        horizons: List of forecast horizons (e.g., [1, 5, 10] means predict t+1, t+5, t+10)
    
    Returns:
        X: (N, window_size, num_sensors) input windows
        y_forecast: (N, num_horizons, num_sensors) multi-horizon targets
        drive_ids: (N,) drive IDs for each window
        scaler: Fitted scaler
    """
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_forecast_list, drive_id_list = [], [], []
    max_horizon = max(horizons)

    for drive_id, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, num_sensors = values.shape
        if T_ <= window_size + max_horizon:
            continue

        # Create consecutive windows: window[t] predicts multiple future timesteps
        for t in range(T_ - window_size - max_horizon):
            X_window = values[t : t + window_size]  # Current window
            
            # Multiple targets for different horizons
            y_targets = []
            for h in horizons:
                y_targets.append(values[t + window_size + h - 1])  # Target at t+window_size+h-1
            
            X_list.append(X_window)
            y_forecast_list.append(np.stack(y_targets))  # (num_horizons, num_sensors)
            drive_id_list.append(drive_id)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y_forecast = torch.tensor(np.stack(y_forecast_list), dtype=torch.float32)  # (N, num_horizons, num_sensors)
    drive_ids = np.array(drive_id_list)
    
    return X, y_forecast, drive_ids, scaler


# ============================================================================
# Contrastive Loss
# ============================================================================


class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss: embeddings from same drive should be similar,
    embeddings from different drives should be dissimilar.
    Implements InfoNCE loss for temporal contrastive learning.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, drive_ids):
        """
        Args:
            embeddings: (B, hidden_dim) window embeddings
            drive_ids: (B,) which drive each window came from (tensor of integers)
        """
        device = embeddings.device
        B = embeddings.size(0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Ensure drive_ids is a tensor on the correct device
        if not isinstance(drive_ids, torch.Tensor):
            drive_ids = torch.tensor(drive_ids, device=device, dtype=torch.long)
        else:
            drive_ids = drive_ids.to(device).long()
        
        drive_ids_tensor = drive_ids
        
        # Similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Positive pairs: same drive
        positive_mask = (drive_ids_tensor.unsqueeze(0) == drive_ids_tensor.unsqueeze(1)).float()
        positive_mask.fill_diagonal_(0)  # Exclude self
        
        # Negative pairs: different drives
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        
        # For each sample, pull positives close, push negatives away
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = (exp_sim * (positive_mask + negative_mask)).sum(dim=1)
        
        # Handle case where no positive pairs exist
        has_pos = positive_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        loss = -torch.log((pos_sim[has_pos] / (all_sim[has_pos] + 1e-8)) + 1e-8).mean()
        
        return loss


# ============================================================================
# Training Function
# ============================================================================


def train_stage1(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    checkpoint_dir="checkpoints",
    use_compile=False,
    compile_mode="reduce-overhead",
    gradient_accumulation_steps=1,
    use_amp=False,
    model_type="enhanced",
):
    """
    Train GDN with forecasting loss only (self-supervised).

    Args:
        model_type: "enhanced" (MultiLabelGDN) or "kag_optimized" (KAGOptimizedGDN)

    Returns:
        model: Trained model with forecasting head
    """
    # Initialize base model based on model_type
    if model_type == "kag_optimized":
        print(f"\nUsing KAG-Optimized Baseline Model (fast, embedding-focused)")
        base_model = KAGOptimizedGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
    else:
        print(f"\nUsing Enhanced MultiLabelGDN Model (full features)")
        base_model = MultiLabelGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)

    # Wrap with forecasting head (with multi-horizon support)
    num_horizons = len(FORECAST_HORIZONS)
    model = GDNWithForecasting(base_model, num_horizons=num_horizons).to(device)
    
    # Apply torch.compile() if requested (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print(f"\nCompiling model with mode='{compile_mode}'...")
        print("  Note: First epoch will be slower due to compilation")
        model = torch.compile(model, mode=compile_mode)
    elif use_compile:
        print("\nWarning: torch.compile() not available (requires PyTorch 2.0+)")
        print("  Continuing without compilation")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    # Mixed precision training (AMP) - only for CUDA
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = GradScaler()
        print("  ✓ Mixed precision training (AMP) enabled for CUDA")
    elif use_amp:
        print("  ⚠ AMP requested but not on CUDA device, disabling AMP")
        use_amp = False

    # Loss functions
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    contrastive_loss_fn = TemporalContrastiveLoss(temperature=0.5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 20

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "stage1_best_forecast.pt")

    print(f"\n{'=' * 80}")
    print("Stage 1: Graph Structure Learning (Self-Supervised Forecasting)")
    print(f"{'=' * 80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Top-K: {TOP_K}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"Model compilation: {use_compile}\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_forecast = 0.0
        train_loss_recon = 0.0
        train_loss_contrastive = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            # Zero gradients at start of epoch (for gradient accumulation)
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                # Batch format: (X, y_forecast, drive_ids)
                X_batch, y_forecast_batch, drive_ids_batch = batch
                
                X_batch = X_batch.to(device)
                y_forecast_batch = y_forecast_batch.to(device)
                drive_ids_batch = drive_ids_batch.to(device)

                # Forward pass with optional AMP
                if use_amp and scaler is not None:
                    with autocast():
                        _, forecast, reconstruction = model(X_batch, return_forecast=True, return_reconstruction=True)

                        # Multi-horizon forecasting loss (averaged across horizons)
                        loss_forecast = forecast_criterion(forecast, y_forecast_batch)

                        # Reconstruction loss
                        loss_recon = recon_criterion(reconstruction, X_batch)

                        # Contrastive loss
                        window_embeddings = model.base_model.get_embeddings(X_batch)
                        loss_contrastive = contrastive_loss_fn(window_embeddings, drive_ids_batch)

                        # Combined loss (scale by accumulation steps)
                        loss = (loss_forecast + 0.5 * loss_recon + 0.3 * loss_contrastive) / gradient_accumulation_steps
                else:
                    # Forward pass: get forecast and reconstruction
                    _, forecast, reconstruction = model(X_batch, return_forecast=True, return_reconstruction=True)

                    # Multi-horizon forecasting loss (averaged across horizons)
                    loss_forecast = forecast_criterion(forecast, y_forecast_batch)

                    # Reconstruction loss
                    loss_recon = recon_criterion(reconstruction, X_batch)

                    # Contrastive loss
                    window_embeddings = model.base_model.get_embeddings(X_batch)
                    loss_contrastive = contrastive_loss_fn(window_embeddings, drive_ids_batch)

                    # Combined loss (scale by accumulation steps)
                    loss = (loss_forecast + 0.5 * loss_recon + 0.3 * loss_contrastive) / gradient_accumulation_steps

                # Backward pass with optional AMP
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update every gradient_accumulation_steps batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update optimizer
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Zero gradients for next accumulation
                    optimizer.zero_grad()

                train_loss += loss.item() * X_batch.size(0) * gradient_accumulation_steps
                train_loss_forecast += loss_forecast.item() * X_batch.size(0) * gradient_accumulation_steps
                train_loss_recon += loss_recon.item() * X_batch.size(0) * gradient_accumulation_steps
                train_loss_contrastive += loss_contrastive.item() * X_batch.size(0) * gradient_accumulation_steps

        train_loss /= len(train_loader.dataset)
        train_loss_forecast /= len(train_loader.dataset)
        train_loss_recon /= len(train_loader.dataset)
        train_loss_contrastive /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_forecast = 0.0
        val_loss_recon = 0.0
        val_loss_contrastive = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Handle batch format: (X, y_forecast, drive_ids)
                X_batch, y_forecast_batch, drive_ids_batch = batch
                
                X_batch = X_batch.to(device)
                y_forecast_batch = y_forecast_batch.to(device)
                drive_ids_batch = drive_ids_batch.to(device)

                _, forecast, reconstruction = model(X_batch, return_forecast=True, return_reconstruction=True)

                loss_forecast = forecast_criterion(forecast, y_forecast_batch)
                loss_recon = recon_criterion(reconstruction, X_batch)
                
                window_embeddings = model.base_model.get_embeddings(X_batch)
                loss_contrastive = contrastive_loss_fn(window_embeddings, drive_ids_batch)
                
                loss = loss_forecast + 0.5 * loss_recon + 0.3 * loss_contrastive

                val_loss += loss.item() * X_batch.size(0)
                val_loss_forecast += loss_forecast.item() * X_batch.size(0)
                val_loss_recon += loss_recon.item() * X_batch.size(0)
                val_loss_contrastive += loss_contrastive.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_forecast /= len(val_loader.dataset)
        val_loss_recon /= len(val_loader.dataset)
        val_loss_contrastive /= len(val_loader.dataset)

        # Update scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Forecast: {train_loss_forecast:.6f} | "
            f"Recon: {train_loss_recon:.6f} | "
            f"Contrastive: {train_loss_contrastive:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Forecast: {val_loss_forecast:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": base_model.sensor_embeddings.data.cpu(),
                    "epoch": epoch + 1,
                    "best_val_loss": val_loss,
                    "stage": 1,
                    "model_type": model_type,
                },
                best_checkpoint_path,
            )
            print(f"  ✓ Best model saved: {best_checkpoint_path} (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no improvement for {max_patience} epochs)"
            )
            break

    # Load best checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n{'=' * 80}")
    print(f"Stage 1 training complete!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"{'=' * 80}\n")

    return model, base_model


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Self-Supervised Graph Structure Learning"
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Force CPU usage (disable CUDA auto-detection)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before updating (default: 1)",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile() to optimize model (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile() mode (default: reduce-overhead)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP) for CUDA devices",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="enhanced",
        choices=["enhanced", "kag_optimized"],
        help="Model type: 'enhanced' (full features) or 'kag_optimized' (fast baseline with essential fixes)",
    )
    args = parser.parse_args()

    # Device detection (CUDA or CPU only - MPS not supported due to PyTorch Geometric incompatibility)
    if args.cpu_only:
        device = "cpu"
        print("Using device: cpu (forced via --cpu_only flag)")
    elif args.device is not None:
        device = args.device
        if device == "mps":
            print("Warning: MPS not supported (PyTorch Geometric incompatibility). Falling back to CPU.")
            device = "cpu"
        print(f"Using device: {device} (specified via --device)")
    else:
        # Auto-detect: prefer CUDA > CPU
        if torch.cuda.is_available():
            device = "cuda"
            print("Using device: cuda (auto-detected)")
        else:
            device = "cpu"
            print("Using device: cpu (auto-detected)")

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    df_list = []
    for file in os.listdir(args.data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{args.data_path}/{file}", index_col=False)
            df["drive_id"] = file
            df_list.append(df)

    print(f"Loaded {len(df_list)} files")

    # Combine all dataframes
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data[ID_COL].nunique()}")

    # Preprocessing
    print("\nPreprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ]
    )
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=[ID_COL]
    )
    data = remove_zero_variance_columns(data, exclude_cols=[ID_COL])
    data = downsample(
        data, time_col=TIME_COL, source_file_col=ID_COL, downsample_factor=1
    )
    data = filter_long_drives(
        data, id_col=ID_COL, min_length=WINDOW_SIZE + max(FORECAST_HORIZONS) + 1
    )
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15)
    print("\nSplitting data by drive...")
    unique_drives = data[ID_COL].unique()
    n_drives = len(unique_drives)

    train_drives = unique_drives[: int(0.70 * n_drives)]
    val_drives = unique_drives[int(0.70 * n_drives) : int(0.85 * n_drives)]
    test_drives = unique_drives[int(0.85 * n_drives) :]

    print(
        f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}"
    )

    train_data = data[data[ID_COL].isin(train_drives)].copy()
    val_data = data[data[ID_COL].isin(val_drives)].copy()
    test_data = data[data[ID_COL].isin(test_drives)].copy()

    print(
        f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}"
    )

    # Build forecast windows (self-supervised, no labels needed)
    print("\nBuilding forecast windows...")
    X_train, y_train_forecast, drive_ids_train, scaler_train = build_forecast_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, horizons=FORECAST_HORIZONS, scaler=None
    )
    X_val, y_val_forecast, drive_ids_val, _ = build_forecast_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, horizons=FORECAST_HORIZONS, scaler=scaler_train
    )

    print(f"Train windows: {len(X_train)}")
    print(f"Val windows: {len(X_val)}")
    print(f"Forecast horizons: {FORECAST_HORIZONS}")

    # Create dataloaders with drive IDs
    # Convert string drive IDs to integer indices for tensor dataset
    all_drive_ids = np.concatenate([drive_ids_train, drive_ids_val])
    unique_drives = np.unique(all_drive_ids)
    drive_to_idx = {drive: idx for idx, drive in enumerate(unique_drives)}
    
    drive_ids_train_idx = np.array([drive_to_idx[drive] for drive in drive_ids_train])
    drive_ids_val_idx = np.array([drive_to_idx[drive] for drive in drive_ids_val])
    
    train_ds = TensorDataset(X_train, y_train_forecast, torch.tensor(drive_ids_train_idx, dtype=torch.long))
    val_ds = TensorDataset(X_val, y_val_forecast, torch.tensor(drive_ids_val_idx, dtype=torch.long))

    # Optimized DataLoader configuration
    pin_memory = device.startswith("cuda")  # Only pin memory for CUDA
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    num_sensors = len(SENSOR_COLS)
    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    # Train model
    model, base_model = train_stage1(
        train_loader,
        val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        model_type=args.model,
    )

    print("✓ Stage 1 training complete!")


if __name__ == "__main__":
    main()
