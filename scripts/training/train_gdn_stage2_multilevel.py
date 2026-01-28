#!/usr/bin/env python3
"""
Stage 2: Multi-Level Center Loss Training
Training script for GDN with hierarchical center loss (window-level + sensor-level).

Objective: Learn separate centers for:
1. Window-level: Normal vs. anomalous window embeddings
2. Sensor-level: Normal vs. anomalous per-sensor embeddings

This enables compatible embedding spaces and better sensor attribution.
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
from models.multi_level_center_loss import MultiLevelCenterLoss

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
FORECAST_HORIZON = 1

# Training hyperparameters
NUM_EPOCHS = 50  # Default: 40-50 epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_CENTER = 1.2  # Increased from 0.8 to emphasize separation
LAMBDA_GLOBAL = 0.3

# Model architecture (must match Stage 1)
EMBED_DIM = 32  # Reduced from 64 for better generalization
TOP_K = 5  # Increased from 3 for better connectivity
HIDDEN_DIM = 64

# Multi-Level Center Loss parameters
MLC_MARGIN = 2.0
MLC_LAMBDA_INTRA = 1.0  # Reduced from 1.5 to allow more natural clustering
MLC_LAMBDA_SENSOR = 0.8  # Increased from 0.5 to push sensor-specific separation

# ============================================================================
# Data Preprocessing Functions (reused from train_gdn_stage2_embedding.py)
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


def build_clean_windows(
    df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None
):
    """Build windows from CLEAN data only. Returns normalized windows."""
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_list = [], []

    for drive_id, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, num_sensors = values.shape
        if T_ <= window_size + horizon:
            continue

        for t in range(T_ - window_size - horizon + 1):
            X_window = values[t : t + window_size]
            y_target = values[t + window_size + horizon - 1]
            X_list.append(X_window)
            y_list.append(y_target)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    return X, y, scaler


# Import shared fault injection with stratified distribution
training_dir = Path(__file__).parent
sys.path.insert(0, str(training_dir))
from fault_injection import inject_faults_with_sensor_labels


# ============================================================================
# Training Function
# ============================================================================


def train_stage2_multilevel(
    train_loader,
    val_loader,
    num_sensors,
    window_size,
    stage1_checkpoint_path,
    num_epochs=NUM_EPOCHS,
    device="cpu",
    lambda_center=LAMBDA_CENTER,
    lambda_global=LAMBDA_GLOBAL,
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
    Train GDN with multi-level center loss (window + sensor levels).

    Args:
        stage1_checkpoint_path: Path to Stage 1 checkpoint to load
        model_type: "enhanced" (MultiLabelGDN) or "kag_optimized" (KAGOptimizedGDN)

    Returns:
        model: Trained model
        center_loss: Trained MultiLevelCenterLoss module
    """
    # Load Stage 1 checkpoint
    print(f"\nLoading Stage 1 checkpoint from {stage1_checkpoint_path}...")
    stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location=device)
    
    # Determine model type from checkpoint or parameter
    # Check if checkpoint has model_type info, otherwise use parameter
    checkpoint_model_type = stage1_checkpoint.get("model_type", None)
    if checkpoint_model_type not in ["enhanced", "kag_optimized"]:
        # Fallback: try to infer from checkpoint structure
        if "base_model_state_dict" in stage1_checkpoint:
            # Check if it has temporal_pooling (enhanced) or not (kag_optimized)
            base_state = stage1_checkpoint["base_model_state_dict"]
            has_temporal_pooling = any("temporal_pooling" in k for k in base_state.keys())
            checkpoint_model_type = "enhanced" if has_temporal_pooling else "kag_optimized"
        elif "model_state_dict" in stage1_checkpoint:
            base_state = stage1_checkpoint["model_state_dict"]
            has_temporal_pooling = any("temporal_pooling" in k for k in base_state.keys())
            checkpoint_model_type = "enhanced" if has_temporal_pooling else "kag_optimized"
        else:
            checkpoint_model_type = model_type
    
    # Use checkpoint model type if available, otherwise use parameter
    final_model_type = checkpoint_model_type if checkpoint_model_type else model_type

    # Initialize model based on type
    if final_model_type == "kag_optimized":
        print(f"\nUsing KAG-Optimized Baseline Model (fast, embedding-focused)")
        model = KAGOptimizedGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
    else:
        print(f"\nUsing Enhanced MultiLabelGDN Model (full features)")
        model = MultiLabelGDN(
            num_nodes=num_sensors,
            window_size=window_size,
            embed_dim=EMBED_DIM,
            top_k=TOP_K,
            hidden_dim=HIDDEN_DIM,
        ).to(device)
    
    # Apply torch.compile() if requested (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print(f"\nCompiling model with mode='{compile_mode}'...")
        print("  Note: First epoch will be slower due to compilation")
        model = torch.compile(model, mode=compile_mode)
    elif use_compile:
        print("\nWarning: torch.compile() not available (requires PyTorch 2.0+)")
        print("  Continuing without compilation")

    # Load base model state (from Stage 1)
    # Handle different checkpoint formats and model architectures
    if "base_model_state_dict" in stage1_checkpoint:
        # Stage 1 used GDNWithForecasting wrapper
        base_state = stage1_checkpoint["base_model_state_dict"]
        
        # For KAG-optimized, filter out enhanced-specific layers
        if final_model_type == "kag_optimized":
            # Remove enhanced-specific layers (temporal_pooling, multi_scale_gat)
            filtered_state = {
                k: v for k, v in base_state.items()
                if not any(x in k for x in ['temporal_pooling', 'multi_scale_gat'])
            }
            # Also filter GAT if heads don't match (4→2)
            filtered_state = {
                k: v for k, v in filtered_state.items()
                if not (k.startswith('gat.') and 'weight' in k and v.shape[0] == 4 * HIDDEN_DIM)
            }
        else:
            # Enhanced model: filter GAT parameters if heads don't match
            filtered_state = {
                k: v for k, v in base_state.items()
                if not (k.startswith('gat.') and 'weight' in k and v.shape[0] == 4 * HIDDEN_DIM)
            }
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"  ✓ Loaded model state (model_type: {final_model_type})")
    elif "model_state_dict" in stage1_checkpoint:
        # Direct model state
        base_state = stage1_checkpoint["model_state_dict"]
        
        # Same filtering logic
        if final_model_type == "kag_optimized":
            filtered_state = {
                k: v for k, v in base_state.items()
                if not any(x in k for x in ['temporal_pooling', 'multi_scale_gat'])
            }
        else:
            filtered_state = base_state
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"  ✓ Loaded model state (model_type: {final_model_type})")

    # CRITICAL: Freeze sensor embeddings and graph structure
    print("\nFreezing sensor embeddings and graph structure...")
    model.sensor_embeddings.requires_grad = False
    print(f"  Sensor embeddings frozen: {not model.sensor_embeddings.requires_grad}")

    # Initialize multi-level center loss
    center_loss = MultiLevelCenterLoss(
        embed_dim=HIDDEN_DIM,
        num_sensors=num_sensors,
        num_classes=2,
        margin=MLC_MARGIN,
        lambda_intra=MLC_LAMBDA_INTRA,
        lambda_sensor=MLC_LAMBDA_SENSOR,
    ).to(device)

    # Initialize centers at opposite ends of hypersphere
    with torch.no_grad():
        # Window-level centers
        center_loss.window_centers[0] = F.normalize(torch.ones(HIDDEN_DIM), dim=0)
        center_loss.window_centers[1] = F.normalize(-torch.ones(HIDDEN_DIM), dim=0)
        
        # Sensor-level centers (initialize all sensors similarly)
        for sensor_idx in range(num_sensors):
            center_loss.sensor_centers[sensor_idx, 0] = F.normalize(torch.ones(HIDDEN_DIM), dim=0)
            center_loss.sensor_centers[sensor_idx, 1] = F.normalize(-torch.ones(HIDDEN_DIM), dim=0)

    # Optimizers: only for trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )
    # Use Adam for center updates (more stable than SGD)
    optimizer_center = torch.optim.Adam(center_loss.parameters(), lr=0.01)
    
    # Mixed precision training (AMP) - only for CUDA
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = GradScaler()
        print("  ✓ Mixed precision training (AMP) enabled for CUDA")
    elif use_amp:
        print("  ⚠ AMP requested but not on CUDA device, disabling AMP")
        use_amp = False

    # Loss functions
    sensor_criterion = nn.BCEWithLogitsLoss(reduction="none")
    global_criterion = nn.BCEWithLogitsLoss()

    # Learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )
    scheduler_center = torch.optim.lr_scheduler.StepLR(
        optimizer_center, step_size=10, gamma=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 15

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "stage2_multilevel.pt")

    print(f"\n{'=' * 80}")
    print("Stage 2: Multi-Level Center Loss Training")
    print(f"{'=' * 80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Lambda_center: {lambda_center}, Lambda_global: {lambda_global}")
    print(f"MLC margin: {MLC_MARGIN}, MLC lambda_intra: {MLC_LAMBDA_INTRA}, MLC lambda_sensor: {MLC_LAMBDA_SENSOR}")
    print(f"Frozen parameters: sensor_embeddings (graph structure)")
    print(f"Device: {device}")
    print(f"Model type: {final_model_type}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Mixed precision (AMP): {use_amp}")
    print(f"Model compilation: {use_compile}\n")

    for epoch in range(num_epochs):
        model.train()
        center_loss.train()

        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_center = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            # Zero gradients at start of epoch (for gradient accumulation)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            for batch_idx, (X_batch, _, sensor_labels_batch, window_labels_batch) in enumerate(pbar):
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                # Forward pass (get both embeddings) with optional AMP
                if use_amp and scaler is not None:
                    with autocast():
                        sensor_logits, global_logits, sensor_embeddings = model(
                            X_batch, 
                            return_global=True,
                            return_sensor_embeddings=True
                        )
                        
                        # Window-level embedding (for window center)
                        window_embeddings = model.get_embeddings(X_batch)

                        # Classification losses
                        loss_sensor_clf = sensor_criterion(
                            sensor_logits, sensor_labels_batch
                        ).mean()
                        loss_global_clf = global_criterion(
                            global_logits, window_labels_batch.float()
                        )

                        # Multi-level center loss
                        loss_center = center_loss(
                            window_embeddings=window_embeddings,
                            sensor_embeddings=sensor_embeddings,
                            window_labels=window_labels_batch,
                            sensor_labels=sensor_labels_batch.long(),  # Use ground-truth sensor labels
                        )

                        # Combined loss (scale by accumulation steps)
                        loss = (
                            loss_sensor_clf +
                            lambda_global * loss_global_clf +
                            lambda_center * loss_center
                        ) / gradient_accumulation_steps
                else:
                    # Forward pass (get both embeddings)
                    sensor_logits, global_logits, sensor_embeddings = model(
                        X_batch, 
                        return_global=True,
                        return_sensor_embeddings=True
                    )
                    
                    # Window-level embedding (for window center)
                    window_embeddings = model.get_embeddings(X_batch)

                    # Classification losses
                    loss_sensor_clf = sensor_criterion(
                        sensor_logits, sensor_labels_batch
                    ).mean()
                    loss_global_clf = global_criterion(
                        global_logits, window_labels_batch.float()
                    )

                    # Multi-level center loss
                    loss_center = center_loss(
                        window_embeddings=window_embeddings,
                        sensor_embeddings=sensor_embeddings,
                        window_labels=window_labels_batch,
                        sensor_labels=sensor_labels_batch.long(),  # Use ground-truth sensor labels
                    )

                    # Combined loss (scale by accumulation steps)
                    loss = (
                        loss_sensor_clf +
                        lambda_global * loss_global_clf +
                        lambda_center * loss_center
                    ) / gradient_accumulation_steps

                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf in loss at epoch {epoch + 1}, skipping batch")
                    continue

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
                        scaler.unscale_(optimizer_center)
                    
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(
                        center_loss.parameters(), max_norm=0.5
                    )

                    # Update optimizers
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.step(optimizer_center)
                        scaler.update()
                    else:
                        optimizer.step()
                        optimizer_center.step()
                    
                    # Zero gradients for next accumulation
                    optimizer.zero_grad()
                    optimizer_center.zero_grad()

                train_loss_sensor += loss_sensor_clf.item() * X_batch.size(0) * gradient_accumulation_steps
                train_loss_global += loss_global_clf.item() * X_batch.size(0) * gradient_accumulation_steps
                train_loss_center += loss_center.item() * X_batch.size(0) * gradient_accumulation_steps

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        train_loss_center /= len(train_loader.dataset)

        # Validation
        model.eval()
        center_loss.eval()
        val_loss_sensor = 0.0
        val_loss_global = 0.0
        val_loss_center = 0.0

        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                sensor_logits, global_logits, sensor_embeddings = model(
                    X_batch,
                    return_global=True,
                    return_sensor_embeddings=True
                )
                window_embeddings = model.get_embeddings(X_batch)

                loss_sensor = sensor_criterion(
                    sensor_logits, sensor_labels_batch
                ).mean()
                loss_global = global_criterion(
                    global_logits, window_labels_batch.float()
                )
                loss_center_val = center_loss(
                    window_embeddings=window_embeddings,
                    sensor_embeddings=sensor_embeddings,
                    window_labels=window_labels_batch,
                    sensor_labels=sensor_labels_batch.long(),
                )

                val_loss_sensor += loss_sensor.item() * X_batch.size(0)
                val_loss_global += loss_global.item() * X_batch.size(0)
                val_loss_center += loss_center_val.item() * X_batch.size(0)

        val_loss_sensor /= len(val_loader.dataset)
        val_loss_global /= len(val_loader.dataset)
        val_loss_center /= len(val_loader.dataset)

        # Total validation loss
        val_total_loss = (
            val_loss_sensor +
            lambda_global * val_loss_global +
            lambda_center * val_loss_center
        )

        # Update scheduler
        scheduler.step(val_total_loss)
        scheduler_center.step()

        # Get separation metrics
        separations = center_loss.get_separations()
        
        # Compute compactness metrics every 5 epochs
        compactness_normal = None
        compactness_anomaly = None
        separation_ratio = None
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                # Collect all validation embeddings
                all_window_embs = []
                all_window_labels = []
                
                for X_batch, _, _, window_labels_batch in val_loader:
                    X_batch = X_batch.to(device)
                    window_labels_batch = window_labels_batch.long().to(device)
                    window_embs = model.get_embeddings(X_batch)
                    window_embs = F.normalize(window_embs, p=2, dim=1)
                    
                    all_window_embs.append(window_embs.cpu())
                    all_window_labels.append(window_labels_batch.cpu())
                
                window_embs = torch.cat(all_window_embs, dim=0)
                window_labels = torch.cat(all_window_labels, dim=0)
                
                # Get centers
                window_centers = center_loss.get_window_centers().cpu()
                
                # Compute compactness
                normal_mask = (window_labels == 0)
                anomaly_mask = (window_labels == 1)
                
                if normal_mask.sum() > 0:
                    normal_embs = window_embs[normal_mask]
                    normal_center = window_centers[0]
                    compactness_normal = torch.norm(
                        normal_embs - normal_center, dim=1
                    ).mean().item()
                
                if anomaly_mask.sum() > 0:
                    anomaly_embs = window_embs[anomaly_mask]
                    anomaly_center = window_centers[1]
                    compactness_anomaly = torch.norm(
                        anomaly_embs - anomaly_center, dim=1
                    ).mean().item()
                
                if compactness_normal is not None and compactness_anomaly is not None:
                    avg_compactness = (compactness_normal + compactness_anomaly) / 2
                    separation_ratio = separations['window_separation'] / avg_compactness

        # Logging (print separations and compactness every 5 epochs)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            compactness_str = ""
            if compactness_normal is not None and compactness_anomaly is not None:
                compactness_str = (
                    f" | Compactness: N={compactness_normal:.4f}, A={compactness_anomaly:.4f}"
                    f" | Ratio: {separation_ratio:.2f}"
                )
            
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Sensor: {train_loss_sensor:.4f} | "
                f"Global: {train_loss_global:.4f} | "
                f"Center: {train_loss_center:.4f} | "
                f"Val Total: {val_total_loss:.4f} | "
                f"Window sep: {separations['window_separation']:.4f} | "
                f"Sensor mean sep: {separations['sensor_mean_separation']:.4f} | "
                f"Sensor range: [{separations['sensor_min_separation']:.4f}, "
                f"{separations['sensor_max_separation']:.4f}]"
                f"{compactness_str}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Sensor: {train_loss_sensor:.4f} | "
                f"Global: {train_loss_global:.4f} | "
                f"Center: {train_loss_center:.4f} | "
                f"Val Total: {val_total_loss:.4f}"
            )

        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "center_loss_state_dict": center_loss.state_dict(),
                    "window_centers": center_loss.get_window_centers().cpu(),
                    "sensor_centers": center_loss.get_sensor_centers().cpu(),
                    "separations": separations,
                    "sensor_names": SENSOR_COLS,
                    "window_size": window_size,
                    "embed_dim": EMBED_DIM,
                    "top_k": TOP_K,
                    "hidden_dim": HIDDEN_DIM,
                    "sensor_embeddings": model.sensor_embeddings.data.cpu(),
                    "lambda_center": lambda_center,
                    "lambda_global": lambda_global,
                    "epoch": epoch + 1,
                    "best_val_loss": val_total_loss,
                    "stage": 2,
                    "model_type": final_model_type,
                },
                best_checkpoint_path,
            )
            print(f"  ✓ Best model saved (Val Loss: {val_total_loss:.4f})")
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
    center_loss.load_state_dict(checkpoint["center_loss_state_dict"])

    print(f"\n{'=' * 80}")
    print(f"Stage 2 multi-level training complete!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Window separation: {checkpoint['separations']['window_separation']:.4f}")
    print(f"Sensor mean separation: {checkpoint['separations']['sensor_mean_separation']:.4f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"{'=' * 80}\n")

    return model, center_loss


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Multi-Level Center Loss Training"
    )
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 checkpoint",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_center", type=float, default=LAMBDA_CENTER, help="Center loss weight"
    )
    parser.add_argument(
        "--lambda_global", type=float, default=LAMBDA_GLOBAL, help="Global loss weight"
    )
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
        help="Model type: 'enhanced' (full features) or 'kag_optimized' (fast baseline with essential fixes). "
             "Will auto-detect from Stage 1 checkpoint if not specified.",
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

    # Load data (same preprocessing as Stage 1)
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
        data, id_col=ID_COL, min_length=WINDOW_SIZE + FORECAST_HORIZON
    )
    data = add_cross_channel_features(data)
    print("Added cross-channel features")

    # Sort data
    data = data.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    # Split by drive (70/15/15) - same as Stage 1
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

    # Build clean windows
    print("\nBuilding clean windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )

    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")

    # Inject faults with sensor-level labels (needed for supervised learning)
    print("\nInjecting faults with sensor-level labels (stratified distribution)...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = (
        inject_faults_with_sensor_labels(
            X_train,
            y_train,
            SENSOR_COLS,
            fault_percentage=0.15,
            random_state=42,
            use_stratified=True,
        )
    )
    X_val_sensor, _, val_sensor_labels, val_window_labels = (
        inject_faults_with_sensor_labels(
            X_val,
            y_val,
            SENSOR_COLS,
            fault_percentage=0.15,
            random_state=43,
            use_stratified=True,
        )
    )

    # Statistics
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()

    print(f"\nTrain: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")

    # Create dataloaders with optimizations
    train_ds = TensorDataset(
        X_train_sensor, y_train, train_sensor_labels, train_window_labels
    )
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)

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
    model, center_loss = train_stage2_multilevel(
        train_loader,
        val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        stage1_checkpoint_path=args.stage1_checkpoint,
        num_epochs=args.epochs,
        device=device,
        lambda_center=args.lambda_center,
        lambda_global=args.lambda_global,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=args.checkpoint_dir,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        model_type=args.model,
    )

    print("✓ Stage 2 multi-level training complete!")


if __name__ == "__main__":
    main()
