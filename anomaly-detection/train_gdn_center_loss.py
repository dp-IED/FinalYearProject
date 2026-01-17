#!/usr/bin/env python3
"""
Training script for MultiLabelGDN with Center Loss

Replicates the exact data preprocessing and training logic from gdn.ipynb.
Saves model as best_center_loss_gdn.pt
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pathlib import Path

# Add path for model import
sys.path.insert(0, str(Path(__file__).parent))
from models.gdn_model import MultiLabelGDN

torch.set_default_dtype(torch.float32)

# ============================================================================
# Constants (matching notebook exactly)
# ============================================================================
DATA_PATH = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata"
SENSOR_COLS = [
    'ENGINE_RPM ()',                    # correlates with speed, load
    'VEHICLE_SPEED ()',                 # VSS glitch detection
    'THROTTLE ()',                      # TPS drift, throttle-load correlation
    'ENGINE_LOAD ()',                   # pair with throttle for mismatch
    'COOLANT_TEMPERATURE ()',           # intermittent sensor faults
    'INTAKE_MANIFOLD_PRESSURE ()',      # MAF/MAP drift detection
    'SHORT_TERM_FUEL_TRIM_BANK_1 ()',  # critical for air/fuel faults
    'LONG_TERM_FUEL_TRIM_BANK_1 ()',   # detects persistent issues
]

ID_COL = "drive_id"
TIME_COL = 'ENGINE_RUN_TINE ()'
WINDOW_SIZE = 300
FORECAST_HORIZON = 1

# Training hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_GLOBAL = 0.3
LAMBDA_CENTER = 0.1
CENTER_LR = 0.5

# ============================================================================
# Data Preprocessing Functions (exact from notebook)
# ============================================================================

def remove_zero_variance_columns(df: pd.DataFrame, exclude_cols: list[str] = None) -> pd.DataFrame:
    """
    Compute std of each std-computable column (numeric only)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [col for col in numeric_cols if col not in exclude_cols]
  
    std_df = df[cols_to_check].std()
    zero_variance_cols = std_df[std_df == 0].index.tolist()
  
    print(f'{len(zero_variance_cols)} columns with zero variance: {zero_variance_cols}')
  
    if len(zero_variance_cols) > 0:
        df = df.drop(columns=zero_variance_cols)
  
    return df


def mean_fill_missing_timestamps_and_remove_duplicates(
    df: pd.DataFrame, 
    time_col: str, 
    id_cols: list[str] = None
) -> pd.DataFrame:
    """
    Remove duplicate timestamps by averaging all numeric columns for each unique timestamp.
    This preserves the overall statistics while removing duplicate entries.
    
    Note: The time column itself is not averaged (it becomes the group key).
    Only numeric columns are averaged when multiple rows share the same timestamp.
    """
    if id_cols is None:
        id_cols = []
    
    existing_id_cols = [col for col in id_cols if col in df.columns]
    
    group_cols = [time_col] + existing_id_cols
    
    agg_dict = {}
    for col in df.columns:
        if col not in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'
  
    df_clean = df.groupby(group_cols, as_index=False).agg(agg_dict)
  
    return df_clean


def downsample(df, time_col, source_file_col, downsample_factor=2):
    """Downsample data by decimation"""
    result_dfs = []
    
    for source_file in df[source_file_col].unique():
        file_df = df[df[source_file_col] == source_file].copy()
        
        if len(file_df) < downsample_factor * 2:
            continue
        
        file_df = file_df.sort_values(time_col).reset_index(drop=True)
        
        # Simple decimation without pre-smoothing
        downsampled = file_df.iloc[::downsample_factor].copy()
        downsampled[time_col] = np.arange(len(downsampled)) * downsample_factor
        
        result_dfs.append(downsampled.reset_index(drop=True))
    
    return pd.concat(result_dfs, ignore_index=True)


def filter_long_drives(df, id_col='drive_id', min_length=608):
    """Keep only drives long enough for your context window"""
    drive_lengths = df.groupby(id_col).size()
    valid_drives = drive_lengths[drive_lengths >= min_length].index
    
    print(f"Keeping {len(valid_drives)}/{df[id_col].nunique()} drives")
    print(f"Dropped {len(df) - df[df[id_col].isin(valid_drives)].shape[0]} timesteps")
    
    return df[df[id_col].isin(valid_drives)].reset_index(drop=True)


def add_cross_channel_features(data):
    """
    Engineer features that capture cross-channel relationships.
    Add these as conditional columns.
    """
    # RPM-to-Speed ratio (gear indicator)
    if 'ENGINE_RPM ()' in data.columns and 'VEHICLE_SPEED ()' in data.columns:
        data['RPM_SPEED_RATIO'] = data['ENGINE_RPM ()'] / (data['VEHICLE_SPEED ()'] + 1)
    
    # Throttle-to-Load ratio (efficiency indicator)
    if 'THROTTLE ()' in data.columns and 'ENGINE_LOAD ()' in data.columns:
        data['THROTTLE_LOAD_RATIO'] = data['THROTTLE ()'] / (data['ENGINE_LOAD ()'] + 1)
    
    # Speed-based categories
    if 'VEHICLE_SPEED ()' in data.columns:
        data['IS_IDLE'] = (data['VEHICLE_SPEED ()'] < 5).astype(float)
        data['IS_HIGHWAY'] = (data['VEHICLE_SPEED ()'] > 60).astype(float)
    
    # RPM acceleration
    if 'ENGINE_RPM ()' in data.columns:
        data['RPM_ACCEL'] = data.groupby('drive_id')['ENGINE_RPM ()'].diff().fillna(0)
    
    return data


def build_clean_windows(df, sensor_cols, id_col, time_col, window_size, horizon=1, scaler=None):
    """Build windows from CLEAN data only. Returns normalized windows."""
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + sensor_cols].copy()
    
    # Normalize BEFORE windowing
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


# ============================================================================
# Fault Injection Function (exact from notebook)
# ============================================================================

def inject_faults_with_sensor_labels(X_windows, y_windows, sensor_cols, 
                                     fault_percentage=0.30, random_state=42):
    """
    Inject faults and return SENSOR-LEVEL labels.
    
    Returns:
    - X_faulty: (N, W, D) window data with injected faults
    - y_windows: (N, D) unchanged target values
    - sensor_labels: (N, D) binary matrix - 1 if sensor i is faulty in window j
    - window_labels: (N,) binary - 1 if any fault exists in window
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
        
        fault_type = np.random.choice([
            'vss_dropout', 'maf_scale_low', 'coolant_dropout', 
            'tps_stuck', 'rpm_speed_decouple'
        ], p=[0.35, 0.25, 0.20, 0.10, 0.10])
        
        affected_sensors = []
        
        # ===== FAULT 1: VSS DROPOUT =====
        if fault_type == 'vss_dropout' and 'VEHICLE_SPEED ()' in pid_idx:
            speed_i = pid_idx['VEHICLE_SPEED ()']
            if win[:, speed_i].mean() > 0.15:
                start = int(W * 0.30)
                end = int(W * 0.70)
                win[start:end, speed_i] = 0.0
                win[start:end, speed_i] += np.random.uniform(0, 0.02, end-start)
                affected_sensors.append(speed_i)
        
        # ===== FAULT 2: MAF SCALE LOW =====
        elif fault_type == 'maf_scale_low' and 'INTAKE_MANIFOLD_PRESSURE ()' in pid_idx:
            map_i = pid_idx['INTAKE_MANIFOLD_PRESSURE ()']
            scale_factor = np.random.uniform(0.75, 0.80)
            win[:, map_i] = win[:, map_i] * scale_factor
            affected_sensors.append(map_i)
            
            if 'SHORT_TERM_FUEL_TRIM_BANK_1 ()' in pid_idx:
                stft_i = pid_idx['SHORT_TERM_FUEL_TRIM_BANK_1 ()']
                win[:, stft_i] = np.clip(win[:, stft_i] + 0.15, 0.0, 1.0)
                affected_sensors.append(stft_i)
        
        # ===== FAULT 3: COOLANT DROPOUT =====
        elif fault_type == 'coolant_dropout' and 'COOLANT_TEMPERATURE ()' in pid_idx:
            cool_i = pid_idx['COOLANT_TEMPERATURE ()']
            if win[:, cool_i].mean() > 0.5:
                n_dropouts = np.random.randint(2, 4)
                for _ in range(n_dropouts):
                    drop_start = np.random.randint(0, W - 60)
                    drop_len = np.random.randint(30, 60)
                    win[drop_start:drop_start + drop_len, cool_i] = np.random.uniform(0.05, 0.15)
                affected_sensors.append(cool_i)
        
        # ===== FAULT 4: TPS STUCK =====
        elif fault_type == 'tps_stuck' and 'THROTTLE ()' in pid_idx:
            thr_i = pid_idx['THROTTLE ()']
            freeze_point = W // 2
            stuck_value = win[freeze_point, thr_i]
            if stuck_value > 0.15 and win[:freeze_point, thr_i].std() > 0.05:
                win[freeze_point:, thr_i] = stuck_value
                affected_sensors.append(thr_i)
        
        # ===== FAULT 5: RPM-SPEED DECOUPLE =====
        elif fault_type == 'rpm_speed_decouple':
            if 'ENGINE_RPM ()' in pid_idx and 'VEHICLE_SPEED ()' in pid_idx:
                speed_i = pid_idx['VEHICLE_SPEED ()']
                rpm_i = pid_idx['ENGINE_RPM ()']
                if win[:, speed_i].mean() > 0.20 and win[:, rpm_i].mean() > 0.30:
                    start = int(W * 0.25)
                    end = int(W * 0.75)
                    win[start:end, speed_i] = win[start:end, speed_i] * np.random.uniform(0.3, 0.5)
                    affected_sensors.append(speed_i)
        
        # Update labels if fault was applied
        if len(affected_sensors) > 0:
            X_faulty[idx] = torch.tensor(win, dtype=torch.float32)
            window_labels[idx] = 1
            for sensor_i in affected_sensors:
                sensor_labels[idx, sensor_i] = 1.0
    
    return X_faulty, y_windows, sensor_labels, window_labels


# ============================================================================
# Center Loss Class (exact from notebook)
# ============================================================================

class CenterLoss(nn.Module):
    '''
    Center Loss for anomaly detection.
    Learns separate centers for normal (class 0) and anomalous (class 1) samples.
    Pulls samples toward their class center, creating better separation.
    
    Reference:
    Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition"
    '''
    def __init__(self, embed_dim, num_classes=2, alpha=0.5):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.alpha = alpha  # Learning rate for center updates (used in SGD optimizer)
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings, labels):
        '''
        Compute center loss: pull embeddings toward their class center.
        
        Args:
            embeddings: (B, D) normalized embeddings from model
            labels: (B,) binary labels (0=normal, 1=anomalous)
        
        Returns:
            loss: scalar tensor
        '''
        batch_size = embeddings.size(0)
        
        # Get the center assigned to each sample
        centers_batch = self.centers.index_select(0, labels.long())  # (B, D)
        
        # Compute distance to assigned centers
        loss = (embeddings - centers_batch).pow(2).sum(dim=1).mean()
        
        return loss


# ============================================================================
# Training Function (exact from notebook)
# ============================================================================

def train_multilabel_with_center_loss(train_loader, val_loader, num_sensors, window_size,
                                      num_epochs=30, device='cpu', lambda_center=0.1):
    '''
    Train with BCE + Center Loss for better separation of normal/anomalous embeddings.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_sensors: Number of sensor channels
        window_size: Temporal window size
        num_epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        lambda_center: Weight for center loss (tune: 0.05-0.2)
    
    Returns:
        model: Trained MultiLabelGDN model
        center_loss: Trained CenterLoss module
    '''
    
    model = MultiLabelGDN(num_nodes=num_sensors, window_size=window_size,
                          embed_dim=32, top_k=3, hidden_dim=32).to(device)
    
    # Initialize center loss (hidden_dim=32 from model)
    center_loss = CenterLoss(embed_dim=32, num_classes=2, alpha=0.5).to(device)
    
    # Separate optimizers for model and centers
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_center = torch.optim.SGD(center_loss.parameters(), lr=CENTER_LR)
    
    # Loss criteria
    sensor_criterion = nn.BCELoss(reduction='none')
    global_criterion = nn.BCELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*80}")
    print("Training Multi-Label GDN with Center Loss")
    print(f"{'='*80}")
    print(f"Lambda_global: {LAMBDA_GLOBAL}, Lambda_center: {lambda_center}")
    print(f"Device: {device}\n")
    
    for epoch in range(num_epochs):
        model.train()
        center_loss.train()
        
        train_loss_sensor = 0.0
        train_loss_global = 0.0
        train_loss_center = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False) as pbar:
            for X_batch, _, sensor_labels_batch, window_labels_batch in pbar:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                # Forward pass
                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)  # (B, hidden_dim)
                
                # Classification losses
                loss_sensor = sensor_criterion(sensor_probs, sensor_labels_batch).mean()
                loss_global = global_criterion(global_prob, window_labels_batch.float())
                
                # Center loss (pulls embeddings to class centers)
                loss_center_val = center_loss(embeddings, window_labels_batch)
                
                # Combined loss
                loss = loss_sensor + LAMBDA_GLOBAL * loss_global + lambda_center * loss_center_val
                
                loss.backward()
                
                # Update both model and centers
                optimizer.step()
                optimizer_center.step()
                
                train_loss_sensor += loss_sensor.item() * X_batch.size(0)
                train_loss_global += loss_global.item() * X_batch.size(0)
                train_loss_center += loss_center_val.item() * X_batch.size(0)
                
                pbar.update(1)
        
        train_loss_sensor /= len(train_loader.dataset)
        train_loss_global /= len(train_loader.dataset)
        train_loss_center /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        center_loss.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, _, sensor_labels_batch, window_labels_batch in val_loader:
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)
                
                sensor_probs, global_prob = model(X_batch, return_global=True)
                embeddings = model.get_embeddings(X_batch)
                
                loss_sensor = sensor_criterion(sensor_probs, sensor_labels_batch).mean()
                loss_global = global_criterion(global_prob, window_labels_batch.float())
                loss_center_val = center_loss(embeddings, window_labels_batch)
                
                loss = loss_sensor + LAMBDA_GLOBAL * loss_global + lambda_center * loss_center_val
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        # Print debug info about center separation
        center_dist = torch.norm(center_loss.centers[0] - center_loss.centers[1]).item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Sensor: {train_loss_sensor:.4f} | "
              f"Global: {train_loss_global:.4f} | "
              f"Center: {train_loss_center:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"Center_dist: {center_dist:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'center_loss': center_loss.state_dict()
            }, "best_center_loss_gdn.pt")
            print("  ✓ Best model saved")
    
    # Load best checkpoint
    checkpoint = torch.load("best_center_loss_gdn.pt")
    model.load_state_dict(checkpoint['model'])
    center_loss.load_state_dict(checkpoint['center_loss'])
    model.eval()
    
    print(f"\n✓ Model saved to: best_center_loss_gdn.pt\n")
    
    return model, center_loss


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    print("="*80)
    print("GDN Training Script with Center Loss")
    print("="*80)
    
    # ========================================================================
    # 1. Load and preprocess data
    # ========================================================================
    print("\n1. Loading data...")
    df_list = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(f'{DATA_PATH}/{file}', index_col=False)
            df['drive_id'] = file
            df_list.append(df)
    
    print(f'{len(df_list)} files loaded out of {len([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])}')
    
    # Combine all dataframes
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")
    print(f"Unique drives: {data['drive_id'].nunique()}")
    
    # Remove useless columns
    data = data.drop(columns=['WARM_UPS_SINCE_CODES_CLEARED ()', 'TIME_SINCE_TROUBLE_CODES_CLEARED ()'])
    
    # Preprocessing steps
    print("\n2. Preprocessing data...")
    data = mean_fill_missing_timestamps_and_remove_duplicates(data, time_col=TIME_COL, id_cols=["drive_id"])
    data = remove_zero_variance_columns(data, exclude_cols=["drive_id"])
    data = downsample(data, time_col=TIME_COL, source_file_col='drive_id', downsample_factor=1)
    data = filter_long_drives(data, min_length=WINDOW_SIZE + FORECAST_HORIZON)
    
    # Add cross-channel features
    data = add_cross_channel_features(data)
    print("Added cross-channel features")
    
    # Sort data
    data = data.sort_values(["drive_id", TIME_COL]).reset_index(drop=True)
    
    # ========================================================================
    # 2. Train/Val/Test split
    # ========================================================================
    print("\n3. Splitting data...")
    unique_drives = data['drive_id'].unique()
    n_drives = len(unique_drives)
    
    train_drives = unique_drives[:int(0.70 * n_drives)]
    val_drives   = unique_drives[int(0.70 * n_drives):int(0.85 * n_drives)]
    test_drives  = unique_drives[int(0.85 * n_drives):]
    
    print(f"Train drives: {len(train_drives)}, Val drives: {len(val_drives)}, Test drives: {len(test_drives)}")
    
    train_data = data[data['drive_id'].isin(train_drives)].copy()
    val_data   = data[data['drive_id'].isin(val_drives)].copy()
    test_data  = data[data['drive_id'].isin(test_drives)].copy()
    
    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")
    
    # ========================================================================
    # 3. Build windows
    # ========================================================================
    print("\n4. Building windows...")
    X_train, y_train, scaler_train = build_clean_windows(
        train_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=None
    )
    
    X_val, y_val, _ = build_clean_windows(
        val_data, SENSOR_COLS, ID_COL, TIME_COL, WINDOW_SIZE, scaler=scaler_train
    )
    
    print(f"Clean train windows: {len(X_train)}")
    print(f"Clean val windows: {len(X_val)}")
    
    # ========================================================================
    # 4. Inject faults
    # ========================================================================
    print("\n5. Injecting faults with sensor-level labels...")
    X_train_sensor, _, train_sensor_labels, train_window_labels = inject_faults_with_sensor_labels(
        X_train, y_train, SENSOR_COLS, fault_percentage=0.15, random_state=42
    )
    
    X_val_sensor, _, val_sensor_labels, val_window_labels = inject_faults_with_sensor_labels(
        X_val, y_val, SENSOR_COLS, fault_percentage=0.15, random_state=43
    )
    
    train_faulty = (train_sensor_labels.sum(dim=1) > 0).sum().item()
    val_faulty = (val_sensor_labels.sum(dim=1) > 0).sum().item()
    
    print(f"Train: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"  Avg sensors per fault: {train_sensor_labels[train_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}")
    print(f"Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")
    print(f"  Avg sensors per fault: {val_sensor_labels[val_sensor_labels.sum(dim=1) > 0].sum(dim=1).mean():.2f}")
    
    # ========================================================================
    # 5. Create DataLoaders
    # ========================================================================
    print("\n6. Creating DataLoaders...")
    train_ds = TensorDataset(X_train_sensor, y_train, train_sensor_labels, train_window_labels)
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    num_sensors = len(SENSOR_COLS)
    print(f"Train windows: {len(train_ds)}, Sensors: {num_sensors}")
    
    # ========================================================================
    # 6. Train model
    # ========================================================================
    print("\n7. Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, center_loss_module = train_multilabel_with_center_loss(
        train_loader, val_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        num_epochs=NUM_EPOCHS,
        device=device,
        lambda_center=LAMBDA_CENTER
    )
    
    print("\n" + "="*80)
    print("✓ Training completed successfully!")
    print(f"✓ Model saved to: best_center_loss_gdn.pt")
    print("="*80)


if __name__ == '__main__':
    main()
