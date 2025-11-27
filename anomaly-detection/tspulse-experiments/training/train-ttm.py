# %%
# Install the tsfm library
! pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"

# %%
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    ForecastDFDataset,
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
    count_parameters,
)
import tempfile
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit.visualization import plot_predictions

# %%
context_length = 512
forecast_length = 96

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
OUT_DIR = "ttm-finetune"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# %%
path = "/Users/darenpalmer/Desktop/UCL/CS/fyp.nosync/data/carOBD/obdiidata" 
time_col = 'ENGINE_RUN_TINE ()'

df_list = []
for file in os.listdir(path):
  if file.endswith('.csv'):
    df = pd.read_csv(f'{path}/{file}', index_col=False)
    df['drive_id'] = file
    df_list.append(df)

print(f'{len(df_list)} files loaded out of {len([f for f in os.listdir(path) if f.endswith(".csv")])}')


# %%
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

# %%
def mean_fill_missing_timestamps_and_remove_duplicates(df: pd.DataFrame, id_cols: list[str] = None) -> pd.DataFrame:
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


# %%
def downsample(df, time_col, source_file_col, downsample_factor=5):
    """
    Improved downsampling with better NaN handling.
    """
    result_dfs = []
    skipped = []
    
    for source_file in df[source_file_col].unique():
        file_df = df[df[source_file_col] == source_file].copy()
        
        if len(file_df) < downsample_factor * 2:
            skipped.append((source_file, 'too_short', len(file_df)))
            continue
        
        file_df = file_df.sort_values(time_col).reset_index(drop=True)
        
        # Check for columns that are already all NaN
        all_nan_cols = file_df.columns[file_df.isna().all()].tolist()
        if all_nan_cols:
            # Remove these columns for this drive
            file_df = file_df.drop(columns=all_nan_cols)
            print(f"⚠️  Drive {source_file}: removed all-NaN columns: {all_nan_cols}")
        
        smoothed = file_df.copy()
        
        for col in smoothed.columns:
            if col in [time_col, source_file_col]:
                continue
            
            if not pd.api.types.is_numeric_dtype(smoothed[col]):
                continue
            
            col_upper = col.upper()
            
            # Determine window size
            if any(kw in col_upper for kw in ['TEMPERATURE', 'COOLANT', 'CATALYST']):
                window = min(20, len(file_df) // 3)
            elif any(kw in col_upper for kw in ['RPM', 'SPEED', 'THROTTLE']):
                window = min(3, len(file_df) // 5)
            elif any(kw in col_upper for kw in ['PRESSURE']):
                window = min(10, len(file_df) // 4)
            else:
                window = min(5, len(file_df) // 4)
            
            if window >= 2:
                # Use min_periods=1 to avoid creating NaNs
                smoothed[col] = smoothed[col].rolling(
                    window=window,
                    center=True,
                    min_periods=1  # KEY: Always produce a value
                ).mean()
        
        # Decimate
        downsampled = smoothed.iloc[::downsample_factor].copy()
        downsampled[time_col] = np.arange(len(downsampled)) * downsample_factor
        
        # Forward fill any remaining NaNs
        downsampled = downsampled.fillna(method='ffill').fillna(method='bfill')
        
        # Final check: ensure no all-NaN columns
        all_nan_after = downsampled.columns[downsampled.isna().all()].tolist()
        if all_nan_after:
            print(f"⚠️  Drive {source_file}: all-NaN after processing: {all_nan_after}")
            # Drop these columns
            downsampled = downsampled.drop(columns=all_nan_after)
        
        # Only keep if we still have target column
        if 'COOLANT_TEMPERATURE ()' not in downsampled.columns:
            skipped.append((source_file, 'missing_target', len(file_df)))
            continue
        
        result_dfs.append(downsampled.reset_index(drop=True))
    
    if len(result_dfs) == 0:
        raise ValueError(f"No valid drives! Skipped: {skipped}")
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    print(f"\nDownsampling: {len(df):,} → {len(result):,} samples ({len(result_dfs)} drives)")
    if skipped:
        print(f"Skipped {len(skipped)} drives: {dict(pd.Series([r for _, r, _ in skipped]).value_counts())}")
    
    return result


# %%
data = pd.concat(df_list, ignore_index=True)

# Clean up

print(f"Total samples: {len(data):,}")
print(f"Unique drives: {data['drive_id'].nunique()}")

data = mean_fill_missing_timestamps_and_remove_duplicates(data, id_cols=["drive_id"])
data = remove_zero_variance_columns(data, exclude_cols=["drive_id"])
data_downsampled = downsample(
    data,
    time_col='ENGINE_RUN_TINE ()',
    source_file_col='drive_id',
    downsample_factor=5
)

# %%
from tsfm_public import get_datasets

target_columns = ['COOLANT_TEMPERATURE ()']

split_params = {"train": 0.75, "test": 0.25}

column_specifiers = {
    "timestamp_column": time_col,
    "id_columns": ["drive_id"],
    "target_columns": target_columns,
    "conditional_columns": [col for col in data.columns if col not in target_columns + ["drive_id", time_col]],
}

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

tsp = tsp.train(data)

train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp,
    data,
    split_params
)

# %%
from tsfm_public import get_model

zeroshot_model = get_model(
    TTM_MODEL_PATH,
    context_length=context_length,
    prediction_length=forecast_length,
    prediction_channel_indices=tsp.prediction_channel_indices,
    num_input_channels=tsp.num_input_channels,
)

# %%
temp_dir = tempfile.mkdtemp()
zeroshot_trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=64,
    ),
)

# %%
zeroshot_trainer.evaluate(test_dataset)

# %%
plot_predictions(
    model=zeroshot_trainer.model,
    dset=test_dataset,
    plot_prefix="test_zeroshot",
    channel=0,
)

# %%
finetune_forecast_model = get_model(
    TTM_MODEL_PATH,
    context_length=context_length,
    prediction_length=forecast_length,
    num_input_channels=tsp.num_input_channels,
    decoder_mode="mix_channel",
    prediction_channel_indices=tsp.prediction_channel_indices,
)

for param in finetune_forecast_model.backbone.parameters():
    param.requires_grad = False

# %%
num_epochs = 100
batch_size = 64

learning_rate, finetune_forecast_model = optimal_lr_finder(
    finetune_forecast_model,
    train_dataset,
    batch_size=batch_size,
    device=device,
    enable_prefix_tuning=False,
)
print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

# %%
print(f"Using learning rate = {learning_rate}")
finetune_forecast_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "output"),
    overwrite_output_dir=True,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    do_eval=True,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=8,
    report_to=None,
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
    greater_is_better=False,  # For loss
)

# Create the early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
)
tracking_callback = TrackingCallback()

# Optimizer and scheduler
optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
scheduler = OneCycleLR(
    optimizer,
    learning_rate,
    epochs=num_epochs,
    steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
)

finetune_forecast_trainer = Trainer(
    model=finetune_forecast_model,
    args=finetune_forecast_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[early_stopping_callback, tracking_callback],
    optimizers=(optimizer, scheduler),
)

# Fine tune
finetune_forecast_trainer.train()

# %%
finetune_forecast_trainer.evaluate(test_dataset)

# %%
plot_predictions(
    model=finetune_forecast_trainer.model,
    dset=test_dataset,
    plot_prefix="test_finetune",
    channel=0,
    num_plots=30
)

# %%



