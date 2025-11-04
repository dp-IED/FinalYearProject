"""
PyTorch data loader for time series sequences.
Converts flat feature arrays to sliding window sequences for deep learning models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path to import ML loader
# Get the project root (two levels up from utils/)
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
ml_path = os.path.join(project_root, 'anomaly-detection', 'ml')
if ml_path not in sys.path:
    sys.path.insert(0, ml_path)
from ml_anomaly_detector import CarOBDMLDataLoader


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, sequences: np.ndarray, labels: Optional[np.ndarray] = None, scaler=None):
        """
        Initialize dataset.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            labels: Optional array of shape (n_samples,) for sequence labels
            scaler: Optional scaler to normalize sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.scaler = scaler
        
        if scaler is not None:
            # Reshape for scaling: (n_samples * seq_len, n_features)
            n_samples, seq_len, n_features = sequences.shape
            reshaped = sequences.reshape(-1, n_features)
            scaled = scaler.transform(reshaped)
            self.sequences = torch.FloatTensor(scaled.reshape(n_samples, seq_len, n_features))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


class SequenceDataLoader:
    """
    Converts flat feature arrays to time series sequences for deep learning models.
    Integrates with existing CarOBDMLDataLoader for data loading and fault injection.
    """
    
    def __init__(self, sequence_length: int = 30, overlap: bool = True, step_size: int = 1):
        """
        Initialize sequence data loader.
        
        Args:
            sequence_length: Length of each sequence window (default: 30)
            overlap: Whether to use overlapping windows (default: True)
            step_size: Step size for sliding window (default: 1, only used if overlap=True)
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = step_size if overlap else sequence_length
    
    def create_sequences(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert flat feature array to sequences using sliding windows.
        
        Args:
            data: Flat feature array of shape (n_samples, n_features)
            labels: Optional labels of shape (n_samples,) for each sample
            
        Returns:
            Tuple of (sequences, sequence_labels) where:
            - sequences: Array of shape (n_sequences, sequence_length, n_features)
            - sequence_labels: Array of shape (n_sequences,) - 1 if ANY timestep in sequence is anomalous
        """
        n_samples, n_features = data.shape
        
        if n_samples < self.sequence_length:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length {self.sequence_length}")
        
        sequences = []
        sequence_labels = [] if labels is not None else None
        
        # Create sliding windows
        for i in range(0, n_samples - self.sequence_length + 1, self.step_size):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Label sequence as anomalous if ANY timestep in sequence is anomalous
            if labels is not None:
                seq_label = int(np.max(labels[i:i + self.sequence_length]) > 0)
                sequence_labels.append(seq_label)
        
        sequences = np.array(sequences)
        if sequence_labels is not None:
            sequence_labels = np.array(sequence_labels)
        
        return sequences, sequence_labels
    
    def load_data_from_ml_loader(
        self,
        ml_loader: CarOBDMLDataLoader,
        data_path: str = "data/carOBD/obdiidata",
        validation_split: float = 0.2,
        test_split: float = 0.1,
        fault_percentage: float = 0.0,
        return_raw_features: bool = False
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load data using existing CarOBDMLDataLoader and convert to sequences.
        
        Args:
            ml_loader: CarOBDMLDataLoader instance
            data_path: Path to data directory
            validation_split: Fraction of data for validation
            test_split: Fraction of data for test
            fault_percentage: Percentage of faults to inject (0.0 for training, >0 for evaluation)
            return_raw_features: If True, also return raw features for manual splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) or 
            if return_raw_features: (train_dataset, val_dataset, test_dataset, train_features, val_features, test_features)
        """
        # Load all data
        idle_data, motion_data = ml_loader.load_all_data()
        all_data = pd.concat([idle_data, motion_data], ignore_index=True)
        
        # Extract features
        features = ml_loader.extract_features(all_data)
        
        # CRITICAL FIX: For time series, we must preserve temporal order!
        # If returning raw features, split WITHOUT shuffling (preserve temporal order)
        # Sequences will be created later in training script, preserving temporal structure
        
        if return_raw_features:
            # For raw features: split temporally (no shuffling) to preserve sequence structure
            # This allows sequences created later to have proper temporal relationships
            n_samples = len(features)
            n_val = int(n_samples * validation_split)
            n_test = int(n_samples * test_split)
            n_train = n_samples - n_val - n_test
            
            # Split temporally (preserve order)
            train_features = features[:n_train]
            val_features = features[n_train:n_train + n_val]
            test_features = features[n_train + n_val:]
            
            # Create temporary sequences for dataset return (won't be used, but needed for signature)
            # These will be recreated in training script from scaled features
            train_sequences, _ = self.create_sequences(train_features, labels=None)
            val_sequences, _ = self.create_sequences(val_features, labels=None)
            test_sequences, _ = self.create_sequences(test_features, labels=None)
            
            train_dataset = TimeSeriesDataset(train_sequences, labels=None)
            val_dataset = TimeSeriesDataset(val_sequences, labels=None)
            test_dataset = TimeSeriesDataset(test_sequences, labels=None)
            
            return train_dataset, val_dataset, test_dataset, train_features, val_features, test_features
        
        # If not returning raw features, create sequences first, then shuffle sequences
        # Create sequences from ALL data first (preserves temporal structure within sequences)
        all_sequences, _ = self.create_sequences(features, labels=None)
        
        # Now shuffle SEQUENCES (not samples) and split
        n_sequences = len(all_sequences)
        sequence_indices = np.arange(n_sequences)
        np.random.seed(42)
        np.random.shuffle(sequence_indices)
        
        n_val = int(n_sequences * validation_split)
        n_test = int(n_sequences * test_split)
        n_train = n_sequences - n_val - n_test
        
        train_seq_indices = sequence_indices[:n_train]
        val_seq_indices = sequence_indices[n_train:n_train + n_val]
        test_seq_indices = sequence_indices[n_train + n_val:]
        
        train_sequences = all_sequences[train_seq_indices]
        val_sequences = all_sequences[val_seq_indices]
        test_sequences = all_sequences[test_seq_indices]
        
        # Handle fault injection if needed
        train_seq_labels = None
        val_seq_labels = None
        test_seq_labels = None
        
        if fault_percentage > 0:
            # Reconstruct feature arrays from sequences (use last timestep of each sequence)
            val_features = val_sequences[:, -1, :]
            test_features = test_sequences[:, -1, :]
            
            val_features_fault, val_labels = ml_loader.create_realistic_fault_data(
                val_features, fault_percentage=fault_percentage
            )
            test_features_fault, test_labels = ml_loader.create_realistic_fault_data(
                test_features, fault_percentage=fault_percentage
            )
            
            # Reconstruct sequences with fault-injected last timestep
            val_sequences_fault = val_sequences.copy()
            val_sequences_fault[:, -1, :] = val_features_fault
            test_sequences_fault = test_sequences.copy()
            test_sequences_fault[:, -1, :] = test_features_fault
            
            val_sequences = val_sequences_fault
            test_sequences = test_sequences_fault
            val_seq_labels = val_labels
            test_seq_labels = test_labels
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_sequences, labels=None)
        val_dataset = TimeSeriesDataset(val_sequences, labels=val_seq_labels)
        test_dataset = TimeSeriesDataset(test_sequences, labels=test_seq_labels)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int = 64,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size for training
            shuffle_train: Whether to shuffle training data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader, test_loader
    
    def load_sequences_from_features(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Direct conversion of features to sequences (for external use).
        
        Args:
            features: Flat feature array of shape (n_samples, n_features)
            labels: Optional labels of shape (n_samples,)
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        return self.create_sequences(features, labels)

