"""
CNN-LSTM hybrid model for anomaly detection.
Combines spatial feature extraction (CNN) with temporal modeling (LSTM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for anomaly detection.
    Uses 1D CNN for spatial feature extraction followed by LSTM for temporal modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        cnn_filters: list = [32, 64],
        cnn_kernel_sizes: list = [3, 3],
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features per timestep
            cnn_filters: List of filter sizes for CNN layers (default: [32, 64])
            cnn_kernel_sizes: List of kernel sizes for CNN layers (default: [3, 3])
            lstm_hidden_size: Number of hidden units in LSTM (default: 64)
            lstm_num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            use_attention: Whether to use attention mechanism (default: False)
        """
        super(CNNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.cnn_filters = cnn_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # CNN layers for spatial feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_size
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2  # Same padding
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM layers for temporal modeling
        # Input to LSTM is the output channels from CNN
        lstm_input_size = cnn_filters[-1]
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism (optional)
        if use_attention:
            lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                batch_first=True
            )
        
        # Output layer
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Anomaly scores of shape (batch, 1)
        """
        batch_size, seq_len, n_features = x.shape
        
        # CNN expects (batch, channels, sequence_length)
        # Reshape: (batch, sequence_length, input_size) -> (batch, input_size, sequence_length)
        x = x.transpose(1, 2)  # (batch, input_size, sequence_length)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)  # (batch, out_channels, sequence_length)
        
        # Reshape back for LSTM: (batch, channels, sequence_length) -> (batch, sequence_length, channels)
        x = x.transpose(1, 2)  # (batch, sequence_length, cnn_output_channels)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, sequence_length, hidden_size)
        
        # Apply attention if enabled
        if self.use_attention:
            lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Fully connected layer for anomaly score
        anomaly_score = self.fc(last_output)  # (batch, 1)
        
        return anomaly_score
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate anomaly scores for input sequences.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Anomaly scores as numpy array of shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
            return scores.cpu().numpy().flatten()
    
    def get_anomaly_scores(self, sequences: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        """
        Get anomaly scores for sequences (numpy input).
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            device: Device to run inference on (default: CPU)
            
        Returns:
            Anomaly scores of shape (n_samples,)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        self.eval()
        
        # Convert to tensor
        x = torch.FloatTensor(sequences).to(device)
        
        # Batch processing for large datasets
        batch_size = 64
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i:i + batch_size]
                batch_scores = self.forward(batch)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores, axis=0).flatten()
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model and metadata.
        
        Args:
            filepath: Path to save model (without extension)
            metadata: Optional metadata dictionary to save
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), f"{filepath}.pth")
        
        # Save metadata
        model_metadata = {
            'input_size': self.input_size,
            'cnn_filters': self.cnn_filters,
            'cnn_kernel_sizes': self.cnn_kernel_sizes,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'model_type': 'cnn_lstm'
        }
        
        if metadata is not None:
            model_metadata.update(metadata)
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"Model saved to {filepath}.pth")
        print(f"Metadata saved to {filepath}_metadata.pkl")
    
    @staticmethod
    def load_model(filepath: str, device: Optional[torch.device] = None) -> 'CNNLSTMModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file (without extension)
            device: Device to load model on (default: CPU)
            
        Returns:
            Loaded CNNLSTMModel instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create model with saved architecture
        model = CNNLSTMModel(
            input_size=metadata['input_size'],
            cnn_filters=metadata['cnn_filters'],
            cnn_kernel_sizes=metadata['cnn_kernel_sizes'],
            lstm_hidden_size=metadata['lstm_hidden_size'],
            lstm_num_layers=metadata['lstm_num_layers'],
            dropout=metadata['dropout'],
            bidirectional=metadata['bidirectional'],
            use_attention=metadata.get('use_attention', False)
        )
        
        # Load state dict
        model.load_state_dict(torch.load(f"{filepath}.pth", map_location=device))
        model.to(device)
        
        print(f"Model loaded from {filepath}.pth")
        return model

