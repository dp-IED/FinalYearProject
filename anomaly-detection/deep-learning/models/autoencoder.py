"""
Autoencoder model for anomaly detection.
Learns normal patterns and flags reconstruction errors as anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any, Tuple


class AutoencoderModel(nn.Module):
    """
    Autoencoder model for unsupervised anomaly detection.
    Uses encoder-decoder architecture with bottleneck to learn normal patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        encoder_dims: list = [64, 32, 16],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize Autoencoder model.
        
        Args:
            input_size: Number of input features per timestep
            sequence_length: Length of input sequences
            encoder_dims: List of dimensions for encoder layers (decoder is symmetric reversed)
            dropout: Dropout rate (default: 0.2)
            activation: Activation function ('relu' or 'tanh', default: 'relu')
        """
        super(AutoencoderModel, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.encoder_dims = encoder_dims
        self.dropout = dropout
        self.activation = activation
        
        # Calculate total input dimension (flattened sequence)
        total_input_dim = input_size * sequence_length
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        prev_dim = total_input_dim
        
        for dim in encoder_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU() if activation == 'relu' else nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
            prev_dim = dim
        
        # Bottleneck (latent representation)
        self.bottleneck_dim = encoder_dims[-1]
        
        # Decoder layers (symmetric to encoder)
        self.decoder_layers = nn.ModuleList()
        decoder_dims = encoder_dims[::-1]  # Reverse encoder dimensions
        
        for i, dim in enumerate(decoder_dims[:-1]):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU() if activation == 'relu' else nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
            prev_dim = dim
        
        # Final decoder layer (reconstruct to original size)
        self.decoder_layers.append(
            nn.Linear(prev_dim, total_input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder (reconstruction).
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Reconstructed tensor of shape (batch, sequence_length, input_size)
        """
        batch_size = x.shape[0]
        
        # Flatten sequence: (batch, seq_len, features) -> (batch, seq_len * features)
        x_flat = x.view(batch_size, -1)
        
        # Encode
        encoded = x_flat
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        # Decode
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer(decoded)
        
        # Reshape back to sequence: (batch, seq_len * features) -> (batch, seq_len, features)
        reconstructed = decoded.view(batch_size, self.sequence_length, self.input_size)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to bottleneck representation.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Encoded representation of shape (batch, bottleneck_dim)
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        encoded = x_flat
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        
        return encoded
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode from bottleneck representation.
        
        Args:
            encoded: Encoded representation of shape (batch, bottleneck_dim)
            
        Returns:
            Reconstructed tensor of shape (batch, sequence_length, input_size)
        """
        batch_size = encoded.shape[0]
        
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = layer(decoded)
        
        reconstructed = decoded.view(batch_size, self.sequence_length, self.input_size)
        return reconstructed
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full encode-decode for reconstruction.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Reconstructed tensor of shape (batch, sequence_length, input_size)
        """
        return self.forward(x)
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE) for anomaly scoring.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Reconstruction error per sample of shape (batch,)
        """
        reconstructed = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction='none')
        # Average over sequence and features
        error = error.mean(dim=(1, 2))
        return error
    
    def get_anomaly_scores(self, sequences: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        """
        Get anomaly scores (reconstruction errors) for sequences.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            device: Device to run inference on (default: CPU)
            
        Returns:
            Anomaly scores (reconstruction errors) of shape (n_samples,)
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
                batch_errors = self.get_reconstruction_error(batch)
                scores.append(batch_errors.cpu().numpy())
        
        return np.concatenate(scores, axis=0)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate anomaly scores (reconstruction errors) for input sequences.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Anomaly scores as numpy array of shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            errors = self.get_reconstruction_error(x)
            return errors.cpu().numpy()
    
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
            'sequence_length': self.sequence_length,
            'encoder_dims': self.encoder_dims,
            'dropout': self.dropout,
            'activation': self.activation,
            'model_type': 'autoencoder'
        }
        
        if metadata is not None:
            model_metadata.update(metadata)
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"Model saved to {filepath}.pth")
        print(f"Metadata saved to {filepath}_metadata.pkl")
    
    @staticmethod
    def load_model(filepath: str, device: Optional[torch.device] = None) -> 'AutoencoderModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file (without extension)
            device: Device to load model on (default: CPU)
            
        Returns:
            Loaded AutoencoderModel instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create model with saved architecture
        model = AutoencoderModel(
            input_size=metadata['input_size'],
            sequence_length=metadata['sequence_length'],
            encoder_dims=metadata['encoder_dims'],
            dropout=metadata['dropout'],
            activation=metadata.get('activation', 'relu')
        )
        
        # Load state dict
        model.load_state_dict(torch.load(f"{filepath}.pth", map_location=device))
        model.to(device)
        
        print(f"Model loaded from {filepath}.pth")
        return model

