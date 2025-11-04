"""
LSTM model for anomaly detection in time series sensor data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any


class LSTMModel(nn.Module):
    """
    LSTM-based anomaly detection model.
    Trained to predict next timestep - prediction error is used as anomaly score.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1
    ):
        """
        Initialize LSTM model with multi-head attention.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in LSTM (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            num_attention_heads: Number of attention heads (default: 4)
            attention_dropout: Dropout for attention layers (default: 0.1)
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.model_type = 'lstm'  # For evaluation function to identify model type
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Multi-head attention for timestamp and column detection
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Output layer to predict next timestep (all features)
        self.fc = nn.Linear(lstm_output_size, input_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Column classifier: predicts which columns are anomalous
        self.column_scorer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, input_size)
        )
        
        # Timestamp classifier: predicts which timesteps are anomalous
        self.timestamp_scorer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Initialize weights properly for normalized data
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence with normalized data."""
        # Initialize LSTM weights (PyTorch handles this, but we can set output layer)
        # Xavier/Glorot initialization for FC layer (good for normalized inputs)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Initialize column and timestamp scorers
        for module in self.column_scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        for module in self.timestamp_scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attentions: bool = False):
        """
        Forward pass through the model with multi-head attention.
        Predicts the next timestep and identifies anomalous columns/timestamps.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            return_attentions: If True, return attention weights and scores
            
        Returns:
            If return_attentions=False:
                predicted_next: Predicted next timestep of shape (batch, input_size)
            If return_attentions=True:
                Tuple of (predicted_next, column_scores, timestamp_scores, attention_weights)
                - predicted_next: (batch, input_size)
                - column_scores: (batch, input_size) - probability each column is anomalous
                - timestamp_scores: (batch, sequence_length, 1) - probability each timestep is anomalous
                - attention_weights: (batch, num_heads, sequence_length, sequence_length)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, sequence_length, hidden_size * num_directions)
        
        # Apply multi-head attention over all timesteps
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        # attended_out shape: (batch, sequence_length, hidden_size * num_directions)
        # attention_weights shape: (batch, num_heads, sequence_length, sequence_length)
        
        # Use the last output from the sequence for next timestep prediction
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size * num_directions)
        
        # Apply dropout
        last_output = self.dropout_layer(last_output)
        
        # Fully connected layer to predict next timestep (all features)
        predicted_next = self.fc(last_output)  # (batch, input_size)
        
        if return_attentions:
            # Column scores: which columns are anomalous (based on last timestep)
            last_attended = attended_out[:, -1, :]  # (batch, hidden_size * num_directions)
            column_logits = self.column_scorer(last_attended)  # (batch, input_size)
            column_scores = torch.sigmoid(column_logits)  # (batch, input_size)
            
            # Timestamp scores: which timesteps are anomalous (based on attended output)
            timestamp_logits = self.timestamp_scorer(attended_out)  # (batch, seq_len, 1)
            timestamp_scores = torch.sigmoid(timestamp_logits)  # (batch, seq_len, 1)
            
            return predicted_next, column_scores, timestamp_scores, attention_weights
        else:
            # Backward compatibility: return only prediction
            return predicted_next
    
    def predict_next_timestep(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict next timestep (wrapper for forward).
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Predicted next timestep of shape (batch, input_size)
        """
        return self.forward(x)
    
    def get_prediction_error(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction error for anomaly scoring.
        Uses sequences[:, :-1] to predict sequences[:, -1] (last timestep).
        
        Args:
            sequences: Input sequences of shape (batch, sequence_length, input_size)
            
        Returns:
            Prediction errors per sample of shape (batch,)
        """
        # Use all but last timestep to predict the last timestep
        input_seq = sequences[:, :-1, :]  # (batch, sequence_length-1, input_size)
        target = sequences[:, -1, :]  # (batch, input_size)
        
        # Predict next timestep
        predicted = self.forward(input_seq)  # (batch, input_size)
        
        # Calculate MSE error per sample
        error = F.mse_loss(predicted, target, reduction='none')  # (batch, input_size)
        error = error.mean(dim=1)  # Average over features: (batch,)
        
        return error
    
    def get_feature_level_errors(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction error per feature for feature-level anomaly detection.
        Uses attention column scores combined with prediction errors.
        
        Args:
            sequences: Input sequences of shape (batch, sequence_length, input_size)
            
        Returns:
            Anomaly scores per feature of shape (batch, input_size)
            Higher score indicates that feature is more anomalous
        """
        # Use all but last timestep to predict the last timestep
        input_seq = sequences[:, :-1, :]  # (batch, sequence_length-1, input_size)
        target = sequences[:, -1, :]  # (batch, input_size)
        
        # Get prediction and column scores from attention
        predicted, column_scores, _, _ = self.forward(input_seq, return_attentions=True)
        
        # Calculate MSE error per feature
        prediction_error = F.mse_loss(predicted, target, reduction='none')  # (batch, input_size)
        
        # Combine prediction error with column scores
        # Normalize both to [0, 1] range and combine
        # Higher values indicate more anomalous
        error_normalized = prediction_error / (prediction_error.max(dim=1, keepdim=True)[0] + 1e-8)
        combined_score = (error_normalized + column_scores) / 2.0
        
        return combined_score  # Return combined anomaly scores per feature
    
    def get_timestamp_scores(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Get timestamp-level anomaly scores - which timesteps are anomalous.
        
        Args:
            sequences: Input sequences of shape (batch, sequence_length, input_size)
            
        Returns:
            Timestamp anomaly scores of shape (batch, sequence_length)
            Higher score indicates that timestep is more anomalous
        """
        # Use all but last timestep
        input_seq = sequences[:, :-1, :]  # (batch, sequence_length-1, input_size)
        
        # Get timestamp scores from attention
        _, _, timestamp_scores, _ = self.forward(input_seq, return_attentions=True)
        # timestamp_scores shape: (batch, seq_len-1, 1)
        
        # Squeeze and return
        return timestamp_scores.squeeze(-1)  # (batch, seq_len-1)
    
    def get_anomaly_scores(self, sequences: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        """
        Get anomaly scores (prediction errors) for sequences.
        
        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            device: Device to run inference on (default: CPU)
            
        Returns:
            Anomaly scores (prediction errors) of shape (n_samples,)
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
                batch_errors = self.get_prediction_error(batch)
                scores.append(batch_errors.cpu().numpy())
        
        return np.concatenate(scores, axis=0)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Generate anomaly scores (prediction errors) for input sequences.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Anomaly scores as numpy array of shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            errors = self.get_prediction_error(x)
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
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'model_type': 'lstm',
            'model_version': '2.0',  # Version with attention
            'num_attention_heads': self.num_attention_heads,
            'attention_dropout': self.attention_dropout
        }
        
        if metadata is not None:
            model_metadata.update(metadata)
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_metadata, f)
        
        print(f"Model saved to {filepath}.pth")
        print(f"Metadata saved to {filepath}_metadata.pkl")
    
    @staticmethod
    def load_model(filepath: str, device: Optional[torch.device] = None) -> 'LSTMModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file (without extension)
            device: Device to load model on (default: CPU)
            
        Returns:
            Loaded LSTMModel instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Check model version for backward compatibility
        model_version = metadata.get('model_version', '1.0')
        num_attention_heads = metadata.get('num_attention_heads', 4)
        attention_dropout = metadata.get('attention_dropout', 0.1)
        
        # Create model with saved architecture
        model = LSTMModel(
            input_size=metadata['input_size'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            dropout=metadata['dropout'],
            bidirectional=metadata['bidirectional'],
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
        )
        
        # Load state dict
        state_dict = torch.load(f"{filepath}.pth", map_location=device)
        
        # Handle backward compatibility: if old model doesn't have attention layers
        if model_version == '1.0':
            # Filter out attention-related keys that don't exist in old model
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"Warning: Skipping {key} due to shape mismatch")
                else:
                    print(f"Warning: Skipping {key} (not in new model)")
            
            # Initialize missing attention layers with random weights
            model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded old model (v1.0) - attention layers initialized randomly")
        else:
            # New model with attention
            model.load_state_dict(state_dict)
        
        model.to(device)
        
        print(f"Model loaded from {filepath}.pth (version {model_version})")
        return model
