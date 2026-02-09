"""
GDN Processor Adapter

Lightweight adapter that wraps existing GARAGE-Final helper functions from kg/create_kg.py
to provide a compatible interface for LLM/RAG code.
"""

import numpy as np
import torch
from typing import List, Optional, Union
from pathlib import Path

# Import existing helper functions from kg.create_kg
from kg.create_kg import (
    load_gdn_model,
    extract_sensor_embeddings,
    compute_adjacency_matrix,
    predict_anomalies,
    extract_window_embeddings,
)
from models.gdn_model import GDN


class GDNPredictor:
    """
    GDN Predictor adapter that wraps existing GARAGE-Final helper functions.
    
    This class provides a compatible interface for LLM/RAG code while reusing
    the existing helper functions from kg/create_kg.py.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        sensor_names: List[str],
        device: str = "cpu",
    ):
        """
        Initialize the GDN Predictor.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            sensor_names: List of sensor names (must match model's num_nodes)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.sensor_names = sensor_names
        self.device = device
        
        # Load model using existing helper function
        self.model, self.metadata = load_gdn_model(str(self.model_path), device=device)
        
        # Store metadata
        self.window_size = self.metadata.get('window_size', 300)
        self.embed_dim = self.metadata.get('embed_dim', 32)
        self.top_k = self.metadata.get('top_k', 5)
        self.hidden_dim = self.metadata.get('hidden_dim', 64)
        self.num_sensors = len(sensor_names)
        
    def get_sensor_embeddings(self) -> np.ndarray:
        """
        Extract learned sensor embeddings from the model.
        
        Returns:
            sensor_embeddings: (num_sensors, embed_dim) numpy array
        """
        return extract_sensor_embeddings(self.model)
    
    def compute_adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix from sensor embeddings using cosine similarity.
        
        Returns:
            adjacency_matrix: (num_sensors, num_sensors) numpy array
        """
        sensor_embeddings = self.get_sensor_embeddings()
        return compute_adjacency_matrix(sensor_embeddings)
    
    def get_corr_embedding(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract window embeddings for distance-based scoring.
        
        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            batch_size: Batch size for inference
            
        Returns:
            embeddings: (num_windows, hidden_dim) numpy array of embeddings
        """
        return extract_window_embeddings(
            self.model, X_windows, batch_size=batch_size, device=self.device
        )
    
    def predict(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Run inference on data windows to get sensor anomaly probabilities.
        
        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            batch_size: Batch size for inference
            
        Returns:
            sensor_probs: (num_windows, num_sensors) numpy array of anomaly probabilities
        """
        return predict_anomalies(
            self.model, X_windows, batch_size=batch_size, device=self.device
        )
