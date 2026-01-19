"""
GDN Processor Class

Extracts GDN model inference logic from gdn.ipynb into a reusable class
that integrates seamlessly with KG.py for knowledge graph construction.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

try:
    # Try relative import first (when used as a module)
    from .models.gdn_model import MultiLabelGDN
except ImportError:
    # Fall back to absolute import (when run as a script)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from models.gdn_model import MultiLabelGDN


class GDNPredictor:
    """
    GDN Predictor class for processing sensor data windows and extracting
    embeddings/adjacency matrices for knowledge graph construction.
    
    This class provides a plug-and-play interface with KG.py's KnowledgeGraphBuilder.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        sensor_names: List[str],
        window_size: int = 300,
        embed_dim: int = 64,
        top_k: int = 3,
        hidden_dim: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize the GDN Predictor.
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            sensor_names: List of sensor names (must match model's num_nodes)
            window_size: Size of input windows (default: 300)
            embed_dim: Embedding dimension (default: 64)
            top_k: Top-K neighbors for graph construction (default: 3)
            hidden_dim: Hidden dimension for GAT layers (default: 32)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = Path(model_path)
        self.sensor_names = sensor_names
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_sensors = len(sensor_names)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Initialize model architecture
        self.model = MultiLabelGDN(
            num_nodes=self.num_sensors,
            window_size=self.window_size,
            embed_dim=self.embed_dim,
            top_k=self.top_k,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        # Try loading with strict=False first to handle architecture mismatches
        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try partial loading
            print(f"  Warning: Strict loading failed: {e}")
            print("  Attempting partial loading...")
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        
        print(f"✓ Loaded model from {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Sensors: {self.num_sensors}")
        print(f"  Window size: {self.window_size}")
    
    def get_sensor_embeddings(self) -> np.ndarray:
        """
        Extract learned sensor embeddings from the model.
        
        Returns:
            sensor_embeddings: (num_sensors, embed_dim) numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        with torch.no_grad():
            embeddings = self.model.sensor_embeddings.cpu().numpy()
        
        return embeddings
    
    def compute_adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix from sensor embeddings using cosine similarity.
        
        This matches the logic used in the notebook for visualization.
        
        Returns:
            adjacency_matrix: (num_sensors, num_sensors) numpy array
                Values are cosine similarities between sensor embeddings,
                scaled to [0.1, 1.0] range for visualization compatibility.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        with torch.no_grad():
            # Get sensor embeddings
            sensor_embeddings = self.model.sensor_embeddings  # (num_sensors, embed_dim)
            
            # Normalize embeddings (L2 normalization for cosine similarity)
            sensor_embeddings_norm = torch.nn.functional.normalize(
                sensor_embeddings, p=2, dim=1
            )
            
            # Compute cosine similarity matrix
            similarity_matrix = torch.mm(
                sensor_embeddings_norm, 
                sensor_embeddings_norm.t()
            )  # (num_sensors, num_sensors)
            
            # Convert to numpy
            similarity_matrix_np = similarity_matrix.cpu().numpy()
            
            # Scale from [-1, 1] to [0.1, 1.0] for compatibility with notebook visualization
            # This matches the scaling used in the notebook's adjacency matrix computation
            adjacency_matrix = (similarity_matrix_np + 1.0) / 2.0
            adjacency_matrix = np.clip(adjacency_matrix, 0.1, 1.0)
            
            # Zero out diagonal (no self-loops)
            np.fill_diagonal(adjacency_matrix, 0.0)
        
        return adjacency_matrix
    
    def predict(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        return_global: bool = False,
        batch_size: int = 32
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on data windows to get sensor anomaly probabilities.
        
        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            return_global: If True, also return global window anomaly probabilities
            batch_size: Batch size for inference
        
        Returns:
            - sensor_probs: (num_windows, num_sensors) numpy array of anomaly probabilities
            - global_probs: (num_windows,) numpy array (optional, if return_global=True)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Convert to tensor if needed
        if isinstance(X_windows, np.ndarray):
            X_windows = torch.from_numpy(X_windows).float()
        
        # Ensure correct shape
        if X_windows.dim() != 3:
            raise ValueError(
                f"Expected 3D input (num_windows, window_size, num_sensors), "
                f"got shape {X_windows.shape}"
            )
        
        num_windows = X_windows.shape[0]
        X_windows = X_windows.to(self.device)
        
        # Run inference in batches
        all_sensor_probs = []
        all_global_probs = []
        
        self.model.eval()
        with torch.no_grad():
            num_batches = (num_windows + batch_size - 1) // batch_size
            with tqdm(total=num_windows, desc="GDN Inference", unit="window", leave=False) as pbar:
                for i in range(0, num_windows, batch_size):
                    batch = X_windows[i:i + batch_size]
                    batch_size_actual = batch.shape[0]
                    
                    if return_global:
                        sensor_probs, global_probs = self.model(batch, return_global=True)
                        all_sensor_probs.append(sensor_probs.cpu().numpy())
                        all_global_probs.append(global_probs.cpu().numpy())
                    else:
                        sensor_probs = self.model(batch, return_global=False)
                    
                    all_sensor_probs.append(sensor_probs.cpu().numpy())
                    pbar.update(batch_size_actual)
        
        sensor_probs = np.concatenate(all_sensor_probs, axis=0)
        
        if return_global:
            global_probs = np.concatenate(all_global_probs, axis=0)
            return sensor_probs, global_probs
        
        return sensor_probs
    
    def process_for_kg(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        sensor_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        window_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Main entry point: Process data and return all information needed for KG construction.
        
        This method returns a dictionary that matches KG.py's KnowledgeGraphBuilder
        input requirements.
        
        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            sensor_labels: (num_windows, num_sensors) binary sensor-level labels (optional)
            window_labels: (num_windows,) binary window-level labels (optional)
            batch_size: Batch size for inference
        
        Returns:
            Dictionary with keys:
                - 'sensor_names': List[str] - sensor names
                - 'sensor_embeddings': np.ndarray (num_sensors, embed_dim)
                - 'adjacency_matrix': np.ndarray (num_sensors, num_sensors)
                - 'X_windows': np.ndarray (num_windows, window_size, num_sensors)
                - 'sensor_labels': np.ndarray (num_windows, num_sensors) - from predictions or provided
                - 'window_labels': np.ndarray (num_windows,) - from predictions or provided
        """
        # Convert inputs to numpy if needed
        if isinstance(X_windows, torch.Tensor):
            X_windows = X_windows.cpu().numpy()
        
        # Ensure X_windows is numpy array
        X_windows = np.asarray(X_windows)
        
        # Get sensor embeddings and adjacency matrix
        sensor_embeddings = self.get_sensor_embeddings()
        adjacency_matrix = self.compute_adjacency_matrix()
        
        # Get predictions if labels not provided
        if sensor_labels is None:
            sensor_probs = self.predict(X_windows, return_global=False, batch_size=batch_size)
            # Convert probabilities to binary labels (threshold at 0.5)
            sensor_labels = (sensor_probs > 0.5).astype(np.float32)
        else:
            if isinstance(sensor_labels, torch.Tensor):
                sensor_labels = sensor_labels.cpu().numpy()
            sensor_labels = np.asarray(sensor_labels)
        
        # Get window labels if not provided
        if window_labels is None:
            # Window is faulty if any sensor is faulty
            window_labels = (sensor_labels.sum(axis=1) > 0).astype(np.int64)
        else:
            if isinstance(window_labels, torch.Tensor):
                window_labels = window_labels.cpu().numpy()
            window_labels = np.asarray(window_labels).astype(np.int64)
        
        return {
            'sensor_names': self.sensor_names,
            'sensor_embeddings': sensor_embeddings,
            'adjacency_matrix': adjacency_matrix,
            'X_windows': X_windows,
            'sensor_labels': sensor_labels,
            'window_labels': window_labels
        }
