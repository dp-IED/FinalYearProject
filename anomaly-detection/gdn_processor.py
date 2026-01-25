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
import matplotlib.pyplot as plt

try:
    # Try relative import first (when used as a module)
    from .models.gdn_model import MultiLabelGDN, ImprovedMultiLabelGDN, FastImprovedMultiLabelGDN, MetricLearningGDN
except ImportError:
    # Fall back to absolute import (when run as a script)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from models.gdn_model import MultiLabelGDN, ImprovedMultiLabelGDN, FastImprovedMultiLabelGDN, MetricLearningGDN


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
        device: Optional[str] = None,
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = None
        self.center_loss = None  # Will be loaded if available in checkpoint
        self.use_distance_scoring = (
            False  # Use distance to normal center instead of probabilities
        )
        self.normal_center = None  # Normal center for distance-based scoring
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Load checkpoint to detect model type
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        
        # Detect model type from checkpoint
        # Check for MetricLearningGDN first (metric learning model)
        use_metric_learning = False
        use_fast_improved = False
        use_improved = False
        num_gat_layers = 2  # Default for improved model
        
        model_path_str = str(self.model_path).lower()
        
        # Check for MetricLearningGDN
        if "metric_learning" in model_path_str or "metric" in model_path_str:
            use_metric_learning = True
        # Check for FastImprovedMultiLabelGDN (simplified)
        elif "fast_improved" in model_path_str or "fast" in model_path_str:
            use_fast_improved = True
        elif isinstance(checkpoint, dict):
            if "num_gat_layers" in checkpoint:
                use_improved = True
                num_gat_layers = checkpoint["num_gat_layers"]
            elif "architecture_improved" in model_path_str:
                use_improved = True
        
        # Initialize model architecture
        if use_metric_learning:
            # MetricLearningGDN uses larger dimensions (128 default)
            # Detect from checkpoint if available
            embed_dim = checkpoint.get("embed_dim", 128) if isinstance(checkpoint, dict) else 128
            hidden_dim = checkpoint.get("hidden_dim", 128) if isinstance(checkpoint, dict) else 128
            self.model = MetricLearningGDN(
                num_nodes=self.num_sensors,
                window_size=self.window_size,
                embed_dim=embed_dim,
                top_k=self.top_k,
                hidden_dim=hidden_dim,
            ).to(self.device)
        elif use_fast_improved:
            self.model = FastImprovedMultiLabelGDN(
                num_nodes=self.num_sensors,
                window_size=self.window_size,
                embed_dim=self.embed_dim,
                top_k=self.top_k,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
        elif use_improved:
            self.model = ImprovedMultiLabelGDN(
                num_nodes=self.num_sensors,
                window_size=self.window_size,
                embed_dim=self.embed_dim,
                top_k=self.top_k,
                hidden_dim=self.hidden_dim,
                num_gat_layers=num_gat_layers,
            ).to(self.device)
        else:
            self.model = MultiLabelGDN(
                num_nodes=self.num_sensors,
                window_size=self.window_size,
                embed_dim=self.embed_dim,
                top_k=self.top_k,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        # Handle both checkpoint formats:
        # 1. Direct state_dict (old format)
        # 2. Dictionary with 'model_state_dict' key (new format with metadata)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

            # Check for Phase 2 checkpoint format (normal_center directly)
            if "normal_center" in checkpoint:
                self.normal_center = checkpoint["normal_center"].to(self.device)
                self.use_distance_scoring = True
                self.center_loss = None  # Not needed for Phase 2

                if "final_separation_ratio" in checkpoint:
                    print(f"  ✓ Distance-based scoring enabled (Phase 2 checkpoint)")
                    print(
                        f"  ✓ Separation ratio: {checkpoint.get('final_separation_ratio', 'N/A'):.2f}×"
                    )
                    print(
                        f"  ✓ Normal mean distance: {checkpoint.get('normal_mean_distance', 'N/A'):.4f}"
                    )
                    print(
                        f"  ✓ Anomalous mean distance: {checkpoint.get('anomalous_mean_distance', 'N/A'):.4f}"
                    )
            # Check for Phase 1 checkpoint format (center_loss_state_dict)
            elif "center_loss_state_dict" in checkpoint:
                try:
                    # Import CenterLoss from training script
                    import importlib.util

                    train_script_path = (
                        Path(__file__).parent / "train_gdn_separation.py"
                    )
                    if train_script_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "train_gdn_separation", train_script_path
                        )
                        train_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(train_module)
                        CenterLoss = train_module.CenterLoss

                        # Load center loss
                        self.center_loss = CenterLoss(embed_dim=32, num_classes=2).to(
                            self.device
                        )
                        self.center_loss.load_state_dict(
                            checkpoint["center_loss_state_dict"]
                        )
                        self.center_loss.eval()

                        # Extract normal center (class 0)
                        self.normal_center = self.center_loss.centers[
                            0
                        ].detach()  # (embed_dim,)
                        self.use_distance_scoring = (
                            checkpoint.get("separation_target", None) is not None
                        )

                        if self.use_distance_scoring:
                            print(
                                f"  ✓ Distance-based scoring enabled (separation target: {checkpoint.get('separation_target', 'N/A')})"
                            )
                            print(
                                f"  ✓ Center separation: {checkpoint.get('center_separation', 'N/A'):.4f}"
                            )
                    else:
                        self.center_loss = None
                        self.use_distance_scoring = False
                except Exception as e:
                    print(f"  Warning: Could not load center loss: {e}")
                    self.center_loss = None
                    self.use_distance_scoring = False
            else:
                self.center_loss = None
                self.use_distance_scoring = False
                self.normal_center = None
        else:
            state_dict = checkpoint
            self.center_loss = None
            self.use_distance_scoring = False

        # Handle GAT layer key mismatch between PyG versions
        # Old format: gat.lin.weight -> New format: gat.lin_src.weight, gat.lin_dst.weight
        if "gat.lin.weight" in state_dict and "gat.lin_src.weight" not in state_dict:
            # Convert old format to new format
            gat_lin_weight = state_dict.pop("gat.lin.weight")
            # Split the weight for source and destination (GATConv uses same weights for both in some versions)
            state_dict["gat.lin_src.weight"] = gat_lin_weight
            state_dict["gat.lin_dst.weight"] = gat_lin_weight.clone()
            print(
                "  ✓ Converted GAT layer from old format (lin.weight) to new format (lin_src/lin_dst)"
            )

        # Try loading with strict=False first to handle architecture mismatches
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            # If strict loading fails, try partial loading
            print(f"  Warning: Strict loading failed: {e}")
            print("  Attempting partial loading...")
            self.model.load_state_dict(state_dict, strict=False)
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
                sensor_embeddings_norm, sensor_embeddings_norm.t()
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

    def get_corr_embedding(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings for distance-based scoring.

        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            batch_size: Batch size for inference

        Returns:
            embeddings: (num_windows, hidden_dim) numpy array of embeddings
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
        all_embeddings = []

        self.model.eval()
        with torch.no_grad():
            num_batches = (num_windows + batch_size - 1) // batch_size
            with tqdm(
                total=num_windows,
                desc="GDN Embeddings",
                unit="window",
                leave=False,
            ) as pbar:
                for i in range(0, num_windows, batch_size):
                    batch = X_windows[i : i + batch_size]
                    batch_size_actual = batch.shape[0]

                    # Get embeddings
                    embeddings = self.model.get_embeddings(batch)  # (B, hidden_dim)
                    all_embeddings.append(embeddings.cpu().numpy())

                    pbar.update(batch_size_actual)

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings

    def predict(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        return_global: bool = False,
        batch_size: int = 32,
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
            with tqdm(
                total=num_windows, desc="GDN Inference", unit="window", leave=False
            ) as pbar:
                for i in range(0, num_windows, batch_size):
                    batch = X_windows[i : i + batch_size]
                    batch_size_actual = batch.shape[0]

                    if self.use_distance_scoring and self.normal_center is not None:
                        # Use embeddings for distance-based scoring
                        embeddings = self.model.get_embeddings(batch)  # (B, hidden_dim)
                        distances = torch.norm(
                            embeddings - self.normal_center.unsqueeze(0), dim=1
                        )  # (B,)

                        # Normalize distances to [0, 1] range for compatibility with existing pipeline
                        # Higher distance = more anomalous
                        # Use min-max normalization based on expected separation
                        # Normal windows should have distance ~0, anomalous ~separation_target (0.3)
                        min_dist = 0.0
                        max_dist = 1.0  # Cap at 1.0 for normalization
                        normalized_distances = torch.clamp(
                            (distances - min_dist) / (max_dist - min_dist), 0.0, 1.0
                        )

                        # Expand to per-sensor scores (same distance for all sensors in a window)
                        # This matches the window-level nature of distance-based scoring
                        sensor_probs = (
                            normalized_distances.unsqueeze(1)
                            .expand(-1, self.num_sensors)
                            .cpu()
                            .numpy()
                        )
                        all_sensor_probs.append(sensor_probs)

                        if return_global:
                            all_global_probs.append(normalized_distances.cpu().numpy())
                    else:
                        # Use standard probability-based scoring
                        if return_global:
                            sensor_probs, global_probs = self.model(
                                batch, return_global=True
                            )
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
        batch_size: int = 32,
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Main entry point: Process data and return all information needed for KG construction.

        This method returns a dictionary that matches KG.py's KnowledgeGraphBuilder
        input requirements. Always returns GDN prediction probabilities (not binary labels)
        for KG construction to avoid ground truth leakage.

        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            sensor_labels: (num_windows, num_sensors) binary sensor-level labels (optional, kept for evaluation purposes only)
            window_labels: (num_windows,) binary window-level labels (optional, kept for evaluation purposes only)
            batch_size: Batch size for inference

        Returns:
            Dictionary with keys:
                - 'sensor_names': List[str] - sensor names
                - 'sensor_embeddings': np.ndarray (num_sensors, embed_dim)
                - 'adjacency_matrix': np.ndarray (num_sensors, num_sensors)
                - 'X_windows': np.ndarray (num_windows, window_size, num_sensors)
                - 'gdn_predictions': np.ndarray (num_windows, num_sensors) - GDN anomaly probabilities (0.0-1.0)
                - 'sensor_labels': np.ndarray (num_windows, num_sensors) - ground truth labels (for evaluation only, not used in KG)
                - 'window_labels': np.ndarray (num_windows,) - ground truth window labels (for evaluation only, not used in KG)
                - 'window_embeddings': np.ndarray (num_windows, hidden_dim) - window embeddings (if center loss available)
                - 'distances_to_normal': np.ndarray (num_windows,) - Euclidean distances to normal center (if available)
                - 'distances_to_anomalous': np.ndarray (num_windows,) - Euclidean distances to anomalous center (if available)
                - 'center_embeddings': np.ndarray (2, hidden_dim) - center embeddings [normal, anomalous] (if available)
        """
        # Convert inputs to numpy if needed
        if isinstance(X_windows, torch.Tensor):
            X_windows = X_windows.cpu().numpy()

        # Ensure X_windows is numpy array
        X_windows = np.asarray(X_windows)

        # Get sensor embeddings and adjacency matrix
        sensor_embeddings = self.get_sensor_embeddings()
        adjacency_matrix = self.compute_adjacency_matrix()

        # Always get GDN predictions (probabilities, not binary labels)
        # This is what the KG should use, not ground truth labels
        gdn_predictions = self.predict(
            X_windows, return_global=False, batch_size=batch_size
        )

        # Extract window embeddings and compute distances to center loss centers
        window_embeddings = None
        distances_to_normal = None
        distances_to_anomalous = None
        center_embeddings = None

        try:
            # Extract embeddings using batch processing
            window_embeddings = self.get_corr_embedding(X_windows, batch_size=batch_size)
            
            # Get center loss centers if available
            if hasattr(self, 'center_loss') and self.center_loss is not None:
                # Centers are available from loaded center_loss
                centers = self.center_loss.centers.detach().cpu().numpy()  # (2, hidden_dim)
                center_normal = centers[0]  # Class 0: normal
                center_anomalous = centers[1]  # Class 1: anomalous
                center_embeddings = centers
            elif hasattr(self, 'normal_center') and self.normal_center is not None:
                # Only normal center available, try to get anomalous from checkpoint
                center_normal = self.normal_center.cpu().numpy() if hasattr(self.normal_center, 'cpu') else self.normal_center
                # Try to load anomalous center from checkpoint
                try:
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    if 'center_loss_state_dict' in checkpoint:
                        # Load center loss temporarily to get centers
                        import importlib.util
                        train_script_path = Path(__file__).parent / "train_gdn_separation.py"
                        if train_script_path.exists():
                            spec = importlib.util.spec_from_file_location("train_gdn_separation", train_script_path)
                            train_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(train_module)
                            CenterLoss = train_module.CenterLoss
                            temp_center_loss = CenterLoss(embed_dim=window_embeddings.shape[1], num_classes=2)
                            temp_center_loss.load_state_dict(checkpoint['center_loss_state_dict'])
                            centers = temp_center_loss.centers.detach().cpu().numpy()
                            center_normal = centers[0]
                            center_anomalous = centers[1]
                            center_embeddings = centers
                        else:
                            # Fallback: create dummy anomalous center
                            center_anomalous = center_normal + np.ones_like(center_normal) * 0.3
                            center_embeddings = np.stack([center_normal, center_anomalous], axis=0)
                    else:
                        # No center loss in checkpoint, create dummy anomalous center
                        center_anomalous = center_normal + np.ones_like(center_normal) * 0.3
                        center_embeddings = np.stack([center_normal, center_anomalous], axis=0)
                except Exception:
                    # Fallback: create dummy anomalous center
                    center_anomalous = center_normal + np.ones_like(center_normal) * 0.3
                    center_embeddings = np.stack([center_normal, center_anomalous], axis=0)
            else:
                # No center loss available, skip embedding extraction
                window_embeddings = None
                center_embeddings = None

            # Compute distances to centers if embeddings and centers are available
            if window_embeddings is not None and center_embeddings is not None:
                # Compute Euclidean distances
                distances_to_normal = np.linalg.norm(
                    window_embeddings - center_normal, axis=1
                )  # (num_windows,)
                distances_to_anomalous = np.linalg.norm(
                    window_embeddings - center_anomalous, axis=1
                )  # (num_windows,)
        except Exception as e:
            # If embedding extraction fails, continue without embeddings
            print(f"  ⚠️  Warning: Could not extract embeddings: {e}")
            window_embeddings = None
            distances_to_normal = None
            distances_to_anomalous = None
            center_embeddings = None

        # Keep sensor_labels and window_labels for evaluation purposes (separate from KG construction)
        if sensor_labels is not None:
            if isinstance(sensor_labels, torch.Tensor):
                sensor_labels = sensor_labels.cpu().numpy()
            sensor_labels = np.asarray(sensor_labels)
        else:
            # If not provided, create empty array (will be used for evaluation if needed)
            sensor_labels = np.zeros(
                (len(X_windows), self.num_sensors), dtype=np.float32
            )

        if window_labels is not None:
            if isinstance(window_labels, torch.Tensor):
                window_labels = window_labels.cpu().numpy()
            window_labels = np.asarray(window_labels).astype(np.int64)
        else:
            # If not provided, derive from GDN predictions (for reference, not used in KG)
            window_labels = (gdn_predictions.max(axis=1) > 0.5).astype(np.int64)

        result = {
            "sensor_names": self.sensor_names,
            "sensor_embeddings": sensor_embeddings,
            "adjacency_matrix": adjacency_matrix,
            "X_windows": X_windows,
            "gdn_predictions": gdn_predictions,  # GDN probabilities for KG construction
            "sensor_labels": sensor_labels,  # Ground truth (for evaluation only)
            "window_labels": window_labels,  # Ground truth (for evaluation only)
        }
        
        # Add embedding data if available
        if window_embeddings is not None:
            result["window_embeddings"] = window_embeddings
            result["distances_to_normal"] = distances_to_normal
            result["distances_to_anomalous"] = distances_to_anomalous
            result["center_embeddings"] = center_embeddings
        
        return result

    def analyze_topk_correlations(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        window_labels: Union[np.ndarray, torch.Tensor],
        top_k: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze and print distribution of top-k learned correlations in normal vs anomalous windows.

        This utility function:
        1. Extracts the learned adjacency matrix (top-k edges from sensor embeddings)
        2. Computes actual correlation matrices for each window
        3. Compares top-k learned correlations with actual correlations
        4. Prints distributions for normal vs anomalous windows

        Args:
            X_windows: (num_windows, window_size, num_sensors) input windows
            window_labels: (num_windows,) binary labels (0=normal, 1=anomalous)
            top_k: Number of top correlations to analyze (default: uses model's top_k)
            sample_size: Maximum number of windows to sample per class (None = use all)

        Returns:
            Dictionary with analysis results:
                - 'learned_topk': Learned top-k correlation values
                - 'actual_topk_normal': Actual top-k correlations in normal windows
                - 'actual_topk_anomalous': Actual top-k correlations in anomalous windows
                - 'learned_topk_normal': Learned top-k correlations in normal windows
                - 'learned_topk_anomalous': Learned top-k correlations in anomalous windows
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Convert to numpy if needed
        if isinstance(X_windows, np.ndarray):
            X_windows_np = X_windows
        else:
            X_windows_np = X_windows.cpu().numpy()

        if isinstance(window_labels, torch.Tensor):
            window_labels_np = window_labels.cpu().numpy()
        else:
            window_labels_np = np.asarray(window_labels)

        # Use model's top_k if not specified
        if top_k is None:
            top_k = self.model.top_k

        num_windows = len(X_windows_np)
        num_sensors = X_windows_np.shape[2]

        # Get learned adjacency matrix (expected correlations)
        learned_adj = self.compute_adjacency_matrix()  # (num_sensors, num_sensors)

        # Extract top-k learned correlations (excluding diagonal)
        # Flatten upper triangle to get all unique pairs
        learned_pairs = []
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                learned_pairs.append((i, j, learned_adj[i, j]))

        # Sort by correlation strength and get top-k pairs
        learned_pairs.sort(key=lambda x: x[2], reverse=True)
        topk_pairs = learned_pairs[:top_k]  # List of (i, j, correlation) tuples
        learned_topk_mean = np.mean([pair[2] for pair in topk_pairs])

        # Separate normal and anomalous windows
        normal_mask = window_labels_np == 0
        anomalous_mask = window_labels_np == 1

        num_normal = normal_mask.sum()
        num_anomalous = anomalous_mask.sum()

        print("=" * 80)
        print("TOP-K LEARNED CORRELATIONS ANALYSIS")
        print("=" * 80)
        print(f"\nTotal windows: {num_windows}")
        print(f"Normal windows: {num_normal}")
        print(f"Anomalous windows: {num_anomalous}")
        print(f"Top-K: {top_k}")
        print(
            f"\nLearned adjacency matrix (mean top-{top_k} correlations): {learned_topk_mean:.4f}"
        )
        print()

        # Sample windows if requested
        normal_indices = np.where(normal_mask)[0]
        anomalous_indices = np.where(anomalous_mask)[0]

        if sample_size is not None:
            if len(normal_indices) > sample_size:
                normal_indices = np.random.choice(
                    normal_indices, size=sample_size, replace=False
                )
            if len(anomalous_indices) > sample_size:
                anomalous_indices = np.random.choice(
                    anomalous_indices, size=sample_size, replace=False
                )

        # Compute actual correlations for normal windows
        print("Computing actual correlations in normal windows...")
        actual_topk_normal = []
        learned_topk_normal = []

        for idx in tqdm(normal_indices, desc="Normal windows", leave=False):
            window_data = X_windows_np[idx]  # (window_size, num_sensors)
            # Compute actual correlation matrix
            actual_corr = np.corrcoef(window_data.T)  # (num_sensors, num_sensors)
            np.fill_diagonal(actual_corr, -np.inf)

            # Extract actual correlations for the same top-k pairs learned by the model
            actual_topk_vals = []
            learned_topk_vals = []
            for i, j, learned_corr in topk_pairs:
                actual_topk_vals.append(actual_corr[i, j])
                learned_topk_vals.append(learned_corr)

            actual_topk_vals = np.array(actual_topk_vals)
            learned_topk_vals = np.array(learned_topk_vals)

            actual_topk_normal.append(actual_topk_vals)
            learned_topk_normal.append(learned_topk_vals)

        # Compute actual correlations for anomalous windows
        print("Computing actual correlations in anomalous windows...")
        actual_topk_anomalous = []
        learned_topk_anomalous = []

        for idx in tqdm(anomalous_indices, desc="Anomalous windows", leave=False):
            window_data = X_windows_np[idx]  # (window_size, num_sensors)
            # Compute actual correlation matrix
            actual_corr = np.corrcoef(window_data.T)  # (num_sensors, num_sensors)
            np.fill_diagonal(actual_corr, -np.inf)

            # Extract actual correlations for the same top-k pairs learned by the model
            actual_topk_vals = []
            learned_topk_vals = []
            for i, j, learned_corr in topk_pairs:
                actual_topk_vals.append(actual_corr[i, j])
                learned_topk_vals.append(learned_corr)

            actual_topk_vals = np.array(actual_topk_vals)
            learned_topk_vals = np.array(learned_topk_vals)

            actual_topk_anomalous.append(actual_topk_vals)
            learned_topk_anomalous.append(learned_topk_vals)

        # Convert to numpy arrays
        actual_topk_normal = np.array(actual_topk_normal)  # (num_normal, top_k)
        actual_topk_anomalous = np.array(
            actual_topk_anomalous
        )  # (num_anomalous, top_k)
        learned_topk_normal = np.array(learned_topk_normal)  # (num_normal, top_k)
        learned_topk_anomalous = np.array(
            learned_topk_anomalous
        )  # (num_anomalous, top_k)

        # Print statistics
        print("\n" + "=" * 80)
        print("DISTRIBUTION STATISTICS")
        print("=" * 80)
        print()

        print("ACTUAL CORRELATIONS (Top-K per window):")
        print("-" * 80)
        print(f"Normal windows (n={len(actual_topk_normal)}):")
        print(f"  Mean: {np.mean(actual_topk_normal):.4f}")
        print(f"  Std:  {np.std(actual_topk_normal):.4f}")
        print(f"  Min:  {np.min(actual_topk_normal):.4f}")
        print(f"  Max:  {np.max(actual_topk_normal):.4f}")
        print(f"  Median: {np.median(actual_topk_normal):.4f}")
        print()

        print(f"Anomalous windows (n={len(actual_topk_anomalous)}):")
        print(f"  Mean: {np.mean(actual_topk_anomalous):.4f}")
        print(f"  Std:  {np.std(actual_topk_anomalous):.4f}")
        print(f"  Min:  {np.min(actual_topk_anomalous):.4f}")
        print(f"  Max:  {np.max(actual_topk_anomalous):.4f}")
        print(f"  Median: {np.median(actual_topk_anomalous):.4f}")
        print()

        mean_diff = np.mean(actual_topk_normal) - np.mean(actual_topk_anomalous)
        print(f"Mean difference (Normal - Anomalous): {mean_diff:.4f}")
        print()

        print("LEARNED CORRELATIONS (Top-K per window):")
        print("-" * 80)
        print(f"Normal windows:")
        print(f"  Mean: {np.mean(learned_topk_normal):.4f}")
        print(f"  Std:  {np.std(learned_topk_normal):.4f}")
        print()

        print(f"Anomalous windows:")
        print(f"  Mean: {np.mean(learned_topk_anomalous):.4f}")
        print(f"  Std:  {np.std(learned_topk_anomalous):.4f}")
        print()

        print("DEVIATION FROM LEARNED (Actual - Learned):")
        print("-" * 80)
        deviation_normal = actual_topk_normal - learned_topk_normal
        deviation_anomalous = actual_topk_anomalous - learned_topk_anomalous

        print(f"Normal windows:")
        print(f"  Mean deviation: {np.mean(deviation_normal):.4f}")
        print(f"  Std deviation:  {np.std(deviation_normal):.4f}")
        print(f"  Mean |deviation|: {np.mean(np.abs(deviation_normal)):.4f}")
        print()

        print(f"Anomalous windows:")
        print(f"  Mean deviation: {np.mean(deviation_anomalous):.4f}")
        print(f"  Std deviation:  {np.std(deviation_anomalous):.4f}")
        print(f"  Mean |deviation|: {np.mean(np.abs(deviation_anomalous)):.4f}")
        print()

        deviation_diff = np.mean(np.abs(deviation_anomalous)) - np.mean(
            np.abs(deviation_normal)
        )
        print(f"Mean |deviation| difference (Anomalous - Normal): {deviation_diff:.4f}")
        if deviation_diff > 0:
            print(
                "  ✓ Anomalous windows show larger deviations from learned correlations"
            )
        else:
            print("  ⚠️  Normal windows show larger deviations (unexpected)")

        print("\n" + "=" * 80)

        return {
            "learned_topk": np.array([pair[2] for pair in topk_pairs]),
            "topk_pairs": topk_pairs,  # List of (i, j, correlation) tuples
            "actual_topk_normal": actual_topk_normal,
            "actual_topk_anomalous": actual_topk_anomalous,
            "learned_topk_normal": learned_topk_normal,
            "learned_topk_anomalous": learned_topk_anomalous,
            "deviation_normal": deviation_normal,
            "deviation_anomalous": deviation_anomalous,
        }
