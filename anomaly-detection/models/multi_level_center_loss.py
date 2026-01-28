"""
Multi-Level Center Loss for Hierarchical Anomaly Detection

Learns separate centers at two levels:
1. Window-level: Normal vs. anomalous window embeddings (coarse-grained)
2. Sensor-level: Normal vs. anomalous per-sensor embeddings (fine-grained)

Both levels use triplet-center formulation for better separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelCenterLoss(nn.Module):
    """
    Hierarchical center loss that learns:
    1. Window-level centers (coarse-grained: normal vs. anomalous window)
    2. Sensor-level centers (fine-grained: normal vs. anomalous per sensor)
    
    Both use triplet-center formulation for better separation.
    """
    
    def __init__(
        self, 
        embed_dim=64, 
        num_sensors=8, 
        num_classes=2,
        margin=2.0,
        lambda_intra=1.5,  # Intra-class compactness weight
        lambda_sensor=0.5,  # Sensor-level loss weight
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.margin = margin
        self.lambda_intra = lambda_intra
        self.lambda_sensor = lambda_sensor
        
        # Window-level centers (2 centers: normal, anomaly)
        self.window_centers = nn.Parameter(
            torch.randn(num_classes, embed_dim)
        )
        nn.init.xavier_uniform_(self.window_centers)
        
        # Sensor-level centers (8 sensors × 2 classes)
        self.sensor_centers = nn.Parameter(
            torch.randn(num_sensors, num_classes, embed_dim)
        )
        nn.init.xavier_uniform_(self.sensor_centers)
        
    def forward(
        self, 
        window_embeddings,      # (B, embed_dim)
        sensor_embeddings,       # (B, num_sensors, embed_dim)
        window_labels,           # (B,) - 0=normal, 1=anomaly
        sensor_labels=None       # (B, num_sensors) - optional, 0=normal, 1=faulty
    ):
        """
        Compute hierarchical center loss.
        
        Args:
            window_embeddings: Aggregated window-level embeddings
            sensor_embeddings: Per-sensor embeddings (post-GAT)
            window_labels: Binary labels for windows
            sensor_labels: Per-sensor binary labels (if available)
        
        Returns:
            total_loss: Combined window + sensor center loss
        """
        # Normalize embeddings and centers
        window_embeddings = F.normalize(window_embeddings, p=2, dim=1)
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)
        window_centers = F.normalize(self.window_centers, p=2, dim=1)
        sensor_centers = F.normalize(self.sensor_centers, p=2, dim=2)
        
        # === Window-level loss ===
        loss_window = self._triplet_center_loss(
            window_embeddings, 
            window_labels, 
            window_centers
        )
        
        # === Sensor-level loss ===
        loss_sensor = 0.0
        
        if sensor_labels is not None:
            # Use ground-truth sensor labels
            for sensor_idx in range(self.num_sensors):
                sensor_emb = sensor_embeddings[:, sensor_idx, :]  # (B, embed_dim)
                sensor_label = sensor_labels[:, sensor_idx]        # (B,)
                sensor_center = sensor_centers[sensor_idx]         # (2, embed_dim)
                
                # Only compute loss if this sensor has faults in batch
                if sensor_label.sum() > 0:
                    loss = self._triplet_center_loss(
                        sensor_emb, 
                        sensor_label, 
                        sensor_center
                    )
                    loss_sensor += loss
            
            loss_sensor /= self.num_sensors
        else:
            # Fallback: use window labels for all sensors
            # (Assumes if window is anomalous, all sensors are)
            for sensor_idx in range(self.num_sensors):
                sensor_emb = sensor_embeddings[:, sensor_idx, :]
                sensor_center = sensor_centers[sensor_idx]
                
                loss = self._triplet_center_loss(
                    sensor_emb,
                    window_labels,  # Use window labels
                    sensor_center
                )
                loss_sensor += loss
            
            loss_sensor /= self.num_sensors
        
        # === Center repulsion (push normal/anomaly centers apart) ===
        # Use exponential penalty to continuously push centers apart (not just to margin)
        # Window-level
        window_sep = torch.norm(
            window_centers[0] - window_centers[1]
        )
        # Exponential penalty: higher loss when separation is small, decreases as sep increases
        # This continuously encourages larger separation even beyond margin
        loss_window_sep = torch.exp(-0.5 * window_sep)
        
        # Sensor-level (average across sensors)
        sensor_seps = torch.norm(
            sensor_centers[:, 0, :] - sensor_centers[:, 1, :],
            dim=1
        )
        # Same exponential penalty for sensors
        loss_sensor_sep = torch.exp(-0.5 * sensor_seps).mean()
        
        # === Combined loss ===
        # Increased repulsion weights since exponential penalty is smaller in magnitude
        total_loss = (
            loss_window +                          # Window-level discrimination
            self.lambda_sensor * loss_sensor +     # Sensor-level discrimination
            0.2 * loss_window_sep +                # Window center separation (increased from 0.1)
            0.2 * loss_sensor_sep                  # Sensor center separation (increased from 0.1)
        )
        
        return total_loss
    
    def _triplet_center_loss(self, embeddings, labels, centers):
        """
        Triplet-center loss: pull toward positive center, push from negative center.
        
        Args:
            embeddings: (B, embed_dim) normalized embeddings
            labels: (B,) binary labels
            centers: (2, embed_dim) [normal_center, anomaly_center]
        
        Returns:
            loss: scalar
        """
        B = embeddings.size(0)
        device = embeddings.device
        eps = 1e-8
        
        # Input validation
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Get positive (same class) and negative (opposite class) centers
        pos_centers = centers[labels.long()]          # (B, embed_dim)
        neg_labels = 1 - labels
        neg_centers = centers[neg_labels.long()]       # (B, embed_dim)
        
        # Distance to positive center (minimize)
        pos_dist = torch.norm(embeddings - pos_centers, p=2, dim=1)
        
        # Distance to negative center (maximize)
        neg_dist = torch.norm(embeddings - neg_centers, p=2, dim=1)
        
        # Clamp distances to prevent overflow
        pos_dist = torch.clamp(pos_dist, min=eps, max=50.0)
        neg_dist = torch.clamp(neg_dist, min=eps, max=50.0)
        
        # Intra-class compactness (pull toward positive center)
        loss_intra = pos_dist.mean()
        
        # Inter-class separation (triplet margin loss)
        loss_inter = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        # Final clamp
        total = loss_inter + self.lambda_intra * loss_intra
        total = torch.clamp(total, min=0.0, max=100.0)
        
        return total
    
    def get_window_centers(self):
        """Return normalized window-level centers."""
        return F.normalize(self.window_centers, p=2, dim=1)
    
    def get_sensor_centers(self):
        """Return normalized sensor-level centers."""
        return F.normalize(self.sensor_centers, p=2, dim=2)
    
    def get_separations(self):
        """
        Compute separation distances between normal/anomaly centers.
        
        Returns:
            dict with window and per-sensor separations
        """
        window_centers = self.get_window_centers()
        sensor_centers = self.get_sensor_centers()
        
        window_sep = torch.norm(
            window_centers[0] - window_centers[1]
        ).item()
        
        sensor_seps = torch.norm(
            sensor_centers[:, 0, :] - sensor_centers[:, 1, :],
            dim=1
        ).detach().cpu().numpy()
        
        return {
            'window_separation': window_sep,
            'sensor_separations': sensor_seps,
            'sensor_mean_separation': sensor_seps.mean(),
            'sensor_min_separation': sensor_seps.min(),
            'sensor_max_separation': sensor_seps.max(),
        }
