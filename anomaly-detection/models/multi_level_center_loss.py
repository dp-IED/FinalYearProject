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
    Multi-level center loss with repulsion (window + sensor levels).

    Learns separate centers at two levels:
    1. Window-level: Normal vs. anomalous window embeddings (coarse-grained)
    2. Sensor-level: Normal vs. anomalous per-sensor embeddings (fine-grained)

    CRITICAL: All embeddings and centers are L2-normalized before computing
    distances to avoid degenerate solutions (centers at opposite poles).
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
        self.window_centers = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.window_centers)

        # Sensor-level centers (8 sensors × 2 classes)
        self.sensor_centers = nn.Parameter(
            torch.randn(num_sensors, num_classes, embed_dim)
        )
        nn.init.xavier_uniform_(self.sensor_centers)

    def forward(
        self,
        window_embeddings,  # (B, embed_dim)
        sensor_embeddings,  # (B, num_sensors, embed_dim)
        window_labels,  # (B,) - 0=normal, 1=anomaly
        sensor_labels=None,  # (B, num_sensors) - optional, 0=normal, 1=faulty
    ):
        """
        Compute hierarchical center loss with repulsion (window + sensor levels).

        CRITICAL FIX: All embeddings and centers must be L2-normalized before
        computing distances to avoid degenerate solutions (centers at poles).

        Args:
            window_embeddings: Aggregated window-level embeddings
            sensor_embeddings: Per-sensor embeddings (post-GAT)
            window_labels: Binary labels for windows
            sensor_labels: Per-sensor binary labels (if available)

        Returns:
            total_loss: Combined window + sensor center loss
        """
        B, N, D = sensor_embeddings.shape

        # ===================================================================
        # CRITICAL FIX: Normalize all embeddings and centers
        # ===================================================================
        window_embeddings = F.normalize(window_embeddings, p=2, dim=1)
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)

        # Normalize centers (both window and sensor)
        window_centers_norm = F.normalize(self.window_centers, p=2, dim=1)
        sensor_centers_norm = F.normalize(self.sensor_centers, p=2, dim=2)

        # ===================================================================
        # 1. Window-level center loss
        # ===================================================================
        window_intra_loss = 0.0
        window_count = 0

        for class_id in range(self.num_classes):
            mask = window_labels == class_id
            if mask.sum() == 0:
                continue

            class_embeddings = window_embeddings[mask]
            class_center = window_centers_norm[class_id]

            # Intra-class: pull embeddings to center
            distances = torch.norm(class_embeddings - class_center, p=2, dim=1)
            window_intra_loss += distances.mean()
            window_count += 1

        if window_count > 0:
            window_intra_loss /= window_count

        # Window-level repulsion: push centers apart
        if self.num_classes == 2:
            center_distance = torch.norm(
                window_centers_norm[0] - window_centers_norm[1], p=2
            )
            # Exponential repulsion (stronger as centers get closer)
            window_repulsion_loss = torch.exp(-center_distance + 1.0)
        else:
            window_repulsion_loss = 0.0

        # ===================================================================
        # 2. Sensor-level center loss
        # ===================================================================
        sensor_intra_loss = 0.0
        sensor_count = 0

        # Use sensor_labels if provided, otherwise fallback to window_labels
        if sensor_labels is None:
            sensor_labels = window_labels.unsqueeze(1).expand(-1, N)

        for sensor_idx in range(N):
            sensor_embs = sensor_embeddings[:, sensor_idx, :]  # (B, D)
            sensor_labs = sensor_labels[:, sensor_idx]  # (B,)

            for class_id in range(self.num_classes):
                mask = sensor_labs == class_id
                if mask.sum() == 0:
                    continue

                class_embeddings = sensor_embs[mask]
                class_center = sensor_centers_norm[sensor_idx, class_id]

                # Intra-class: pull embeddings to center
                distances = torch.norm(class_embeddings - class_center, p=2, dim=1)
                sensor_intra_loss += distances.mean()
                sensor_count += 1

        if sensor_count > 0:
            sensor_intra_loss /= sensor_count

        # Sensor-level repulsion: push centers apart (per sensor)
        sensor_repulsion_loss = 0.0
        if self.num_classes == 2:
            for sensor_idx in range(N):
                center_distance = torch.norm(
                    sensor_centers_norm[sensor_idx, 0]
                    - sensor_centers_norm[sensor_idx, 1],
                    p=2,
                )
                # Exponential repulsion
                sensor_repulsion_loss += torch.exp(-center_distance + 1.0)
            sensor_repulsion_loss /= N

        # ===================================================================
        # 3. Combined loss
        # ===================================================================
        total_loss = (
            self.lambda_intra * window_intra_loss
            + window_repulsion_loss
            + self.lambda_sensor * sensor_intra_loss
            + self.lambda_sensor * sensor_repulsion_loss
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
        pos_centers = centers[labels.long()]  # (B, embed_dim)
        neg_labels = 1 - labels
        neg_centers = centers[neg_labels.long()]  # (B, embed_dim)

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

        window_sep = torch.norm(window_centers[0] - window_centers[1]).item()

        sensor_seps = (
            torch.norm(sensor_centers[:, 0, :] - sensor_centers[:, 1, :], dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        return {
            "window_separation": window_sep,
            "sensor_separations": sensor_seps,
            "sensor_mean_separation": sensor_seps.mean(),
            "sensor_min_separation": sensor_seps.min(),
            "sensor_max_separation": sensor_seps.max(),
        }


class SensorOnlyCenterLoss(nn.Module):
    """
    Sensor-level center loss ONLY (no window-level centers).

    This is simpler, faster, and better for sensor attribution in KAG.
    Learns separate normal/anomaly centers for each sensor type.

    CRITICAL: All embeddings and centers are L2-normalized before computing
    distances to avoid degenerate solutions (centers at opposite poles).
    """

    def __init__(
        self,
        embed_dim=64,
        num_sensors=8,
        num_classes=2,
        margin=2.0,
        lambda_intra=1.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.margin = margin
        self.lambda_intra = lambda_intra

        # Only sensor-level centers (num_sensors × num_classes)
        self.sensor_centers = nn.Parameter(
            torch.randn(num_sensors, num_classes, embed_dim)
        )
        nn.init.xavier_uniform_(self.sensor_centers)

    def forward(self, sensor_embeddings, sensor_labels):
        """
        Compute sensor-only center loss.

        Args:
            sensor_embeddings: (B, N, embed_dim) - sensor-level embeddings
            sensor_labels: (B, N) - sensor labels (0=normal, 1=anomaly)

        Returns:
            total_loss: scalar
        """
        B, N, D = sensor_embeddings.shape

        # Normalize embeddings and centers
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=2)
        sensor_centers_norm = F.normalize(self.sensor_centers, p=2, dim=2)

        # Intra-class loss (pull embeddings to centers)
        intra_loss = 0.0
        count = 0

        for sensor_idx in range(N):
            sensor_embs = sensor_embeddings[:, sensor_idx, :]  # (B, D)
            sensor_labs = sensor_labels[:, sensor_idx]  # (B,)

            for class_id in range(self.num_classes):
                mask = sensor_labs == class_id
                if mask.sum() == 0:
                    continue

                class_embeddings = sensor_embs[mask]
                class_center = sensor_centers_norm[sensor_idx, class_id]

                # Euclidean distance to center
                distances = torch.norm(class_embeddings - class_center, p=2, dim=1)
                intra_loss += distances.mean()
                count += 1

        if count > 0:
            intra_loss /= count

        # Repulsion loss (push normal/anomaly centers apart per sensor)
        repulsion_loss = 0.0
        if self.num_classes == 2:
            for sensor_idx in range(N):
                center_distance = torch.norm(
                    sensor_centers_norm[sensor_idx, 0]
                    - sensor_centers_norm[sensor_idx, 1],
                    p=2,
                )
                # Exponential repulsion (stronger as centers get closer)
                repulsion_loss += torch.exp(-center_distance + 1.0)
            repulsion_loss /= N

        # Combined loss
        total_loss = self.lambda_intra * intra_loss + repulsion_loss

        return total_loss

    def get_sensor_centers(self):
        """Return normalized sensor centers."""
        return F.normalize(self.sensor_centers, p=2, dim=2)

    def get_separations(self):
        """Compute per-sensor separation metrics."""
        sensor_centers_norm = F.normalize(self.sensor_centers, p=2, dim=2)

        sensor_seps = []
        if self.num_classes == 2:
            for sensor_idx in range(self.num_sensors):
                sep = torch.norm(
                    sensor_centers_norm[sensor_idx, 0]
                    - sensor_centers_norm[sensor_idx, 1],
                    p=2,
                ).item()
                sensor_seps.append(sep)

        return {
            "sensor_mean_separation": sum(sensor_seps) / len(sensor_seps)
            if sensor_seps
            else 0.0,
            "sensor_min_separation": min(sensor_seps) if sensor_seps else 0.0,
            "sensor_max_separation": max(sensor_seps) if sensor_seps else 0.0,
            "sensor_separations": sensor_seps,  # Individual per sensor
        }
