"""
GDN Model and Loss Functions

Extracted from gdn.ipynb for reuse in scripts.
Contains the MultiLabelGDN model and all associated loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class MultiLabelGDN(nn.Module):
    """
    Multi-Label GDN: Predicts which sensors are anomalous.

    Key differences from original GDN:
    1. Per-sensor anomaly classifier (not just reconstruction)
    2. Global window classifier (auxiliary task)
    3. Returns probabilities for each sensor
    """

    def __init__(self, num_nodes, window_size, embed_dim=64, top_k=3, hidden_dim=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = min(top_k, num_nodes - 1)
        self.hidden_dim = hidden_dim

        # Learned sensor embeddings
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)

        # Temporal encoder
        self.temporal_encoder = nn.GRU(
            input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        # Graph attention
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=False,
        )

        # Per-sensor anomaly classifier
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Global window classifier (auxiliary)
        self.global_classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def build_graph_from_embeddings(self):
        """Build Top-K graph from learned embeddings."""
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        sim_matrix.fill_diagonal_(-1e9)

        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)

        src_nodes = torch.arange(
            self.num_nodes, device=sim_matrix.device
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten()

        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        return edge_index

    def forward(self, x, return_global=False):
        """
        Forward pass through the model.

        Args:
            x: (B, W, N) input tensor where B=batch_size, W=window_size, N=num_sensors
            return_global: If True, also return global window anomaly probability

        Returns:
            - sensor_probs: (B, N) probability each sensor is anomalous
            - global_prob: (B,) probability window has any anomaly (optional, if return_global=True)
        """
        B, W, N = x.shape

        # Temporal encoding per sensor
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention
        edge_index = self.build_graph_from_embeddings()

        h_graph_list = []
        for i in range(B):
            h_gat = self.gat(h_last[i], edge_index)
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Per-sensor anomaly probability
        sensor_probs = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)

        if return_global:
            # Global window anomaly probability
            h_flat = h_graph.flatten(1)  # (B, N * hidden_dim)
            global_prob = self.global_classifier(h_flat).squeeze(-1)  # (B,)
            return sensor_probs, global_prob

        return sensor_probs

    def get_embeddings(self, x):
        """
        Extract normalized embeddings for center loss and distance-based scoring.

        Args:
            x: (B, W, N) input tensor

        Returns:
            embeddings: (B, hidden_dim) L2-normalized embeddings per window
        """
        B, W, N = x.shape

        # Temporal encoding (same as forward)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention
        edge_index = self.build_graph_from_embeddings()

        h_graph_list = []
        for i in range(B):
            h_gat = self.gat(h_last[i], edge_index)
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Aggregate across sensors: mean pooling
        embeddings = h_graph.mean(dim=1)  # (B, hidden_dim)

        # L2 normalize for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class CenterLoss(nn.Module):
    """
    Center Loss for anomaly detection.
    Learns separate centers for normal (class 0) and anomalous (class 1) samples.
    Pulls samples toward their class center, creating better separation.

    Reference:
    Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """

    def __init__(self, embed_dim, num_classes=2, alpha=0.5):
        super().__init__()
        # Initialize centers further apart to prevent collapse
        # Normal center near origin, anomalous center offset
        centers_init = torch.zeros(num_classes, embed_dim)
        centers_init[0] = torch.randn(embed_dim) * 0.05  # Normal center near origin
        centers_init[1] = (
            torch.randn(embed_dim) * 0.05 + torch.ones(embed_dim) * 0.3
        )  # Anomalous center offset
        self.centers = nn.Parameter(centers_init)
        self.alpha = alpha  # Learning rate for center updates (used in SGD optimizer)

    def forward(self, embeddings, labels):
        """
        Compute center loss: pull embeddings toward their class center.

        Args:
            embeddings: (B, D) normalized embeddings from model
            labels: (B,) binary labels (0=normal, 1=anomalous)

        Returns:
            loss: scalar tensor
        """
        # Get the center assigned to each sample
        centers_batch = self.centers.index_select(0, labels.long())  # (B, D)

        # Compute distance to assigned centers
        loss = (embeddings - centers_batch).pow(2).sum(dim=1).mean()

        return loss

    def get_center_separation(self):
        """L2 distance between normal and anomalous centers."""
        return torch.norm(self.centers[0] - self.centers[1]).item()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, *) predictions (logits or probabilities)
            targets: (N, *) binary labels (0 or 1)
        """
        # Ensure inputs are probabilities
        p = torch.sigmoid(inputs) if inputs.min() < 0 or inputs.max() > 1 else inputs

        # Compute focal term: (1 - p_t)^gamma
        # For positive samples: use p
        # For negative samples: use (1 - p)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross entropy
        ce_loss = F.binary_cross_entropy(p, targets, reduction="none")

        # Apply focal weight and alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricFocalContrastiveLoss(nn.Module):
    """
    Combines focal loss weighting with contrastive learning.
    Emphasizes hard-to-classify samples in embedding space.
    Supports both single-label (B,) and multi-label (B, D) inputs.

    IMPROVEMENTS:
    - Consistent numerical stability (max subtraction) for all branches
    - Proper diagonal masking to exclude self-pairs
    - Clipping focal weights to prevent gradient explosion
    - Better epsilon handling for log stability
    """

    def __init__(self, temperature=0.07, gamma=2.0, alpha=0.25):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, D) normalized embeddings/predictions (values in [0,1])
            labels: (B,) binary labels or (B, D) multi-label
        """
        device = embeddings.device
        B = embeddings.size(0)

        # Handle multi-label case: compute contrastive loss per sensor
        if labels.dim() == 2:
            # labels: (B, num_sensors), embeddings: (B, num_sensors)
            num_sensors = labels.size(1)
            total_loss = 0.0
            num_valid_sensors = 0

            for sensor_idx in range(num_sensors):
                sensor_labels = labels[:, sensor_idx]  # (B,)
                sensor_embeddings = embeddings[:, sensor_idx].unsqueeze(1)  # (B, 1)

                # Skip if all labels are same (no positive pairs possible)
                if sensor_labels.min() == sensor_labels.max():
                    continue

                num_valid_sensors += 1

                # Create masks for this sensor
                sensor_labels_expanded = sensor_labels.unsqueeze(1)  # (B, 1)
                mask_pos = (
                    sensor_labels_expanded == sensor_labels_expanded.t()
                ).float()  # (B, B)
                mask_pos.fill_diagonal_(0)

                # Compute similarity matrix
                sim_matrix = (
                    torch.mm(sensor_embeddings, sensor_embeddings.t())
                    / self.temperature
                )  # (B, B)

                # Compute probabilities with numerical stability
                max_sim = sim_matrix.max(dim=1, keepdim=True)[0]
                sim_matrix_stable = sim_matrix - max_sim

                # Mask diagonal out during softmax computation
                diag_mask = torch.eye(B, device=device)
                sim_matrix_stable = sim_matrix_stable - diag_mask * 1e10

                exp_sim = torch.exp(sim_matrix_stable)
                pos_sim = (exp_sim * mask_pos).sum(dim=1)
                all_sim = exp_sim.sum(dim=1)

                # Probability of correct pairing with clipping for stability
                p_t = torch.clamp(pos_sim / (all_sim + 1e-10), min=1e-7, max=1.0 - 1e-7)

                # Apply focal weighting with clipping to prevent explosion
                focal_weight = torch.clamp(
                    (1.0 - p_t) ** self.gamma, min=1e-4, max=100.0
                )

                # Contrastive loss with focal weighting
                has_pos = mask_pos.sum(dim=1) > 0
                if has_pos.sum() > 0:
                    loss_per_sample = -focal_weight[has_pos] * torch.log(
                        p_t[has_pos] + 1e-10
                    )
                    loss_per_sample = torch.clamp(loss_per_sample, min=0.0, max=100.0)
                    sensor_loss = loss_per_sample.mean()
                    total_loss = total_loss + sensor_loss

            if num_valid_sensors == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            return total_loss / num_valid_sensors

        else:
            # Single-label case: (B,)
            sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

            # Create masks
            labels = labels.unsqueeze(1)
            mask_pos = (labels == labels.t()).float()
            mask_pos.fill_diagonal_(0)

            # Compute probabilities with numerical stability
            max_sim = sim_matrix.max(dim=1, keepdim=True)[0]
            sim_matrix_stable = sim_matrix - max_sim

            # Mask diagonal out
            diag_mask = torch.eye(B, device=device)
            sim_matrix_stable = sim_matrix_stable - diag_mask * 1e10

            exp_sim = torch.exp(sim_matrix_stable)
            pos_sim = (exp_sim * mask_pos).sum(dim=1)
            all_sim = exp_sim.sum(dim=1)

            # Probability with clipping
            p_t = torch.clamp(pos_sim / (all_sim + 1e-10), min=1e-7, max=1.0 - 1e-7)

            # Apply focal weighting with clipping
            focal_weight = torch.clamp((1.0 - p_t) ** self.gamma, min=1e-4, max=100.0)

            # Contrastive loss
            has_pos = mask_pos.sum(dim=1) > 0
            if has_pos.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            loss_per_sample = -focal_weight[has_pos] * torch.log(p_t[has_pos] + 1e-10)
            loss_per_sample = torch.clamp(loss_per_sample, min=0.0, max=100.0)

            return loss_per_sample.mean()


# ============================================================================
# Metric Learning Losses
# ============================================================================

class ProxyNCALoss(nn.Module):
    """
    Proxy-NCA Loss: Learnable class proxies for metric learning.
    
    More stable and faster than triplet loss. Learns proxies (class representatives)
    in embedding space and minimizes distance to correct proxy while maximizing
    distance to incorrect proxies.
    
    This is ideal for training embedding models on classification labels because:
    1. Directly optimizes embedding separation (not classification accuracy)
    2. More stable than triplet sampling
    3. Faster than computing all pairwise distances
    4. Works well with normalized embeddings
    """
    
    def __init__(self, embed_dim, num_classes=2, alpha=32.0):
        """
        Args:
            embed_dim: Embedding dimension
            num_classes: Number of classes (2 for normal/anomalous)
            alpha: Temperature parameter (higher = sharper distribution)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Learnable proxies: one per class
        # These represent the "ideal" embedding for each class
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.proxies)
    
    def forward(self, embeddings, labels):
        """
        Compute Proxy-NCA loss using squared Euclidean distance (standard implementation).
        
        Standard Proxy-NCA formula:
        Loss = -log(exp(-alpha * ||x - p_y||^2) / sum_{p} exp(-alpha * ||x - p||^2))
             = alpha * ||x - p_y||^2 - log(sum_{p} exp(-alpha * ||x - p||^2))
        
        Args:
            embeddings: (B, D) embeddings (can be normalized or not)
            labels: (B,) class labels (0=normal, 1=anomalous)
        
        Returns:
            loss: Scalar loss value
        """
        # Compute squared Euclidean distances to all proxies
        # ||x - p||^2 = ||x||^2 + ||p||^2 - 2*x^T*p
        # For normalized embeddings/proxies: ||x||^2 = ||p||^2 = 1, so:
        # ||x - p||^2 = 2 - 2*x^T*p = 2*(1 - x^T*p)
        
        # Normalize embeddings and proxies for stability
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (B, D)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)  # (num_classes, D)
        
        # Compute squared Euclidean distances: ||x - p||^2
        # Using expanded form: ||x||^2 + ||p||^2 - 2*x^T*p
        # For normalized vectors: 2 - 2*x^T*p = 2*(1 - x^T*p)
        similarities = torch.mm(embeddings_norm, proxies_norm.t())  # (B, num_classes)
        squared_distances = 2.0 * (1.0 - similarities)  # (B, num_classes)
        
        # Get distance to correct proxy
        correct_dist_sq = squared_distances[range(len(labels)), labels.long()]  # (B,)
        
        # Proxy-NCA loss: -log(exp(-alpha * d_correct^2) / sum(exp(-alpha * d_all^2)))
        #                  = alpha * d_correct^2 - log(sum(exp(-alpha * d_all^2)))
        # Using log-sum-exp trick for numerical stability
        log_sum_exp = torch.logsumexp(-self.alpha * squared_distances, dim=1)  # (B,)
        
        # Final loss: alpha * d_correct^2 - log(sum(exp(-alpha * d_all^2)))
        loss = (self.alpha * correct_dist_sq.mean() - log_sum_exp.mean())
        
        return loss
    
    def get_proxy_separation(self):
        """Get separation between class proxies (for monitoring)."""
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)
        separation = torch.norm(proxies_norm[0] - proxies_norm[1]).item()
        return separation


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Maximizes margin between normal and anomalous embeddings.
    More explicit than Proxy-NCA but requires careful triplet mining.
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: Minimum margin between positive and negative pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Compute triplet loss.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) binary labels (0=normal, 1=anomalous)
        
        Returns:
            loss: Scalar loss value
        """
        normal_mask = labels == 0
        anomalous_mask = labels == 1
        
        if not (normal_mask.any() and anomalous_mask.any()):
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        normal_emb = embeddings[normal_mask]  # (N, D)
        anomalous_emb = embeddings[anomalous_mask]  # (A, D)
        
        if len(normal_emb) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Sample anchor-positive pairs from normal samples
        # Use random sampling (can be improved with hard negative mining)
        anchor_idx = torch.randint(0, len(normal_emb), (len(normal_emb),), device=embeddings.device)
        positive_idx = torch.randint(0, len(normal_emb), (len(normal_emb),), device=embeddings.device)
        # Ensure anchor != positive
        positive_idx = torch.where(anchor_idx == positive_idx,
                                   (positive_idx + 1) % len(normal_emb),
                                   positive_idx)
        
        anchor = normal_emb[anchor_idx]
        positive = normal_emb[positive_idx]
        
        # Find hardest negative (closest anomalous sample to anchor)
        neg_distances = torch.cdist(anchor.unsqueeze(0), anomalous_emb.unsqueeze(0)).squeeze(0)  # (N, A)
        hard_neg_idx = neg_distances.argmin(dim=1)  # (N,)
        negative = anomalous_emb[hard_neg_idx]  # (N, D)
        
        # Compute distances
        pos_dist = torch.norm(anchor - positive, dim=1)  # (N,)
        neg_dist = torch.norm(anchor - negative, dim=1)  # (N,)
        
        # Triplet loss: maximize margin
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning.
    
    Pulls similar samples together and pushes dissimilar samples apart.
    Simpler than triplet loss, works well for binary classification.
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: Margin for negative pairs (dissimilar samples should be > margin apart)
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Compute contrastive loss.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) binary labels (0=normal, 1=anomalous)
        
        Returns:
            loss: Scalar loss value
        """
        # Create pairs: (i, j) where i < j
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings)  # (B, B)
        
        # Create mask for pairs (upper triangle, excluding diagonal)
        pair_mask = torch.triu(torch.ones(B, B, device=embeddings.device), diagonal=1).bool()
        
        # Get pairwise labels: 1 if same class, 0 if different
        labels_expanded = labels.unsqueeze(0).expand(B, B)
        same_class = (labels_expanded == labels_expanded.t()).float()
        same_class_pairs = same_class[pair_mask]  # (num_pairs,)
        
        # Get distances for pairs
        pair_distances = distances[pair_mask]  # (num_pairs,)
        
        # Contrastive loss:
        # - For positive pairs (same class): minimize distance
        # - For negative pairs (different class): maximize distance (with margin)
        pos_loss = same_class_pairs * pair_distances ** 2  # Pull similar together
        neg_loss = (1 - same_class_pairs) * F.relu(self.margin - pair_distances) ** 2  # Push dissimilar apart
        
        loss = (pos_loss + neg_loss).mean()
        
        return loss


# ============================================================================
# Fast Improved GDN Model (Simplified for Speed)
# ============================================================================

class FastImprovedMultiLabelGDN(nn.Module):
    """
    Fast version of ImprovedMultiLabelGDN with reduced complexity.
    
    Optimizations:
    1. Unidirectional GRU (instead of bidirectional) - 2x faster
    2. Single GAT layer (instead of multiple) - faster forward pass
    3. Simplified sensor encoders (shared encoder only, no adapters) - less computation
    4. Still includes: LayerNorm, GATv2Conv, efficient batched processing
    
    This trades some model capacity for 3-5x faster training while maintaining
    key architectural improvements.
    """
    
    def __init__(self, num_nodes, window_size, embed_dim=64, top_k=3, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        # Simplified temporal encoder: shared unidirectional GRU (no sensor adapters)
        self.temporal_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,  # Single layer for speed
            batch_first=True,
            bidirectional=False,  # Unidirectional for speed
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Single GAT layer (instead of multiple)
        self.gat = GATv2Conv(
            hidden_dim,
            hidden_dim,  # Single head for speed
            heads=1,
            concat=False,
            dropout=0.2,
            add_self_loops=True,
        )
        self.gat_norm = nn.LayerNorm(hidden_dim)
        
        # Sensor embeddings for graph construction (properly initialized)
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)
        
        # Classifiers (output logits, not probabilities - for better training stability)
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Simple attention aggregation (simpler than full AttentionAggregation)
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Initialize all linear layers with xavier_uniform
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all linear layers with xavier_uniform for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def build_graph_from_embeddings(self):
        """Build Top-K graph from learned embeddings."""
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        sim_matrix.fill_diagonal_(-1e9)
        
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)
        
        src_nodes = torch.arange(
            self.num_nodes, device=sim_matrix.device
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten()
        
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        return edge_index
    
    def forward(self, x, return_global=False):
        """
        Forward pass through the model.
        
        Args:
            x: (B, W, N) input tensor where B=batch_size, W=window_size, N=num_sensors
            return_global: If True, also return global window anomaly probability
        
        Returns:
            - sensor_probs: (B, N) probability each sensor is anomalous
            - global_prob: (B,) probability window has any anomaly (optional, if return_global=True)
        """
        B, W, N = x.shape
        
        # Simplified temporal encoding: process all sensors through shared GRU
        # Reshape: (B*N, W, 1)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)  # (B, N, hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # Efficient batched GAT processing
        edge_index = self.build_graph_from_embeddings()
        
        # Reshape for batched GAT: (B*N, hidden_dim)
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        # Create batched edge_index (vectorized)
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # (B, 1)
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)  # (B, 2, num_edges)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)  # Add offset
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)  # (2, B*num_edges)
        
        # Single GAT layer
        h_gat = self.gat(h_flat, edge_index_batched)
        
        # Reshape back: (B, N, hidden_dim)
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        
        # Normalization
        h_graph = self.gat_norm(h_graph)
        
        # Simple attention aggregation for global representation
        attention_scores = self.attention_weights(h_graph)  # (B, N, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, N, 1)
        h_global = (h_graph * attention_weights).sum(dim=1)  # (B, hidden_dim)
        
        # Per-sensor anomaly logits (output logits, convert to probabilities for return)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)
        sensor_probs = torch.sigmoid(sensor_logits)  # Convert to probabilities
        
        if return_global:
            # Global window anomaly logits
            global_logits = self.global_classifier(h_global).squeeze(-1)  # (B,)
            global_prob = torch.sigmoid(global_logits)  # Convert to probabilities
            return sensor_probs, global_prob
        
        return sensor_probs
    
    def forward_logits(self, x):
        """
        Forward pass returning logits instead of probabilities (for training efficiency).
        
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            sensor_logits: (B, N) logits for each sensor
            global_logits: (B,) logits for global window
            embeddings: (B, hidden_dim) normalized embeddings
        """
        B, W, N = x.shape
        
        # Temporal encoding
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        h_gat = self.gat(h_flat, edge_index_batched)
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        h_graph = self.gat_norm(h_graph)
        
        # Attention aggregation
        attention_scores = self.attention_weights(h_graph)
        attention_weights = F.softmax(attention_scores, dim=1)
        h_global = (h_graph * attention_weights).sum(dim=1)
        
        # Get logits directly (before sigmoid)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)
        global_logits = self.global_classifier(h_global).squeeze(-1)  # (B,)
        
        # Get embeddings
        embeddings = F.normalize(h_global, p=2, dim=1)
        
        return sensor_logits, global_logits, embeddings
    
    def get_embeddings(self, x):
        """
        Get normalized embeddings for a batch of windows.
        
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            embeddings: (B, hidden_dim) normalized embeddings
        """
        B, W, N = x.shape
        
        # Temporal encoding
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        h_gat = self.gat(h_flat, edge_index_batched)
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        h_graph = self.gat_norm(h_graph)
        
        # Attention aggregation
        attention_scores = self.attention_weights(h_graph)
        attention_weights = F.softmax(attention_scores, dim=1)
        h_global = (h_graph * attention_weights).sum(dim=1)
        
        # Normalize embeddings
        return F.normalize(h_global, p=2, dim=1)


# ============================================================================
# Metric Learning GDN Model (Optimized for Embedding Quality)
# ============================================================================

class MetricLearningGDN(nn.Module):
    """
    GDN model optimized for metric learning.
    
    Key differences from classification-focused models:
    1. Larger embedding dimension (128-256) for better separation
    2. Stronger encoder (bidirectional GRU, 2 layers)
    3. Multi-layer GAT with residuals for richer representations
    4. Batch normalization for stable training
    5. Designed to maximize embedding separation, not classification accuracy
    
    This model prioritizes learning good embeddings that can be separated
    in embedding space, rather than optimizing classification directly.
    """
    
    def __init__(self, num_nodes, window_size, embed_dim=128, top_k=3, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        
        # Strong temporal encoder: bidirectional GRU, 2 layers
        self.temporal_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim // 2,  # Bidirectional halves hidden size
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-layer GAT with residuals
        self.gat1 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=True,
        )
        self.gat1_norm = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=True,
        )
        self.gat2_norm = nn.LayerNorm(hidden_dim)
        
        # Sensor embeddings for graph construction
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)
        
        # Attention aggregation for global embedding
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Embedding projection (for metric learning)
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        
        # Classifiers (auxiliary, for regularization)
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all linear layers with xavier_uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def build_graph_from_embeddings(self):
        """Build Top-K graph from learned embeddings."""
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        sim_matrix.fill_diagonal_(-1e9)
        
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)
        
        src_nodes = torch.arange(
            self.num_nodes, device=sim_matrix.device
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten()
        
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        return edge_index
    
    def forward(self, x, return_global=False):
        """
        Forward pass.
        
        Args:
            x: (B, W, N) input tensor
            return_global: If True, also return global window probability
        
        Returns:
            sensor_probs: (B, N) probability each sensor is anomalous
            global_prob: (B,) probability window has any anomaly (optional)
        """
        B, W, N = x.shape
        
        # Temporal encoding
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing (batched)
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        # First GAT layer
        h_gat1 = self.gat1(h_flat, edge_index_batched)
        h_gat1 = h_gat1.reshape(B, N, self.hidden_dim)
        h_gat1 = self.gat1_norm(h_gat1)
        
        # Residual connection
        h_gat1 = h_gat1 + h_last
        
        # Second GAT layer
        h_gat1_flat = h_gat1.reshape(B * N, self.hidden_dim)
        h_gat2 = self.gat2(h_gat1_flat, edge_index_batched)
        h_gat2 = h_gat2.reshape(B, N, self.hidden_dim)
        h_gat2 = self.gat2_norm(h_gat2)
        
        # Residual connection
        h_graph = h_gat2 + h_gat1
        
        # Attention aggregation
        attention_scores = self.attention_weights(h_graph)  # (B, N, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        h_global = (h_graph * attention_weights).sum(dim=1)  # (B, hidden_dim)
        
        # Per-sensor logits (auxiliary)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)
        sensor_probs = torch.sigmoid(sensor_logits)
        
        if return_global:
            global_logits = self.global_classifier(h_global).squeeze(-1)  # (B,)
            global_prob = torch.sigmoid(global_logits)
            return sensor_probs, global_prob
        
        return sensor_probs
    
    def forward_logits(self, x):
        """
        Forward pass returning logits (for training efficiency).
        
        Returns:
            sensor_logits: (B, N) logits for each sensor
            global_logits: (B,) logits for global window
            embeddings: (B, hidden_dim) normalized embeddings
        """
        B, W, N = x.shape
        
        # Temporal encoding
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        # Multi-layer GAT with residuals
        h_gat1 = self.gat1(h_flat, edge_index_batched)
        h_gat1 = h_gat1.reshape(B, N, self.hidden_dim)
        h_gat1 = self.gat1_norm(h_gat1)
        h_gat1 = h_gat1 + h_last
        
        h_gat1_flat = h_gat1.reshape(B * N, self.hidden_dim)
        h_gat2 = self.gat2(h_gat1_flat, edge_index_batched)
        h_gat2 = h_gat2.reshape(B, N, self.hidden_dim)
        h_gat2 = self.gat2_norm(h_gat2)
        h_graph = h_gat2 + h_gat1
        
        # Attention aggregation
        attention_scores = self.attention_weights(h_graph)
        attention_weights = F.softmax(attention_scores, dim=1)
        h_global = (h_graph * attention_weights).sum(dim=1)
        
        # Embedding projection and normalization
        embeddings = self.embedding_proj(h_global)  # (B, hidden_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize for metric learning
        
        # Get logits
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)
        global_logits = self.global_classifier(h_global).squeeze(-1)
        
        return sensor_logits, global_logits, embeddings
    
    def get_embeddings(self, x):
        """
        Get normalized embeddings for a batch of windows.
        
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            embeddings: (B, hidden_dim) normalized embeddings
        """
        B, W, N = x.shape
        
        # Temporal encoding
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, self.hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        h_gat1 = self.gat1(h_flat, edge_index_batched)
        h_gat1 = h_gat1.reshape(B, N, self.hidden_dim)
        h_gat1 = self.gat1_norm(h_gat1)
        h_gat1 = h_gat1 + h_last
        
        h_gat1_flat = h_gat1.reshape(B * N, self.hidden_dim)
        h_gat2 = self.gat2(h_gat1_flat, edge_index_batched)
        h_gat2 = h_gat2.reshape(B, N, self.hidden_dim)
        h_gat2 = self.gat2_norm(h_gat2)
        h_graph = h_gat2 + h_gat1
        
        # Attention aggregation
        attention_scores = self.attention_weights(h_graph)
        attention_weights = F.softmax(attention_scores, dim=1)
        h_global = (h_graph * attention_weights).sum(dim=1)
        
        # Embedding projection and normalization
        embeddings = self.embedding_proj(h_global)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# ============================================================================
# Improved GDN Model Classes (Full Version)
# ============================================================================


class SensorSpecificTemporalEncoder(nn.Module):
    """
    Sensor-specific temporal encoder with shared bidirectional GRU + sensor-specific adapters.
    
    More parameter-efficient than fully independent encoders while still allowing
    sensor-specific pattern learning. The shared encoder learns common temporal patterns,
    while sensor-specific adapters transform features to sensor-aware representations.
    
    This helps separation because:
    1. Each sensor (RPM, temperature, pressure) has unique temporal patterns
    2. Sensor-specific adapters learn sensor-aware fault signatures
    3. Creates multi-dimensional separation in embedding space
    """
    
    def __init__(self, num_sensors, window_size, hidden_dim, num_layers=2, bidirectional=True):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # Shared encoder learns common temporal patterns
        self.shared_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Sensor-specific adapters learn sensor-specific transformations
        # Each adapter projects shared features to sensor-specific space
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.sensor_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for _ in range(num_sensors)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            h_last: (B, N, hidden_dim) sensor-specific temporal features
        """
        B, W, N = x.shape
        
        # Process each sensor through shared encoder + sensor-specific adapter
        h_list = []
        for sensor_idx in range(N):
            # Extract sensor data: (B, W, 1)
            x_sensor = x[:, :, sensor_idx:sensor_idx+1]
            
            # Shared encoder: learns common temporal patterns
            h_temporal, _ = self.shared_encoder(x_sensor)  # (B, W, hidden_dim*2)
            
            # Use last timestep
            h_shared = h_temporal[:, -1, :]  # (B, hidden_dim*2)
            
            # Sensor-specific adapter: transforms to sensor-specific space
            h_sensor = self.sensor_adapters[sensor_idx](h_shared)  # (B, hidden_dim)
            
            h_list.append(h_sensor)
        
        # Stack: (B, N, hidden_dim)
        h_last = torch.stack(h_list, dim=1)
        
        return h_last


class ResidualGATLayer(nn.Module):
    """
    GAT layer with residual connection and normalization.
    
    Improves gradient flow and enables deeper networks by allowing
    information to bypass layers through residual connections.
    """
    
    def __init__(self, hidden_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,  # Per-head dimension
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,  # Important for self-attention
        )
        # Projection to match dimensions after concat
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        residual = x
        x = self.gat(x, edge_index)
        x = self.proj(x)  # Project back to hidden_dim
        x = self.norm(x + residual)  # Residual + normalization
        x = self.dropout(x)
        return x


class AttentionAggregation(nn.Module):
    """
    Attention-based aggregation for sensor embeddings.
    
    Computes adaptive weights for each sensor instead of simple mean pooling.
    This allows the model to focus on more informative sensors.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, h_graph):
        """
        Args:
            h_graph: (B, N, hidden_dim) sensor features
        
        Returns:
            aggregated: (B, hidden_dim) weighted aggregation
        """
        attn_weights = F.softmax(self.attention(h_graph), dim=1)  # (B, N, 1)
        aggregated = (h_graph * attn_weights).sum(dim=1)  # (B, hidden_dim)
        return aggregated


class ImprovedMultiLabelGDN(nn.Module):
    """
    Improved Multi-Label GDN with best practices.
    
    Key improvements:
    1. HIGH PRIORITY:
       - Efficient batched GAT processing (5-10x faster)
       - LayerNorm for training stability
       - GATv2Conv for better attention
    
    2. MEDIUM PRIORITY:
       - Multi-layer GAT with residual connections
       - Attention-based aggregation
       - Bidirectional GRU for temporal patterns
    
    3. SENSOR-SPECIFIC ENCODERS:
       - Shared encoder + sensor-specific adapters
       - Better separation in embedding space
    
    Maintains backward compatibility with MultiLabelGDN interface.
    """
    
    def __init__(
        self,
        num_nodes,
        window_size,
        embed_dim=64,
        top_k=3,
        hidden_dim=64,
        num_gat_layers=2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = min(top_k, num_nodes - 1)
        self.hidden_dim = hidden_dim
        
        # Learned sensor embeddings for graph construction
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)
        
        # Sensor-specific temporal encoder (shared + adapters)
        self.temporal_encoder = SensorSpecificTemporalEncoder(
            num_sensors=num_nodes,
            window_size=window_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            bidirectional=True
        )
        
        # Normalization after temporal encoding
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-layer GAT with residuals
        self.gat_layers = nn.ModuleList([
            ResidualGATLayer(hidden_dim, heads=4, dropout=0.2)
            for _ in range(num_gat_layers)
        ])
        
        # Final GAT layer (no residual, output dimension)
        self.gat_final = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=True
        )
        
        # Normalization after GAT
        self.gat_norm = nn.LayerNorm(hidden_dim)
        
        # Attention-based aggregation
        self.aggregation = AttentionAggregation(hidden_dim)
        
        # Per-sensor anomaly classifier
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Global window classifier (auxiliary)
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Using aggregated hidden_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def build_graph_from_embeddings(self):
        """Build Top-K graph from learned embeddings."""
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        sim_matrix.fill_diagonal_(-1e9)
        
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)
        
        src_nodes = torch.arange(
            self.num_nodes, device=sim_matrix.device
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten()
        
        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        return edge_index
    
    def forward(self, x, return_global=False):
        """
        Forward pass with efficient batched processing.
        
        Args:
            x: (B, W, N) input tensor
            return_global: If True, also return global window anomaly probability
        
        Returns:
            - sensor_probs: (B, N) probability each sensor is anomalous
            - global_prob: (B,) probability window has any anomaly (optional)
        """
        B, W, N = x.shape
        
        # Sensor-specific temporal encoding
        h_last = self.temporal_encoder(x)  # (B, N, hidden_dim)
        h_last = self.temporal_norm(h_last)
        
        # Efficient batched GAT processing
        edge_index = self.build_graph_from_embeddings()
        
        # Reshape for batched GAT: (B*N, hidden_dim)
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        # Create batched edge_index (vectorized - much faster than loop)
        # Each graph in batch gets offset by b*N
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # (B, 1)
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)  # (B, 2, num_edges)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)  # Add offset to each batch
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)  # (2, B*num_edges)
        
        # Multi-layer GAT with residuals
        h_gat = h_flat
        for gat_layer in self.gat_layers:
            h_gat = gat_layer(h_gat, edge_index_batched)
        
        # Final GAT layer
        h_gat = self.gat_final(h_gat, edge_index_batched)
        
        # Reshape back: (B, N, hidden_dim)
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        
        # Normalization
        h_graph = self.gat_norm(h_graph)
        
        # Per-sensor anomaly probability
        sensor_probs = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)
        
        if return_global:
            # Attention-based aggregation
            h_aggregated = self.aggregation(h_graph)  # (B, hidden_dim)
            global_prob = self.global_classifier(h_aggregated).squeeze(-1)  # (B,)
            return sensor_probs, global_prob
        
        return sensor_probs
    
    def get_embeddings(self, x):
        """
        Extract normalized embeddings for center loss and distance-based scoring.
        
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            embeddings: (B, hidden_dim) L2-normalized embeddings per window
        """
        B, W, N = x.shape
        
        # Temporal encoding (same as forward)
        h_last = self.temporal_encoder(x)
        h_last = self.temporal_norm(h_last)
        
        # Graph attention (batched)
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        batch_edge_indices = []
        for b in range(B):
            offset = b * N
            batch_edges = edge_index + offset
            batch_edge_indices.append(batch_edges)
        edge_index_batched = torch.cat(batch_edge_indices, dim=1)
        
        h_gat = h_flat
        for gat_layer in self.gat_layers:
            h_gat = gat_layer(h_gat, edge_index_batched)
        h_gat = self.gat_final(h_gat, edge_index_batched)
        
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        h_graph = self.gat_norm(h_graph)
        
        # Attention-based aggregation instead of mean
        embeddings = self.aggregation(h_graph)  # (B, hidden_dim)
        
        # L2 normalize for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward_logits(self, x):
        """
        Forward pass returning logits instead of probabilities (for training efficiency).
        
        Args:
            x: (B, W, N) input tensor
        
        Returns:
            sensor_logits: (B, N) logits for each sensor
            global_logits: (B,) logits for global window
            embeddings: (B, hidden_dim) normalized embeddings
        """
        B, W, N = x.shape
        
        # Temporal encoding
        h_last = self.temporal_encoder(x)
        h_last = self.temporal_norm(h_last)
        
        # GAT processing
        edge_index = self.build_graph_from_embeddings()
        h_flat = h_last.reshape(B * N, self.hidden_dim)
        
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)
        
        h_gat = h_flat
        for gat_layer in self.gat_layers:
            h_gat = gat_layer(h_gat, edge_index_batched)
        h_gat = self.gat_final(h_gat, edge_index_batched)
        
        h_graph = h_gat.reshape(B, N, self.hidden_dim)
        h_graph = self.gat_norm(h_graph)
        
        # Attention aggregation
        h_aggregated = self.aggregation(h_graph)
        
        # Get logits (before sigmoid) - need to modify classifiers to output logits
        # For now, get probabilities and convert to logits
        sensor_probs = self.sensor_classifier(h_graph).squeeze(-1)
        global_probs = self.global_classifier(h_aggregated).squeeze(-1)
        
        # Convert probabilities to logits (inverse sigmoid)
        sensor_logits = torch.logit(sensor_probs.clamp(1e-7, 1-1e-7))
        global_logits = torch.logit(global_probs.clamp(1e-7, 1-1e-7))
        
        # Get embeddings
        embeddings = F.normalize(h_aggregated, p=2, dim=1)
        
        return sensor_logits, global_logits, embeddings
