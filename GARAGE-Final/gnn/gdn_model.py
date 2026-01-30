"""
GDN Model and Loss Functions

Extracted from gdn.ipynb for reuse in scripts.
Contains the MultiLabelGDN model and all associated loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class TemporalAttentionPooling(nn.Module):
    """
    Attention-weighted pooling over temporal dimension.
    Learns which timesteps are important for anomaly detection.

    Optimized: Now supports vectorized processing of all sensors at once.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_temporal):
        """
        Vectorized attention pooling that processes all sensors at once.

        Args:
            h_temporal: (B, T, hidden_dim) or (B, N, T, hidden_dim) GRU outputs
                       If 4D, processes all sensors in batch
        Returns:
            h_pooled: (B, hidden_dim) or (B, N, hidden_dim) attention-weighted average
            attention_weights: (B, T) or (B, N, T) for interpretability
        """
        if h_temporal.dim() == 4:
            # Vectorized mode: (B, N, T, hidden_dim) -> (B, N, hidden_dim)
            B, N, T, H = h_temporal.shape

            # Reshape for batch processing: (B*N, T, H)
            h_flat = h_temporal.reshape(B * N, T, H)

            # Compute attention scores for ALL sensors at once
            scores = self.attention(h_flat)  # (B*N, T, 1)
            weights = F.softmax(scores, dim=1)  # (B*N, T, 1)

            # Weighted sum
            h_pooled = (h_flat * weights).sum(dim=1)  # (B*N, H)

            # Reshape back
            h_pooled = h_pooled.reshape(B, N, H)
            weights = weights.squeeze(-1).reshape(B, N, T)

            return h_pooled, weights
        else:
            # Original mode: (B, T, hidden_dim) -> (B, hidden_dim)
            # Compute attention scores
            scores = self.attention(h_temporal)  # (B, T, 1)
            weights = F.softmax(scores, dim=1)  # (B, T, 1)

            # Weighted sum
            h_pooled = (h_temporal * weights).sum(dim=1)  # (B, hidden_dim)

            return h_pooled, weights.squeeze(-1)


class MultiScaleGAT(nn.Module):
    """
    Multi-scale GAT: applies GAT at different hops.
    - 1-hop: immediate neighbors
    - 2-hop: neighbors of neighbors
    Preserves local and global structure.
    """

    def __init__(self, hidden_dim, heads=2, dropout=0.4):
        super().__init__()
        self.gat_1hop = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.gat_2hop = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,  # Already covered by 1-hop
        )
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, edge_index):
        # 1-hop aggregation
        h1 = self.gat_1hop(x, edge_index)

        # 2-hop aggregation (apply GAT again on edge_index)
        # This reaches neighbors of neighbors
        h2 = self.gat_2hop(h1, edge_index)

        # Combine multi-scale features
        h_combined = torch.cat([h1, h2], dim=1)
        h_out = self.combine(h_combined)

        return h_out


class MultiLabelGDN(nn.Module):
    """
    Multi-Label GDN: Predicts which sensors are anomalous.

    Key differences from original GDN:
    1. Per-sensor anomaly classifier (not just reconstruction)
    2. Global window classifier (auxiliary task)
    3. Returns probabilities for each sensor
    """

    def __init__(
        self,
        num_nodes,
        window_size,
        embed_dim=32,
        top_k=5,
        hidden_dim=64,
        rebuild_graph_every=10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = min(top_k, num_nodes - 1)
        self.hidden_dim = hidden_dim
        self.rebuild_graph_every = rebuild_graph_every

        # Graph caching
        self.cached_edge_index = None
        self._graph_step_counter = 0

        # Learned sensor embeddings
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)

        # Enhanced temporal encoder: 2-layer bidirectional GRU
        self.temporal_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim // 2,  # Half size because bidirectional
            num_layers=2,  # 2 layers
            batch_first=True,
            dropout=0.2,  # Dropout between layers
            bidirectional=True,  # Look forward and backward
        )

        # Temporal attention pooling
        self.temporal_pooling = TemporalAttentionPooling(hidden_dim)

        # Multi-scale GAT to reduce over-smoothing
        self.multi_scale_gat = MultiScaleGAT(
            hidden_dim=hidden_dim, heads=2, dropout=0.4
        )

        # Normalization after GAT to stabilize training
        self.gat_norm = nn.LayerNorm(hidden_dim)

        # Per-sensor anomaly classifier (returns logits, no sigmoid for numerical stability)
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

        # Global window classifier (auxiliary, returns logits)
        self.global_classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

    def build_graph_from_embeddings(self, threshold=0.5, force_rebuild=False):
        """
        Build graph with ADAPTIVE K based on embedding similarity.
        Instead of fixed K, connect sensors above similarity threshold.

        Optimized: Caches graph structure and rebuilds only when needed.

        Args:
            threshold: Similarity threshold for connecting sensors (default 0.5)
                      If None, uses original Top-K only
            force_rebuild: If True, force rebuild even if cached

        Returns:
            edge_index: (2, num_edges) tensor of graph edges
        """
        # Check cache
        if self.cached_edge_index is not None and not force_rebuild:
            return self.cached_edge_index

        # Compute pairwise similarities
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        similarity = torch.mm(emb_norm, emb_norm.t())  # (N, N)

        # Remove self-loops
        similarity.fill_diagonal_(-1.0)

        if threshold is None:
            # Original Top-K only (backward compatibility)
            topk_values, topk_indices = torch.topk(similarity, self.top_k, dim=1)
            src_nodes = torch.arange(
                self.num_nodes, device=similarity.device
            ).repeat_interleave(self.top_k)
            dst_nodes = topk_indices.flatten()
            edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
        else:
            # Adaptive approach: connect if similarity > threshold OR top-K
            edges = []
            for i in range(self.num_nodes):
                # Get top-K
                topk_values, topk_indices = torch.topk(similarity[i], self.top_k)

                # Get all above threshold
                above_thresh = torch.where(similarity[i] > threshold)[0]

                # Union of both (ensures minimum connectivity)
                connected = torch.cat([topk_indices, above_thresh]).unique()

                # Add edges
                for j in connected:
                    edges.append([i, j.item()])

            if len(edges) == 0:
                # Fallback to Top-K if no edges found
                topk_values, topk_indices = torch.topk(similarity, self.top_k, dim=1)
                src_nodes = torch.arange(
                    self.num_nodes, device=similarity.device
                ).repeat_interleave(self.top_k)
                dst_nodes = topk_indices.flatten()
                edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = edge_index.to(self.sensor_embeddings.device)

        # Cache the result
        self.cached_edge_index = edge_index
        return edge_index

    def forward(self, x, return_global=False, return_sensor_embeddings=False):
        """
        Forward pass through the model.

        Args:
            x: (B, W, N) input tensor where B=batch_size, W=window_size, N=num_sensors
            return_global: If True, also return global window anomaly logits
            return_sensor_embeddings: If True, also return per-sensor embeddings (post-GAT)

        Returns:
            - sensor_logits: (B, N) logits for each sensor being anomalous
            - global_logits: (B,) logits for window having any anomaly (optional, if return_global=True)
            - sensor_embeddings: (B, N, hidden_dim) normalized post-GAT embeddings (optional, if return_sensor_embeddings=True)
        """
        B, W, N = x.shape

        # Enhanced temporal encoding per sensor with attention pooling
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)  # (B*N, W, hidden_dim)

        # Reshape for attention pooling: (B, N, W, hidden_dim)
        h_temporal = h_temporal.reshape(B, N, W, -1)

        # Vectorized temporal attention pooling (processes all sensors at once)
        h_last, _ = self.temporal_pooling(h_temporal)  # (B, N, hidden_dim)

        # Graph attention with residual connection to preserve sensor distinctiveness
        # Check if we need to rebuild graph (based on step counter)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT for stability
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        # Batched GAT processing (eliminates batch loop)
        # Reshape to (B*N, hidden_dim) - treat all batch samples as one big graph
        h_flat = h_last_norm.reshape(B * N, -1)  # (B*N, hidden_dim)

        # Create batched edge_index (offset edges for each batch sample)
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # (B, 1)
        edge_index_batched = edge_index.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, 2, num_edges)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(
            2
        )  # Add offset to each batch
        edge_index_batched = edge_index_batched.reshape(
            2, B * num_edges
        )  # (2, B*num_edges)

        # Single GAT call for entire batch!
        h_gat_flat = self.multi_scale_gat(
            h_flat, edge_index_batched
        )  # (B*N, hidden_dim)

        # Reshape back
        h_gat = h_gat_flat.reshape(B, N, -1)  # (B, N, hidden_dim)

        # Residual connection and normalization
        h_graph = h_gat + h_last_norm  # Residual
        h_graph = self.gat_norm(h_graph)  # Normalize

        # Per-sensor anomaly logits (no sigmoid - use BCEWithLogitsLoss)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)

        # Prepare return values
        return_values = [sensor_logits]

        if return_global:
            # Global window anomaly logits
            h_flat = h_graph.flatten(1)  # (B, N * hidden_dim)
            global_logits = self.global_classifier(h_flat).squeeze(-1)  # (B,)
            return_values.append(global_logits)

        if return_sensor_embeddings:
            # Return normalized post-GAT sensor embeddings (same space as window embeddings)
            sensor_embeddings = F.normalize(h_graph, p=2, dim=2)  # (B, N, hidden_dim)
            return_values.append(sensor_embeddings)

        if len(return_values) == 1:
            return return_values[0]
        else:
            return tuple(return_values)

    def get_embeddings(self, x):
        """
        Extract normalized embeddings for center loss and distance-based scoring.

        Args:
            x: (B, W, N) input tensor

        Returns:
            embeddings: (B, hidden_dim) L2-normalized embeddings per window
        """
        B, W, N = x.shape

        # Temporal encoding with attention pooling (same as forward)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)  # (B*N, W, hidden_dim)

        # Reshape for attention pooling: (B, N, W, hidden_dim)
        h_temporal = h_temporal.reshape(B, N, W, -1)

        # Vectorized temporal attention pooling (processes all sensors at once)
        h_last, _ = self.temporal_pooling(h_temporal)  # (B, N, hidden_dim)

        # Graph attention with residual connection (same as forward)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT for stability
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        # Batched GAT processing (eliminates batch loop)
        h_flat = h_last_norm.reshape(B * N, -1)  # (B*N, hidden_dim)

        # Create batched edge_index
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N
        edge_index_batched = edge_index.unsqueeze(0).expand(B, -1, -1)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)
        edge_index_batched = edge_index_batched.reshape(2, B * num_edges)

        # Single GAT call for entire batch
        h_gat_flat = self.multi_scale_gat(h_flat, edge_index_batched)
        h_gat = h_gat_flat.reshape(B, N, -1)

        # Residual connection and normalization
        h_graph = h_gat + h_last_norm
        h_graph = self.gat_norm(h_graph)

        # Aggregate across sensors: mean pooling
        embeddings = h_graph.mean(dim=1)  # (B, hidden_dim)

        # L2 normalize for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_sensor_embeddings(self, x):
        """
        Extract per-sensor embeddings for sensor-level fault attribution.

        Uses POST-GAT embeddings (h_graph) to match the window embedding space.
        Window embeddings use normalize(mean(h_graph)), so sensor embeddings use
        normalize(h_graph) to ensure they're in the same embedding space.

        This ensures compatible embedding spaces for multi-level center loss,
        where sensor-specific centers can be compared directly to sensor embeddings.

        Args:
            x: (B, W, N) input tensor
                B = batch size
                W = window size (300 timesteps)
                N = num sensors (8)

        Returns:
            sensor_embeddings: (B, N, hidden_dim)
                L2-normalized post-GAT embeddings (same space as window embeddings)
        """
        B, W, N = x.shape

        # Temporal encoding with attention pooling (same as forward and get_embeddings)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)  # (B*N, W, hidden_dim)

        # Reshape for attention pooling: (B, N, W, hidden_dim)
        h_temporal = h_temporal.reshape(B, N, W, -1)

        # Apply temporal attention pooling per sensor
        h_last_list = []
        for i in range(N):
            h_sensor_temporal = h_temporal[:, i, :, :]  # (B, W, hidden_dim)
            h_sensor_pooled, _ = self.temporal_pooling(
                h_sensor_temporal
            )  # (B, hidden_dim)
            h_last_list.append(h_sensor_pooled)

        h_last = torch.stack(h_last_list, dim=1)  # (B, N, hidden_dim)

        # Graph attention with residual connection (same as forward and get_embeddings)
        edge_index = self.build_graph_from_embeddings()

        # Normalize before GAT for stability
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        h_graph_list = []
        for i in range(B):
            h_gat = self.multi_scale_gat(h_last_norm[i], edge_index)
            # Residual connection: preserve sensor distinctiveness
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Use POST-GAT embeddings to match window embedding space
        # Window embeddings: normalize(mean(h_graph))
        # Sensor embeddings: normalize(h_graph) - same space, just not aggregated
        h_sensor = F.normalize(h_graph, p=2, dim=2)  # Normalize over hidden_dim

        return h_sensor


class KAGOptimizedGDN(nn.Module):
    """
    KAG-Optimized Baseline GDN: Simple architecture optimized for embedding quality.

    Key differences from enhanced MultiLabelGDN:
    - Single-layer unidirectional GRU (fast, no bidirectional)
    - Last hidden state pooling (no temporal attention)
    - Single GAT layer (no multi-scale)
    - But includes essential fixes: LayerNorm, residual, graph caching, no sigmoid

    Optimized for KAG integration:
    - Contrastive learning (in Stage 1) provides embedding separation
    - Residual connections preserve sensor distinctiveness
    - LayerNorm stabilizes training
    - Graph caching speeds up training

    """

    def __init__(
        self,
        num_nodes,
        window_size,
        embed_dim=64,
        top_k=5,
        hidden_dim=64,
        rebuild_graph_every=50,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.top_k = min(top_k, num_nodes - 1)
        self.hidden_dim = hidden_dim
        self.rebuild_graph_every = rebuild_graph_every

        # Graph caching
        self.cached_edge_index = None
        self._graph_step_counter = 0

        # Learned sensor embeddings
        self.sensor_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.sensor_embeddings)

        # Single-layer unidirectional GRU (baseline - fast)
        self.temporal_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Single GAT layer with improvements
        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,  # Reduced from 4 (less over-smoothing)
            concat=False,
            dropout=0.4,  # Increased from 0.2 (better regularization)
            add_self_loops=True,  # Changed from False (preserve own signal)
        )

        # LayerNorm after GAT for stability
        self.gat_norm = nn.LayerNorm(hidden_dim)

        # Per-sensor anomaly classifier (returns logits, no sigmoid)
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

        # Global window classifier (returns logits, no sigmoid)
        self.global_classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            # No sigmoid - use BCEWithLogitsLoss instead
        )

    def build_graph_from_embeddings(self, force_rebuild=False):
        """
        Build Top-K graph from learned embeddings with caching.

        Args:
            force_rebuild: If True, force rebuild even if cached

        Returns:
            edge_index: (2, num_edges) tensor of graph edges
        """
        # Check cache
        if self.cached_edge_index is not None and not force_rebuild:
            return self.cached_edge_index

        # Compute pairwise similarities
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())  # (N, N)

        # Remove self-loops
        sim_matrix.fill_diagonal_(-1e9)

        # Top-K graph
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)

        # PyTorch Geometric requires CPU edge_index (doesn't fully support MPS/CUDA edge_index)
        src_nodes = torch.arange(
            self.num_nodes, device="cpu", dtype=torch.long
        ).repeat_interleave(self.top_k)
        dst_nodes = topk_indices.flatten().cpu().to(torch.long)

        edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

        # Cache the result
        self.cached_edge_index = edge_index
        return edge_index

    def forward(self, x, return_global=False, return_sensor_embeddings=False):
        """
        Forward pass through the model.

        Args:
            x: (B, W, N) input tensor where B=batch_size, W=window_size, N=num_sensors
            return_global: If True, also return global window anomaly logits
            return_sensor_embeddings: If True, also return per-sensor embeddings (post-GAT)

        Returns:
            - sensor_logits: (B, N) logits for each sensor being anomalous
            - global_logits: (B,) logits for window having any anomaly (optional, if return_global=True)
            - sensor_embeddings: (B, N, hidden_dim) normalized post-GAT embeddings (optional, if return_sensor_embeddings=True)
        """
        B, W, N = x.shape

        # Temporal encoding per sensor (simple: last hidden state)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)  # (B*N, W, hidden_dim)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention with caching
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT for stability
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        # GAT processing (per-batch loop, but with residual)
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            # Residual connection: preserve sensor distinctiveness
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Per-sensor anomaly logits (no sigmoid - use BCEWithLogitsLoss)
        sensor_logits = self.sensor_classifier(h_graph).squeeze(-1)  # (B, N)

        # Prepare return values
        return_values = [sensor_logits]

        if return_global:
            # Global window anomaly logits
            h_flat = h_graph.flatten(1)  # (B, N * hidden_dim)
            global_logits = self.global_classifier(h_flat).squeeze(-1)  # (B,)
            return_values.append(global_logits)

        if return_sensor_embeddings:
            # Return normalized post-GAT sensor embeddings
            sensor_embeddings = F.normalize(h_graph, p=2, dim=2)  # (B, N, hidden_dim)
            return_values.append(sensor_embeddings)

        if len(return_values) == 1:
            return return_values[0]
        else:
            return tuple(return_values)

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

        # Graph attention (same as forward)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        # GAT processing with residual
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Aggregate across sensors: mean pooling
        embeddings = h_graph.mean(dim=1)  # (B, hidden_dim)

        # L2 normalize for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_sensor_embeddings(self, x):
        """
        Extract per-sensor embeddings for sensor-level fault attribution.

        Uses POST-GAT embeddings (h_graph) to match the window embedding space.
        Window embeddings use normalize(mean(h_graph)), so sensor embeddings use
        normalize(h_graph) to ensure they're in the same embedding space.

        Args:
            x: (B, W, N) input tensor

        Returns:
            sensor_embeddings: (B, N, hidden_dim)
                L2-normalized post-GAT embeddings (same space as window embeddings)
        """
        B, W, N = x.shape

        # Temporal encoding (same as forward)
        x_flat = x.permute(0, 2, 1).reshape(B * N, W, 1)
        h_temporal, _ = self.temporal_encoder(x_flat)
        h_last = h_temporal[:, -1, :].reshape(B, N, -1)  # (B, N, hidden_dim)

        # Graph attention (same as forward)
        force_rebuild = self._graph_step_counter % self.rebuild_graph_every == 0
        edge_index = self.build_graph_from_embeddings(force_rebuild=force_rebuild)
        self._graph_step_counter += 1

        # Normalize before GAT
        h_last_norm = F.normalize(h_last, p=2, dim=2)

        # GAT processing with residual
        # PyTorch Geometric requires CPU for node features and edge_index
        h_graph_list = []
        for i in range(B):
            h_node = h_last_norm[i]  # (N, hidden_dim)
            # Move to CPU for GAT (PyG doesn't fully support CUDA/MPS)
            h_node_cpu = h_node.cpu()
            h_gat_cpu = self.gat(h_node_cpu, edge_index)
            h_gat = h_gat_cpu.to(h_node.device)
            h_gat = h_gat + h_last_norm[i]  # Residual
            h_gat = self.gat_norm(h_gat)  # Normalize
            h_graph_list.append(h_gat)

        h_graph = torch.stack(h_graph_list, dim=0)  # (B, N, hidden_dim)

        # Use POST-GAT embeddings to match window embedding space
        # Window embeddings: normalize(mean(h_graph))
        # Sensor embeddings: normalize(h_graph) - same space, just not aggregated
        h_sensor = F.normalize(h_graph, p=2, dim=2)  # Normalize over hidden_dim

        return h_sensor


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


class TripletCenterLoss(nn.Module):
    """
    Triplet-Center Loss: Combines center pull with triplet margin.
    Superior to standard center loss by explicitly pushing opposite classes apart.

    Reference:
    He et al. "Triplet-Center Loss for Multi-View 3D Object Retrieval" (CVPR 2018)
    """

    def __init__(self, embed_dim, num_classes=2, margin=1.0, lambda_c=0.5):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.centers)
        self.margin = margin
        self.lambda_c = lambda_c

    def forward(self, embeddings, labels):
        """
        Compute triplet-center loss with enhanced numerical stability.

        Args:
            embeddings: (B, D) normalized embeddings from model
            labels: (B,) binary labels (0=normal, 1=anomalous)

        Returns:
            loss: scalar tensor
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        eps = 1e-8

        # Input validation
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Center loss component: pull embeddings toward their class center
        centers_batch = self.centers[labels.long()]
        center_loss = (embeddings - centers_batch).pow(2).sum(dim=1)
        center_loss = torch.clamp(center_loss, min=0.0, max=100.0)  # Clamp distances
        center_loss = center_loss.mean()

        # Triplet margin component: push opposite class away
        dist_to_own = torch.norm(embeddings - self.centers[labels.long()], dim=1)
        dist_to_other = torch.norm(embeddings - self.centers[1 - labels.long()], dim=1)

        # Clamp distances to prevent overflow
        dist_to_own = torch.clamp(dist_to_own, min=eps, max=50.0)
        dist_to_other = torch.clamp(dist_to_other, min=eps, max=50.0)

        triplet_loss = torch.clamp(
            dist_to_own - dist_to_other + self.margin, min=0.0, max=100.0
        )
        triplet_loss = triplet_loss.mean()

        total_loss = center_loss + self.lambda_c * triplet_loss

        # Final clamp to prevent NaN/Inf
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)

        return total_loss

    def get_center_separation(self):
        """L2 distance between normal and anomalous centers."""
        return torch.norm(self.centers[0] - self.centers[1]).item()


class CenterLossWithRepulsion(nn.Module):
    """
    Center Loss with explicit repulsion between centers.
    Forces centers apart explicitly to prevent collapse.

    Reference:
    Various metric learning papers on center repulsion
    """

    def __init__(self, embed_dim, num_classes=2, alpha=0.5, beta=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.alpha = alpha
        self.beta = beta  # Repulsion weight
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings, labels):
        """
        Compute center loss with repulsion and enhanced numerical stability.

        Args:
            embeddings: (B, D) normalized embeddings from model
            labels: (B,) binary labels (0=normal, 1=anomalous)

        Returns:
            loss: scalar tensor
        """
        device = embeddings.device
        eps = 1e-8

        # Input validation
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        centers_batch = self.centers[labels.long()]
        pull_loss = (embeddings - centers_batch).pow(2).sum(dim=1)
        pull_loss = torch.clamp(pull_loss, min=0.0, max=100.0)  # Clamp distances
        pull_loss = pull_loss.mean()

        # Explicit repulsion between centers: target distance = 4.0
        center_dist = torch.norm(self.centers[0] - self.centers[1])
        center_dist = torch.clamp(center_dist, min=eps, max=50.0)  # Prevent overflow
        repulsion_loss = torch.clamp(4.0 - center_dist, min=0.0, max=100.0)

        total_loss = pull_loss + self.beta * repulsion_loss

        # Final clamp to prevent NaN/Inf
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)

        return total_loss

    def get_center_separation(self):
        """L2 distance between normal and anomalous centers."""
        return torch.norm(self.centers[0] - self.centers[1]).item()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for robust metric learning.
    More robust than center loss for anomaly detection.

    Reference:
    Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = max(temperature, 0.1)  # Minimum temperature for stability
        self.eps = 1e-8  # Numerical stability epsilon

    def forward(self, embeddings, labels):
        """
        Compute supervised contrastive loss with enhanced numerical stability.

        Args:
            embeddings: (B, D) embeddings from model (will be normalized)
            labels: (B,) binary labels (0=normal, 1=anomalous)

        Returns:
            loss: scalar tensor
        """
        device = embeddings.device
        B = embeddings.size(0)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1, p=2, eps=self.eps)

        # Input validation
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create masks for positive pairs (same class)
        labels_expanded = labels.unsqueeze(1)
        mask_pos = (labels_expanded == labels_expanded.T).float()
        mask_pos.fill_diagonal_(0)  # Exclude self

        # Check if any positive pairs exist
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Enhanced numerical stability: subtract max
        max_sim = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        sim_matrix_stable = sim_matrix - max_sim

        # Clamp to prevent overflow BEFORE exp
        sim_matrix_stable = torch.clamp(sim_matrix_stable, min=-50.0, max=50.0)

        # Mask diagonal during softmax
        diag_mask = torch.eye(B, device=device)
        sim_matrix_stable = sim_matrix_stable - diag_mask * 1e10

        # Compute exp and probabilities
        exp_sim = torch.exp(sim_matrix_stable)
        pos_sim = (exp_sim * mask_pos).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        # Add epsilon for log stability
        pt = torch.clamp(
            pos_sim / (all_sim + self.eps), min=self.eps, max=1.0 - self.eps
        )

        # Compute loss only for samples with positive pairs
        has_pos = mask_pos.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_sample = -torch.log(pt[has_pos] + self.eps)
        loss_per_sample = torch.clamp(loss_per_sample, min=0.0, max=100.0)

        # Check for NaN/Inf
        loss_per_sample = torch.where(
            torch.isfinite(loss_per_sample),
            loss_per_sample,
            torch.zeros_like(loss_per_sample),
        )

        if len(loss_per_sample) == 0 or loss_per_sample.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss_per_sample.mean()

    # ============================================================================
    # Loss Functions
    # ============================================================================
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
        h_last = h_temporal[:, -1, :].reshape(
            B, N, self.hidden_dim
        )  # (B, N, hidden_dim)
        h_last = self.temporal_norm(h_last)

        # Efficient batched GAT processing
        edge_index = self.build_graph_from_embeddings()

        # Reshape for batched GAT: (B*N, hidden_dim)
        h_flat = h_last.reshape(B * N, self.hidden_dim)

        # Create batched edge_index (vectorized)
        num_edges = edge_index.shape[1]
        offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # (B, 1)
        edge_index_batched = edge_index.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, 2, num_edges)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(2)  # Add offset
        edge_index_batched = edge_index_batched.reshape(
            2, B * num_edges
        )  # (2, B*num_edges)

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
        embeddings = F.normalize(
            embeddings, p=2, dim=1
        )  # Normalize for metric learning

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

    def __init__(
        self, num_sensors, window_size, hidden_dim, num_layers=2, bidirectional=True
    ):
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
            dropout=0.2 if num_layers > 1 else 0,
        )

        # Sensor-specific adapters learn sensor-specific transformations
        # Each adapter projects shared features to sensor-specific space
        encoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.sensor_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(encoder_output_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                for _ in range(num_sensors)
            ]
        )

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
            x_sensor = x[:, :, sensor_idx : sensor_idx + 1]

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
            nn.Linear(hidden_dim // 2, 1),
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
            bidirectional=True,
        )

        # Normalization after temporal encoding
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Multi-layer GAT with residuals
        self.gat_layers = nn.ModuleList(
            [
                ResidualGATLayer(hidden_dim, heads=4, dropout=0.2)
                for _ in range(num_gat_layers)
            ]
        )

        # Final GAT layer (no residual, output dimension)
        self.gat_final = GATv2Conv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=0.2,
            add_self_loops=True,
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
        edge_index_batched = edge_index.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, 2, num_edges)
        edge_index_batched = edge_index_batched + offsets.unsqueeze(
            2
        )  # Add offset to each batch
        edge_index_batched = edge_index_batched.reshape(
            2, B * num_edges
        )  # (2, B*num_edges)

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
        sensor_logits = torch.logit(sensor_probs.clamp(1e-7, 1 - 1e-7))
        global_logits = torch.logit(global_probs.clamp(1e-7, 1 - 1e-7))

        # Get embeddings
        embeddings = F.normalize(h_aggregated, p=2, dim=1)

        return sensor_logits, global_logits, embeddings
