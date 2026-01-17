"""
MultiLabelGDN Model Definition

Extracted from gdn.ipynb for reuse in scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Graph attention
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, 
                          dropout=0.2, add_self_loops=False)
        
        # Per-sensor anomaly classifier
        self.sensor_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Global window classifier (auxiliary)
        self.global_classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def build_graph_from_embeddings(self):
        """Build Top-K graph from learned embeddings."""
        emb_norm = F.normalize(self.sensor_embeddings, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t())
        sim_matrix.fill_diagonal_(-1e9)
        
        topk_values, topk_indices = torch.topk(sim_matrix, self.top_k, dim=1)
        
        src_nodes = torch.arange(self.num_nodes, device=sim_matrix.device).repeat_interleave(self.top_k)
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
