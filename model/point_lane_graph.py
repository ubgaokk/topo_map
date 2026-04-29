"""
Hierarchical Point-Lane Graph Module

Implements hierarchical message passing:
1. Point → Lane: Aggregate point features into lane representation
2. Lane → Lane: Propagate lane-level topology information
3. Endpoint-aware attention for topology reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MessagePassingLayer(nn.Module):
    """
    GNN Message Passing Layer with Edge Types
    
    Handles different relation types:
    - successor: End of one lane connects to start of another
    - predecessor: Inverse of successor
    - self-loop: Lane self-connection
    
    Args:
        dim: Feature dimension
        edge_types: Number of edge type embeddings
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int = 256,
        edge_types: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.edge_types = edge_types
        
        # Edge-type specific transformations
        self.edge_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            for _ in range(edge_types)
        ])
        
        # Edge type predictor (optional, can be used to learn edge weights)
        self.edge_type_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, edge_types)
        )
        
        # Node update
        self.node_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, dim]
            adj_matrix: [B, N, N] Adjacency matrix
            edge_weights: Optional edge weights for attention
        Returns:
            updated_features: [B, N, dim]
        """
        B, N, D = node_features.shape
        
        # Add self-loops
        adj_with_self = adj_matrix + torch.eye(N, device=adj_matrix.device).unsqueeze(0)
        
        # Normalize adjacency
        degrees = adj_with_self.sum(dim=-1, keepdim=True) + 1e-8
        adj_norm = adj_with_self / degrees
        
        # Aggregate neighbor features
        # [B, N, N] @ [B, N, D] → [B, N, D]
        neighbor_feat = torch.matmul(adj_norm, node_features)
        
        # For each edge type, compute transformed features
        # Simplified: use uniform transform
        transformed = self.edge_transforms[0](neighbor_feat)
        
        # Node update: combine self features with aggregated neighbor features
        combined = torch.cat([node_features, transformed], dim=-1)
        out = self.node_update(combined)
        
        return out


class PointLaneGraph(nn.Module):
    """
    Hierarchical Point-Lane Graph
    
    Implements two-level message passing:
    1. Point → Lane: Aggregates point-level features into lane queries
    2. Lane → Lane: Propagates lane topology information
    
    The graph is built with endpoint-awareness: lane connections are
    weighted by endpoint proximity and geometry.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads for point aggregation
        gnn_layers: Number of GNN layers for lane-level reasoning
        edge_types: Number of edge type categories
        adjacency_threshold: Distance threshold for lane adjacency (meters)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        gnn_layers: int = 3,
        edge_types: int = 3,
        adjacency_threshold: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.gnn_layers = gnn_layers
        self.adjacency_threshold = adjacency_threshold
        
        # Point to Lane attention
        self.point_to_lane_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.point_to_lane_norm = nn.LayerNorm(dim)
        
        # Lane to Lane GNN
        self.lane_gnn_layers = nn.ModuleList([
            MessagePassingLayer(dim, edge_types, dropout)
            for _ in range(gnn_layers)
        ])
        
        # Endpoint-aware attention for lane connections
        self.endpoint_attention = nn.Sequential(
            nn.Linear(dim + 4, dim),  # +4 for start/end positions
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
        # Cross-level fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(dim)
    
    def build_lane_adjacency(
        self,
        lane_geometry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build adjacency matrix based on endpoint proximity
        
        A lane i connects to lane j if the end of i is close to the start of j.
        
        Args:
            lane_geometry: [B, N_lane, 11, 3] Lane centerline points
            
        Returns:
            adj_matrix: [B, N_lane, N_lane] Binary adjacency
            dist_matrix: [B, N_lane, N_lane] Distance values
        """
        B, N_lane = lane_geometry.shape[0], lane_geometry.shape[1]
        
        # Extract start and end points
        starts = lane_geometry[:, :, 0, :2]   # [B, N, 2] xy
        ends   = lane_geometry[:, :, -1, :2]  # [B, N, 2] xy
        
        # Compute end-to-start distance matrix
        # dist[i,j] = ||end_i - start_j||_2
        ends_expanded = ends.unsqueeze(2)       # [B, N, 1, 2]
        starts_expanded = starts.unsqueeze(1)   # [B, 1, N, 2]
        
        dist_matrix = torch.norm(ends_expanded - starts_expanded, dim=-1)  # [B, N, N]
        
        # Binary adjacency based on threshold
        adj_matrix = (dist_matrix < self.adjacency_threshold).float()
        
        return adj_matrix, dist_matrix
    
    def endpoint_aware_edge_weights(
        self,
        lane_features: torch.Tensor,
        lane_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge weights based on endpoint proximity and feature similarity
        
        Args:
            lane_features: [B, N, dim]
            lane_geometry: [B, N, 11, 3]
            
        Returns:
            edge_weights: [B, N, N] Attention weights for each potential edge
        """
        B, N, D = lane_features.shape
        
        # Get endpoint positions
        starts = lane_geometry[:, :, 0, :2]   # [B, N, 2]
        ends   = lane_geometry[:, :, -1, :2]  # [B, N, 2]
        
        # For each pair (i, j), compute if i's end is close to j's start
        ends_i = ends.unsqueeze(2).expand(-1, -1, N, -1)      # [B, N, N, 2]
        starts_j = starts.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, 2]
        
        endpoint_dist = torch.norm(ends_i - starts_j, dim=-1)  # [B, N, N]
        
        # Convert distance to similarity weight
        edge_weights = torch.exp(-endpoint_dist / self.adjacency_threshold)  # [B, N, N]
        
        # Also consider feature similarity
        feat_i = lane_features.unsqueeze(2).expand(-1, -1, N, -1)      # [B, N, N, dim]
        feat_j = lane_features.unsqueeze(1).expand(-1, N, -1, -1)      # [B, N, N, dim]
        
        feat_sim = torch.cosine_similarity(feat_i, feat_j, dim=-1)     # [B, N, N]
        
        # Combine geometry and feature similarity
        combined_weights = edge_weights * (0.5 + 0.5 * feat_sim)
        
        return combined_weights
    
    def forward(
        self,
        lane_features: torch.Tensor,
        point_features: Optional[torch.Tensor],
        lane_geometry: torch.Tensor,
        endpoint_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hierarchical point-lane graph forward pass
        
        Args:
            lane_features: [B, N_lane, dim] Initial lane query features
            point_features: [B, N_lane, N_points, dim] Point-level features
            lane_geometry: [B, N_lane, 11, 3] Lane centerline geometry
            endpoint_tokens: [B, N_lane, dim] Endpoint-aware tokens
            
        Returns:
            updated_lane_features: [B, N_lane, dim]
        """
        B, N_lane, _ = lane_features.shape
        
        # === Step 1: Point → Lane Aggregation ===
        if point_features is not None:
            # Reshape for attention
            # point_features: [B, N_lane, N_points, dim]
            # lane_features: [B, N_lane, dim]
            
            # Each lane attends to its own points
            point_flat = point_features.view(B * N_lane, -1, self.dim)
            lane_expanded = lane_features.view(B * N_lane, 1, self.dim)
            
            # Cross attention: lane query attends to its points
            aggregated, _ = self.point_to_lane_attn(
                lane_expanded, point_flat, point_flat
            )
            aggregated = aggregated.squeeze(1).view(B, N_lane, self.dim)
            
            # Residual connection
            lane_features = self.point_to_lane_norm(lane_features + aggregated)
        
        # === Step 2: Build Endpoint-Aware Adjacency ===
        adj_matrix, dist_matrix = self.build_lane_adjacency(lane_geometry)
        
        # Compute endpoint-aware edge weights
        edge_weights = self.endpoint_aware_edge_weights(lane_features, lane_geometry)
        
        # Combine adjacency with learned weights
        adj_weighted = adj_matrix * edge_weights
        
        # === Step 3: Lane → Lane GNN ===
        lane_out = lane_features
        
        # Fuse endpoint tokens into lane features
        lane_out = lane_out + endpoint_tokens
        
        for gnn_layer in self.lane_gnn_layers:
            lane_out = gnn_layer(lane_out, adj_weighted, edge_weights)
        
        # === Step 4: Cross-Level Fusion ===
        if point_features is not None:
            # Gate between point-aggregated and lane-gnn features
            combined = torch.cat([lane_features, lane_out], dim=-1)
            gate = self.fusion_gate(combined)
            lane_out = gate * lane_features + (1 - gate) * lane_out
        
        # Output normalization
        lane_out = self.output_norm(lane_out)
        
        return lane_out