"""
Topology Prediction Heads

Predicts lane-lane and lane-traffic-element topology relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TopologyHead(nn.Module):
    """
    Topology Relation Prediction Head
    
    Computes pairwise topology scores:
    - Lane-Lane: Connection/succession relationships
    - Lane-TE: Association between lanes and traffic elements
    
    Args:
        dim: Feature dimension
        hidden_dim: Hidden dimension for MLPs
    """
    
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Lane-Lane topology prediction
        self.lane_lane_head = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Lane-TE topology prediction
        self.lane_te_head = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def pairwise_lane_score(
        self,
        lane_features_i: torch.Tensor,
        lane_features_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute connection scores for all lane pairs
        
        Args:
            lane_features_i: [B, N, dim] Source lane features
            lane_features_j: [B, M, dim] Target lane features
            
        Returns:
            scores: [B, N, M] Connection scores
        """
        # Expand for pairwise computation
        lane_i_expanded = lane_features_i.unsqueeze(2)    # [B, N, 1, dim]
        lane_j_expanded = lane_features_j.unsqueeze(1)    # [B, 1, M, dim]
        
        # Concatenate pairs
        pair_features = torch.cat([lane_i_expanded, lane_j_expanded], dim=-1)  # [B, N, M, dim*2]
        
        # Compute scores
        scores = self.lane_lane_head(pair_features)  # [B, N, M, 1]
        
        return scores.squeeze(-1)  # [B, N, M]
    
    def pairwise_lane_te_score(
        self,
        lane_features: torch.Tensor,
        te_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute association scores for lane-TE pairs
        
        Args:
            lane_features: [B, N_lane, dim]
            te_features: [B, N_te, dim]
            
        Returns:
            scores: [B, N_lane, N_te]
        """
        # Expand for pairwise
        lane_expanded = lane_features.unsqueeze(2)    # [B, N_lane, 1, dim]
        te_expanded   = te_features.unsqueeze(1)      # [B, 1, N_te, dim]
        
        # Concatenate pairs
        pair_features = torch.cat([lane_expanded, te_expanded], dim=-1)  # [B, N_lane, N_te, dim*2]
        
        # Compute scores
        scores = self.lane_te_head(pair_features)  # [B, N_lane, N_te, 1]
        
        return scores.squeeze(-1)  # [B, N_lane, N_te]
    
    def forward(
        self,
        lane_features: torch.Tensor,
        te_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lane_features: [B, N_lane, dim]
            te_features: [B, N_te, dim]
            
        Returns:
            topo_lclc: [B, N_lane, N_lane] Lane-lane topology scores
            topo_lcte: [B, N_lane, N_te] Lane-TE topology scores
        """
        # Lane-lane topology
        topo_ll = self.pairwise_lane_score(lane_features, lane_features)
        
        # Lane-TE topology
        topo_lte = self.pairwise_lane_te_score(lane_features, te_features)
        
        return topo_ll, topo_lte


class LaneDetectionHead(nn.Module):
    """
    Lane Detection Head
    
    Predicts lane bounding boxes and geometry.
    
    Args:
        dim: Feature dimension
    """
    
    def __init__(self, dim: int = 256):
        super().__init__()
        
        # Presence confidence
        self.conf_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
        
        # Geometry regression (11 centerline points * 3 coords = 33)
        self.geometry_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 11 * 3),
        )
    
    def forward(self, lane_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_queries: [B, N_lane, dim]
            
        Returns:
            pred_boxes: [B, N_lane, 34] (conf + geometry)
        """
        conf = self.conf_head(lane_queries)  # [B, N_lane, 1]
        geometry = self.geometry_head(lane_queries)  # [B, N_lane, 33]
        
        return torch.cat([conf, geometry], dim=-1)  # [B, N_lane, 34]


class TEDetectionHead(nn.Module):
    """
    Traffic Element Detection Head
    
    Predicts TE bounding boxes and semantic attributes.
    
    Args:
        dim: Feature dimension
        num_attributes: Number of TE attribute classes
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_attributes: int = 13,
    ):
        super().__init__()
        
        # Box regression [x1, y1, x2, y2]
        self.box_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 4),
        )
        
        # Attribute classification
        self.attribute_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_attributes),
        )
        
        # Confidence
        self.conf_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )
    
    def forward(self, te_queries: torch.Tensor) -> dict:
        """
        Args:
            te_queries: [B, N_te, dim]
            
        Returns:
            dict with 'boxes', 'attributes', 'conf'
        """
        return {
            'boxes': self.box_head(te_queries),
            'attributes': self.attribute_head(te_queries),
            'conf': self.conf_head(te_queries),
        }