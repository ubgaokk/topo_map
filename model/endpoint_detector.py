"""
Endpoint Detection Module

Detects lane start/end points and generates endpoint-aware tokens
for topology reasoning. This is the core innovation inspired by TopoPoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EndpointDetector(nn.Module):
    """
    Endpoint Detection Module
    
    Predicts the start and end points of each lane centerline.
    These endpoints are crucial for topology reasoning as they
    determine lane connectivity (successor/predecessor relationships).
    
    Args:
        dim: Feature dimension
        hidden_dim: Hidden dimension for MLPs
        num_classes: Number of output classes (x, y offset + confidence)
    """
    
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,  # x, y offset + confidence
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Shared feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # Start point detection head
        self.start_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for position encoding
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # End point detection head
        self.end_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Endpoint token generator
        # Creates a unified token that encodes both start and end info
        self.endpoint_token_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, dim),  # start + end features + positions
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        lane_queries: torch.Tensor,
        lane_geometry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect endpoints for each lane query
        
        Args:
            lane_queries: [B, N_lane, dim] Lane query features
            lane_geometry: [B, N_lane, 11, 3] Lane centerline points (11 points per lane)
            
        Returns:
            start_points: [B, N_lane, 3] Start point predictions (x, y, conf)
            end_points: [B, N_lane, 3] End point predictions (x, y, conf)
            endpoint_tokens: [B, N_lane, dim] Endpoint-aware tokens for graph
        """
        B, N_lane, num_pts, C_geo = lane_geometry.shape
        
        # Extract start and end positions from geometry
        start_xyz = lane_geometry[:, :, 0, :]      # [B, N_lane, 3]
        end_xyz   = lane_geometry[:, :, -1, :]     # [B, N_lane, 3]
        
        # Encode geometry features
        # Use entire lane geometry context, not just endpoints
        geom_mean = lane_geometry.mean(dim=2)      # [B, N_lane, 3]
        geom_diff = end_xyz - start_xyz             # [B, N_lane, 3] direction
        
        # Feature encoding with geometry context
        start_feat = self.feature_encoder(lane_queries) + geom_mean
        end_feat   = self.feature_encoder(lane_queries) + geom_mean
        
        # Add position encoding
        start_pos_encoding = torch.cat([start_xyz, geom_diff], dim=-1)  # [B, N, 6]
        end_pos_encoding   = torch.cat([end_xyz, -geom_diff], dim=-1)   # negate for opposite direction
        
        # Predict start/end points
        start_pred = self.start_head(
            torch.cat([start_feat, start_pos_encoding], dim=-1)
        )  # [B, N, 2] → will be [x_offset, y_offset, conf]
        
        end_pred = self.end_head(
            torch.cat([end_feat, end_pos_encoding], dim=-1)
        )  # [B, N, 2]
        
        # Generate endpoint tokens for graph aggregation
        start_token = torch.cat([start_feat, start_xyz], dim=-1)
        end_token   = torch.cat([end_feat, end_xyz], dim=-1)
        
        endpoint_tokens = self.endpoint_token_generator(
            torch.cat([start_token, end_token], dim=-1)
        )  # [B, N, dim]
        
        # Combine offsets with base positions to get absolute predictions
        start_pred_absolute = start_pred.clone()
        start_pred_absolute[..., :2] += start_xyz[..., :2]  # Add offset to start
        
        end_pred_absolute = end_pred.clone()
        end_pred_absolute[..., :2] += end_xyz[..., :2]  # Add offset to end
        
        return start_pred_absolute, end_pred_absolute, endpoint_tokens


class PointSampler(nn.Module):
    """
    Point Sampler along Lane Centerlines
    
    Samples dense points along each lane for point-level graph reasoning.
    
    Args:
        num_points: Number of points to sample per lane
    """
    
    def __init__(self, num_points: int = 32):
        super().__init__()
        self.num_points = num_points
    
    def sample_along_lane(
        self,
        lane_geometry: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample dense points along each lane centerline
        
        Args:
            lane_geometry: [B, N_lane, 11, 3] Original 11 centerline points
            
        Returns:
            dense_points: [B, N_lane, num_points, 3] Dense sampled points
            point_features: None (features should be sampled from BEV)
        """
        B, N_lane, N_orig, C = lane_geometry.shape
        
        # Interpolate from 11 points to num_points
        # Reshape for interpolate: [B*N_lane, C, N_orig] → [B*N_lane, C, num_points]
        lane_geometry_flat = lane_geometry.permute(0, 1, 3, 2).reshape(B * N_lane, C, N_orig)
        
        dense_points = F.interpolate(
            lane_geometry_flat,
            size=self.num_points,
            mode='linear',
            align_corners=False,
        )  # [B*N_lane, C, num_points]
        
        dense_points = dense_points.reshape(B, N_lane, C, self.num_points)
        dense_points = dense_points.permute(0, 1, 3, 2)  # [B, N_lane, num_points, 3]
        
        return dense_points, None
    
    def sample_from_bev(
        self,
        bev_features: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample features from BEV feature map at given point locations
        
        Args:
            bev_features: [B, dim, H, W] BEV feature map
            points: [B, N_lane, num_points, 3] 3D point coordinates
            
        Returns:
            point_features: [B, N_lane, num_points, dim] Sampled features
        """
        B, dim, H, W = bev_features.shape
        N_lane, N_pts = points.shape[1], points.shape[2]
        
        # Convert 3D points to BEV grid coordinates
        # Assuming BEV range: x ∈ [-50, 50], y ∈ [-25, 25]
        x = (points[..., 0] / 100.0 + 0.5) * W  # Normalize to [0, W]
        y = (points[..., 1] / 50.0 + 0.5) * H   # Normalize to [0, H]
        
        # Clamp to valid range
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
        
        # Create grid for grid_sample: [B, H, W, 2]
        # Note: grid_sample expects [x, y] format, normalized to [-1, 1]
        grid_x = (2 * x.unsqueeze(-1) / (W - 1) - 1)  # [B, N_lane, N_pts, 1]
        grid_y = (2 * y.unsqueeze(-1) / (H - 1) - 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)     # [B, N_lane, N_pts, 2]
        
        # Reshape bev for batch processing
        # Need: [B, dim, H*W] and points as [B, H*W, 2]
        bev_flat = bev_features.flatten(2)  # [B, dim, H*W]
        grid_flat = grid.reshape(B, N_lane * N_pts, 2)
        
        # For simplicity, sample at grid points
        # This is a simplified version; real implementation needs 
        # proper bilinear sampling for each point
        point_features_list = []
        for b in range(B):
            bev_b = bev_flat[b]  # [dim, H*W]
            grid_b = grid_flat[b]  # [N_lane*N_pts, 2]
            
            # Simple nearest neighbor for now
            grid_b_x = ((grid_b[:, 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
            grid_b_y = ((grid_b[:, 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
            
            idx = grid_b_y * W + grid_b_x  # [N_lane*N_pts]
            feat = bev_b[:, idx]  # [dim, N_lane*N_pts]
            point_features_list.append(feat.T)  # [N_lane*N_pts, dim]
        
        point_features = torch.stack(point_features_list, dim=0)  # [B, N_lane*N_pts, dim]
        point_features = point_features.reshape(B, N_lane, N_pts, -1)  # [B, N_lane, N_pts, dim]
        
        return point_features