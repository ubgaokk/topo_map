"""
BEV Feature Encoder with View Transformer

Implements a simplified BEVFormer-style view transformer for 
multi-view to Bird's-Eye-View feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ViewTransformer(nn.Module):
    """
    Simplified BEVFormer-style View Transformer
    
    Converts multi-view camera features into a unified BEV representation.
    
    Args:
        bev_h: BEV feature map height
        bev_w: BEV feature map width  
        bev_z: Number of height planes
        dim: Feature dimension
    """
    
    def __init__(
        self,
        bev_h: int = 100,
        bev_w: int = 200,
        bev_z: int = 4,
        dim: int = 256,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.dim = dim
        
        # BEV query grid
        self.bev_query = nn.Parameter(
            torch.randn(bev_h, bev_w, dim)
        )
        
        # Reference point encoder
        self.refpoint_encoder = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Camera feature projection
        self.camera_proj = nn.Linear(dim, dim)
        self.bev_proj = nn.Linear(dim, dim)
        
        # Cross-attention layers
        self.attention = nn.MultiheadAttention(
            dim, num_heads=8, batch_first=True, dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
    
    def get_bev_features(
        self,
        multi_view_features: List[torch.Tensor],
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        ego_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract BEV features from multi-view images
        
        Args:
            multi_view_features: List of [B, C, H, W] tensors per camera
            camera_intrinsics: [B, N_cam, 3, 3] camera intrinsic matrices
            camera_extrinsics: [B, N_cam, 4, 4] camera extrinsic matrices
            ego_pose: [B, 4, 4] ego vehicle pose (optional)
            
        Returns:
            bev_features: [B, dim, bev_h, bev_w]
        """
        B = multi_view_features[0].shape[0]
        N_cam = len(multi_view_features)
        
        # Initialize BEV queries
        bev_query = self.bev_query.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, dim]
        bev_query = bev_query.flatten(1, 2)  # [B, H*W, dim]
        
        # Project camera features
        # multi_view_features: list of [B, C, H, W]
        # Flatten spatial dimensions for attention
        camera_features = []
        for feat in multi_view_features:
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            camera_features.append(self.camera_proj(feat_flat))
        
        # Concatenate all camera features: [B, N_cam*H*W, dim]
        camera_features = torch.cat(camera_features, dim=1)
        
        # Cross-attention: BEV queries attend to camera features
        # Note: This is simplified. Real implementation needs 
        # spatial-aware attention with geometric encoding
        bev_out, _ = self.attention(
            bev_query, camera_features, camera_features
        )
        
        # Project back to grid
        bev_out = bev_out.transpose(1, 2).view(B, self.dim, self.bev_h, self.bev_w)
        
        bev_out = self.output_proj(
            bev_out.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        
        return bev_out
    
    def _simple_bev_encode(
        self,
        multi_view_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fallback simple BEV encoding when proper view transform is too complex.
        Uses first camera as proxy for BEV features.
        """
        B = multi_view_features[0].shape[0]
        C = multi_view_features[0].shape[1]
        
        # Take first camera and resize to BEV dimensions
        feat = multi_view_features[0]  # [B, C, H, W]
        
        # Downsample to BEV grid
        feat = F.interpolate(
            feat, 
            size=(self.bev_h, self.bev_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Project to target dimension
        feat = self.bev_proj(feat.permute(0, 2, 3, 1))  # [B, H, W, dim]
        
        return feat.permute(0, 3, 1, 2)  # [B, dim, H, W]


class BevEncoder(nn.Module):
    """
    Complete BEV Encoder with backbone and view transformer
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        bev_h: int = 100,
        bev_w: int = 200,
        dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # Backbone
        if backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            backbone_dim = 2048
        elif backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            backbone_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # FPN for multi-scale features
        self.fpn = nn.Sequential(
            nn.Conv2d(backbone_dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        
        # View transformer
        self.view_transformer = ViewTransformer(
            bev_h=bev_h,
            bev_w=bev_w,
            dim=dim,
        )
    
    def forward(
        self,
        multi_view_features: List[torch.Tensor],
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract BEV features from multi-view images
        
        Returns:
            bev_features: [B, dim, bev_h, bev_w]
        """
        # Extract backbone features from each view
        multi_scale_features = []
        for feat in multi_view_features:
            feat = self.backbone(feat)  # [B, C, H/32, W/32]
            feat = self.fpn(feat)       # [B, dim, H/32, W/32]
            multi_scale_features.append(feat)
        
        # View transformer to BEV
        bev_feat = self.view_transformer.get_bev_features(
            multi_scale_features,
            camera_intrinsics,
            camera_extrinsics,
        )
        
        return bev_feat