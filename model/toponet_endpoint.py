"""
Complete Endpoint-Aware Topology Network

Integrates all components:
1. BEV Encoder
2. Query Decoders (Lane + TE)
3. Endpoint Detector
4. Point Sampler
5. Hierarchical Point-Lane Graph
6. Topology Prediction Heads
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from model.bev_encoder import ViewTransformer, BevEncoder
from model.endpoint_detector import EndpointDetector, PointSampler
from model.point_lane_graph import PointLaneGraph
from model.topology_head import TopologyHead, LaneDetectionHead, TEDetectionHead


class LaneDecoder(nn.Module):
    """
    Lane Query Decoder with Transformer Layers
    
    Args:
        dim: Feature dimension
        num_queries: Number of lane queries
        num_layers: Number of decoder layers
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_queries: int = 200,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, dim)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim=dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Geometry prediction head (11 centerline points * 3 coords)
        self.geometry_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 11 * 3),
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, bev_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev_features: [B, dim, bev_h, bev_w]
            
        Returns:
            queries: [B, num_queries, dim] Refined lane queries
            geometry: [B, num_queries, 11, 3] Predicted centerline points
        """
        B = bev_features.shape[0]
        
        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Flatten BEV for cross-attention
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # [B, H*W, dim]
        
        # Iterative decoding
        for layer in self.decoder_layers:
            queries = layer(queries, bev_flat)
        
        # Final normalization
        queries = self.norm(queries)
        
        # Predict geometry
        geometry = self.geometry_head(queries)  # [B, N, 33]
        geometry = geometry.view(B, self.num_queries, 11, 3)
        
        return queries, geometry


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with Self-Attention and Cross-Attention
    """
    
    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Self attention (query-query)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross attention (query to BEV features)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, N, dim] Query features
            memory: [B, M, dim] Key/value features (BEV)
            
        Returns:
            updated queries: [B, N, dim]
        """
        # Self attention
        q = self.norm1(queries)
        queries = queries + self.dropout(self.self_attn(q, q, q)[0])
        
        # Cross attention
        q = self.norm2(queries)
        queries = queries + self.dropout(self.cross_attn(q, memory, memory)[0])
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries


class TEDecoder(nn.Module):
    """
    Traffic Element Query Decoder
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_queries: int = 100,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, dim)
        
        # Decoder layers (fewer than lane decoder)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim=dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_features: [B, dim, bev_h, bev_w]
            
        Returns:
            queries: [B, num_queries, dim]
        """
        B = bev_features.shape[0]
        
        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Flatten BEV
        bev_flat = bev_features.flatten(2).permute(0, 2, 1)
        
        # Decoding
        for layer in self.decoder_layers:
            queries = layer(queries, bev_flat)
        
        return self.norm(queries)


class EndpointAwareTopologyNet(nn.Module):
    """
    Complete Endpoint-Aware Topology Network
    
    Architecture:
    1. BEV Feature Extraction (backbone + view transformer)
    2. Query Decoding (lane + traffic element queries)
    3. Endpoint Detection (predicts lane start/end points)
    4. Point Sampling (samples dense points along lanes)
    5. Hierarchical Point-Lane Graph (message passing)
    6. Topology Prediction (lane-lane + lane-TE)
    
    Args:
        config: Model configuration dict
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # === BEV Encoder ===
        self.bev_encoder = ViewTransformer(
            bev_h=config.get('bev_h', 100),
            bev_w=config.get('bev_w', 200),
            bev_z=config.get('bev_z', 4),
            dim=config.get('dim', 256),
        )
        
        # === Decoders ===
        self.lane_decoder = LaneDecoder(
            dim=config.get('dim', 256),
            num_queries=config.get('num_lane_queries', 200),
            num_layers=config.get('num_decoder_layers', 6),
            num_heads=config.get('num_heads', 8),
        )
        
        self.te_decoder = TEDecoder(
            dim=config.get('dim', 256),
            num_queries=config.get('num_te_queries', 100),
            num_layers=config.get('num_decoder_layers', 4),
            num_heads=config.get('num_heads', 8),
        )
        
        # === Endpoint Detection (Core Innovation) ===
        self.use_endpoint_detector = config.get('use_endpoint_detector', True)
        if self.use_endpoint_detector:
            self.endpoint_detector = EndpointDetector(
                dim=config.get('dim', 256),
                hidden_dim=config.get('endpoint_hidden_dim', 128),
            )
        
        # === Point Sampling ===
        self.point_sampler = PointSampler(
            num_points=config.get('num_points', 32)
        )
        
        # === Hierarchical Point-Lane Graph (Core Innovation) ===
        self.use_point_lane_graph = config.get('use_point_lane_graph', True)
        if self.use_point_lane_graph:
            self.point_lane_graph = PointLaneGraph(
                dim=config.get('dim', 256),
                num_heads=config.get('num_heads', 8),
                gnn_layers=config.get('gnn_layers', 3),
                edge_types=config.get('edge_types', 3),
                adjacency_threshold=config.get('adjacency_threshold', 5.0),
                dropout=config.get('dropout', 0.1),
            )
        
        # === Detection Heads ===
        self.lane_detection_head = LaneDetectionHead(
            dim=config.get('dim', 256)
        )
        self.te_detection_head = TEDetectionHead(
            dim=config.get('dim', 256),
            num_attributes=13,  # OpenLane-V2 has 13 TE attributes
        )
        
        # === Topology Heads ===
        self.topology_head = TopologyHead(
            dim=config.get('dim', 256),
            hidden_dim=config.get('dim', 256) // 2,
        )
    
    def forward(
        self,
        multi_view_features: List[torch.Tensor],
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
        return_aux: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass
        
        Args:
            multi_view_features: List of [B, C, H, W] tensors per camera
            camera_intrinsics: [B, N_cam, 3, 3] (optional)
            camera_extrinsics: [B, N_cam, 4, 4] (optional)
            return_aux: Whether to return auxiliary predictions
            
        Returns:
            Dictionary with predictions
        """
        # === 1. BEV Feature Extraction ===
        bev_feat = self.bev_encoder.get_bev_features(
            multi_view_features,
            camera_intrinsics,
            camera_extrinsics,
        )
        
        # === 2. Query Decoding ===
        lane_queries, lane_geometry = self.lane_decoder(bev_feat)
        te_queries = self.te_decoder(bev_feat)
        
        # === 3. Endpoint Detection ===
        start_pred = None
        end_pred = None
        endpoint_tokens = torch.zeros_like(lane_queries)
        
        if self.use_endpoint_detector:
            start_pred, end_pred, endpoint_tokens = self.endpoint_detector(
                lane_queries, lane_geometry
            )
        
        # === 4. Point Sampling ===
        dense_points, point_features = self.point_sampler.sample_along_lane(lane_geometry)
        
        # Sample point features from BEV
        if point_features is None and return_aux:
            point_features = self.point_sampler.sample_from_bev(bev_feat, dense_points)
        
        # === 5. Hierarchical Point-Lane Graph ===
        if self.use_point_lane_graph:
            updated_lane_queries = self.point_lane_graph(
                lane_queries,
                point_features,
                lane_geometry,
                endpoint_tokens,
            )
        else:
            updated_lane_queries = lane_queries + endpoint_tokens
        
        # === 6. Topology Prediction ===
        topo_ll, topo_lte = self.topology_head(updated_lane_queries, te_queries)
        
        # === 7. Detection Predictions ===
        lane_boxes = self.lane_detection_head(lane_queries)
        te_pred = self.te_detection_head(te_queries)
        
        # === Build Output ===
        predictions = {
            'lane_geometry': lane_geometry,
            'lane_queries': updated_lane_queries,
            'lane_boxes': lane_boxes,
            'topology_lclc': torch.sigmoid(topo_ll),  # [B, N_lane, N_lane]
            'topology_lcte': torch.sigmoid(topo_lte), # [B, N_lane, N_te]
            'te_predictions': te_pred,
        }
        
        if return_aux:
            predictions.update({
                'te_queries': te_queries,
                'start_points': start_pred,
                'end_points': end_pred,
                'endpoint_tokens': endpoint_tokens,
                'dense_points': dense_points,
            })
        
        return predictions
    
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)