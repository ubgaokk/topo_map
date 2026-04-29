"""Unit tests for model components"""

import unittest
import torch
import torch.nn as nn

from model.endpoint_detector import EndpointDetector, PointSampler
from model.point_lane_graph import PointLaneGraph, MessagePassingLayer
from model.topology_head import TopologyHead, LaneDetectionHead, TEDetectionHead
from model.bev_encoder import ViewTransformer


class TestEndpointDetector(unittest.TestCase):
    """Test EndpointDetector module"""
    
    def setUp(self):
        self.detector = EndpointDetector(dim=256, hidden_dim=128)
        self.batch_size = 2
        self.num_lanes = 10
        
        # Create dummy lane geometry: [B, N_lane, 11, 3]
        self.lane_geometry = torch.randn(self.batch_size, self.num_lanes, 11, 3)
        
        # Create dummy lane queries: [B, N_lane, dim]
        self.lane_queries = torch.randn(self.batch_size, self.num_lanes, 256)
    
    def test_output_shapes(self):
        """Test endpoint detector output shapes"""
        start_pred, end_pred, endpoint_tokens = self.detector(
            self.lane_queries, self.lane_geometry
        )
        
        # start_pred, end_pred: [B, N_lane, 3] (x, y, conf)
        self.assertEqual(start_pred.shape, (self.batch_size, self.num_lanes, 3))
        self.assertEqual(end_pred.shape, (self.batch_size, self.num_lanes, 3))
        
        # endpoint_tokens: [B, N_lane, dim]
        self.assertEqual(endpoint_tokens.shape, (self.batch_size, self.num_lanes, 256))
    
    def test_start_end_differentiable(self):
        """Test outputs are differentiable"""
        start_pred, end_pred, endpoint_tokens = self.detector(
            self.lane_queries, self.lane_geometry
        )
        
        loss = start_pred.sum() + end_pred.sum() + endpoint_tokens.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(start_pred.grad)
        self.assertIsNotNone(end_pred.grad)
        self.assertIsNotNone(endpoint_tokens.grad)
    
    def test_different_input_sizes(self):
        """Test with different batch sizes"""
        for bs in [1, 2, 4]:
            for nl in [5, 20, 100]:
                geom = torch.randn(bs, nl, 11, 3)
                queries = torch.randn(bs, nl, 256)
                
                start, end, tokens = self.detector(queries, geom)
                
                self.assertEqual(start.shape, (bs, nl, 3))
                self.assertEqual(end.shape, (bs, nl, 3))
                self.assertEqual(tokens.shape, (bs, nl, 256))


class TestPointSampler(unittest.TestCase):
    """Test PointSampler module"""
    
    def setUp(self):
        self.sampler = PointSampler(num_points=32)
    
    def test_sample_along_lane_shapes(self):
        """Test point sampling output shapes"""
        # Original geometry: [B, N_lane, 11, 3]
        lane_geometry = torch.randn(2, 10, 11, 3)
        
        dense_points, point_features = self.sampler.sample_along_lane(lane_geometry)
        
        # dense_points: [B, N_lane, num_points, 3]
        self.assertEqual(dense_points.shape, (2, 10, 32, 3))
        self.assertIsNone(point_features)  # Features come from BEV sampling
    
    def test_different_num_points(self):
        """Test with different num_points"""
        for num_pts in [8, 16, 32, 64]:
            sampler = PointSampler(num_points=num_pts)
            geom = torch.randn(1, 5, 11, 3)
            
            dense, _ = sampler.sample_along_lane(geom)
            self.assertEqual(dense.shape[2], num_pts)


class TestMessagePassingLayer(unittest.TestCase):
    """Test MessagePassingLayer"""
    
    def setUp(self):
        self.layer = MessagePassingLayer(dim=128, edge_types=3)
    
    def test_output_shape(self):
        """Test output shape matches input"""
        # [B, N, dim]
        node_features = torch.randn(2, 10, 128)
        adj_matrix = torch.zeros(2, 10, 10)
        adj_matrix[:, 0, 1] = 1  # Add some edges
        adj_matrix[:, 1, 2] = 1
        
        out = self.layer(node_features, adj_matrix)
        
        self.assertEqual(out.shape, (2, 10, 128))
    
    def test_differentiable(self):
        """Test gradient flow"""
        node_features = torch.randn(1, 5, 64, requires_grad=True)
        adj_matrix = torch.ones(1, 5, 5)
        
        out = self.layer(node_features, adj_matrix)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(node_features.grad)


class TestPointLaneGraph(unittest.TestCase):
    """Test PointLaneGraph module"""
    
    def setUp(self):
        self.graph = PointLaneGraph(dim=128, num_heads=4, gnn_layers=2)
    
    def test_output_shape(self):
        """Test graph output shape"""
        B, N_lane = 2, 10
        lane_features = torch.randn(B, N_lane, 128)
        lane_geometry = torch.randn(B, N_lane, 11, 3)
        endpoint_tokens = torch.randn(B, N_lane, 128)
        
        # Without point features
        out = self.graph(
            lane_features,
            point_features=None,
            lane_geometry=lane_geometry,
            endpoint_tokens=endpoint_tokens,
        )
        
        self.assertEqual(out.shape, (B, N_lane, 128))
    
    def test_with_point_features(self):
        """Test with point features provided"""
        B, N_lane, N_pts = 2, 10, 32
        lane_features = torch.randn(B, N_lane, 128)
        point_features = torch.randn(B, N_lane, N_pts, 128)
        lane_geometry = torch.randn(B, N_lane, 11, 3)
        endpoint_tokens = torch.randn(B, N_lane, 128)
        
        out = self.graph(
            lane_features,
            point_features=point_features,
            lane_geometry=lane_geometry,
            endpoint_tokens=endpoint_tokens,
        )
        
        self.assertEqual(out.shape, (B, N_lane, 128))
    
    def test_adjacency_building(self):
        """Test adjacency matrix construction"""
        lane_geometry = torch.randn(2, 10, 11, 3)
        
        adj, dist = self.graph.build_lane_adjacency(lane_geometry)
        
        self.assertEqual(adj.shape, (2, 10, 10))
        self.assertTrue(torch.allclose(adj, adj.transpose(1, 2)))  # Symmetric
        # Diagonal should be 1 (self-loops)
        self.assertTrue(torch.all(adj.diagonal(dim1=1, dim2=2) == 1))


class TestTopologyHead(unittest.TestCase):
    """Test TopologyHead"""
    
    def setUp(self):
        self.head = TopologyHead(dim=128, hidden_dim=64)
    
    def test_lane_lane_topology_shapes(self):
        """Test lane-lane topology output shape"""
        B, N = 2, 10
        lane_features = torch.randn(B, N, 128)
        
        topo_ll = self.head.lane_lane_head(
            lane_features.unsqueeze(2),  # [B, N, 1, dim]
            lane_features.unsqueeze(1),  # [B, 1, N, dim]
        )
        
        # Output: [B, N, N, 1]
        self.assertEqual(topo_ll.shape, (B, N, N, 1))
    
    def test_full_forward(self):
        """Test full forward pass"""
        B = 2
        lane_features = torch.randn(B, 10, 128)
        te_features = torch.randn(B, 5, 128)
        
        topo_ll, topo_lte = self.head(lane_features, te_features)
        
        # topo_ll: [B, N_lane, N_lane]
        self.assertEqual(topo_ll.shape, (B, 10, 10))
        # topo_lte: [B, N_lane, N_te]
        self.assertEqual(topo_lte.shape, (B, 10, 5))


class TestLaneDetectionHead(unittest.TestCase):
    """Test LaneDetectionHead"""
    
    def setUp(self):
        self.head = LaneDetectionHead(dim=128)
    
    def test_output_shape(self):
        """Test detection head output"""
        queries = torch.randn(2, 10, 128)
        out = self.head(queries)
        
        # [B, N_lane, 34] = 1 conf + 33 geometry
        self.assertEqual(out.shape, (2, 10, 34))


class TestTEDetectionHead(unittest.TestCase):
    """Test TEDetectionHead"""
    
    def setUp(self):
        self.head = TEDetectionHead(dim=128, num_attributes=13)
    
    def test_output_dict_keys(self):
        """Test TE head output structure"""
        queries = torch.randn(2, 5, 128)
        out = self.head(queries)
        
        self.assertIn('boxes', out)
        self.assertIn('attributes', out)
        self.assertIn('conf', out)
        
        # boxes: [B, N, 4]
        self.assertEqual(out['boxes'].shape, (2, 5, 4))
        # attributes: [B, N, num_attributes]
        self.assertEqual(out['attributes'].shape, (2, 5, 13))
        # conf: [B, N, 1]
        self.assertEqual(out['conf'].shape, (2, 5, 1))


class TestViewTransformer(unittest.TestCase):
    """Test ViewTransformer"""
    
    def setUp(self):
        self.vt = ViewTransformer(bev_h=100, bev_w=200, dim=128)
    
    def test_output_shape(self):
        """Test BEV feature output shape"""
        B, C, H, W = 2, 256, 64, 64
        # List of camera features
        multi_view = [torch.randn(B, C, H, W) for _ in range(6)]
        
        bev_feat = self.vt.get_bev_features(
            multi_view,
            camera_intrinsics=None,
            camera_extrinsics=None,
        )
        
        # Output: [B, dim, bev_h, bev_w]
        self.assertEqual(bev_feat.shape, (B, 128, 100, 200))


if __name__ == '__main__':
    unittest.main()