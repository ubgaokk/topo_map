"""Integration tests for the complete model pipeline"""

import unittest
import torch
import tempfile
import yaml
from pathlib import Path

from model.toponet_endpoint import EndpointAwareTopologyNet, LaneDecoder, TEDecoder
from loss.topo_loss import TopologyLoss
from evaluation.openlane_v2_eval import compute_ols_metrics


class TestEndToEndForward(unittest.TestCase):
    """Test complete model forward pass"""
    
    def setUp(self):
        self.config = {
            'dim': 128,  # Smaller for faster tests
            'num_lane_queries': 20,
            'num_te_queries': 10,
            'bev_h': 50,
            'bev_w': 100,
            'num_decoder_layers': 2,
            'num_heads': 4,
            'num_points': 16,
            'use_endpoint_detector': True,
            'use_point_lane_graph': True,
            'endpoint_hidden_dim': 64,
            'gnn_layers': 2,
            'edge_types': 3,
            'adjacency_threshold': 5.0,
            'dropout': 0.1,
        }
        self.model = EndpointAwareTopologyNet(self.config)
        self.model.eval()
    
    def test_full_forward_shapes(self):
        """Test complete forward pass produces correct shapes"""
        B = 2
        N_cam = 6
        
        # Create dummy multi-view images
        multi_view = [
            torch.randn(B, 256, 64, 64)
            for _ in range(N_cam)
        ]
        
        with torch.no_grad():
            predictions = self.model(
                multi_view,
                camera_intrinsics=None,
                camera_extrinsics=None,
                return_aux=True
            )
        
        # Check output shapes
        self.assertIn('lane_geometry', predictions)
        self.assertIn('topology_lclc', predictions)
        self.assertIn('topology_lcte', predictions)
        
        # Lane geometry: [B, N_lane, 11, 3]
        self.assertEqual(predictions['lane_geometry'].shape, (B, 20, 11, 3))
        
        # Topology matrices: [B, N_lane, N_lane] and [B, N_lane, N_te]
        self.assertEqual(predictions['topology_lclc'].shape, (B, 20, 20))
        self.assertEqual(predictions['topology_lcte'].shape, (B, 20, 10))
        
        # Endpoint predictions
        self.assertIn('start_points', predictions)
        self.assertIn('end_points', predictions)
        self.assertIsNotNone(predictions['start_points'])
    
    def test_without_aux(self):
        """Test forward with return_aux=False"""
        multi_view = [torch.randn(1, 256, 64, 64) for _ in range(6)]
        
        with torch.no_grad():
            predictions = self.model(multi_view, return_aux=False)
        
        # Should not have aux outputs
        self.assertIn('start_points', predictions)
        # But they might be None or not in dict depending on implementation
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes"""
        for B in [1, 2, 4]:
            multi_view = [torch.randn(B, 256, 64, 64) for _ in range(6)]
            
            with torch.no_grad():
                predictions = self.model(multi_view)
            
            self.assertEqual(predictions['lane_geometry'].shape[0], B)


class TestModelComponents(unittest.TestCase):
    """Test individual model components"""
    
    def test_lane_decoder(self):
        """Test LaneDecoder"""
        decoder = LaneDecoder(dim=64, num_queries=10, num_layers=2)
        
        bev_feat = torch.randn(2, 64, 50, 100)
        queries, geometry = decoder(bev_feat)
        
        self.assertEqual(queries.shape, (2, 10, 64))
        self.assertEqual(geometry.shape, (2, 10, 11, 3))
    
    def test_te_decoder(self):
        """Test TEDecoder"""
        decoder = TEDecoder(dim=64, num_queries=5, num_layers=2)
        
        bev_feat = torch.randn(2, 64, 50, 100)
        queries = decoder(bev_feat)
        
        self.assertEqual(queries.shape, (2, 5, 64))


class TestTrainingLoss(unittest.TestCase):
    """Test that loss can be computed during training"""
    
    def setUp(self):
        self.config = {
            'dim': 64,
            'num_lane_queries': 10,
            'num_te_queries': 5,
            'bev_h': 25,
            'bev_w': 50,
            'use_endpoint_detector': True,
            'use_point_lane_graph': True,
        }
        self.model = EndpointAwareTopologyNet(self.config)
        self.loss_fn = TopologyLoss(
            lambda_detection=1.0,
            lambda_topology=1.0,
            lambda_endpoint=0.5,
        )
    
    def test_training_step(self):
        """Test a single training step"""
        B = 1
        multi_view = [torch.randn(B, 256, 32, 32) for _ in range(6)]
        
        # Forward
        predictions = self.model(multi_view, return_aux=True)
        
        # Create mock targets
        targets = {
            'lane_gt': {
                'geometry': torch.randn(B, 10, 11, 3),
                'start_points': torch.randn(B, 10, 2),
                'end_points': torch.randn(B, 10, 2),
            },
            'topology_lclc': torch.randint(0, 2, (B, 10, 10)).float(),
            'topology_lcte': torch.randint(0, 2, (B, 10, 5)).float(),
        }
        
        # Compute loss
        losses, total_loss = self.loss_fn(predictions, targets)
        
        self.assertIsInstance(total_loss.item(), float)
        self.assertGreater(total_loss.item(), 0)
        
        # Backward
        total_loss.backward()
        
        # Check gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
    
    def test_loss_without_endpoint_detection(self):
        """Test training without endpoint detection"""
        config = self.config.copy()
        config['use_endpoint_detector'] = False
        model = EndpointAwareTopologyNet(config)
        
        multi_view = [torch.randn(1, 256, 32, 32) for _ in range(6)]
        predictions = model(multi_view, return_aux=True)
        
        targets = {
            'topology_lclc': torch.randint(0, 2, (1, 10, 10)).float(),
            'topology_lcte': torch.randint(0, 2, (1, 10, 5)).float(),
        }
        
        losses, total_loss = self.loss_fn(predictions, targets)
        
        self.assertIsInstance(total_loss.item(), float)


class TestConfigLoading(unittest.TestCase):
    """Test configuration handling"""
    
    def test_config_from_yaml(self):
        """Test loading config from YAML"""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / 'test_config.yaml'
        
        config_data = {
            'model': {
                'dim': 256,
                'num_lane_queries': 200,
                'use_endpoint_detector': True,
                'use_point_lane_graph': True,
            },
            'training': {
                'batch_size': 4,
                'lr': 2e-4,
                'epochs': 30,
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        self.assertEqual(loaded['model']['dim'], 256)
        self.assertEqual(loaded['training']['batch_size'], 4)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()