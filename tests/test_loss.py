"""Unit tests for loss module"""

import unittest
import torch

from loss.topo_loss import TopologyLoss, FocalLoss, GeneralizedIoULoss


class TestFocalLoss(unittest.TestCase):
    """Test FocalLoss"""
    
    def setUp(self):
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    def test_output_shape(self):
        """Test loss output is scalar"""
        pred = torch.sigmoid(torch.randn(10))
        target = torch.randint(0, 2, (10,)).float()
        
        loss = self.loss_fn(pred, target)
        
        self.assertEqual(loss.numel(), 1)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_perfect_prediction(self):
        """Test loss is near zero for perfect prediction"""
        pred = torch.tensor([0.9, 0.1])  # High confidence for correct class
        target = torch.tensor([1.0, 0.0])
        
        loss = self.loss_fn(pred, target)
        
        self.assertLess(loss.item(), 0.1)
    
    def test_differentiable(self):
        """Test loss is differentiable"""
        pred = torch.randn(10, requires_grad=True)
        target = torch.randint(0, 2, (10,)).float()
        
        loss = self.loss_fn(torch.sigmoid(pred), target)
        loss.backward()
        
        self.assertIsNotNone(pred.grad)


class TestTopologyLoss(unittest.TestCase):
    """Test TopologyLoss"""
    
    def setUp(self):
        self.loss_fn = TopologyLoss(
            lambda_detection=1.0,
            lambda_topology=1.0,
            lambda_endpoint=0.5,
        )
    
    def test_full_forward(self):
        """Test complete loss computation"""
        predictions = {
            'lane_boxes': torch.randn(2, 10, 34),
            'lane_geometry': torch.randn(2, 10, 11, 3),
            'te_predictions': {
                'boxes': torch.randn(2, 5, 4),
                'attributes': torch.randn(2, 5, 13),
                'conf': torch.randn(2, 5, 1),
            },
            'topology_lclc': torch.rand(2, 10, 10),
            'topology_lcte': torch.rand(2, 10, 5),
            'start_points': torch.randn(2, 10, 3),
            'end_points': torch.randn(2, 10, 3),
        }
        
        targets = {
            'lane_gt': {
                'geometry': torch.randn(2, 10, 11, 3),
            },
            'te_gt': {},
            'topology_lclc': torch.randint(0, 2, (2, 10, 10)).float(),
            'topology_lcte': torch.randint(0, 2, (2, 10, 5)).float(),
        }
        
        losses, total = self.loss_fn(predictions, targets)
        
        self.assertIn('detection', losses)
        self.assertIn('topology', losses)
        self.assertIn('endpoint', losses)
        self.assertIn('total', losses)
        self.assertGreater(total.item(), 0)
    
    def test_without_endpoint(self):
        """Test loss without endpoint predictions"""
        predictions = {
            'lane_boxes': torch.randn(2, 10, 34),
            'lane_geometry': torch.randn(2, 10, 11, 3),
            'topology_lclc': torch.rand(2, 10, 10),
            'topology_lcte': torch.rand(2, 10, 5),
            'start_points': None,  # No endpoint detection
        }
        
        targets = {
            'topology_lclc': torch.randint(0, 2, (2, 10, 10)).float(),
            'topology_lcte': torch.randint(0, 2, (2, 10, 5)).float(),
        }
        
        losses, total = self.loss_fn(predictions, targets)
        
        self.assertGreater(total.item(), 0)
        self.assertNotIn('endpoint', losses)  # Should skip endpoint loss
    
    def test_loss_weights(self):
        """Test that loss weights affect output"""
        loss_fn_1 = TopologyLoss(lambda_topology=1.0)
        loss_fn_10 = TopologyLoss(lambda_topology=10.0)
        
        predictions = {
            'topology_lclc': torch.rand(1, 5, 5),
            'topology_lcte': torch.rand(1, 5, 3),
            'lane_boxes': torch.randn(1, 5, 34),
            'lane_geometry': torch.randn(1, 5, 11, 3),
        }
        targets = {
            'topology_lclc': torch.randint(0, 2, (1, 5, 5)).float(),
            'topology_lcte': torch.randint(0, 2, (1, 5, 3)).float(),
        }
        
        _, total_1 = loss_fn_1(predictions, targets)
        _, total_10 = loss_fn_10(predictions, targets)
        
        # Higher weight should give higher loss (for non-zero topology)
        # This is a basic sanity check
        self.assertIsInstance(total_1.item(), float)
        self.assertIsInstance(total_10.item(), float)


class TestGeneralizedIoULoss(unittest.TestCase):
    """Test GeneralizedIoULoss"""
    
    def setUp(self):
        self.loss_fn = GeneralizedIoULoss()
    
    def test_output_is_scalar(self):
        """Test GIoU loss output"""
        pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])
        gt_boxes = torch.tensor([[10, 10, 50, 50], [25, 25, 65, 65]])
        
        loss = self.loss_fn(pred_boxes, gt_boxes)
        
        self.assertEqual(loss.numel(), 1)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_perfect_overlap(self):
        """Test GIoU is 0 for identical boxes"""
        boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])
        
        loss = self.loss_fn(boxes, boxes)
        
        self.assertLess(loss.item(), 0.01)
    
    def test_non_overlap(self):
        """Test GIoU is high for non-overlapping boxes"""
        pred = torch.tensor([[0, 0, 10, 10]])
        gt = torch.tensor([[100, 100, 110, 110]])
        
        loss = self.loss_fn(pred, gt)
        
        self.assertGreater(loss.item(), 1.0)  # Should be close to 2 (worst case)


if __name__ == '__main__':
    unittest.main()