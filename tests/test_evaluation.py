"""Unit tests for evaluation module"""

import unittest
import torch
import numpy as np

from evaluation.openlane_v2_eval import (
    compute_ols_metrics,
    compute_lane_detection_score,
    compute_topology_ap,
    compute_frame_metrics,
)


class TestComputeOLSMets(unittest.TestCase):
    """Test OLS metrics computation"""
    
    def test_empty_results(self):
        """Test with empty results"""
        results = []
        metrics = compute_ols_metrics(results)
        
        self.assertIn('ols', metrics)
        self.assertEqual(metrics['ols'], 0.0)
    
    def test_single_frame(self):
        """Test with single frame"""
        results = [{
            'det_lane_score': 0.5,
            'det_te_score': 0.6,
            'top_ll_score': 0.4,
            'top_lte_score': 0.7,
        }]
        
        metrics = compute_ols_metrics(results)
        
        # OLS = sqrt(0.5 * 0.4) * 0.5 + 0.6 * 0.25 + 0.5 * 0.25
        expected_ols = np.sqrt(0.5 * 0.4) * 0.5 + 0.6 * 0.25 + 0.5 * 0.25
        self.assertAlmostEqual(metrics['ols'], expected_ols, places=5)
    
    def test_multiple_frames(self):
        """Test averaging across frames"""
        results = [
            {'det_lane_score': 0.5, 'det_te_score': 0.5, 'top_ll_score': 0.5, 'top_lte_score': 0.5},
            {'det_lane_score': 0.6, 'det_te_score': 0.6, 'top_ll_score': 0.6, 'top_lte_score': 0.6},
        ]
        
        metrics = compute_ols_metrics(results)
        
        # Should average the inputs
        self.assertEqual(metrics['det_lane'], 0.55)


class TestComputeLaneDetectionScore(unittest.TestCase):
    """Test lane detection scoring"""
    
    def test_empty_prediction(self):
        """Test with no predictions"""
        score = compute_lane_detection_score([], [])
        self.assertEqual(score, 1.0)  # Perfect if both empty
    
    def test_no_prediction_no_gt(self):
        """Test with predictions but no GT"""
        pred = [np.random.randn(11, 3) for _ in range(3)]
        score = compute_lane_detection_score(pred, [])
        self.assertEqual(score, 0.0)
    
    def test_no_prediction_with_gt(self):
        """Test with GT but no predictions"""
        gt = [np.random.randn(11, 3) for _ in range(3)]
        score = compute_lane_detection_score([], gt)
        self.assertEqual(score, 0.0)
    
    def test_matching_lanes(self):
        """Test with perfectly matching lanes"""
        # Same lane repeated
        lane = np.array([[0, 0, 0], [1, 1, 0]] * 6)  # 12 points
        pred = [lane]
        gt = [lane]
        
        score = compute_lane_detection_score(pred, gt, thresholds=[1.0, 2.0, 3.0])
        
        self.assertGreater(score, 0.9)


class TestComputeTopologyAP(unittest.TestCase):
    """Test topology AP computation"""
    
    def test_empty_matrices(self):
        """Test with empty matrices"""
        ap = compute_topology_ap(
            np.zeros((0, 0)),
            np.zeros((0, 0))
        )
        self.assertEqual(ap, 1.0)
    
    def test_no_predictions(self):
        """Test with no predictions but GT exists"""
        ap = compute_topology_ap(
            np.zeros((5, 5)),
            np.ones((5, 5))
        )
        self.assertEqual(ap, 0.0)
    
    def test_perfect_prediction(self):
        """Test with perfect prediction"""
        matrix = np.eye(5)  # Perfect diagonal
        ap = compute_topology_ap(matrix, matrix)
        self.assertEqual(ap, 1.0)
    
    def test_partial_prediction(self):
        """Test with partial correct predictions"""
        pred = np.zeros((5, 5))
        pred[0, 1] = 1
        pred[1, 2] = 1
        
        gt = np.zeros((5, 5))
        gt[0, 1] = 1
        gt[1, 2] = 1
        gt[2, 3] = 1  # Extra in GT
        
        ap = compute_topology_ap(pred, gt)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)


class TestComputeFrameMetrics(unittest.TestCase):
    """Test frame-level metrics"""
    
    def test_empty_frame(self):
        """Test with minimal data"""
        result = {
            'pred_lanes': np.array([]),
            'gt_lanes': [],
            'pred_topology_lclc': np.zeros((0, 0)),
            'gt_topology_lclc': np.zeros((0, 0)),
        }
        
        metrics = compute_frame_metrics(result)
        
        self.assertIn('det_lane_score', metrics)
        self.assertIn('top_ll_score', metrics)
    
    def test_frame_with_lanes(self):
        """Test with actual lane data"""
        # Create matching prediction and GT
        lane = np.array([[0, 0, 0], [1, 1, 0]] * 6)
        
        result = {
            'pred_lanes': [lane],
            'gt_lanes': [lane],
            'pred_topology_lclc': np.array([[0.9]]),
            'gt_topology_lclc': np.array([[1]]),
        }
        
        metrics = compute_frame_metrics(result)
        
        self.assertGreater(metrics['det_lane_score'], 0.5)
        self.assertGreater(metrics['top_ll_score'], 0.5)


if __name__ == '__main__':
    unittest.main()