"""Unit tests for dataset module"""

import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import pickle
import tempfile
import os

from dataset.openlane_v2 import OpenLaneV2Dataset, collate_fn


class TestOpenLaneV2Dataset(unittest.TestCase):
    """Test OpenLane-V2 dataset loading"""
    
    @classmethod
    def setUpClass(cls):
        """Create a mock dataset for testing"""
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create mock data
        cls.mock_data = []
        for i in range(10):
            frame = {
                'frame_id': i,
                'source': 'argoverse',
                'camera_intrinsics': np.eye(3),
                'camera_extrinsics': np.eye(4),
                'ego_pose': np.eye(4),
                'lane_centerline': [
                    {
                        'id': f'lane_{i}_{j}',
                        'points': np.random.randn(11, 3).tolist(),
                        'start_point': [0, 0, 0],
                        'end_point': [1, 1, 0],
                    }
                    for j in range(3)
                ],
                'traffic_element': [
                    {
                        'id': f'te_{i}_{j}',
                        'category': 'stop_sign',
                        'attribute': 'stop',
                        'box_2d': [100, 100, 200, 200],
                    }
                    for j in range(2)
                ],
                'topology_lclc': np.zeros((3, 3)),
                'topology_lcte': np.zeros((3, 2)),
            }
            # Add some connections
            frame['topology_lclc'][0, 1] = 1
            frame['topology_lclc'][1, 2] = 1
            cls.mock_data.append(frame)
        
        # Add some invalid frames (empty lanes) for filtering test
        for i in range(5):
            cls.mock_data.append({
                'frame_id': 10 + i,
                'source': 'argoverse',
                'lane_centerline': [],
                'traffic_element': [],
            })
        
        # Save to pickle
        cls.pkl_path = Path(cls.temp_dir) / 'train_info.pkl'
        with open(cls.pkl_path, 'wb') as f:
            pickle.dump(cls.mock_data, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory"""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_dataset_loads(self):
        """Test dataset can be initialized"""
        dataset = OpenLaneV2Dataset(
            root_dir=self.temp_dir,
            split='train',
            task='topology',
        )
        self.assertIsNotNone(dataset)
    
    def test_dataset_length(self):
        """Test dataset length after filtering empty frames"""
        dataset = OpenLaneV2Dataset(
            root_dir=self.temp_dir,
            split='train',
            task='topology',
        )
        # Should be 10 (only frames with lanes)
        self.assertEqual(len(dataset), 10)
    
    def test_dataset_getitem(self):
        """Test getting a sample"""
        dataset = OpenLaneV2Dataset(
            root_dir=self.temp_dir,
            split='train',
            task='topology',
        )
        sample = dataset[0]
        
        self.assertIn('frame_id', sample)
        self.assertIn('lane_centerline', sample)
        self.assertIn('traffic_element', sample)
        self.assertIn('topology_lclc', sample)
        self.assertEqual(len(sample['lane_centerline']), 3)
    
    def test_dataset_with_empty_root(self):
        """Test error handling for non-existent path"""
        with self.assertRaises(FileNotFoundError):
            dataset = OpenLaneV2Dataset(
                root_dir='/nonexistent/path',
                split='train',
            )
    
    def test_collate_fn(self):
        """Test collate function"""
        batch = [{'a': 1}, {'a': 2}]
        result = collate_fn(batch)
        self.assertEqual(len(result), 2)


class TestOpenLaneV2MockedIntegration(unittest.TestCase):
    """Integration tests with mocked file operations"""
    
    def test_dataset_iteration(self):
        """Test iterating through dataset"""
        temp_dir = tempfile.mkdtemp()
        
        # Create minimal mock data
        mock_data = [
            {
                'frame_id': 0,
                'source': 'test',
                'lane_centerline': [{
                    'id': 'lane_0',
                    'points': [[0, 0, 0], [1, 1, 0]] * 6,  # 12 points
                    'start_point': [0, 0, 0],
                    'end_point': [1, 1, 0],
                }],
                'traffic_element': [],
                'topology_lclc': np.zeros((1, 1)),
                'topology_lcte': np.zeros((1, 0)),
            }
        ]
        
        pkl_path = Path(temp_dir) / 'val_info.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(mock_data, f)
        
        dataset = OpenLaneV2Dataset(root_dir=temp_dir, split='val')
        
        # Iterate
        count = 0
        for sample in dataset:
            count += 1
            self.assertIn('frame_id', sample)
        
        self.assertEqual(count, 1)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()