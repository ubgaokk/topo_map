"""OpenLane-V2 Dataset Loader"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class OpenLaneV2Dataset(Dataset):
    """
    OpenLane-V2 Dataset Loader
    
    Supports:
    - OpenLane Topology task (centerline-based)
    - Driving Scene Topology task (lane-segment-based)
    
    Args:
        root_dir: Root directory of OpenLane-V2 dataset
        split: 'train', 'val', or 'test'
        task: 'topology' (original centerline) or 'segment' (lane segments)
        grid_size: BEV grid dimensions (H, W)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        task: str = 'topology',
        grid_size: Tuple[int, int] = (100, 200),
    ):
        self.root = Path(root_dir)
        self.split = split
        self.task = task
        self.grid_h, self.grid_w = grid_size
        
        # Load preprocessed data
        self.data = self._load_data()
        
        # Filter valid frames (frames with at least one lane)
        self.valid_indices = self._filter_valid_frames()
    
    def _load_data(self) -> List[Dict]:
        """Load preprocessed pickle data"""
        pkl_path = self.root / f'{self.split}_info.pkl'
        
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {pkl_path}. "
                f"Please run preprocessing script first."
            )
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def _filter_valid_frames(self) -> List[int]:
        """Filter frames with valid annotations"""
        valid = []
        
        for i, info in enumerate(self.data):
            if self.task == 'topology':
                num_lanes = len(info.get('lane_centerline', []))
            else:
                num_lanes = len(info.get('lane_segment', []))
            
            if num_lanes > 0:
                valid.append(i)
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        real_idx = self.valid_indices[idx]
        info = self.data[real_idx]
        
        sample = {
            'frame_id': info.get('frame_id', real_idx),
            'source': info.get('source', 'unknown'),
        }
        
        # === Multi-view Images ===
        # Stored as dict: camera_id -> tensor [C, H, W]
        sample['images'] = info.get('images', {})
        
        # === Camera Parameters ===
        sample['camera_intrinsics'] = info.get('camera_intrinsics')
        sample['camera_extrinsics'] = info.get('camera_extrinsics')
        sample['ego_pose'] = info.get('ego_pose')
        
        # === Lane Annotations ===
        if self.task == 'topology':
            sample['lane_centerline'] = info.get('lane_centerline', [])
            sample['topology_lclc'] = info.get('topology_lclc')  # [N_lane, N_lane]
            sample['topology_lcte'] = info.get('topology_lcte')  # [N_lane, N_te]
        else:
            sample['lane_segment'] = info.get('lane_segment', [])
            sample['topology_lsls'] = info.get('topology_lsls')
            sample['topology_lste'] = info.get('topology_lste')
        
        # === Traffic Element Annotations ===
        sample['traffic_element'] = info.get('traffic_element', [])
        
        return sample
    
    def get_annotations(self, idx: int) -> Dict:
        """Get ground truth annotations for evaluation"""
        real_idx = self.valid_indices[idx]
        info = self.data[real_idx]
        
        annotations = {}
        
        # Lane centerlines with topology
        if self.task == 'topology':
            lanes = []
            for lane in info.get('lane_centerline', []):
                lane_data = {
                    'id': lane.get('id'),
                    'points': np.array(lane.get('points', [])),  # [N, 3]
                    'start_point': np.array(lane.get('start_point', [0, 0, 0])),
                    'end_point': np.array(lane.get('end_point', [0, 0, 0])),
                }
                lanes.append(lane_data)
            
            annotations['lanes'] = lanes
            annotations['topology_lclc'] = info.get('topology_lclc')
            annotations['topology_lcte'] = info.get('topology_lcte')
        
        # Traffic elements
        te_list = []
        for te in info.get('traffic_element', []):
            te_data = {
                'id': te.get('id'),
                'category': te.get('category'),
                'attribute': te.get('attribute'),
                'box_2d': te.get('box_2d'),  # [x1, y1, x2, y2] in image
            }
            te_list.append(te_data)
        
        annotations['traffic_elements'] = te_list
        
        return annotations


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching
    
    Handles variable number of lanes and traffic elements per frame
    """
    # For now, return batch as list (model handles batching internally)
    # In production, would pad to fixed length
    return batch


def visualize_sample(sample: Dict, save_path: Optional[str] = None):
    """
    Visualize a sample for debugging
    
    Args:
        sample: Sample dict from __getitem__
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Left: Lane topology
    ax = axes[0]
    
    if 'lane_centerline' in sample:
        lanes = sample['lane_centerline']
        topology = sample.get('topology_lclc')
        
        # Draw lanes
        for i, lane in enumerate(lanes):
            points = np.array(lane.get('points', []))
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.7)
                ax.scatter(points[0, 0], points[0, 1], c='g', s=50, marker='o')  # start
                ax.scatter(points[-1, 0], points[-1, 1], c='r', s=50, marker='x')  # end
        
        # Draw topology edges
        if topology is not None:
            for i in range(len(lanes)):
                for j in range(len(lanes)):
                    if topology[i, j] > 0.5:
                        lane_i = lanes[i]
                        lane_j = lanes[j]
                        end = np.array(lane_i.get('end_point', [0, 0]))[:2]
                        start = np.array(lane_j.get('start_point', [0, 0]))[:2]
                        ax.arrow(end[0], end[1], start[0] - end[0], start[1] - end[1],
                                head_width=1, head_length=0.5, fc='green', ec='green')
    
    ax.set_title('Lane Topology')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Right: Traffic elements
    ax = axes[1]
    ax.set_title('Traffic Elements (2D Image Plane)')
    
    for te in sample.get('traffic_element', []):
        box = te.get('box_2d', [])
        if len(box) == 4:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height,
                            linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{te.get('category', '?')}\n{te.get('attribute', '')}",
                   fontsize=8, va='bottom')
    
    ax.set_xlim(0, 1600)
    ax.set_ylim(900, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()