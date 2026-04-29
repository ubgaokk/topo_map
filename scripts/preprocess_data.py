#!/usr/bin/env python
"""
Data Preprocessing Script for OpenLane-V2

Converts raw OpenLane-V2 annotations to pickle format for efficient loading.

Usage:
    python scripts/preprocess_data.py --root /path/to/openlane_v2 --output /path/to/output
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess OpenLane-V2 data')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of OpenLane-V2 dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process')
    return parser.parse_args()


def load_openlane_topology_annotations(annotation_path: Path) -> Dict[str, Any]:
    """
    Load OpenLane Topology task annotations
    
    Returns:
        Dictionary with keys:
            - lane_centerline: list of lane dicts with 'id', 'points', 'start_point', 'end_point'
            - traffic_element: list of TE dicts with 'id', 'category', 'attribute', 'box_2d'
            - topology_lclc: [N_lane, N_lane] connection matrix
            - topology_lcte: [N_lane, N_te] association matrix
    """
    # This is a placeholder - actual implementation depends on
    # the specific annotation format of OpenLane-V2
    # 
    # Reference: https://github.com/OpenDriveLab/OpenLane-V2
    
    annotations = {
        'lane_centerline': [],
        'traffic_element': [],
        'topology_lclc': np.zeros((0, 0)),
        'topology_lcte': np.zeros((0, 0)),
    }
    
    # TODO: Implement actual annotation loading
    # The official format uses JSON files with:
    # - lane_centerline: {id: {points: [[x,y,z]...], start_point: [x,y,z], end_point: [x,y,z]}}
    # - traffic_element: {id: {category: str, attribute: str, box_2d: [x1,y1,x2,y2]}}
    # - topology_lclc: adjacency matrix for lane-lane connections
    # - topology_lcte: association matrix for lane-TE relations
    
    return annotations


def preprocess_scene(scene_info: Dict, source: str = 'argoverse') -> Dict:
    """
    Preprocess a single scene's annotations
    
    Args:
        scene_info: Raw scene annotation dictionary
        source: 'argoverse' or 'nuscenes'
        
    Returns:
        Processed scene info suitable for pickle serialization
    """
    processed = {
        'frame_id': scene_info.get('frame_id', 0),
        'source': source,
        'camera_intrinsics': scene_info.get('camera_intrinsics'),
        'camera_extrinsics': scene_info.get('camera_extrinsics'),
        'ego_pose': scene_info.get('ego_pose'),
        'lane_centerline': [],
        'traffic_element': [],
        'topology_lclc': np.zeros((0, 0)),
        'topology_lcte': np.zeros((0, 0)),
    }
    
    # Process lane centerlines
    if 'lane_centerline' in scene_info:
        for lane_id, lane_data in scene_info['lane_centerline'].items():
            centerline = {
                'id': lane_id,
                'points': np.array(lane_data.get('points', [])),
                'start_point': np.array(lane_data.get('start_point', [0, 0, 0])),
                'end_point': np.array(lane_data.get('end_point', [0, 0, 0])),
            }
            processed['lane_centerline'].append(centerline)
    
    # Process traffic elements
    if 'traffic_element' in scene_info:
        for te_id, te_data in scene_info['traffic_element'].items():
            te = {
                'id': te_id,
                'category': te_data.get('category'),
                'attribute': te_data.get('attribute'),
                'box_2d': te_data.get('box_2d'),
            }
            processed['traffic_element'].append(te)
    
    # Process topology matrices
    if 'topology_lclc' in scene_info:
        processed['topology_lclc'] = np.array(scene_info['topology_lclc'])
    
    if 'topology_lcte' in scene_info:
        processed['topology_lcte'] = np.array(scene_info['topology_lcte'])
    
    # Note: 'images' are not stored in pickle, they are loaded separately
    # from the raw data directory when needed
    
    return processed


def process_split(root_dir: Path, split: str, output_dir: Path):
    """
    Process all scenes in a split and save to pickle
    """
    print(f"Processing {split} split...")
    
    # Paths for OpenLane-V2 data structure
    annotation_dir = root_dir / split
    
    if not annotation_dir.exists():
        print(f"Warning: {annotation_dir} does not exist, skipping...")
        return
    
    # Collect all frame annotations
    all_frames = []
    
    # Iterate over all annotation files
    # The actual file structure depends on OpenLane-V2 format
    # Usually: /split/annotations/ or similar
    
    annotation_files = list(annotation_dir.rglob('*.json'))
    
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                scene_data = json.load(f)
            
            # Determine source dataset
            source = 'argoverse' if 'argoverse' in str(ann_file) else 'nuscenes'
            
            # Preprocess each frame
            if 'frames' in scene_data:
                for frame in scene_data['frames']:
                    processed = preprocess_scene(frame, source)
                    all_frames.append(processed)
            elif isinstance(scene_data, dict):
                # Single frame or scene-level annotation
                processed = preprocess_scene(scene_data, source)
                all_frames.append(processed)
                
        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue
    
    print(f"  Processed {len(all_frames)} frames")
    
    # Save to pickle
    output_file = output_dir / f'{split}_info.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_frames, f)
    
    print(f"  Saved to {output_file}")
    
    return len(all_frames)


def main():
    args = parse_args()
    
    root_dir = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"OpenLane-V2 Preprocessing")
    print(f"Root: {root_dir}")
    print(f"Output: {output_dir}")
    print()
    
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    total_frames = 0
    for split in splits:
        count = process_split(root_dir, split, output_dir)
        if count:
            total_frames += count
    
    print(f"\nTotal frames processed: {total_frames}")
    print("\nPreprocessing complete!")
    print(f"\nTo train, use:")
    print(f"  python scripts/train.py --config configs/experiments/exp_4_full.yaml")
    print(f"  (Make sure to update 'data_root' in the config to: {output_dir})")


if __name__ == '__main__':
    main()