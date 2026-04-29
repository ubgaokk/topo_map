#!/usr/bin/env python
"""
Evaluation Script for TopoMap

Usage:
    python scripts/eval.py --checkpoint outputs/best_model.pth --data /path/to/data
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from topo_map import EndpointAwareTopologyNet, OpenLaneV2Dataset
from evaluation.openlane_v2_eval import evaluate_openlane_v2


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TopoMap model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file used for training (optional)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Build model config (use defaults if not provided)
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
    else:
        model_config = {
            'dim': 256,
            'num_lane_queries': 200,
            'num_te_queries': 100,
            'bev_h': 100,
            'bev_w': 200,
            'use_endpoint_detector': True,
            'use_point_lane_graph': True,
        }
    
    # Build model
    model = EndpointAwareTopologyNet(model_config)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Build dataset
    dataset = OpenLaneV2Dataset(
        root_dir=args.data,
        split=args.split,
        task='topology',
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Run evaluation
    metrics = evaluate_openlane_v2(model, dataloader, device)
    
    # Print results
    print("\n" + "=" * 50)
    print("OpenLane-V2 Evaluation Results")
    print("=" * 50)
    print(f"Detection (Lane): {metrics['det_lane']:.4f}")
    print(f"Detection (TE):   {metrics['det_te']:.4f}")
    print(f"TOP_ll:           {metrics['top_ll']:.4f}")
    print(f"TOP_lte:          {metrics['top_lte']:.4f}")
    print(f"OLS:              {metrics['ols']:.4f}")
    print("=" * 50)
    
    # Save results
    results_path = Path(args.checkpoint).parent / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("OpenLane-V2 Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()