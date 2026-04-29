#!/usr/bin/env python
"""
Main Training Script for TopoMap

Usage:
    python scripts/train.py --config configs/experiments/exp_4_full.yaml
    
Options:
    --config     : Path to experiment config file
    --resume     : Path to checkpoint to resume from
    --amp        : Enable automatic mixed precision (AMP)
    --compile    : Enable torch.compile() for faster training (PyTorch 2.0+)
    --debug      : Enable debug mode
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from topo_map import EndpointAwareTopologyNet, OpenLaneV2Dataset, TopologyLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train TopoMap model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision training')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile() for faster training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict, use_compile: bool = False) -> nn.Module:
    """Build model from config"""
    model = EndpointAwareTopologyNet(config.get('model', {}))
    
    # Apply torch.compile if requested (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    return model


def build_dataloader(config: dict, split: str = 'train') -> DataLoader:
    """Build dataloader from config"""
    dataset = OpenLaneV2Dataset(
        root_dir=config['data']['data_root'],
        split=split,
        task='topology',
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=lambda x: x,  # Return list of samples
    )
    
    return dataloader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: TopologyLoss,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
    writer: SummaryWriter = None,
    use_amp: bool = False,
) -> float:
    """
    Train for one epoch
    
    Args:
        use_amp: Enable automatic mixed precision
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    # AMP gradient scaler
    scaler = GradScaler() if use_amp else None
    
    for batch_idx, samples in enumerate(dataloader):
        batch_losses = []
        
        for sample in samples:
            # Move sample to device
            images = sample.get('images', {})
            if isinstance(images, dict):
                images = [img.to(device) for img in images.values()]
            elif isinstance(images, list):
                images = [img.to(device) for img in images]
            
            camera_params = {
                'intrinsics': sample.get('camera_intrinsics', torch.eye(3, device=device),
                                        dtype=torch.float32),
                'extrinsics': sample.get('camera_extrinsics', torch.eye(4, device=device),
                                        dtype=torch.float32),
            }
            
            # Forward with optional mixed precision
            if use_amp:
                with autocast(device_type='cuda'):
                    predictions = model(images, **camera_params, return_aux=True)
                    targets = build_targets_from_sample(sample)
                    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in targets.items()}
                    losses, loss = loss_fn(predictions, targets)
                batch_losses.append(loss)
                
                # Backward with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Unscale and clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler update
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(images, **camera_params, return_aux=True)
                targets = build_targets_from_sample(sample)
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in targets.items()}
                losses, loss = loss_fn(predictions, targets)
                batch_losses.append(loss)
                
                # Standard backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Average loss over batch
        batch_loss = torch.stack(batch_losses).mean()
        total_loss += batch_loss.item()
        total_batches += 1
        
        # Logging
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {batch_loss.item():.4f}")
            
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('train/loss', batch_loss.item(), global_step)
    
    return total_loss / total_batches


def build_targets_from_sample(sample: dict) -> dict:
    """
    Build target dictionary from raw sample
    
    This is a simplified version. Real implementation would need
    proper post-processing of annotations.
    """
    targets = {}
    
    # Lane ground truth
    if 'lane_centerline' in sample:
        lanes = sample['lane_centerline']
        if len(lanes) > 0:
            # Stack all lane points
            all_points = []
            for lane in lanes:
                points = lane.get('points', [])
                if len(points) >= 11:
                    all_points.append(points[:11])
                elif len(points) > 0:
                    # Pad to 11 points
                    pts = points + [points[-1]] * (11 - len(points))
                    all_points.append(pts[:11])
            
            if all_points:
                targets['lane_gt'] = {
                    'geometry': torch.tensor(all_points, dtype=torch.float32),
                }
                
                # Start/end points
                starts = [[l.get('start_point', [0, 0, 0])[:2] for l in lanes]]
                ends = [[l.get('end_point', [0, 0, 0])[:2] for l in lanes]]
                targets['lane_gt']['start_points'] = torch.tensor(starts, dtype=torch.float32)
                targets['lane_gt']['end_points'] = torch.tensor(ends, dtype=torch.float32)
    
    # Topology ground truth
    if 'topology_lclc' in sample:
        targets['topology_lclc'] = torch.tensor(sample['topology_lclc'], dtype=torch.float32)
    
    if 'topology_lcte' in sample:
        targets['topology_lcte'] = torch.tensor(sample['topology_lcte'], dtype=torch.float32)
    
    return targets


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: TopologyLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
) -> float:
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for samples in dataloader:
            for sample in samples:
                images = sample.get('images', {})
                if isinstance(images, dict):
                    images = [img.to(device) for img in images.values()]
                elif isinstance(images, list):
                    images = [img.to(device) for img in images]
                
                camera_params = {
                    'intrinsics': sample.get('camera_intrinsics', torch.eye(3, device=device),
                                            dtype=torch.float32),
                    'extrinsics': sample.get('camera_extrinsics', torch.eye(4, device=device),
                                            dtype=torch.float32),
                }
                
                predictions = model(images, **camera_params, return_aux=False)
                targets = build_targets_from_sample(sample)
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in targets.items()}
                
                _, loss = loss_fn(predictions, targets)
                total_loss += loss.item()
                total_batches += 1
    
    avg_loss = total_loss / max(total_batches, 1)
    
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"AMP enabled: {args.amp}")
    print(f"torch.compile enabled: {args.compile}")
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config.get('logging', {}).get('experiment_name', 'default')
    output_dir = Path('./outputs') / f'{exp_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Build model
    model = build_model(config, use_compile=args.compile)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build dataloaders
    train_loader = build_dataloader(config, split='train')
    val_loader = build_dataloader(config, split='val')
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Build optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01),
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
    )
    
    # Build loss
    loss_fn = TopologyLoss(
        lambda_detection=config['training'].get('lambda_detection', 1.0),
        lambda_topology=config['training'].get('lambda_topology', 1.0),
        lambda_endpoint=config['training'].get('lambda_endpoint', 0.5),
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            log_interval=config.get('logging', {}).get('log_interval', 50),
            writer=writer,
            use_amp=args.amp,
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validate every N epochs
        if epoch % config.get('training', {}).get('eval_interval', 5) == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, writer)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint
        if epoch % config.get('training', {}).get('save_interval', 5) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()