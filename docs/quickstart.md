# Quick Start Guide

## Prerequisites

- Python 3.9+
- CUDA 11.8+ / PyTorch 2.0+
- ~50GB disk space for dataset and outputs
- GPU with 8GB+ VRAM (16GB recommended for training)

## Step 1: Environment Setup

```bash
# Create conda environment
conda create -n topo_map python=3.9
conda activate topo_map

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone this repository
git clone https://github.com/ubgaokk/topo_map.git
cd topo_map

# Install dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```

## Step 2: Data Preparation

### Download OpenLane-V2 Dataset

1. Visit [OpenLane-V2 GitHub](https://github.com/OpenDriveLab/OpenLane-V2)
2. Download subset_A (recommended, ~151GB)
3. Extract to a convenient location

### Preprocess Data

```bash
python scripts/preprocess_data.py \
    --root /path/to/openlane_v2/subset_A \
    --output /path/to/preprocessed_data \
    --split all
```

This creates pickle files:
- `train_info.pkl`
- `val_info.pkl`  
- `test_info.pkl`

## Step 3: Configure Training

Edit your experiment config, e.g., `configs/experiments/exp_4_full.yaml`:

```yaml
data:
  data_root: /path/to/preprocessed_data  # Update this!

training:
  batch_size: 4  # Adjust based on GPU memory
  lr: 2.0e-4
  epochs: 30

logging:
  project_name: topo_map
  experiment_name: exp_4_full
```

## Step 4: Train Baseline First

Before training the full model, verify baseline works:

```bash
# Train baseline (faster, ~2 hours)
python scripts/train.py --config configs/experiments/exp_1_baseline.yaml

# Monitor with TensorBoard
tensorboard --logdir ./outputs/exp_1_baseline_*/
```

## Step 5: Train Full Model

```bash
# Train full model (~6-8 hours on 1x A100)
python scripts/train.py --config configs/experiments/exp_4_full.yaml

# Or resume from checkpoint
python scripts/train.py \
    --config configs/experiments/exp_4_full.yaml \
    --resume outputs/exp_4_full_*/checkpoint_epoch_10.pth
```

## Step 6: Evaluate

```bash
python scripts/eval.py \
    --checkpoint outputs/exp_4_full_*/best_model.pth \
    --data /path/to/preprocessed_data \
    --split val
```

Expected output:
```
==================================================
OpenLane-V2 Evaluation Results
==================================================
Detection (Lane): 0.52xx
Detection (TE):   0.61xx
TOP_ll:           0.39xx
TOP_lte:          0.45xx
OLS:              0.46xx
==================================================
```

## Common Issues

### Out of Memory (OOM)

If you get CUDA OOM errors:
1. Reduce batch size: `batch_size: 2`
2. Enable gradient checkpointing (not implemented yet, coming soon)
3. Use mixed precision training (not implemented yet, coming soon)

### Data Loading Errors

If you see pickle loading errors:
1. Verify data path in config
2. Try re-running preprocessing
3. Check file permissions

### Slow Training

Training should process ~2-3 batches/second on A100. If slower:
1. Check data loading workers: `num_workers: 8`
2. Verify dataset fits in memory
3. Use faster storage (NVMe SSD)

## Next Steps

1. **Read the Architecture Docs**: `docs/architecture.md`
2. **Review Experiment Design**: `docs/experiments.md`
3. **Try Different Configs**: Test exp_2, exp_3 for ablation study
4. **Customize**: Modify endpoint detector or graph architecture

## File Structure Overview

```
topo_map/
├── configs/              # Experiment configurations
├── dataset/              # Data loading
├── model/                # Model architecture
│   ├── toponet_endpoint.py  # Main model
│   ├── endpoint_detector.py # Endpoint detection
│   ├── point_lane_graph.py  # Point-lane graph
│   └── topology_head.py     # Prediction heads
├── loss/                 # Loss functions
├── evaluation/           # Metrics and evaluation
├── scripts/              # Training/eval scripts
└── docs/                 # Documentation
```

## Getting Help

- Open an issue on GitHub
- Check OpenLane-V2 docs: https://github.com/OpenDriveLab/OpenLane-V2
- Review TopoNet paper: https://arxiv.org/abs/2309.16784