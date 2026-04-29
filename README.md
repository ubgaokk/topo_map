# TopoMap: Endpoint-Aware Hierarchical Point-Lane Graph for Driving Scene Topology Reasoning

A research-focused implementation of **endpoint-aware topology reasoning** with hierarchical point-lane graphs, built on top of OpenLane-V2 benchmark.

## 🔑 Core Innovation

This project explores **endpoint-aware + hierarchical point-lane graph** architecture for lane topology reasoning, inspired by TopoPoint's findings that:

1. **Endpoint instability** is the primary bottleneck in topology reasoning
2. **Point-level graph structure** provides finer granularity than lane-level graphs alone
3. **Hierarchical aggregation** (point → lane → topology) captures multi-scale spatial relationships

## 📁 Project Structure

```
topo_map/
├── configs/                 # Experiment configurations
│   ├── default_config.py    # Default hyperparameters
│   └── experiments/         # Ablation experiment configs
├── dataset/                 # OpenLane-V2 data loader
├── model/                   # Model architecture
│   ├── toponet_endpoint.py  # Complete model
│   ├── endpoint_detector.py # Endpoint detection (core innovation)
│   ├── point_lane_graph.py  # Hierarchical graph (core innovation)
│   └── ...
├── loss/                    # Combined loss functions
├── evaluation/              # OLS/TOP metrics
├── scripts/                 # Training & evaluation scripts
├── tests/                   # Unit tests (1000+ lines)
├── docs/                    # Architecture & experiment docs
└── CHANGELOG.md             # Version history
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n topo_map python=3.9
conda activate topo_map

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```

### 2. Verify Installation

```bash
# Run quick sanity check (no GPU required for this)
./quick_sanity_check.sh

# Or run all unit tests
./run_tests.sh
```

Expected output (without PyTorch):
```
NOTE: Install PyTorch first...
```

With PyTorch installed:
```
==========================================
TopoMap Training Sanity Check
==========================================
[1/5] Testing imports... All imports successful!
[2/5] Testing model creation... Model created with XXX parameters
[3/5] Testing forward pass... Forward pass successful!
[4/5] Testing loss computation... Loss computation successful!
[5/5] Testing backward pass... Backward pass successful!
==========================================
All sanity checks passed!
```

### 3. Data Preparation

```bash
# Download OpenLane-V2 from https://github.com/OpenDriveLab/OpenLane-V2
# subset_A (~151GB) recommended

# Preprocess data
python scripts/preprocess_data.py \
    --root /path/to/openlane_v2/subset_A \
    --output /path/to/preprocessed \
    --split all
```

### 4. Training

```bash
# Update configs/experiments/exp_4_full.yaml with your data path:
# data:
#   data_root: /path/to/preprocessed

# Train baseline first (faster, ~2 hours)
python scripts/train.py --config configs/experiments/exp_1_baseline.yaml

# Train full model (~6-8 hours on A100)
python scripts/train.py --config configs/experiments/exp_4_full.yaml

# Resume from checkpoint
python scripts/train.py \
    --config configs/experiments/exp_4_full.yaml \
    --resume outputs/exp_4_full_*/checkpoint_epoch_10.pth
```

### 5. Evaluation

```bash
python scripts/eval.py \
    --checkpoint outputs/exp_4_full_*/best_model.pth \
    --data /path/to/preprocessed \
    --split val
```

## 📊 Expected Results (OLS on subset_A val)

| Experiment | Model | Expected OLS |
|------------|-------|--------------|
| Exp-1 | TopoNet Baseline | ~39.8 |
| Exp-2 | + Endpoint Detection | ~43.0 |
| Exp-3 | + Point-Lane Graph | ~45.0 |
| Exp-4 | Full Model | ~46.5+ |

## 🧪 Testing

```bash
# Run all unit tests
python -m pytest tests/ -v

# Or use the provided script
./run_tests.sh

# Expected output:
# test_dataset.py::TestOpenLaneV2Dataset::test_dataset_loads PASSED
# test_model.py::TestEndpointDetector::test_output_shapes PASSED
# test_integration.py::TestEndToEndForward::test_full_forward_shapes PASSED
# ...
```

## 🔬 Architecture Overview

```
Input: Multi-view Images
         ↓
┌────────────────────────┐
│   BEV Feature Encoder  │  (ResNet-50 + FPN + View Transformer)
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Query Decoders       │  (Lane + Traffic Element queries)
└────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  ① Endpoint Detector                   │  ← Core Innovation #1
│     - Predicts lane start/end points   │
│     - Generates endpoint tokens        │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  ② Point Sampler                       │  ← Core Innovation #2
│     - Samples 32 dense points per lane │
│     - Enables point-level reasoning    │
└────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  ③ Hierarchical Point-Lane Graph       │  ← Core Innovation #3
│     - Point → Lane aggregation         │
│     - Lane → Lane topology edges       │
│     - Endpoint-aware attention         │
└────────────────────────────────────────┘
         ↓
┌────────────────────────┐
│   Topology Heads       │  (Lane-Lane + Lane-TE prediction)
└────────────────────────┘
         ↓
    OLS / TOP Metrics
```

## 📝 Key Configuration

```yaml
# Key hyperparameters
model:
  dim: 256                    # Feature dimension
  num_lane_queries: 200       # Number of lane queries
  num_te_queries: 100         # Number of traffic element queries
  num_points: 32              # Points sampled per lane
  bev_h: 100                  # BEV grid height
  bev_w: 200                  # BEV grid width
  use_endpoint_detector: true # Enable endpoint detection
  use_point_lane_graph: true  # Enable hierarchical graph

training:
  batch_size: 4
  lr: 2.0e-4
  epochs: 30
  lambda_detection: 1.0       # Detection loss weight
  lambda_topology: 1.0        # Topology loss weight
  lambda_endpoint: 0.5        # Endpoint loss weight
```

## 📚 References

- [TopoNet: Graph-based Topology Reasoning for Driving Scenes](https://arxiv.org/abs/2309.16784)
- [TopoPoint: Endpoint-Aware Topology Reasoning with Point-level Graph](https://arxiv.org/abs/xxx)
- [OpenLane-V2 Dataset](https://github.com/OpenDriveLab/OpenLane-V2)

## 📄 License

MIT License

## 🤝 Acknowledgments

This project builds upon:
- [TopoNet](https://github.com/OpenDriveLab/TopoNet) by OpenDriveLab
- [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) benchmark
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) view transformer