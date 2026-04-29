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
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default_config.py          # Default hyperparameters
│   └── experiments/
│       ├── exp_1_baseline.yaml    # TopoNet baseline
│       ├── exp_2_endpoint.yaml    # + Endpoint detection
│       ├── exp_3_point_lane.yaml  # + Point-lane graph
│       └── exp_4_full.yaml        # Complete model
├── dataset/
│   ├── __init__.py
│   ├── openlane_v2.py             # OpenLane-V2 dataset loader
│   └── preprocessing.py           # Data preprocessing tools
├── model/
│   ├── __init__.py
│   ├── bever_encoder.py           # BEV feature extraction
│   ├── endpoint_detector.py       # Endpoint detection module
│   ├── point_lane_graph.py        # Hierarchical point-lane graph
│   ├── topology_head.py           # Topology prediction heads
│   └── toponet_endpoint.py        # Complete model
├── loss/
│   ├── __init__.py
│   └── topo_loss.py               # Combined loss function
├── evaluation/
│   ├── __init__.py
│   └── openlane_v2_eval.py        # Official OLS/TOP metrics
├── scripts/
│   ├── train.py                   # Main training script
│   ├── eval.py                    # Evaluation script
│   └── preprocess_data.py         # Data preprocessing runner
└── docs/
    ├── architecture.md            # Detailed architecture docs
    ├── experiments.md             # Experiment design details
    └── quickstart.md              # Quick start guide
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n topo_map python=3.9
conda activate topo_map

# Install dependencies
pip install -r requirements.txt

# Install this package
pip install -e .
```

### 2. Data Preparation

```bash
# Download OpenLane-V2 subset_A from official source
# https://github.com/OpenDriveLab/OpenLane-V2

# Preprocess data
python scripts/preprocess_data.py \
    --root /path/to/openlane_v2/subset_A \
    --output /path/to/preprocessed \
    --split train
```

### 3. Training

```bash
# Baseline (TopoNet-style)
python scripts/train.py --config configs/experiments/exp_1_baseline.yaml

# With endpoint detection
python scripts/train.py --config configs/experiments/exp_2_endpoint.yaml

# Full model (endpoint + point-lane graph)
python scripts/train.py --config configs/experiments/exp_4_full.yaml
```

### 4. Evaluation

```bash
python scripts/eval.py \
    --checkpoint /path/to/checkpoint.pth \
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