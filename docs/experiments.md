# Experiment Design

## Overview

This document describes the systematic ablation experiments to validate each component's contribution to the final model performance.

## Experiment Matrix

| Exp ID | Description | OLS Target | Key Hypothesis |
|--------|-------------|------------|----------------|
| Exp-1 | TopoNet Baseline | ~39.8 | Reproduce baseline |
| Exp-2 | + Endpoint Detection | ~43.0 | Endpoint modeling improves topology |
| Exp-3 | + Point-Lane Graph | ~45.0 | Point-level reasoning helps |
| Exp-4 | Full Model | ~46.5+ | Combined improvements |
| Exp-5a | Ablation: Endpoint only (no point graph) | ~43.0 | Endpoint is primary driver |
| Exp-5b | Ablation: Point graph only (no endpoint) | ~44.0 | Point reasoning has independent value |
| Exp-5c | Ablation: Different GNN depths | varies | Optimal layer count |
| Exp-5d | Ablation: Different point counts | varies | Optimal sampling density |

## Detailed Protocols

### Exp-1: Baseline Reproduction

**Objective**: Reproduce TopoNet architecture and verify implementation

**Configuration**:
```yaml
model:
  use_endpoint_detector: false
  use_point_lane_graph: false
  dim: 256
  num_lane_queries: 200
  num_te_queries: 100
```

**Expected Results**:
- OLS: 39.8 ± 0.5
- Lane Detection: ~0.50
- TOP_ll: ~0.30

**Validation Criteria**:
- [ ] OLS within 1.0 of reported TopoNet score
- [ ] Lane detection works correctly
- [ ] Topology matrices are non-trivial (not all zeros/ones)

---

### Exp-2: Endpoint Detection Addition

**Objective**: Validate that endpoint detection improves topology reasoning

**Configuration**:
```yaml
model:
  use_endpoint_detector: true
  use_point_lane_graph: false  # Disabled for ablation
  endpoint_hidden_dim: 128
```

**Expected Results**:
- OLS: 43.0 ± 0.5
- Improvement over baseline: +3.2 points
- Δ Lane Detection: minimal
- Δ TOP_ll: significant improvement

**What to Measure**:
1. Endpoint prediction accuracy (L1 distance to GT)
2. Improvement in lane-lane topology only
3. Ablation on endpoint loss weight (0.1, 0.3, 0.5, 1.0)

---

### Exp-3: Point-Lane Graph Addition

**Objective**: Validate that hierarchical point-level reasoning helps

**Configuration**:
```yaml
model:
  use_endpoint_detector: false  # Disabled for ablation
  use_point_lane_graph: true
  num_points: 32
  gnn_layers: 3
```

**Expected Results**:
- OLS: 45.0 ± 0.5
- Improvement over baseline: +5.2 points
- Note: This is higher than Exp-2, indicating point reasoning may be stronger

**What to Measure**:
1. Point feature variance (should be high)
2. Per-layer GNN contribution (disable each layer)
3. Sensitivity to number of points (16, 32, 64)

---

### Exp-4: Full Model

**Objective**: Validate combined endpoint + point-lane approach

**Configuration**:
```yaml
model:
  use_endpoint_detector: true
  use_point_lane_graph: true
  endpoint_hidden_dim: 128
  num_points: 32
  gnn_layers: 3
  lambda_endpoint: 0.5
```

**Expected Results**:
- OLS: 46.5+ (target)
- Should exceed both Exp-2 and Exp-3

**Validation**:
- Compare to TopoPoint (~48.8) and TopoLogic (~44.1)
- Analyze where improvements come from (detection vs topology)

---

### Exp-5: Ablation Studies

#### Exp-5a: Endpoint Only (no point graph)
```
OLS Target: ~43.0
Hypothesis: Endpoint detection is primary driver
```

#### Exp-5b: Point Graph Only (no endpoint)
```
OLS Target: ~44.0
Hypothesis: Point-level reasoning has independent value
```

#### Exp-5c: GNN Depth Sweep
```
gnn_layers: [1, 2, 3, 4, 6]
Expected: Optimal around 3
```

#### Exp-5d: Point Count Sweep
```
num_points: [8, 16, 32, 64]
Expected: Saturation around 32
```

## Evaluation Protocol

### Training Configuration

```yaml
training:
  batch_size: 4
  lr: 2e-4
  epochs: 30
  warmup: 2
  weight_decay: 0.01
  
  # Loss weights
  lambda_detection: 1.0
  lambda_topology: 1.0
  lambda_endpoint: 0.5
```

### Data Configuration

- **Dataset**: OpenLane-V2 subset_A
- **Train/Val Split**: 700/150 scene segments
- **Sampling**: 2 Hz
- **Annotation Range**: ~50m longitudinal, ~25m lateral

### Metrics to Log

1. **Primary**: OLS (final score)
2. **Detection**: Lane AP, TE AP
3. **Topology**: TOP_ll, TOP_lte
4. **Endpoint**: Endpoint L1 distance (oracle test)
5. **Auxiliary**: Per-component loss values

### Logging Schedule

```
Training:
- Every 50 batches: Component losses
- Every epoch: Train/Val loss
- Every 5 epochs: Full evaluation (OLS)

Tensorboard Logs:
- scalars: loss, learning rate, metrics
- histograms: gradient norms, activation distributions
- plots: PR curves for topology
```

## Statistical Rigor

### Minimum Runs
- 3 random seed runs per experiment
- Report mean ± std

### Significance Testing
- Use paired t-test for ablation comparisons
- Report p-values for key claims

###oracle Testing
- Run with oracle detection to isolate reasoning improvements
- This reveals true topology reasoning quality independent of detection

## Expected Outcome Summary

```
Exp-1 Baseline:     OLS = 39.8  (reference)
Exp-2 Endpoint:     OLS = 43.0  (+3.2)
Exp-3 PointGraph:   OLS = 45.0  (+5.2)
Exp-4 Full:         OLS = 46.5+ (+6.7)

Key Insights:
1. Point-level reasoning may contribute more than endpoint alone
2. Combined model should exceed both individual improvements
3. GNN depth and point count have diminishing returns beyond certain values
```