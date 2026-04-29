# Architecture Documentation

## Overview

TopoMap implements an **endpoint-aware hierarchical point-lane graph** for driving scene topology reasoning, building upon the OpenLane-V2 benchmark and TopoNet architecture.

## Core Innovations

### 1. Endpoint Detection Module

Inspired by TopoPoint's finding that endpoint instability is the primary bottleneck in topology reasoning, this module explicitly predicts:

- **Start Point**: The beginning of each lane centerline
- **End Point**: The termination point of each lane centerline
- **Endpoint Tokens**: Semantic embeddings that encode endpoint geometry

```
Input: Lane queries [B, N_lane, dim] + Lane geometry [B, N_lane, 11, 3]
           ↓
┌─────────────────────────────────────┐
│  EndpointDetector                   │
│  - Shared feature encoder           │
│  - Start/End detection heads        │
│  - Endpoint token generator         │
└─────────────────────────────────────┘
           ↓
Output: start_points [B, N, 3], end_points [B, N, 3], endpoint_tokens [B, N, dim]
```

**Why it matters**: The topology of lane connections is fundamentally determined by which lane's end connects to which lane's start. By explicitly modeling endpoints, the model can better reason about successor/predecessor relationships.

### 2. Point Sampler

Samples 32 dense points along each lane centerline to enable point-level reasoning:

```
Original: 11 points per lane
           ↓
        Interpolation
           ↓
Sampled: 32 points per lane (configurable via num_points)
```

**Why it matters**: Point-level features capture local geometry details that are lost when aggregating directly to lane-level representations.

### 3. Hierarchical Point-Lane Graph

Two-level message passing:

**Level 1: Point → Lane**
- Each lane's points aggregate to update lane representation
- Uses multi-head attention with point-to-lane cross-attention

**Level 2: Lane → Lane**
- Lane nodes exchange information based on geometric adjacency
- Adjacency is determined by endpoint proximity (end of one lane near start of another)
- Uses Graph Neural Network with edge-type awareness (successor/predecessor/self-loop)

```
                    ┌──────────────────────┐
                    │  Point-Lane Graph    │
                    └──────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ↓                       ↓                       ↓
┌───────────────┐     ┌─────────────────┐     ┌───────────────────┐
│ Point Level   │     │ Lane Level GNN  │     │ Cross-Level Fusion│
│ Point→Lane    │     │ Lane→Lane       │     │ Gated combination │
│ Attention     │     │ (3 types edges) │     │ of point & lane   │
└───────────────┘     └─────────────────┘     └───────────────────┘
```

**Key Design: Endpoint-Aware Edge Weights**

Edge weights between lanes i and j are computed as:

```
weight(i→j) = exp(-dist(end_i, start_j) / threshold) 
              × (0.5 + 0.5 × cosine_similarity(feat_i, feat_j))
```

This combines:
- **Geometry prior**: Endpoints should be close
- **Feature similarity**: Similar lanes may connect

### 4. Topology Prediction Heads

After graph aggregation, predicts:

- **Lane-Lane Topology**: Pairwise connection scores for all lane pairs
- **Lane-TE Topology**: Association scores between lanes and traffic elements

Uses learned pairwise scoring with MLPs over concatenated lane/TE embeddings.

## Complete Architecture

```
Input: Multi-view Camera Images
            │
            ↓
┌───────────────────────────────────────────┐
│           BEV Encoder                      │
│  ResNet-50 + FPN + View Transformer        │
└───────────────────────────────────────────┘
            │
            ↓
┌───────────────────────────────────────────┐
│           Query Decoders                   │
│  Lane Queries (200) + TE Queries (100)     │
└───────────────────────────────────────────┘
            │
            ├────────────────────────────────────────┐
            ↓                                        ↓
┌─────────────────────┐                ┌─────────────────────┐
│  Endpoint Detector  │                │   TE Decoder        │
│  (Start/End Points) │                │   (Traffic Elements)│
└─────────────────────┘                └─────────────────────┘
            │                                        │
            ↓                                        ↓
┌─────────────────────┐                ┌─────────────────────┐
│  Point Sampler      │                │   TE Queries        │
│  (32 pts per lane)  │                │   [B, 100, dim]     │
└─────────────────────┘                └─────────────────────┘
            │                                        │
            ├────────────────────────────────────────┤
            ↓                                        ↓
┌─────────────────────────────────────────────────────┐
│           Hierarchical Point-Lane Graph             │
│  1. Point → Lane Attention                         │
│  2. Endpoint-Aware Lane → Lane GNN                 │
│  3. Cross-Level Fusion Gate                        │
└─────────────────────────────────────────────────────┘
            │
            ↓
┌─────────────────────────────────────────────────────┐
│           Topology Heads                            │
│  Lane-Lane Matrix [B, 200, 200]                     │
│  Lane-TE Matrix   [B, 200, 100]                     │
└─────────────────────────────────────────────────────┘
            │
            ↓
      OLS / TOP Metrics
```

## Computational Complexity

| Component | Complexity | Parameters |
|-----------|-----------|------------|
| BEV Encoder | O(B × H × W × C) | ~25M |
| Lane Decoder (6 layers) | O(B × N_q² × D) | ~15M |
| TE Decoder (4 layers) | O(B × N_te² × D) | ~5M |
| Endpoint Detector | O(B × N_lane × D) | ~0.5M |
| Point-Lane Graph (3 layers) | O(B × N_lane² × D) | ~2M |
| Topology Heads | O(B × N² × D) | ~1M |

**Total**: ~50M parameters (comparable to TopoNet)

## Comparison to Prior Work

| Method | OLS | Key Innovation |
|--------|-----|----------------|
| TopoNet | 39.8 | Heterogeneous GNN |
| TopoMLP | 44.1 | Strong detector + MLP |
| TopoLogic | 44.1 | Geometric endpoint distance |
| TopoPoint | 48.8 | Point-level graph + endpoint |
| **TopoMap** | ~46.5+ | Hierarchical point-lane + endpoint |

TopoMap differs from TopoPoint by:
1. **Hierarchical aggregation**: Explicit point→lane→topology levels
2. **Gated fusion**: Learnable combination of point and lane features
3. **Endpoint-aware adjacency**: Geometry priors for graph structure