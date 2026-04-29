#!/bin/bash
# Quick training sanity check (without actual data)
# Usage: ./quick_sanity_check.sh

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "TopoMap Training Sanity Check"
echo "=========================================="

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# Check if torch is available
if ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch not installed. Run:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    echo "Skipping forward/backward pass tests..."
    
    # Still test imports if possible
    echo "[1/5] Testing static imports..."
    if $PYTHON -c "import sys; sys.path.insert(0, '.'); from pathlib import Path; import pickle" 2>/dev/null; then
        echo "  Basic imports OK"
    fi
    
    echo "[2/5] Skipping model creation (PyTorch required)"
    echo "[3/5] Skipping forward pass (PyTorch required)"
    echo "[4/5] Skipping loss computation (PyTorch required)"
    echo "[5/5] Skipping backward pass (PyTorch required)"
    
    echo ""
    echo "NOTE: Install PyTorch first, then run this script again."
    echo "=========================================="
    exit 0
fi

# Test 1: Import all modules
echo "[1/5] Testing imports..."
$PYTHON -c "
import topo_map
from model.toponet_endpoint import EndpointAwareTopologyNet
from model.endpoint_detector import EndpointDetector, PointSampler
from model.point_lane_graph import PointLaneGraph
from model.topology_head import TopologyHead
from loss.topo_loss import TopologyLoss
from evaluation.openlane_v2_eval import compute_ols_metrics
print('All imports successful!')
"

# Test 2: Create model
echo "[2/5] Testing model creation..."
$PYTHON -c "
import torch
from model.toponet_endpoint import EndpointAwareTopologyNet

config = {
    'dim': 64,
    'num_lane_queries': 10,
    'num_te_queries': 5,
    'bev_h': 25,
    'bev_w': 50,
    'use_endpoint_detector': True,
    'use_point_lane_graph': True,
}

model = EndpointAwareTopologyNet(config)
print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
"

# Test 3: Forward pass
echo "[3/5] Testing forward pass..."
$PYTHON -c "
import torch
from model.toponet_endpoint import EndpointAwareTopologyNet

config = {
    'dim': 64,
    'num_lane_queries': 10,
    'num_te_queries': 5,
    'bev_h': 25,
    'bev_w': 50,
    'use_endpoint_detector': True,
    'use_point_lane_graph': True,
}

model = EndpointAwareTopologyNet(config)
multi_view = [torch.randn(1, 256, 32, 32) for _ in range(6)]

with torch.no_grad():
    predictions = model(multi_view)

print(f'Forward pass successful!')
print(f'  lane_geometry shape: {predictions[\"lane_geometry\"].shape}')
print(f'  topology_lclc shape: {predictions[\"topology_lclc\"].shape}')
print(f'  topology_lcte shape: {predictions[\"topology_lcte\"].shape}')
"

# Test 4: Loss computation
echo "[4/5] Testing loss computation..."
$PYTHON -c "
import torch
from model.toponet_endpoint import EndpointAwareTopologyNet
from loss.topo_loss import TopologyLoss

config = {
    'dim': 64,
    'num_lane_queries': 10,
    'num_te_queries': 5,
    'bev_h': 25,
    'bev_w': 50,
    'use_endpoint_detector': True,
    'use_point_lane_graph': True,
}

model = EndpointAwareTopologyNet(config)
loss_fn = TopologyLoss(lambda_detection=1.0, lambda_topology=1.0, lambda_endpoint=0.5)

multi_view = [torch.randn(1, 256, 32, 32) for _ in range(6)]
predictions = model(multi_view, return_aux=True)

targets = {
    'lane_gt': {
        'geometry': torch.randn(1, 10, 11, 3),
        'start_points': torch.randn(1, 10, 2),
        'end_points': torch.randn(1, 10, 2),
    },
    'topology_lclc': torch.randint(0, 2, (1, 10, 10)).float(),
    'topology_lcte': torch.randint(0, 2, (1, 10, 5)).float(),
}

losses, total = loss_fn(predictions, targets)
print(f'Loss computation successful!')
print(f'  Total loss: {total.item():.4f}')
print(f'  Components: {list(losses.keys())}')
"

# Test 5: Backward pass
echo "[5/5] Testing backward pass..."
$PYTHON -c "
import torch
from model.toponet_endpoint import EndpointAwareTopologyNet
from loss.topo_loss import TopologyLoss

config = {
    'dim': 64,
    'num_lane_queries': 10,
    'num_te_queries': 5,
    'bev_h': 25,
    'bev_w': 50,
    'use_endpoint_detector': True,
    'use_point_lane_graph': True,
}

model = EndpointAwareTopologyNet(config)
loss_fn = TopologyLoss(lambda_detection=1.0, lambda_topology=1.0, lambda_endpoint=0.5)

multi_view = [torch.randn(1, 256, 32, 32) for _ in range(6)]
predictions = model(multi_view, return_aux=True)

targets = {
    'lane_gt': {'geometry': torch.randn(1, 10, 11, 3)},
    'topology_lclc': torch.randint(0, 2, (1, 10, 10)).float(),
    'topology_lcte': torch.randint(0, 2, (1, 10, 5)).float(),
}

losses, total = loss_fn(predictions, targets)
total.backward()

# Check gradient
has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f'Backward pass successful!')
print(f'  Has gradients: {has_grad}')
"

echo ""
echo "=========================================="
echo "All sanity checks passed!"
echo "Model is ready for training."
echo "=========================================="