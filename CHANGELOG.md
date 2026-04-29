# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-29

### Added
- **Complete training pipeline** - Full train.py script with:
  - Config-based model building
  - Mixed batch processing
  - TensorBoard logging
  - Learning rate scheduling (CosineAnnealing)
  - Gradient clipping
  - Model checkpointing (best + periodic)
  - Validation loop

- **Unit tests** - Comprehensive test suite:
  - `tests/test_dataset.py` - OpenLane-V2 dataset loading
  - `tests/test_model.py` - Model components (EndpointDetector, PointSampler, PointLaneGraph, etc.)
  - `tests/test_loss.py` - Loss functions (FocalLoss, TopologyLoss, GIoU)
  - `tests/test_evaluation.py` - Metrics computation (OLS, TOP, AP)
  - `tests/test_integration.py` - End-to-end forward/backward pass

- **Training sanity check** - `quick_sanity_check.sh` script to verify:
  - Module imports
  - Model creation
  - Forward pass
  - Loss computation
  - Backward pass (gradient flow)

- **Test runner scripts**:
  - `tests/run_tests.py` - Python unittest runner
  - `run_tests.sh` - Bash wrapper for pytest or unittest

### Improved
- **Code structure** - Clean separation between:
  - `topo_map/` - Package root with public API
  - `model/` - Model components
  - `dataset/` - Data loading
  - `loss/` - Loss functions
  - `evaluation/` - Metrics
  - `scripts/` - Training/evaluation scripts
  - `tests/` - Unit tests

- **Configuration** - YAML-based experiment configs for:
  - Baseline (TopoNet-style)
  - Endpoint detection only
  - Point-lane graph only
  - Full model

- **Documentation**:
  - `docs/architecture.md` - Detailed architecture
  - `docs/experiments.md` - Experiment design
  - `docs/quickstart.md` - Getting started guide

## [0.1.0] - 2026-04-29

### Added
- Initial project structure
- Core model components:
  - `EndpointDetector` - Lane start/end point detection
  - `PointSampler` - Dense point sampling along lanes
  - `PointLaneGraph` - Hierarchical point-lane message passing
  - `TopologyHead` - Lane-lane and lane-TE topology prediction
  - `ViewTransformer` - BEV feature extraction
- OpenLane-V2 dataset loader
- Combined loss function (detection + topology + endpoint)
- OLS/TOP metrics evaluation
- Preprocessing script template