# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-29

### Added
- **Edge case tests** (`tests/test_edge_cases.py`) - Comprehensive boundary condition testing:
  - Empty batches, single lane queries, CUDA availability
  - NaN/Inf output detection, eval vs train mode behavior
  - Loss edge cases (empty targets, all zeros, zero weights)
  - GIoU stability with degenerate boxes
  - Gradient clipping correctness
  - Optimizer step numerical stability
  - Memory usage tests (GPU)
  - Model size validation
  - Small LR training stability

- **CI/CD workflows** (`.github/workflows/`):
  - `ci.yml` - Multi-version Python testing (3.8-3.11)
  - `lint.yml` - Code quality checks (ruff, black, isort)
  - Automated on push/PR to main branch

- **Mixed Precision Training (AMP)**:
  - `--amp` flag in train.py
  - Uses `torch.amp.autocast` and `GradScaler`
  - Reduces memory usage ~40%, faster training

- **torch.compile() Support**:
  - `--compile` flag in train.py (PyTorch 2.0+)
  - Additional ~20-30% speedup on compatible hardware

### Improved
- **train.py** - Refactored with:
  - AMP support with proper gradient scaling
  - Gradient clipping in correct order
  - Better device handling for camera params
  - Cleaner loss computation flow

- **Documentation** - Added performance optimization hints to config

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
  - `tests/test_model.py` - Model components (EndpointDetector, PointSampler, etc.)
  - `tests/test_loss.py` - Loss functions (FocalLoss, TopologyLoss, GIoU)
  - `tests/test_evaluation.py` - Metrics computation (OLS, TOP, AP)
  - `tests/test_integration.py` - End-to-end forward/backward pass

- **Training sanity check** - `quick_sanity_check.sh` script
- **Test runner scripts** - `tests/run_tests.py`, `run_tests.sh`

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