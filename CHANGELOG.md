# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-29

### Added
- **Presentation** (`presentation/TopoMap_Presentation.pptx`):
  - 11 slides covering motivation, contributions, architecture, comparisons, and conclusions
  - Two-column layouts for architecture comparison
  - Created using python-pptx

- **Paper Draft** (`paper/`):
  - `topo_map_paper.tex` - IEEE format LaTeX paper draft
  - `TopoMap_Paper.pdf` - Generated PDF using ReportLab
  - `generate_paper_pdf.py` - Python script to generate PDF
  - Contents: Abstract, Introduction, Related Work, Method, Experiments, Conclusion, References

- **Documentation Updates**:
  - Presentation covers model advantages and improvements
  - Paper draft includes full technical description with equations and tables

## [0.3.0] - 2026-04-29

### Added
- **Edge case tests** (`tests/test_edge_cases.py`) - Comprehensive boundary condition testing
- **CI/CD workflows** (`.github/workflows/`) - Multi-version testing and linting
- **Mixed Precision Training (AMP)** - `--amp` flag support
- **torch.compile() Support** - `--compile` flag support

## [0.2.0] - 2026-04-29

### Added
- Complete training pipeline with train.py
- Unit tests for all modules
- Training sanity check script
- YAML-based experiment configurations
- Documentation (architecture, experiments, quickstart)

## [0.1.0] - 2026-04-29

### Added
- Core model components (EndpointDetector, PointSampler, PointLaneGraph, TopologyHead)
- OpenLane-V2 dataset loader
- Combined loss function
- OLS/TOP metrics evaluation
- Preprocessing script template