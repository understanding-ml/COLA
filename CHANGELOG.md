# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation with Sphinx
- FAQ with common questions and troubleshooting
- Tutorial system with step-by-step guides
- Read the Docs integration
- Installation guide with troubleshooting section

### Changed
- Restructured documentation for better organization
- Enhanced README with clearer examples
- Improved CONTRIBUTING guidelines with detailed workflows

### Removed
- Removed temporary Q&A files from development process
- Cleaned up redundant documentation files

## [0.1.0] - 2024-01-15

### Added
- Initial release of COLA (COunterfactual explanations with Limited Actions)
- Core COLA functionality for refining counterfactual explanations
- Support for multiple counterfactual explainers:
  - DiCE (Diverse Counterfactual Explanations)
  - DisCount (Distributional Counterfactuals)
  - Alibi-CFI (Counterfactual Instances)
  - ARecourseS (Actionable Recourse)
  - WachterCF (Gradient-based optimization)
  - Growing Sphere
  - K-Nearest Neighbors based explainer
- Support for multiple matching policies:
  - Optimal Transport (OT)
  - Exact Matching (ECT)
  - Nearest Neighbor (NN)
  - Coarsened Exact Matching (CEM)
- Feature attribution methods:
  - PSHAP (Probability-weighted Shapley values)
  - Joint distribution-informed feature importance
- Visualization tools:
  - Binary heatmaps (which features changed)
  - Directional heatmaps (increase/decrease patterns)
  - Highlighted DataFrames with color-coded changes
  - Stacked bar charts for modification comparison
  - Diversity analysis showing minimal feature combinations
- Built-in datasets:
  - German Credit dataset
  - COMPAS recidivism dataset
  - HELOC credit risk dataset
  - Hotel Bookings dataset
- Data interface:
  - COLAData class for unified data handling
  - Support for Pandas DataFrames and NumPy arrays
  - Automatic preprocessing with OHE, z-score, min-max normalization
  - Inverse transformation support
- Model interface:
  - Support for scikit-learn models
  - Support for PyTorch models
  - Support for TensorFlow 1.x and 2.x models
  - Factory pattern for automatic model adapter selection
- Feature selection:
  - `features_to_vary` parameter to specify modifiable features
  - `features_to_keep` parameter for immutable features
  - Action limiting with `limited_actions` parameter
- Documentation:
  - API Reference with complete function documentation
  - Quick Start guide
  - Architecture overview (Chinese)
  - Project structure documentation

### Infrastructure
- Package structure following best practices
- Setup with pyproject.toml and setup.py
- Test framework with pytest
- CI/CD configuration files
- MIT License

[Unreleased]: https://github.com/your-repo/COLA/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-repo/COLA/releases/tag/v0.1.0

