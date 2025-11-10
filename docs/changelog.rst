=========
Changelog
=========

All notable changes to this project are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

Unreleased
==========

Added
-----

- Comprehensive documentation with Sphinx
- FAQ with common questions and troubleshooting
- Tutorial system with step-by-step guides
- Read the Docs integration
- Installation guide with troubleshooting section
- User guides for all major components
- Complete API reference documentation

Changed
-------

- Restructured documentation for better organization
- Enhanced README with clearer examples
- Improved CONTRIBUTING guidelines with detailed workflows
- Updated data interface to support both pandas and numpy
- Improved error messages throughout the codebase

Fixed
-----

- Fixed column transformer inverse transform handling
- Improved preprocessor wrapper compatibility

Removed
-------

- Removed temporary Q&A files from development process
- Cleaned up redundant documentation files
- Removed deprecated ARES and WachterCF explainers (moved to external)

[0.1.0] - 2024-01-15
====================

Added
-----

**Core Features**

- Initial release of COLA (COunterfactual explanations with Limited Actions)
- Core COLA functionality for refining counterfactual explanations
- ``COLAData`` class for unified data management
- ``Model`` interface supporting multiple ML frameworks

**Counterfactual Explainers**

- DiCE (Diverse Counterfactual Explanations) integration
- DisCount (Distributional Counterfactuals) integration
- Support for external explainers (Alibi, custom implementations)

**Matching Policies**

- Optimal Transport (OT) matcher - globally optimal matching
- Exact Class Transition (ECT) matcher - fast deterministic matching
- Nearest Neighbor (NN) matcher - simple proximity-based
- Soft CEM matcher - probabilistic soft matching

**Feature Attribution**

- PSHAP (Pair-wise SHAP) attributor
- Joint distribution-informed feature importance
- Automatic feature ranking for refinement

**Visualization Tools**

- Binary heatmaps (which features changed)
- Directional heatmaps (increase/decrease patterns)
- Highlighted DataFrames with color-coded changes
- Stacked bar charts comparing CE vs ACE
- Diversity analysis visualizations

**ML Framework Support**

- scikit-learn pipelines and models
- PyTorch neural networks
- TensorFlow 1.x and 2.x / Keras models
- Custom models via base interface

**Data Processing**

- Support for pandas DataFrames
- Support for NumPy arrays
- Automatic feature type inference
- Integration with sklearn preprocessors
- ColumnTransformer support

**Documentation**

- Basic README with usage examples
- API reference documentation
- Example notebooks
- German Credit dataset for testing

[0.0.1] - 2024-01-01
====================

Added
-----

- Initial project structure
- Basic implementation of COLA algorithm
- Proof of concept code

Version History
===============

Summary of releases:

- **0.1.0** (2024-01-15): First public release with full feature set
- **0.0.1** (2024-01-01): Initial development version

Upgrade Guide
=============

From 0.0.x to 0.1.0
-------------------

**Breaking Changes:**

None - 0.1.0 is the first public release.

**New Features:**

If you were using internal development versions:

1. Update import paths:

   .. code-block:: python

       # Old (internal)
       from cola import COLA

       # New (public)
       from xai_cola import COLA

2. Update data interface:

   .. code-block:: python

       # Old
       cola = COLA(factual=X, counterfactual=CF, model=model)

       # New
       data = COLAData(factual_data=X, label_column='target')
       data.add_counterfactuals(CF)
       cola = COLA(data=data, ml_model=model)

Deprecation Notices
===================

Current Version (0.1.0)
-----------------------

No deprecations in current version.

Planned Deprecations
--------------------

None currently planned. We maintain backward compatibility within major versions.

Migration Path
==============

We're committed to smooth upgrades:

1. **Deprecated features** are marked in documentation
2. **Warnings** are issued for deprecated usage
3. **One major version** of overlap before removal
4. **Migration guides** provided for breaking changes

Contributing
============

See :doc:`contributing` for how to contribute to COLA.

See Also
========

- :doc:`installation` - Installation instructions
- :doc:`quickstart` - Quick start guide
- :doc:`faq` - Frequently asked questions
- `GitHub Releases <https://github.com/understanding-ml/COLA/releases>`_ - Download releases
- `PyPI <https://pypi.org/project/xai-cola/>`_ - Python Package Index
