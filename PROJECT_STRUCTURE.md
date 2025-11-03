# COLA Project Structure

This document describes the standardized structure of the COLA project.

## Directory Layout

```
COLA/
├── xai_cola/                  # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── version.py             # Version information
│   ├── counterfactual_limited_actions.py  # Main COLA class
│   ├── data_interface/        # Data handling interfaces
│   │   ├── base_data.py
│   │   ├── pandas_data.py
│   │   ├── numpy_data.py
│   │   └── ares_dataset_info.py
│   ├── ml_model_interface/    # Model interfaces
│   │   ├── base_model.py
│   │   ├── model.py
│   │   ├── sklearn_model.py
│   │   └── pytorch_model.py
│   ├── cola_policy/           # COLA policy implementations
│   │   ├── data_composer/     # Counterfactual composition
│   │   ├── feature_attributor/ # Feature attribution
│   │   ├── matching/          # Matching strategies
│   │   │   ├── base_matcher.py
│   │   │   ├── ect_matcher.py  # Exact matching
│   │   │   ├── nn_matcher.py   # Nearest neighbor
│   │   │   ├── ot_matcher.py   # Optimal transport
│   │   │   └── cem_matcher.py  # Coarsened exact matching
│   ├── plot/                  # Visualization utilities
│   │   ├── heatmap.py
│   │   └── highlight_dataframe.py
│   └── utils/                  # Utility functions
│       └── logger_config.py
│
├── counterfactual_explainer/   # Counterfactual explanation algorithms
│   ├── __init__.py
│   ├── base_explainer.py      # Base class
│   ├── dice.py                # DiCE implementation
│   ├── discount.py             # DisCount implementation
│   ├── alibi_cfi.py            # Alibi-CFI implementation
│   ├── ares.py                 # ARecourseS implementation
│   ├── knn.py                  # KNN implementation
│   └── auxiliary.py            # Auxiliary functions
│
├── datasets/                   # Built-in datasets
│   ├── german_credit.py
│   ├── compas.py
│   ├── heloc.py
│   ├── hotel_bookings.py
│   ├── data_loader.py
│   └── rawdata/               # Raw data files
│
├── tests/                      # Test suite
│   ├── conftest.py            # Pytest configuration
│   ├── test_data_interface.py
│   └── test_model_interface.py
│
├── docs/                       # Documentation
│   ├── images/                # Figures and images
│
├── scripts/                    # Build and release scripts
│   ├── build_release.sh
│   └── publish_to_pypi.sh
│
├── setup.py                    # Package setup (legacy)
├── pyproject.toml             # Modern Python packaging
├── setup.cfg                  # Configuration
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── MANIFEST.in                # Manifest for distribution
├── Makefile                   # Build automation
│
├── README.rst                  # Original README
├── README.md                   # Main README
├── INSTALLATION.md             # Installation guide
├── QUICKSTART.md               # Quick start guide
├── API_REFERENCE.md            # API documentation
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
├── RELEASE_GUIDE.md             # Release instructions
└── PROJECT_STRUCTURE.md         # This file
```

## Key Files Explained

### Package Configuration

- **setup.py**: Legacy package configuration for pip install
- **pyproject.toml**: Modern Python packaging configuration
- **requirements.txt**: Python dependencies for pip
- **environment.yml**: Conda environment definition
- **MANIFEST.in**: Specifies which files to include in distribution
- **Makefile**: Build automation commands

### Documentation

- **README.md**: Main project documentation
- **README.rst**: Original reStructuredText README
- **INSTALLATION.md**: Detailed installation instructions
- **QUICKSTART.md**: Quick start guide
- **API_REFERENCE.md**: Complete API documentation
- **CONTRIBUTING.md**: Contribution guidelines
- **CHANGELOG.md**: Version history
- **RELEASE_GUIDE.md**: Instructions for releasing new versions

### Testing and CI/CD

- **tests/**: Test suite directory
- **.github/workflows/ci.yml**: GitHub Actions CI
- **.circleci/config.yml**: CircleCI configuration (alternative)

### Build Scripts

- **scripts/build_release.sh**: Build distribution packages
- **scripts/publish_to_pypi.sh**: Publish to PyPI
- **Makefile**: Build automation

## Module Organization

### xai_cola (Main Package)

Core functionality for counterfactual refinement with limited actions.

### counterfactual_explainer

Various counterfactual explanation algorithms that can generate initial counterfactuals.

### datasets

Built-in datasets for testing and demonstration.

### tests

Unit and integration tests for the package.

## Naming Conventions

- **Package names**: lowercase_with_underscores (e.g., `xai_cola`)
- **Class names**: PascalCase (e.g., `CounterFactualExplainer`)
- **Function names**: lowercase_with_underscores (e.g., `generate_counterfactuals`)
- **Constants**: UPPERCASE_WITH_UNDERSCORES (e.g., `MAX_ITERATIONS`)

## Import Organization

Modules are organized with clear separation of concerns:

1. **Data handling**: `xai_cola.data_interface`
2. **Model interfaces**: `xai_cola.ml_model_interface`
3. **COLA policies**: `xai_cola.cola_policy`
4. **Visualization**: `xai_cola.plot`
5. **Counterfactual generation**: `counterfactual_explainer`

## Adding New Features

When adding new features:

1. Place in appropriate subdirectory
2. Follow naming conventions
3. Add docstrings
4. Add tests in `tests/`
5. Update documentation
6. Update `CHANGELOG.md`

## Distribution

The package can be distributed via:

1. **PyPI**: Main Python package index
   - Install: `pip install xai-cola`

2. **GitHub**: Source code repository
   - Clone: `git clone https://github.com/your-repo/COLA.git`

3. **Conda**: Conda package manager
   - Environment: `conda env create -f environment.yml`

