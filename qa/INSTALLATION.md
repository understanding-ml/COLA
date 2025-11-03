# Installation Guide

This guide provides detailed instructions for installing COLA.

## System Requirements

- Python 3.8 or higher
- pip or conda package manager
- Minimum 4GB RAM recommended

## Installation Methods

### Method 1: pip (Recommended)

The easiest way to install COLA is using pip:

```bash
pip install xai-cola
```

To upgrade to the latest version:

```bash
pip install --upgrade xai-cola
```

### Method 2: Install from Source

If you want the latest development version or need to modify the code:

```bash
# Clone the repository
git clone https://github.com/your-repo/COLA.git
cd COLA

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Method 3: Using Conda

Create a new conda environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate cola
```

## Dependencies

COLA requires the following packages (automatically installed with pip):

- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- lightgbm >= 4.0.0
- joblib >= 1.0.0
- dice-ml >= 0.9.0
- alibi >= 0.9.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

Optional dependencies:
- shap >= 0.40.0 (for feature attributions)
- POT >= 0.8.0 (for optimal transport)
- torch >= 1.9.0 (for PyTorch support)

## Verification

To verify your installation, run:

```python
import xai_cola
print(f"COLA version: {xai_cola.__version__}")
```

## Troubleshooting

### Issue: Import errors after installation

If you encounter import errors, try:
```bash
pip install --force-reinstall xai-cola
```

### Issue: Compatibility with Python version

COLA requires Python 3.8 or higher. Check your Python version:
```bash
python --version
```

### Issue: Missing dependencies

Reinstall with all optional dependencies:
```bash
pip install xai-cola[all]
```

## Development Installation

For development and contributing:

```bash
# Clone repository
git clone https://github.com/your-repo/COLA.git
cd COLA

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install additional development tools
pip install pytest black flake8 mypy
```

## Docker Installation (Coming Soon)

A Docker image will be available soon for easy deployment and reproducibility.

