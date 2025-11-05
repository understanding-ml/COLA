# Installation Guide

This guide provides detailed instructions for installing COLA (COunterfactual explanations with Limited Actions).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [1. Using pip (Recommended)](#1-using-pip-recommended)
  - [2. Using conda](#2-using-conda)
  - [3. From source](#3-from-source)
- [Verifying Installation](#verifying-installation)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## Prerequisites

COLA requires:
- Python 3.7 or higher
- pip or conda package manager

## Installation Methods

### 1. Using pip (Recommended)

The simplest way to install COLA is via pip:

```bash
pip install xai-cola
```

To install with all optional dependencies:

```bash
pip install xai-cola[all]
```

To install specific optional dependencies:

```bash
# For PyTorch support
pip install xai-cola[torch]

# For TensorFlow support
pip install xai-cola[tensorflow]

# For development tools
pip install xai-cola[dev]
```

### 2. Using conda

If you prefer using conda, you can create a dedicated environment for COLA:

#### Option A: From environment file (Recommended)

```bash
# Clone the repository first
git clone https://github.com/your-repo/COLA.git
cd COLA

# Create conda environment from file
conda env create -f environment.yml

# Activate the environment
conda activate cola
```

#### Option B: Manual conda installation

```bash
# Create a new environment
conda create -n cola python=3.9

# Activate the environment
conda activate cola

# Install COLA via pip in the conda environment
pip install xai-cola
```

### 3. From source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/your-repo/COLA.git
cd COLA

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Verifying Installation

After installation, verify that COLA is correctly installed:

```python
import xai_cola
print(xai_cola.__version__)

# Test basic import
from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model
print("COLA installation successful!")
```

You should see the version number and success message without any errors.

## Dependencies

### Core Dependencies

COLA requires the following core packages:

- `numpy >= 1.19.0`
- `pandas >= 1.1.0`
- `scikit-learn >= 0.23.0`
- `scipy >= 1.5.0`
- `matplotlib >= 3.3.0`
- `seaborn >= 0.11.0`
- `joblib >= 0.16.0`
- `shap >= 0.40.0`

### Optional Dependencies

Depending on your use case, you may need:

**For PyTorch models:**
- `torch >= 1.7.0`
- `torchvision >= 0.8.0`

**For TensorFlow models:**
- `tensorflow >= 2.4.0` (or `tensorflow-gpu`)

**For development:**
- `pytest >= 6.0.0`
- `pytest-cov >= 2.10.0`
- `black >= 20.8b1`
- `flake8 >= 3.8.0`
- `sphinx >= 3.5.0`

**For counterfactual explainers:**
- `dice-ml >= 0.8` (for DiCE)
- `alibi >= 0.6.0` (for Alibi-CFI)

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'xai_cola'

**Solution:**
- Ensure the package is installed: `pip list | grep xai-cola`
- If not listed, reinstall: `pip install xai-cola`
- Check you're in the correct Python environment: `which python`

#### 2. Version Conflicts

**Problem:** Dependency conflicts with existing packages.

**Solution:**
```bash
# Create a fresh environment
conda create -n cola_env python=3.9
conda activate cola_env
pip install xai-cola
```

#### 3. SHAP Installation Issues

**Problem:** SHAP may fail to build on some systems.

**Solution:**
```bash
# On Windows, you may need Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# On Linux/Mac, ensure you have build tools
sudo apt-get install build-essential  # Ubuntu/Debian
```

#### 4. PyTorch/TensorFlow Issues

**Problem:** CUDA/GPU support not working.

**Solution:**
```bash
# For PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For TensorFlow with GPU
pip install tensorflow[and-cuda]
```

#### 5. Permission Denied Errors

**Problem:** Permission errors during installation.

**Solution:**
```bash
# Use --user flag
pip install --user xai-cola

# Or use virtual environment (recommended)
python -m venv cola_venv
source cola_venv/bin/activate  # On Windows: cola_venv\Scripts\activate
pip install xai-cola
```

#### 6. Slow Installation

**Problem:** pip installation is very slow.

**Solution:**
```bash
# Use a mirror (example for China)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xai-cola

# Or upgrade pip
pip install --upgrade pip
```

### Platform-Specific Notes

#### Windows
- Some packages may require Microsoft Visual C++ 14.0 or greater
- Consider using Anaconda for easier dependency management

#### macOS
- Xcode Command Line Tools may be required: `xcode-select --install`
- For M1/M2 Macs, ensure you're using ARM-compatible packages

#### Linux
- Ubuntu/Debian: `sudo apt-get install python3-dev build-essential`
- CentOS/RHEL: `sudo yum install python3-devel gcc gcc-c++`

## Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](FAQ.md) for common questions
2. Search existing [GitHub Issues](https://github.com/your-repo/COLA/issues)
3. Create a new issue with:
   - Your Python version: `python --version`
   - Your OS and version
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. Check out the [Quick Start Guide](QUICKSTART.md)
2. Explore the [API Reference](API_REFERENCE.md)
3. Try the [example notebooks](examples/)
4. Read the [tutorials](docs/tutorials/)

## Updating COLA

To update to the latest version:

```bash
# Using pip
pip install --upgrade xai-cola

# From source
cd COLA
git pull
pip install -e .
```

To check your current version:

```python
import xai_cola
print(xai_cola.__version__)
```
