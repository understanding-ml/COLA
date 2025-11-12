============
Installation
============

COLA can be installed via pip or from source. Choose the method that best suits your needs.

Quick Install
=============

Install from PyPI (recommended):

.. code-block:: bash

    pip install xai-cola

This will install COLA and its core dependencies.

Installation Options
====================

Option 1: Install from PyPI
----------------------------

**For most users:**

.. code-block:: bash

    pip install xai-cola

**With specific dependencies:**

.. code-block:: bash

    # With PyTorch support
    pip install xai-cola torch

    # With TensorFlow support
    pip install xai-cola tensorflow

    # With all optional dependencies
    pip install xai-cola[all]

Option 2: Install from Source
------------------------------

**For contributors or latest development version:**

.. code-block:: bash

    git clone https://github.com/understanding-ml/COLA.git
    cd COLA
    pip install -e .

**Install with development dependencies:**

.. code-block:: bash

    pip install -e .
    pip install -r requirements.txt

Requirements
============

**Python Version:**

- Python >= 3.9
- Python < 3.13

**Core Dependencies (Required):**

- numpy >= 1.26.4, < 2.0
- pandas >= 2.0.0, <= 2.3.0
- scikit-learn >= 1.3.0, <= 1.7.0
- scipy >= 1.13.0, <= 1.16.0

**Counterfactual Explainers:**

- dice-ml >= 0.10, <= 0.12
- cem == 1.1.0

**Visualization:**

- matplotlib >= 3.8.0
- seaborn >= 0.13.0

**Feature Attributions:**

- shap >= 0.41.0

**Optimal Transport:**

- POT >= 0.9.0

**Progress Bar:**

- tqdm >= 4.67.0

**Deep learning model**

- torch >= 2.3.0 (for PyTorch models)

**Optional Dependencies:**

- jupyter (for notebooks)

Verifying Installation
======================

Test your installation:

.. code-block:: python

    import xai_cola
    print(xai_cola.__version__)

    # Test basic import
    from xai_cola import COLA
    from xai_cola.ce_sparsifier.data import COLAData
    from xai_cola.ce_sparsifier.models import Model
    from xai_cola.ce_generator import DiCE

    print("✓ COLA installed successfully!")

Quick Test
----------

Run a quick test with the built-in German Credit dataset:

.. code-block:: python

    from xai_cola.datasets.german_credit import GermanCreditDataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Load built-in dataset
    dataset = GermanCreditDataset()
    X_train, y_train, X_test, y_test = dataset.get_original_train_test_split()

    # Train a simple model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    pipe.fit(X_train, y_train)

    print(f"Model accuracy: {pipe.score(X_test, y_test):.3f}")
    print("✓ Everything works!")

Troubleshooting
===============

Issue 1: ImportError for POT
-----------------------------

**Error:**

.. code-block:: text

    ImportError: No module named 'ot'

**Solution:**

Install Python Optimal Transport:

.. code-block:: bash

    pip install POT

Issue 2: NumPy Version Conflict
--------------------------------

**Error:**

.. code-block:: text

    ImportError: numpy.core.multiarray failed to import

**Solution:**

Update NumPy:

.. code-block:: bash

    pip install --upgrade numpy

Issue 3: PyTorch Not Found
---------------------------

**Error:**

.. code-block:: text

    ModuleNotFoundError: No module named 'torch'

**Solution:**

Install PyTorch (if you need PyTorch support):

.. code-block:: bash

    # CPU version
    pip install torch

    # GPU version (CUDA 11.8)
    pip install torch --index-url https://download.pytorch.org/whl/cu118

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for more options.


Virtual Environment Setup
=========================

Using venv (Recommended)
------------------------

.. code-block:: bash

    # Create virtual environment
    python -m venv cola_env

    # Activate (Linux/Mac)
    source cola_env/bin/activate

    # Activate (Windows)
    cola_env\Scripts\activate

    # Install COLA
    pip install xai-cola

    # Deactivate when done
    deactivate

Using conda
-----------

.. code-block:: bash

    # Create conda environment
    conda create -n cola_env python=3.10

    # Activate environment
    conda activate cola_env

    # Install COLA
    pip install xai-cola

    # Deactivate when done
    conda deactivate

Docker (TBA)
-----------------

If you prefer Docker:

.. code-block:: dockerfile

    To be added in future releases.

Development Installation
========================

For contributors:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/understanding-ml/COLA.git
    cd COLA

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install in editable mode with dev dependencies
    pip install -e .
    pip install -r requirements.txt

    # Install pre-commit hooks (optional)
    pip install pre-commit
    pre-commit install

    # Run tests
    pytest tests/

Upgrading
=========

Upgrade to the latest version:

.. code-block:: bash

    pip install --upgrade xai-cola

Check current version:

.. code-block:: python

    import xai_cola
    print(xai_cola.__version__)

Uninstallation
==============

Remove COLA:

.. code-block:: bash

    pip uninstall xai-cola

Next Steps
==========

After installation:

1. :doc:`quickstart` - 5-minute quick start guide
2. :doc:`tutorials/01_basic_tutorial` - Complete tutorial
3. :doc:`user_guide/data_interface` - Learn about data management

Getting Help
============

If you encounter issues:

1. Check :doc:`faq` - Common questions
2. Search `GitHub Issues <https://github.com/understanding-ml/COLA/issues>`_
3. Open a new issue with:
   - Python version
   - COLA version
   - Full error message
   - Minimal code to reproduce

Platform-Specific Notes
=======================

Windows
-------

- Use PowerShell or Command Prompt
- Path separators are ``\`` instead of ``/``
- Some dependencies may require Visual C++ Build Tools

macOS
-----

- May need to install Xcode Command Line Tools: ``xcode-select --install``
- For M1/M2 Macs, ensure you're using ARM-compatible packages

Linux
-----

- May need to install build essentials: ``sudo apt-get install build-essential``
- For GPU support, ensure CUDA is properly installed

See Also
========

- `PyPI Page <https://pypi.org/project/xai-cola/>`_
- `GitHub Repository <https://github.com/understanding-ml/COLA>`_
- `Requirements File <https://github.com/understanding-ml/COLA/blob/main/requirements.txt>`_
